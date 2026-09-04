"""
Regression / functional test for the aarch64 CPU-MoE implementation in
exllamav3_ext.cpu.moe_mul1_arm.cpp.

The ARM port replaces the previous x86-only no-op stub with a real scalar/NEON
implementation. This test is CPU-only and runnable without a GPU:

    PATH="/usr/local/cuda/bin:$PWD/.venv/bin:$PATH" \
    TORCH_EXTENSIONS_DIR=/tmp/torch_ext TORCH_CUDA_ARCH_LIST="10.0+PTX" \
    .venv/bin/python tests/test_cpu_moe_arm.py

The extension loads via torch's JIT (`exllamav3.ext`), so the run needs the same
environment as a build (`ninja` on PATH, and CUDA visible or `TORCH_CUDA_ARCH_LIST`
set; run outside any sandbox that hides `/dev/nvidia*`). On x86 the underlying kernel
(`moe_mul1.cpp`) is exercised instead — the assertions are dtype/boundary-based there.

It builds a deterministic layer (torch.manual_seed fixed), runs the CPU MoE forward
across the gated and gateless configurations and exercises multiple threads, then asserts
the output against a deterministic golden captured from the (now validated) ARM implementation.
A regression in the ARM kernel (e.g. the hadamard_128 butterfly bug that corrupted
output scale and the heap) changes these values by orders of magnitude, so the bounds
below are meaningful. The first test's golden is exact for the ARM scalar path only; on x86
(VNNI/VBMI) results differ in scale, so that single test is skipped unless running on aarch64.
"""

import os
import platform
import sys

import torch

try:
    import pytest
    _HAVE_PYTEST = True
except ImportError:
    _HAVE_PYTEST = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.ext import exllamav3_ext as ext

IS_ARM = platform.machine() in ("aarch64", "arm64")

if _HAVE_PYTEST:
    def _skipif(cond, reason):
        return pytest.mark.skipif(cond, reason=reason)
else:
    class _skipif:
        def __init__(self, cond, reason):
            self.cond = cond
            self.reason = reason
        def __call__(self, fn):
            return fn


def build_layer(k, n, bits, n_experts, activation, act_limit=0.0):
    """Create a deterministic layer (seeded RNG must already be fixed by caller)."""
    trellis = torch.randint(0, 1 << bits,
                           (k // 16, n // 16, 16 * bits)).to(torch.uint16).contiguous()
    suh = torch.randn(k).half().contiguous()
    svh = torch.randn(n).half().contiguous()
    gated = activation != 2
    gate = ([trellis.clone() for _ in range(n_experts)],
            [suh.clone() for _ in range(n_experts)],
            [svh.clone() for _ in range(n_experts)]) if gated else ([], [], [])
    up = ([trellis.clone() for _ in range(n_experts)],
          [suh.clone() for _ in range(n_experts)],
          [svh.clone() for _ in range(n_experts)])
    down = ([trellis.clone() for _ in range(n_experts)],
            [svh.clone() for _ in range(n_experts)],   # k of down == n of up
            [suh.clone() for _ in range(n_experts)])   # n of down == k of up
    return ext.exl3_moe_cpu_make_layer(
        *gate, *up, *down, [], [], [], activation, act_limit, 0)


def _forward(h, x, sel, w, N, threads):
    out = torch.zeros(x.size(0), N, dtype=torch.float32)
    ext.exl3_moe_cpu_forward(h, x, sel, w, out, threads)
    return out


def _check_close(actual, expected, rtol):
    diff = (actual - expected).abs()
    scale = expected.abs() + 1e-8
    assert bool((diff <= rtol * scale).all()), (
        f"output deviates from golden: max rel diff "
        f"{(diff / scale).max().item():.6g}")


@_skipif(not IS_ARM, reason="exact golden is specific to the aarch64 scalar path")
def test_gated_golden_single_thread():
    """Deterministic gated layer, single-thread forward must match captured golden."""
    torch.manual_seed(42)
    K = N = 128
    handle = build_layer(K, N, 1, 2, activation=0)
    x = torch.randn(6, K, dtype=torch.float16)
    sel = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]],
                      dtype=torch.int64)
    w = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.5, 0.5],
                     [0.9, 0.1], [0.2, 0.8], [0.6, 0.4]],
                    dtype=torch.float16)
    out = _forward(handle, x, sel, w, N, 1)

    assert not torch.isnan(out).any() and torch.isfinite(out).all().item()

    golden_norm = 43584828.0
    assert abs(out.norm().item() - golden_norm) / golden_norm < 1e-5

    golden_first = [
        -4372871.0, -80179.594, 132148.188, -11111.889, 11857.686,
        -7800.341, 127838.297, -24495.729, 30080.266, -1710.703,
        37.592, -511.501, -56473.875, 10248.144, -668.789, -3608.524,
        25457.773, 27978.697, -78840.422, 121470.023, -2356.354,
        15274.017, 40813.586, 37874.836,
    ]
    _check_close(out.flatten()[:24], torch.tensor(golden_first), rtol=1e-4)
    ext.exl3_moe_cpu_free_layer(handle)


def test_multi_thread_matches_single_thread():
    """Multi-thread result must be close to the single-thread result and NaN-free."""
    torch.manual_seed(123)
    K = N = 128
    handle = build_layer(K, N, 2, 3, activation=3, act_limit=6.0)
    x = torch.randn(8, K, dtype=torch.float16)
    sel = torch.randint(0, 3, (8, 2), dtype=torch.int64)
    w = torch.rand(8, 2, dtype=torch.float16) * 0.5
    single = _forward(handle, x, sel, w, N, 1)
    multi = _forward(handle, x, sel, w, N, 2)
    assert not torch.isnan(multi).any() and torch.isfinite(multi).all().item()
    # tolerate fp ordering differences under different thread counts
    _check_close(multi, single, rtol=1e-2)
    ext.exl3_moe_cpu_free_layer(handle)


def _max_abs(o):
    return float(o.abs().max().item())


def _assert_act_limit_active(activation, act_limit, scale):
    """A nonzero act_limit must change silu/gelu (cases 0/1) kernel output.

    Mirrors the x86 moe_mul1.cpp clamp (min(g_act, lim) and clamp(u, -lim, lim)):
    without it, a large-magnitude input saturates the activation and up path, so the
    unlimited and clamped forward must differ. A regressed ARM kernel that drops the
    act_limit for cases 0/1 yields identical outputs and fails here.
    """
    torch.manual_seed(99)
    K = N = 128
    hi = build_layer(K, N, 2, 3, activation=activation, act_limit=act_limit)
    lo = build_layer(K, N, 2, 3, activation=activation, act_limit=0.0)
    x = (torch.randn(6, K, dtype=torch.float16) * scale).contiguous()
    sel = torch.randint(0, 3, (6, 2), dtype=torch.int64)
    w = torch.rand(6, 2, dtype=torch.float16)
    out_hi = _forward(hi, x, sel, w, N, 2)
    out_lo = _forward(lo, x, sel, w, N, 2)
    assert not torch.isnan(out_hi).any() and torch.isfinite(out_hi).all().item()
    assert not torch.isnan(out_lo).any() and torch.isfinite(out_lo).all().item()
    assert _max_abs(out_hi) > 0
    # force |g| and |u| well past the clamp so the limit is observable
    assert _max_abs(out_lo) > act_limit
    diff = (out_hi - out_lo).abs().max().item()
    assert diff > 0.5 * act_limit, (
        f"act_limit={act_limit} has no effect on activation {activation} "
        f"(max diff {diff:.4g})")


def test_act_limit_silu():
    """Case 0 (silu) honours the act_limit clamp like the x86 kernel."""
    _assert_act_limit_active(0, act_limit=10.0, scale=50.0)


def test_act_limit_gelu():
    """Case 1 (gelu) honours the act_limit clamp like the x86 kernel."""
    _assert_act_limit_active(1, act_limit=10.0, scale=50.0)


def test_gateless_relu2():
    """Activation 2 (gateless, relu2) path must run cleanly."""
    torch.manual_seed(7)
    K = N = 128
    handle = build_layer(K, N, 1, 2, activation=2)
    x = torch.randn(4, K, dtype=torch.float16)
    sel = torch.randint(0, 2, (4, 1), dtype=torch.int64)
    w = torch.rand(4, 1, dtype=torch.float16)
    out = _forward(handle, x, sel, w, N, 3)
    assert not torch.isnan(out).any() and torch.isfinite(out).all().item()
    assert out.norm().item() > 0
    # no heap corruption / double-free on teardown
    for _ in range(3):
        ext.exl3_moe_cpu_forward(handle, x, sel, w, out, 3)
    ext.exl3_moe_cpu_free_layer(handle)


if __name__ == "__main__":
    if not IS_ARM:
        print("SKIPPED exact-golden test (x86 VNNI/VBMI differs in scale); running the rest")
    else:
        test_gated_golden_single_thread()
    test_act_limit_silu()
    test_act_limit_gelu()
    test_multi_thread_matches_single_thread()
    test_gateless_relu2()
    print("ALL CPU-MoE ARM TESTS PASSED")
