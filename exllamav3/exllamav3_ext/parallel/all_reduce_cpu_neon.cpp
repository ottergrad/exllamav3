// aarch64 (NEON) implementation of the CPU-helper all-reduce accumulate path.
//
// The x86 code implements perform_cpu_reduce in all_reduce_cpu_avx2.cpp (which dispatches
// AVX-512 vs AVX2). Here we provide the single aarch64 definition of perform_cpu_reduce, which
// runs the identical chunked reduce loop but accumulates BF16/FP16 with NEON instead of AVX.
//
// This TU is compiled only for aarch64 (guarded below). On x86 the x86 TU provides the symbol.

#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include <torch/extension.h>   // TORCH_CHECK (host cpp)
#include "all_reduce_cpu_avx2.h"   // cpusum_* helpers, atomic_ref, MAX_DEVICES, perform_cpu_reduce decl
#include "all_reduce_cpu_avx512.h"
#include "../arm_target.h"
#include "../util.h"

// Portable pause/barrier for ARM spin loops (analog of _mm_pause()).
static inline void cpu_pause_neon()
{
    __asm__ volatile("" ::: "memory");
}

// A += B (BF16), 8 values. Integer path mirrors the AVX2 do16(): widen BF16->uint32, shift
// left 16 to form the fp32 bit pattern, add in 32-bit, round-to-nearest (+0x8000), shift
// right 16, narrow back to uint16.
static inline void bf16_add8_neon(uint16_t* __restrict ap, const uint16_t* __restrict bp)
{
    uint16x8_t va = vld1q_u16(ap);
    uint16x8_t vb = vld1q_u16(bp);

    uint32x4_t a_lo = vshlq_n_u32(vmovl_u16(vget_low_u16(va)), 16);
    uint32x4_t a_hi = vshlq_n_u32(vmovl_u16(vget_high_u16(va)), 16);
    uint32x4_t b_lo = vshlq_n_u32(vmovl_u16(vget_low_u16(vb)), 16);
    uint32x4_t b_hi = vshlq_n_u32(vmovl_u16(vget_high_u16(vb)), 16);

    uint32x4_t s_lo = vaddq_u32(a_lo, b_lo);
    uint32x4_t s_hi = vaddq_u32(a_hi, b_hi);
    const uint32x4_t rnd = vdupq_n_u32(0x8000);

    uint16x4_t o_lo = vmovn_u32(vshrq_n_u32(vaddq_u32(s_lo, rnd), 16));
    uint16x4_t o_hi = vmovn_u32(vshrq_n_u32(vaddq_u32(s_hi, rnd), 16));
    vst1q_u16(ap, vcombine_u16(o_lo, o_hi));
}

// A += B (BF16), in-place, round-toward-zero. count % 32 == 0.
static inline void bf16_add_inplace_neon(uint16_t* __restrict a, const uint16_t* __restrict b, size_t count)
{
    size_t i = 0;
    for (; i + 8 <= count; i += 8)
        bf16_add8_neon(a + i, b + i);
}

// A += B (FP16), 8 values. Widen via fp32, add, narrow with round-to-nearest.
static inline void fp16_add8_neon(uint16_t* __restrict ap, const uint16_t* __restrict bp)
{
    float16x4_t a_lo = vcvt_f16_f32(vcvt_f32_f16(vld1_f16((const __fp16*)(ap))));
    float16x4_t a_hi = vcvt_f16_f32(vcvt_f32_f16(vld1_f16((const __fp16*)(ap + 4))));
    float16x4_t b_lo = vcvt_f16_f32(vcvt_f32_f16(vld1_f16((const __fp16*)(bp))));
    float16x4_t b_hi = vcvt_f16_f32(vcvt_f32_f16(vld1_f16((const __fp16*)(bp + 4))));
    float32x4_t sa_lo = vcvt_f32_f16(a_lo), sa_hi = vcvt_f32_f16(a_hi);
    float32x4_t sb_lo = vcvt_f32_f16(b_lo), sb_hi = vcvt_f32_f16(b_hi);
    vst1_f16((__fp16*)(ap),     vcvt_f16_f32(vaddq_f32(sa_lo, sb_lo)));
    vst1_f16((__fp16*)(ap + 4), vcvt_f16_f32(vaddq_f32(sa_hi, sb_hi)));
}

// A += B (FP16), in-place. count % 32 == 0.
static inline void fp16_add_inplace_neon(uint16_t* __restrict a, const uint16_t* __restrict b, size_t count)
{
    size_t i = 0;
    for (; i + 8 <= count; i += 8)
        fp16_add8_neon(a + i, b + i);
}

// Perform reduction on current job (NEON). Mirrors perform_cpu_reduce_avx2().
void perform_cpu_reduce
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
    uint32_t wire_dtype,
    uint8_t* shbuf_ptr,
    size_t shbuf_size
)
{
    const uint32_t buf_slot_size = (shbuf_size / (MAX_DEVICES + 1) / 1024) * 1024;
    const uint32_t max_buf_stages = buf_slot_size / CPUREDUCE_CHUNK_SIZE;
    TORCH_CHECK(max_buf_stages >= 2, "Shared buffer too small for chunk size (need at least 2 stages)");
    auto host_ptr = [&] (int device, uint32_t stage_idx)
    {
        return shbuf_ptr + buf_slot_size * device + (stage_idx % max_buf_stages) * CPUREDUCE_CHUNK_SIZE;
    };

    int num_chunks = (int) CEIL_DIVIDE(data_size, CPUREDUCE_CHUNK_SIZE);
    size_t rem_data_size = data_size;
    const bool multi = num_chunks > 1;

    atomic_ref<uint32_t> stage_(&ctx->cpusum_stage_cpu);
    uint32_t stage = stage_.load_acquire();
    uint32_t next_stage = (stage + 1u) & 0x7fffffffu;

    const auto start = std::chrono::high_resolution_clock::now();

    int chunk_idx = 0;
    while (num_chunks)
    {
        size_t stage_size = MIN(rem_data_size, CPUREDUCE_CHUNK_SIZE);
        rem_data_size = MAX(rem_data_size - stage_size, 0);

        if (multi && chunk_idx >= (int)max_buf_stages)
        {
            uint32_t recv_target = ((stage - max_buf_stages) & 0x7fffffffu) + 1u;
            int throttle_spin = 0;
            while (!cpusum_recv_ready(ctx, device_mask, recv_target))
            {
                cpu_pause_neon();
                if (++throttle_spin > 10000)
                {
                    throttle_spin = 0;
                    const auto now = std::chrono::high_resolution_clock::now();
                    const std::chrono::duration<double, std::milli> elapsed = now - start;
                    if (elapsed > std::chrono::duration<double, std::milli>(45000.0))
                    {
                        printf(" ## CPU reduce process timeout (recv throttle)\n");
                        TORCH_CHECK(false, "CPU reduce process timeout");
                    }
                }
            }
        }

        uint32_t rem_devices = device_mask;
        bool first_contribution = true;
        int timeout_spin = 0;
        while (true)
        {
            for (int device = 0; device < MAX_DEVICES; ++device)
            {
                if (!(rem_devices & (1 << device))) continue;

                bool no_contrib;
                if (cpusum_device_arrived(ctx, device, stage, multi, &no_contrib))
                {
                    rem_devices &= ~(1 << device);
                    if (!no_contrib)
                    {
                        uint8_t* src = host_ptr(device, stage);
                        uint8_t* dst = host_ptr(MAX_DEVICES, stage);
                        if (first_contribution)
                        {
                            memcpy(dst, src, stage_size);
                            first_contribution = false;
                        }
                        else
                        {
                            size_t elem_count = CEIL_DIVIDE(stage_size, 64) * 32;
                            if (wire_dtype == REDUCE_WIRE_FP16)
                                fp16_add_inplace_neon((uint16_t*) dst, (uint16_t*) src, elem_count);
                            else
                                bf16_add_inplace_neon((uint16_t*) dst, (uint16_t*) src, elem_count);
                        }
                    }
                }
            }
            if (!rem_devices) break;

            cpu_pause_neon();
            timeout_spin++;
            if (timeout_spin > 10000)
            {
                timeout_spin = 0;
                const auto now = std::chrono::high_resolution_clock::now();
                const std::chrono::duration<double, std::milli> elapsed = now - start;
                if (elapsed > std::chrono::duration<double, std::milli>(45000.0))
                {
                    printf(" ## CPU reduce process timeout\n");
                    TORCH_CHECK(false, "CPU reduce process timeout");
                }
            }
        }

        stage = next_stage;
        next_stage = (stage + 1u) & 0x7fffffffu;
        stage_.store_release(stage);

        num_chunks--;
        chunk_idx++;
    }
}

#endif // defined(__aarch64__) || defined(_M_ARM64)
