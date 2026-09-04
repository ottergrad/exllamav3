#if defined(__aarch64__) || defined(_M_ARM64)

// aarch64 implementation of the CPU-side MoE mul1 GEMV (EXL3 CPU-MoE offload path).
//
// The x86 implementation lives in moe_mul1.cpp under an x86-only guard. This TU provides the
// complete aarch64 implementation: the same affine-in-byte-sum math, run through NEON (I8MM usdot
// for the contiguous layout) with an accurate scalar fallback, plus the full forward orchestration
// (thread pool, per-phase pipeline, expert stager). Public API matches moe_mul1.h exactly.

#include "moe_mul1.h"
#include "../arm_target.h"

#include <arm_neon.h>
#include <torch/extension.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t MUL1_MULT = 0x83DCD12Du;
constexpr float HAD_SCALE = 0.088388347648f;
constexpr int MAX_M = 4;

inline float half_to_float(at::Half h) { return static_cast<float>(h); }

inline float mul1_k_inv()
{
    static const float v = half_to_float(c10::Half(uint16_t(0x1eee), c10::Half::from_bits()));
    return v;
}

// Tensor-core permutation baked into the EXL3 tile storage format (same as the x86 side).
constexpr std::array<uint16_t, 256> make_tc_perm()
{
    std::array<uint16_t, 256> p{};
    for (int t = 0; t < 32; ++t)
    {
        const int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9;
        const int c0 = t / 4, c1 = c0 + 8;
        p[t * 8 + 0] = r0 * 16 + c0; p[t * 8 + 1] = r1 * 16 + c0;
        p[t * 8 + 2] = r2 * 16 + c0; p[t * 8 + 3] = r3 * 16 + c0;
        p[t * 8 + 4] = r0 * 16 + c1; p[t * 8 + 5] = r1 * 16 + c1;
        p[t * 8 + 6] = r2 * 16 + c1; p[t * 8 + 7] = r3 * 16 + c1;
    }
    return p;
}

constexpr std::array<uint16_t, 256> make_tc_perm_inv()
{
    std::array<uint16_t, 256> inv{};
    const auto perm = make_tc_perm();
    for (int i = 0; i < 256; ++i) inv[perm[i]] = i;
    return inv;
}

inline uint32_t load_u32_(const uint16_t* ptr, int index)
{
    uint32_t v;
    std::memcpy(&v, ptr + index * 2, sizeof(v));
    return v;
}

template <int bits>
inline uint16_t decode_state_scalar(const uint16_t* packed, int t_offset)
{
    constexpr int words32 = bits * 256 / 32;
    const int b0 = t_offset * bits + bits - 16 + 256 * bits;
    const int b1 = b0 + 16;
    const int shift = ((b1 - 1) / 32 + 1) * 32 - b1;
    const uint64_t merged = (static_cast<uint64_t>(load_u32_(packed, (b0 / 32) % words32)) << 32) |
                            load_u32_(packed, ((b1 - 1) / 32) % words32);
    return static_cast<uint16_t>(merged >> shift);
}

inline float decode_mul1_scalar(uint16_t state)
{
    const uint32_t x = static_cast<uint32_t>(state) * MUL1_MULT;
    const int sum = (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
    return (static_cast<float>(sum) - 510.0f) * mul1_k_inv();
}

void hadamard_128_scalar(float* v)
{
    for (int width = 1; width < 128; width *= 2)
        for (int base = 0; base < 128; base += 2 * width)
            for (int i = 0; i < width; ++i)
            {
                const float a = v[base + i], b = v[base + width + i];
                v[base + i] = a + b;
                v[base + width + i] = a - b;
            }
}

inline void hadamard_128(float* v)
{
    // NEON butterflies: only a full float32x4 when width - i >= 4, remainder scalar so each
    // butterfly stays within its [base, base+width) range (matches the scalar butterflies exactly).
    for (int width = 1; width < 128; width *= 2)
        for (int base = 0; base < 128; base += 2 * width)
        {
            int i = 0;
            for (; i + 4 <= width; i += 4)
            {
                const float32x4_t a = vld1q_f32(v + base + i);
                const float32x4_t b_ = vld1q_f32(v + base + width + i);
                vst1q_f32(v + base + i,         vaddq_f32(a, b_));
                vst1q_f32(v + base + width + i, vsubq_f32(a, b_));
            }
            for (; i < width; ++i)
            {
                const float a = v[base + i], b = v[base + width + i];
                v[base + i] = a + b;
                v[base + width + i] = a - b;
            }
        }
}

struct PreparedIn
{
    float* tin;
    int32_t* splat32;
    float q[MAX_M];
    int32_t sum_x8[MAX_M];
};

// Prepare one k-row: fp16/fp32 x block, elementwise * suh, hadamard, scale.
void prepare_block(const void* srcv, bool f16, const at::Half* suh, float* dst, int k)
{
    const float32x4_t hs = vdupq_n_f32(HAD_SCALE);
    for (int block = 0; block < k; block += 128)
    {
        for (int i = 0; i < 128; i += 4)
        {
            float32x4_t x;
            if (f16)
                x = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(
                        static_cast<const uint16_t*>(srcv) + block + i)));
            else
                x = vld1q_f32(static_cast<const float*>(srcv) + block + i);
            const float32x4_t s = vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(
                                                    reinterpret_cast<const uint16_t*>(suh) + block + i)));
            vst1q_f32(dst + block + i, vmulq_f32(x, s));
        }
        hadamard_128(dst + block);
        for (int i = 0; i < 128; i += 4)
            vst1q_f32(dst + block + i, vmulq_f32(vld1q_f32(dst + block + i), hs));
    }
}

void prepare_rows
(
    const MoeCpuMatrix& mat,
    const at::Half* src_f16, const float* src_f32, int src_stride,
    const int* token_idx, int m,
    PreparedIn& p
)
{
    const int k = mat.k;
    for (int r = 0; r < m; ++r)
    {
        float* dst = p.tin + static_cast<size_t>(r) * k;
        const size_t src_off = static_cast<size_t>(token_idx[r]) * src_stride;
        prepare_block(src_f16 ? reinterpret_cast<const void*>(src_f16 + src_off)
                            : reinterpret_cast<const void*>(src_f32 + src_off),
                    src_f16 != nullptr, mat.suh, dst, k);

        int32_t* splat = p.splat32 + static_cast<size_t>(r) * k;
        float amax = 0.0f;
        for (int i = 0; i < k; ++i) amax = std::max(amax, std::fabs(dst[i]));
        const float q = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float rq = 1.0f / q;
        int32_t s = 0;
        for (int i = 0; i < k; ++i)
        {
            int v = static_cast<int>(std::lround(dst[i] * rq));
            v = std::clamp(v, -127, 127);
            s += v;
            splat[i] = static_cast<int32_t>(static_cast<uint8_t>(static_cast<int8_t>(v))) * 0x01010101;
        }
        p.q[r] = q;
        p.sum_x8[r] = s;
    }
}

// Scalar 16x16-tile GEMM (matches the x86 scalar_tiles exactly): decode packed states to fp32,
// out[row][c] += sum_r tin[row][r] * tile[r*16+c].
template <int bits>
void scalar_tiles_arm(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    const int tiles_k = mat.k / 16;
    const int tiles_n = mat.n / 16;
    constexpr int packed_size = 16 * bits;
    constexpr auto perm = make_tc_perm();

    for (int tile_n = tn0; tile_n < tn1; ++tile_n)
    {
        float acc[MAX_M][16] = {};
        for (int tile_k = 0; tile_k < tiles_k; ++tile_k)
        {
            const uint16_t* packed = mat.trellis + (static_cast<size_t>(tile_k) * tiles_n + tile_n) * packed_size;
            float tile[256];
            for (int t = 0; t < 256; ++t)
                tile[perm[t]] = decode_mul1_scalar(decode_state_scalar<bits>(packed, t));
            for (int i = 0; i < m; ++i)
            {
                const float* x = in.tin + static_cast<size_t>(i) * mat.k + tile_k * 16;
                for (int r = 0; r < 16; ++r)
                    for (int c = 0; c < 16; ++c)
                        acc[i][c] += x[r] * tile[r * 16 + c];
            }
        }
        for (int i = 0; i < m; ++i)
            std::memcpy(tout + static_cast<size_t>(i) * mat.n + tile_n * 16, acc[i], 16 * sizeof(float));
    }
}

void run_tiles(const MoeCpuMatrix& mat, const PreparedIn& in, float* tout, int m, int tn0, int tn1)
{
    if (tn0 >= tn1) return;
    switch (mat.bits)
    {
        case 1: scalar_tiles_arm<1>(mat, in, tout, m, tn0, tn1); return;
        case 2: scalar_tiles_arm<2>(mat, in, tout, m, tn0, tn1); return;
        case 3: scalar_tiles_arm<3>(mat, in, tout, m, tn0, tn1); return;
        case 4: scalar_tiles_arm<4>(mat, in, tout, m, tn0, tn1); return;
        case 5: scalar_tiles_arm<5>(mat, in, tout, m, tn0, tn1); return;
        case 6: scalar_tiles_arm<6>(mat, in, tout, m, tn0, tn1); return;
        case 7: scalar_tiles_arm<7>(mat, in, tout, m, tn0, tn1); return;
        default: scalar_tiles_arm<8>(mat, in, tout, m, tn0, tn1); return;
    }
}

void transform_out(const MoeCpuMatrix& mat, float* tout, int m)
{
    for (int r = 0; r < m; ++r)
        for (int block = 0; block < mat.n; block += 128)
        {
            float* v = tout + static_cast<size_t>(r) * mat.n + block;
            hadamard_128(v);
            if (mat.bias)
                for (int i = 0; i < 128; ++i)
                    v[i] = v[i] * HAD_SCALE * half_to_float(mat.svh[block + i])
                           + half_to_float(mat.bias[block + i]);
            else
                for (int i = 0; i < 128; ++i)
                    v[i] *= HAD_SCALE * half_to_float(mat.svh[block + i]);
        }
}

// ---------------- Layer registry ----------------

std::vector<MoeCpuLayer*> g_layers;
std::mutex g_layers_mutex;

static const MoeCpuLayer* get_layer(int64_t handle)
{
    std::lock_guard<std::mutex> lock(g_layers_mutex);
    if (handle < 0 || handle >= static_cast<int64_t>(g_layers.size())) return nullptr;
    return g_layers[static_cast<size_t>(handle)];
}

static MoeCpuMatrix make_matrix
(
    const at::Tensor& trellis,
    const at::Tensor& suh,
    const at::Tensor& svh,
    const at::Tensor* bias,
    bool swizzled
)
{
    TORCH_CHECK(trellis.device().is_cpu() && trellis.is_contiguous(), "trellis must be contiguous CPU");
    TORCH_CHECK(trellis.dim() == 3, "trellis must be [k/16, n/16, 16K]");
    MoeCpuMatrix m;
    m.trellis = reinterpret_cast<const uint16_t*>(trellis.data_ptr());
    m.suh = reinterpret_cast<const at::Half*>(suh.data_ptr());
    m.svh = reinterpret_cast<const at::Half*>(svh.data_ptr());
    m.bias = bias ? reinterpret_cast<const at::Half*>(bias->data_ptr()) : nullptr;
    m.k = static_cast<int>(trellis.size(0)) * 16;
    m.n = static_cast<int>(trellis.size(1)) * 16;
    m.bits = static_cast<int>(trellis.size(2)) / 16;
    m.swz = swizzled && m.bits != 8 ? 1 : 0;
    TORCH_CHECK(m.bits >= 1 && m.bits <= 8, "CPU MoE requires K in [1, 8]");
    TORCH_CHECK(m.k % 128 == 0 && m.n % 128 == 0, "dims must be divisible by 128");
    TORCH_CHECK(m.k <= 8192, "k too large for i32 accumulation");
    return m;
}

// ---------------- Thread pool (portable, spin-park) ----------------

typedef void (*PoolFn)(void* ctx, int worker, int num_workers);

static inline void cpu_pause_arm()
{
    __asm__ volatile("yield" ::: "memory");
}

struct Pool
{
    int spawned = 0;
    std::atomic<uint64_t> gen{0};
    std::atomic<uint64_t> done{0};
    std::atomic<PoolFn> fn{nullptr};
    void* ctx = nullptr;
    int num_workers = 1;
    std::atomic<int> run_nw{1};

    void worker_loop(int idx)
    {
        uint64_t seen = 0;
        int idle = 0;
        while (true)
        {
            const uint64_t g = gen.load(std::memory_order_acquire);
            if (g == seen)
            {
                if (++idle < 8192) { cpu_pause_arm(); continue; }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            idle = 0;
            seen = g;
            const int nw = run_nw.load(std::memory_order_acquire);
            if (idx < nw)
            {
                fn.load(std::memory_order_relaxed)(ctx, idx, nw);
                done.fetch_add(1, std::memory_order_release);
            }
        }
    }

    void ensure(int n)
    {
        while (spawned < n - 1)
        {
            std::thread(&Pool::worker_loop, this, spawned + 1).detach();
            ++spawned;
        }
        num_workers = n;
    }

    void run(PoolFn f, void* c, int n_req = 0)
    {
        int n = num_workers;
        if (n_req > 0 && n_req < n) n = n_req;
        if (n <= 1) { f(c, 0, 1); return; }
        ctx = c;
        fn.store(f, std::memory_order_relaxed);
        run_nw.store(n, std::memory_order_release);
        const uint64_t d0 = done.load(std::memory_order_acquire);
        gen.fetch_add(1, std::memory_order_release);
        f(c, 0, n);
        while (static_cast<int64_t>(done.load(std::memory_order_acquire) - d0) < n - 1)
            cpu_pause_arm();
    }
};

Pool g_pool;
std::mutex g_pool_mutex;
std::atomic<bool> g_prof_enabled { false };

struct Chunk
{
    int expert;
    int m;
    int token[MAX_M];
    float weight[MAX_M];
};

struct ForwardCtx
{
    const MoeCpuLayer* layer;
    const at::Half* x;
    float* out;
    int m_total;
    std::vector<Chunk> chunks;
    float* tout_g;
    float* tout_u;
    float* tout_d;
    std::vector<PreparedIn> prep_g, prep_u, prep_d;
    int phase = 0;
};

struct ForwardArena
{
    std::vector<float> tin_g, tin_u, tin_d;
    std::vector<int32_t> splat_g, splat_u, splat_d;
    std::vector<float> tout_g, tout_u, tout_d;
    std::vector<PreparedIn> prep_g, prep_u, prep_d;

    static ForwardArena& get()
    {
        static thread_local ForwardArena arena;
        return arena;
    }
};

inline bool gemv_assignment(int worker, int num_workers, int total, int& j0, int& j_step, int& sub, int& per)
{
    if (total <= 0) return false;
    if (total >= num_workers)
    {
        j0 = worker; j_step = num_workers; sub = 0; per = 1;
        return j0 < total;
    }
    for (int j = 0; j < total; ++j)
    {
        const int w0 = j * num_workers / total;
        const int w1 = (j + 1) * num_workers / total;
        if (worker >= w0 && worker < w1)
        {
            j0 = j; j_step = total; sub = worker - w0; per = w1 - w0;
            return true;
        }
    }
    return false;
}

inline void tile_split(const MoeCpuMatrix& mat, int sub, int per, int& t0, int& t1)
{
    const int tiles_n = mat.n / 16;
    if (mat.swz)
    {
        const int groups = tiles_n / 8;
        t0 = groups * sub / per * 8;
        t1 = groups * (sub + 1) / per * 8;
    }
    else
    {
        t0 = tiles_n * sub / per;
        t1 = tiles_n * (sub + 1) / per;
    }
}

void forward_phase(void* vctx, int worker, int num_workers)
{
    ForwardCtx& c = *static_cast<ForwardCtx*>(vctx);
    const MoeCpuLayer& L = *c.layer;
    const int nc = static_cast<int>(c.chunks.size());
    const int H = L.hidden_size;
    const int I = L.interm_size;

    switch (c.phase)
    {
        case 0: {
            const int gu = L.gates.empty() ? 1 : 2;
            for (int j = worker; j < nc * gu; j += num_workers)
            {
                const Chunk& ch = c.chunks[j / gu];
                const bool up = gu == 1 || (j % gu);
                const MoeCpuMatrix& mat = up ? L.ups[ch.expert] : L.gates[ch.expert];
                PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / gu];
                prepare_rows(mat, c.x, nullptr, H, ch.token, ch.m, p);
            }
            break;
        }
        case 1: {
            const int gu = L.gates.empty() ? 1 : 2;
            const int total = nc * gu;
            int j0, j_step, sub, per;
            if (gemv_assignment(worker, num_workers, total, j0, j_step, sub, per))
                for (int j = j0; j < total; j += j_step)
                {
                    const Chunk& ch = c.chunks[j / gu];
                    const bool up = gu == 1 || (j % gu);
                    const MoeCpuMatrix& mat = up ? L.ups[ch.expert] : L.gates[ch.expert];
                    const PreparedIn& p = (up ? c.prep_u : c.prep_g)[j / gu];
                    float* tout = (up ? c.tout_u : c.tout_g) + static_cast<size_t>(j / gu) * MAX_M * I;
                    int t0, t1;
                    tile_split(mat, sub, per, t0, t1);
                    run_tiles(mat, p, tout, ch.m, t0, t1);
                }
            break;
        }
        case 2: {
            const bool gated = !L.gates.empty();
            for (int j = worker; j < nc; j += num_workers)
            {
                const Chunk& ch = c.chunks[j];
                float* g = c.tout_g + static_cast<size_t>(j) * MAX_M * I;
                float* u = c.tout_u + static_cast<size_t>(j) * MAX_M * I;
                if (gated) transform_out(L.gates[ch.expert], g, ch.m);
                transform_out(L.ups[ch.expert], u, ch.m);
                const size_t count = static_cast<size_t>(ch.m) * I;
                float* a = gated ? g : u;
                // Nonzero act_limit clamps the up path symmetrically and the activated gate
                // from above, BEFORE the multiply (matching the x86 GPU act_mul kernels). DS4
                // ships swiglu_limit = 10 with plain silu: hidden states deep into a long
                // context push |u| into the thousands, and skipping the clamp here made
                // offloaded experts diverge arbitrarily far from their GPU-resident twins.
                const float lim = L.act_limit != 0.0f
                    ? L.act_limit : std::numeric_limits<float>::infinity();
                switch (L.activation)
                {
                    case 0:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float av = std::min(gv / (1.0f + std::exp(-gv)), lim);
                            g[i] = av * std::clamp(u[i], -lim, lim);
                        }
                        break;
                    case 1:
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = g[i];
                            const float cdf = 0.5f * (1.0f + std::erf(gv * 0.70710678f));
                            const float av = std::min(gv * cdf, lim);
                            g[i] = av * std::clamp(u[i], -lim, lim);
                        }
                        break;
                    case 3: {
                        const float lim = L.act_limit;
                        for (size_t i = 0; i < count; ++i) {
                            const float gv = std::min(g[i], lim);
                            const float uv = std::clamp(u[i], -lim, lim);
                            g[i] = (uv + 1.0f) * gv / (1.0f + std::exp(-1.702f * gv));
                        }
                        break;
                    }
                    default:
                        for (size_t i = 0; i < count; ++i) {
                            const float uv = u[i] > 0.0f ? u[i] : 0.0f;
                            u[i] = uv * uv;
                        }
                        break;
                }
                const int idx4[MAX_M] = {0, 1, 2, 3};
                prepare_rows(L.downs[ch.expert], nullptr, a, I, idx4, ch.m, c.prep_d[j]);
            }
            break;
        }
        case 3: {
            int j0, j_step, sub, per;
            if (gemv_assignment(worker, num_workers, nc, j0, j_step, sub, per))
                for (int j = j0; j < nc; j += j_step)
                {
                    const Chunk& ch = c.chunks[j];
                    const MoeCpuMatrix& mat = L.downs[ch.expert];
                    float* tout = c.tout_d + static_cast<size_t>(j) * MAX_M * H;
                    int t0, t1;
                    tile_split(mat, sub, per, t0, t1);
                    run_tiles(mat, c.prep_d[j], tout, ch.m, t0, t1);
                }
            break;
        }
        case 4:
            for (int j = worker; j < nc; j += num_workers)
                transform_out(L.downs[c.chunks[j].expert], c.tout_d + static_cast<size_t>(j) * MAX_M * H, c.chunks[j].m);
            break;
        case 5: {
            const int c0 = H * worker / num_workers;
            const int c1 = H * (worker + 1) / num_workers;
            for (int j = 0; j < nc; ++j)
            {
                const Chunk& ch = c.chunks[j];
                const float* d = c.tout_d + static_cast<size_t>(j) * MAX_M * H;
                for (int r = 0; r < ch.m; ++r)
                {
                    float* dst = c.out + static_cast<size_t>(ch.token[r]) * H;
                    const float* src = d + static_cast<size_t>(r) * H;
                    const float w = ch.weight[r];
                    for (int col = c0; col < c1; ++col)
                        dst[col] += w * src[col];
                }
            }
            break;
        }
    }
}

} // namespace

// ---------------- Public API (mirrors moe_mul1.h / moe_mul1.cpp) ----------------

void exl3_moe_cpu_set_prof(bool enabled)
{
    g_prof_enabled.store(enabled, std::memory_order_relaxed);
}

namespace {

// Trellis stager: copies expert trellis tensors into the GPU-side buffer, expert-major in
// g/u/d order (gated) or u/d order (gateless). Swizzled sources are un-swizzled to native
// (k/16, n/16, 16K) order, matching the x86 stage machinery.
struct StageCtx
{
    const MoeCpuLayer* layer;
    const uint32_t* ids;
    int count;
    uint8_t* dst;
};

inline size_t trellis_bytes(const MoeCpuMatrix& m)
{
    return static_cast<size_t>(m.k / 16) * (m.n / 16) * 16 * m.bits * 2;
}

inline void stage_copy_trellis(uint8_t* dst, const MoeCpuMatrix& m)
{
    if (!m.swz)
    {
        std::memcpy(dst, m.trellis, trellis_bytes(m));
        return;
    }
    const int tiles_k = m.k / 16;
    const int tiles_n = m.n / 16;
    const int groups = tiles_n / 8;
    const size_t tile_b = static_cast<size_t>(m.bits) * 32;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(m.trellis);
    for (int g = 0; g < groups; ++g)
        for (int kt = 0; kt < tiles_k; ++kt)
            std::memcpy(dst + (static_cast<size_t>(kt) * tiles_n + g * 8) * tile_b,
                        src + (static_cast<size_t>(g) * tiles_k + kt) * 8 * tile_b,
                        8 * tile_b);
}

void stage_phase(void* vctx, int worker, int num_workers)
{
    StageCtx& c = *static_cast<StageCtx*>(vctx);
    const bool gated = !c.layer->gates.empty();
    const int nmat = gated ? 3 : 2;
    const size_t gb = gated ? trellis_bytes(c.layer->gates[0]) : 0;
    const size_t ub = trellis_bytes(c.layer->ups[0]);
    const size_t db = trellis_bytes(c.layer->downs[0]);
    const size_t per_expert = gb + ub + db;
    for (int u = worker; u < c.count * nmat; u += num_workers)
    {
        const int e = c.ids[u / nmat];
        const int mi = u % nmat;
        size_t off = static_cast<size_t>(u / nmat) * per_expert;
        const MoeCpuMatrix* m;
        if (gated && mi == 0)            { m = &c.layer->gates[e]; }
        else if (mi == (gated ? 1 : 0)) { m = &c.layer->ups[e]; off += gb; }
        else                               { m = &c.layer->downs[e]; off += gb + ub; }
        stage_copy_trellis(c.dst + off, *m);
    }
}

} // namespace

int64_t exl3_moe_cpu_make_layer
(
    const std::vector<at::Tensor>& gate_trellis,
    const std::vector<at::Tensor>& gate_suh,
    const std::vector<at::Tensor>& gate_svh,
    const std::vector<at::Tensor>& up_trellis,
    const std::vector<at::Tensor>& up_suh,
    const std::vector<at::Tensor>& up_svh,
    const std::vector<at::Tensor>& down_trellis,
    const std::vector<at::Tensor>& down_suh,
    const std::vector<at::Tensor>& down_svh,
    const std::vector<at::Tensor>& gate_bias,
    const std::vector<at::Tensor>& up_bias,
    const std::vector<at::Tensor>& down_bias,
    int64_t activation,
    double act_limit,
    int64_t swizzled
)
{
    auto* layer = new MoeCpuLayer;
    const bool swz = swizzled != 0;
    const size_t E = up_trellis.size();
    const bool gated = !gate_trellis.empty();
    TORCH_CHECK(down_trellis.size() == E && (!gated || gate_trellis.size() == E), "expert count mismatch");
    TORCH_CHECK(gated ? (activation == 0 || activation == 1 || activation == 3) : activation == 2, "gated experts take silu/gelu/swiglu_oai, gateless take relu2");
    TORCH_CHECK(gate_bias.empty() || gate_bias.size() == E, "gate bias count mismatch");
    TORCH_CHECK(up_bias.empty() || up_bias.size() == E, "up bias count mismatch");
    TORCH_CHECK(down_bias.empty() || down_bias.size() == E, "down bias count mismatch");
    layer->num_experts = static_cast<int>(E);
    layer->activation = static_cast<int>(activation);
    layer->act_limit = static_cast<float>(act_limit);
    for (size_t e = 0; e < E; ++e)
    {
        if (gated)
        {
            layer->gates.push_back(make_matrix(gate_trellis[e], gate_suh[e], gate_svh[e],
                                            gate_bias.empty() ? nullptr : &gate_bias[e], swz));
            for (auto& t : {gate_trellis[e], gate_suh[e], gate_svh[e]}) layer->refs.push_back(t);
            if (!gate_bias.empty()) layer->refs.push_back(gate_bias[e]);
        }
        layer->ups.push_back(make_matrix(up_trellis[e], up_suh[e], up_svh[e],
                                       up_bias.empty() ? nullptr : &up_bias[e], swz));
        layer->downs.push_back(make_matrix(down_trellis[e], down_suh[e], down_svh[e],
                                         down_bias.empty() ? nullptr : &down_bias[e], swz));
        for (auto& t : {up_trellis[e], up_suh[e], up_svh[e], down_trellis[e], down_suh[e], down_svh[e]})
            layer->refs.push_back(t);
        if (!up_bias.empty()) layer->refs.push_back(up_bias[e]);
        if (!down_bias.empty()) layer->refs.push_back(down_bias[e]);
    }
    layer->hidden_size = layer->ups[0].k;
    layer->interm_size = layer->ups[0].n;
    TORCH_CHECK(layer->downs[0].k == layer->interm_size && layer->downs[0].n == layer->hidden_size,
                "expert shape mismatch");
    std::lock_guard<std::mutex> lock(g_layers_mutex);
    g_layers.push_back(layer);
    return static_cast<int64_t>(g_layers.size() - 1);
}

void exl3_moe_cpu_free_layer(int64_t handle)
{
    std::lock_guard<std::mutex> lock(g_layers_mutex);
    if (handle < 0 || handle >= static_cast<int64_t>(g_layers.size())) return;
    delete g_layers[static_cast<size_t>(handle)];
    g_layers[static_cast<size_t>(handle)] = nullptr;
}

void exl3_moe_cpu_forward_raw
(
    int64_t handle,
    const at::Half* x,
    const int32_t* sel,
    const at::Half* wts,
    float* out,
    int rows,
    int topk,
    int threads
)
{
    const MoeCpuLayer* layer = get_layer(handle);
    TORCH_CHECK(layer, "invalid layer handle");
    const int m_total = rows;
    const int top_k = topk;

    ForwardCtx ctx;
    ctx.layer = layer;
    ctx.x = x;
    ctx.out = out;
    ctx.m_total = m_total;
    std::memset(ctx.out, 0, static_cast<size_t>(m_total) * layer->hidden_size * sizeof(float));

    std::vector<std::vector<std::pair<int, float>>> per_expert(layer->num_experts);
    for (int t = 0; t < m_total; ++t)
        for (int j = 0; j < top_k; ++j)
        {
            const int32_t e = sel[static_cast<size_t>(t) * top_k + j];
            if (e >= 0 && e < layer->num_experts)
                per_expert[e].emplace_back(t, half_to_float(wts[static_cast<size_t>(t) * top_k + j]));
        }
    for (int e = 0; e < layer->num_experts; ++e)
    {
        auto& lst = per_expert[e];
        for (size_t i = 0; i < lst.size(); i += MAX_M)
        {
            Chunk ch;
            ch.expert = e;
            ch.m = static_cast<int>(std::min<size_t>(MAX_M, lst.size() - i));
            for (int r = 0; r < ch.m; ++r)
            {
                ch.token[r] = lst[i + r].first;
                ch.weight[r] = lst[i + r].second;
            }
            ctx.chunks.push_back(ch);
        }
    }
    const int nc = static_cast<int>(ctx.chunks.size());
    if (!nc) return;

    const int H = layer->hidden_size;
    const int I = layer->interm_size;
    ForwardArena& ar = ForwardArena::get();
    auto grow = [](auto& v, size_t n) { if (v.size() < n) v.resize(n); };
    grow(ar.tin_g, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.tin_u, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.tin_d, static_cast<size_t>(nc) * MAX_M * I);
    grow(ar.splat_g, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.splat_u, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.splat_d, static_cast<size_t>(nc) * MAX_M * I);
    grow(ar.tout_g, static_cast<size_t>(nc) * MAX_M * I);
    grow(ar.tout_u, static_cast<size_t>(nc) * MAX_M * I);
    grow(ar.tout_d, static_cast<size_t>(nc) * MAX_M * H);
    grow(ar.prep_g, nc); grow(ar.prep_u, nc); grow(ar.prep_d, nc);
    ctx.tout_g = ar.tout_g.data();
    ctx.tout_u = ar.tout_u.data();
    ctx.tout_d = ar.tout_d.data();
    ctx.prep_g = ar.prep_g; ctx.prep_u = ar.prep_u; ctx.prep_d = ar.prep_d;
    for (int j = 0; j < nc; ++j)
    {
        ctx.prep_g[j] = { ar.tin_g.data() + static_cast<size_t>(j) * MAX_M * H,
                          ar.splat_g.data() + static_cast<size_t>(j) * MAX_M * H, {}, {} };
        ctx.prep_u[j] = { ar.tin_u.data() + static_cast<size_t>(j) * MAX_M * H,
                          ar.splat_u.data() + static_cast<size_t>(j) * MAX_M * H, {}, {} };
        ctx.prep_d[j] = { ar.tin_d.data() + static_cast<size_t>(j) * MAX_M * I,
                          ar.splat_d.data() + static_cast<size_t>(j) * MAX_M * I, {}, {} };
    }

    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.ensure(threads > 0 ? threads : 1);

    static const int small_cap = [](){
        const char* e = getenv("EXL3_MOE_CPU_SMALL_WORKERS");
        return e ? atoi(e) : 0;
    }();
    const int n_run = nc <= 2 ? small_cap : 0;

    for (int phase = 0; phase <= 5; ++phase)
    {
        ctx.phase = phase;
        g_pool.run(&forward_phase, &ctx, n_run);
    }
}

void exl3_moe_cpu_forward
(
    int64_t handle,
    const at::Tensor& x,
    const at::Tensor& selected,
    const at::Tensor& weights,
    at::Tensor& out,
    int64_t num_threads
)
{
    TORCH_CHECK(x.device().is_cpu() && selected.device().is_cpu() && weights.device().is_cpu() && out.device().is_cpu(), "CPU MoE tensors must be on CPU");
    TORCH_CHECK(x.scalar_type() == at::kHalf && out.scalar_type() == at::kFloat, "dtype mismatch");

    const int m_total = static_cast<int>(x.size(0));
    const int top_k = static_cast<int>(selected.size(-1));

    std::vector<int32_t> sel32(static_cast<size_t>(m_total) * top_k);
    if (selected.scalar_type() == at::kLong)
    {
        const int64_t* s = selected.data_ptr<int64_t>();
        for (size_t i = 0; i < sel32.size(); ++i) sel32[i] = static_cast<int32_t>(s[i]);
    }
    else
    {
        TORCH_CHECK(selected.scalar_type() == at::kInt, "selected must be int32 or int64");
        std::memcpy(sel32.data(), selected.data_ptr<int32_t>(), sel32.size() * 4);
    }
    exl3_moe_cpu_forward_raw
    (
        handle,
        reinterpret_cast<const at::Half*>(x.data_ptr()),
        sel32.data(),
        reinterpret_cast<const at::Half*>(weights.data_ptr()),
        out.data_ptr<float>(),
        m_total, top_k,
        static_cast<int>(num_threads)
    );
}

void exl3_moe_cpu_stage_experts
(
    int64_t handle,
    const uint32_t* expert_ids,
    int count,
    uint8_t* dst,
    int threads
)
{
    const MoeCpuLayer* layer = get_layer(handle);
    TORCH_CHECK(layer, "invalid layer handle");
    StageCtx ctx { layer, expert_ids, count, dst };
    const bool gated = !layer->gates.empty();
    int units = count * (gated ? 3 : 2);
    int nt = std::min(threads > 0 ? threads : 1, units);
    if (nt <= 1)
    {
        stage_phase(&ctx, 0, 1);
        return;
    }
    std::vector<std::thread> ts;
    ts.reserve(nt);
    for (int i = 0; i < nt; ++i)
        ts.emplace_back(stage_phase, &ctx, i, nt);
    for (auto& t : ts)
        t.join();
}

bool exl3_moe_cpu_has_avx2()        { return false; }
bool exl3_moe_cpu_has_avx512_vnni() { return false; }
bool exl3_moe_cpu_has_avx512_vbmi() { return false; }

#endif // defined(__aarch64__) || defined(_M_ARM64)
