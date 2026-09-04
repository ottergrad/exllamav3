#pragma once
#include <cstdint>
#include "context.cuh"

#define NUM_THREADS 1024
#define CPUREDUCE_CHUNK_SIZE (NUM_THREADS * 128)

// Generic path. On x86 the definition is in all_reduce_cpu_avx2.cpp (dispatches AVX-512/AVX2);
// on aarch64 it lives in all_reduce_cpu_neon.cpp (NEON). Exactly one TU provides the definition.
void perform_cpu_reduce
(
    PGContext* ctx,
    size_t data_size,
    uint32_t device_mask,
    uint32_t wire_dtype,
    uint8_t* shbuf_ptr,
    size_t shbuf_size
);

#if defined(__x86_64__) || defined(_M_X64)
    void enable_fast_fp();
    void enable_fast_fp_avx2();

    void perform_cpu_reduce_avx2
    (
        PGContext* ctx,
        size_t data_size,
        uint32_t device_mask,
        uint32_t wire_dtype,
        uint8_t* shbuf_ptr,
        size_t shbuf_size
    );
#else
    // Non-x86_64: no-op stubs so all_reduce_cpu.cu and the AVX TUs link.
    inline void enable_fast_fp() {}
    inline void enable_fast_fp_avx2() {}

    inline void perform_cpu_reduce_avx2
    (
        PGContext*,
        size_t,
        uint32_t,
        uint32_t,
        uint8_t*,
        size_t
    )
    {
    }
#endif

// Basic process-safe atomic reference with acquire/release semantics, Linux/Windows compatible
template <typename T>
struct atomic_ref
{
    T* p;
    explicit atomic_ref(T* ptr) : p(ptr) {}

    T load_relaxed() const noexcept
    {
        return *p;
    }

    T load_acquire() const noexcept
    {
        #if defined(_MSC_VER) && !defined(__clang__)
            static_assert(sizeof(T) == 4, "MSVC path assumes 32-bit T");
            long v = _InterlockedCompareExchange(reinterpret_cast<volatile long*>(p), 0L, 0L);
            return static_cast<T>(static_cast<uint32_t>(v));
        #else
            return __atomic_load_n(p, __ATOMIC_ACQUIRE);
        #endif
    }

    void store_release(T v)
    {
        #if defined(_MSC_VER) && !defined(__clang__)
            static_assert(sizeof(T) == 4, "MSVC path assumes 32-bit T");
            (void)_InterlockedExchange(reinterpret_cast<volatile long*>(p), static_cast<long>(static_cast<uint32_t>(v)));
        #else
            __atomic_store_n(p, v, __ATOMIC_RELEASE);
        #endif
    }
};

inline bool cpusum_device_arrived(PGContext* ctx, int device, uint32_t stage, bool multi, bool* no_contrib)
{
    if (!multi)
    {
        atomic_ref<uint32_t> f_(&ctx->cpusum_stage_device[device * REDUCE_STAGE_STRIDE]);
        uint32_t v = f_.load_acquire();
        if ((v & 0x7fffffffu) == stage) return false;
        *no_contrib = (v & 0x80000000u) != 0;
        return true;
    }
    uint32_t target = (stage + 1u) & 0x7fffffffu;
    bool nc = false;
    for (int b = 0; b < CPUREDUCE_MB_BLOCKS; ++b)
    {
        atomic_ref<uint32_t> f_(&ctx->cpusum_stage_device_mb[(device * CPUREDUCE_MB_BLOCKS + b) * REDUCE_STAGE_STRIDE]);
        uint32_t v = f_.load_acquire();
        if ((int32_t)((v & 0x7fffffffu) - target) < 0) return false;
        if (b == 0) nc = (v & 0x80000000u) != 0;
    }
    *no_contrib = nc;
    return true;
}

inline bool cpusum_recv_ready(PGContext* ctx, uint32_t device_mask, uint32_t target)
{
    for (int device = 0; device < MAX_DEVICES; ++device)
    {
        if (!(device_mask & (1u << device))) continue;
        for (int b = 0; b < CPUREDUCE_MB_BLOCKS; ++b)
        {
            atomic_ref<uint32_t> f_(&ctx->cpusum_stage_recv_mb[(device * CPUREDUCE_MB_BLOCKS + b) * REDUCE_STAGE_STRIDE]);
            if ((int32_t)((f_.load_acquire() & 0x7fffffffu) - target) < 0) return false;
        }
    }
    return true;
}

// Run an accumulate split across worker threads. Implemented in all_reduce_cpu_avx2.cpp (x86).
void cpu_reduce_parallel
(
    void (*fn3)(uint16_t*, const uint16_t*, const uint16_t*, size_t),
    void (*fn2)(uint16_t*, const uint16_t*, size_t),
    uint16_t* dst,
    const uint16_t* a,
    const uint16_t* b,
    size_t count,
    int threads
);
