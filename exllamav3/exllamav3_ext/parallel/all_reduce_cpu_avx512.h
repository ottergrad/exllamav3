#pragma once
#include <cstdint>
#include <cstddef>
#include "context.cuh"

#if defined(__x86_64__) || defined(_M_X64)
    // AVX-512 versions
    void enable_fast_fp_avx512();

    void bf16_add_inplace_avx512(
        uint16_t* __restrict a,
        const uint16_t* __restrict b,
        size_t count
    );

    void bf16_add_twosrc_avx512(
        uint16_t* __restrict dst,
        const uint16_t* __restrict src_a,
        const uint16_t* __restrict src_b,
        size_t count
    );

    void fp16_add_inplace_avx512(
        uint16_t* __restrict a,
        const uint16_t* __restrict b,
        size_t count
    );

    void fp16_add_twosrc_avx512(
        uint16_t* __restrict dst,
        const uint16_t* __restrict src_a,
        const uint16_t* __restrict src_b,
        size_t count
    );

    void perform_cpu_reduce_avx512(
        PGContext* ctx,
        size_t data_size,
        uint32_t device_mask,
        uint32_t wire_dtype,
        uint8_t* shbuf_ptr,
        size_t shbuf_size
    );
#else
    // Non-x86_64 stubs: these paths are only reachable from the x86-only AVX code.
    inline void enable_fast_fp_avx512() {}
    inline void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) {}
    inline void bf16_add_twosrc_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t) {}
    inline void fp16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) {}
    inline void fp16_add_twosrc_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t) {}
    inline void perform_cpu_reduce_avx512(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) {}
#endif
