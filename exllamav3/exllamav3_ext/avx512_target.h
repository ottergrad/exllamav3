#pragma once

// AVX-512 intrinsic detection is only implemented on x86_64
#if defined(__x86_64__) || defined(_M_X64)

    #ifndef __linux__
        #include <intrin.h>
    #endif

    bool is_avx512_supported();

    #ifdef __linux__
        #define AVX512_TARGET __attribute__((target("avx512f,avx512bw")))
        #define AVX512_TARGET_OPTIONAL __attribute__((target_clones("avx512f,avx512bw","default")))
    #else
        #define AVX512_TARGET
        #define AVX512_TARGET_OPTIONAL
    #endif

#else

    // Non-x86_64: never supported
    inline bool is_avx512_supported() { return false; }
    #define AVX512_TARGET
    #define AVX512_TARGET_OPTIONAL

#endif
