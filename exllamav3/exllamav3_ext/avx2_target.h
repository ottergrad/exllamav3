#pragma once

// AVX2 intrinsic detection is only implemented on x86_64
#if defined(__x86_64__) || defined(_M_X64)

    #ifndef __linux__
        #include <intrin.h>
    #endif

    bool is_avx2_supported();
    bool is_f16c_supported();

    #ifdef __linux__
        #define AVX2_TARGET __attribute__((target("avx2")))
        #define AVX2_F16C_TARGET __attribute__((target("avx2,f16c")))
        #define AVX2_TARGET_OPTIONAL __attribute__((target_clones("avx2","default")))
    #else
        #define AVX2_TARGET
        #define AVX2_F16C_TARGET
        #define AVX2_TARGET_OPTIONAL
    #endif

#else

    // Non-x86_64: never supported
    inline bool is_avx2_supported() { return false; }
    inline bool is_f16c_supported() { return false; }
    #define AVX2_TARGET
    #define AVX2_F16C_TARGET
    #define AVX2_TARGET_OPTIONAL

#endif
