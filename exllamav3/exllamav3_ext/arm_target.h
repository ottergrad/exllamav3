#pragma once

// aarch64 SIMD feature detection. On non-aarch64 only never-true stubs are produced so shared
// dispatch code compiles everywhere; the real implementations live in aarch64-only TUs
// (all_reduce_cpu_neon.cpp / the ARM branch of cpu/moe_mul1.cpp).
//
// EXLLAMA_ARM_SIMD is defined by the build (setup.py and ext.py) when compiling aarch64 with
// the optional SIMD extensions (-march=armv9-a+i8mm+bf16+sve2). NEON is mandatory on
// aarch64, so is_arm_neon_supported() is always true there. The optional extension tests default
// to matching EXLLAMA_ARM_SIMD rather than relying on GCC's inconsistent __ARM_FEATURE_I8MM
// macro.

#if defined(__aarch64__) || defined(_M_ARM64)

    #include <arm_neon.h>
    #include <stdint.h>

    inline bool is_arm_neon_supported() { return true; }

    #if defined(EXLLAMA_ARM_SIMD)
        inline bool is_arm_i8mm_supported()   { return true; }   // usdot (v8.4 I8MM)
        inline bool is_arm_bf16_supported()    { return true; }   // bfdot (v8.4 BF16)
        inline bool is_arm_sve_supported()      { return true; }
        inline bool is_arm_sve2_supported()     { return true; }
    #else
        inline bool is_arm_i8mm_supported()   { return false; }
        inline bool is_arm_bf16_supported()    { return false; }
        inline bool is_arm_sve_supported()      { return false; }
        inline bool is_arm_sve2_supported()    { return false; }
    #endif

#else

    inline bool is_arm_neon_supported() { return false; }
    inline bool is_arm_i8mm_supported()   { return false; }
    inline bool is_arm_bf16_supported()    { return false; }
    inline bool is_arm_sve_supported()     { return false; }
    inline bool is_arm_sve2_supported()   { return false; }

#endif
