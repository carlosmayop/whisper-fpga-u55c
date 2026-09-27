#pragma once

#include <hls_math.h>
#include "whisper_types.h"
#ifndef __SYNTHESIS__
#include <cmath>
#endif

// GELU approximation used in Whisper:
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline half apply_gelu(half x) {
    #pragma HLS INLINE

    half x3 = x * x * x;
    // Constants precomputed in fp16
    half c1 = (half)0.044715;
    half c2 = (half)0.797884; // sqrt(2/pi)

    half inner = c2 * (x + c1 * x3);
    // hls::tanh(half) produces NaN in csim; use std::tanh(float) in sim.
#ifdef __SYNTHESIS__
    half tanh_val = hls::tanh(inner);
#else
    half tanh_val = (half)std::tanh((float)inner);
#endif

    return (half)0.5 * x * ((half)1.0 + tanh_val);
}

// BLOCK_SIZE is already defined in whisper_types.h as const int BLOCK_SIZE = 32