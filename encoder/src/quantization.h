#pragma once

#include "whisper_types.h"
#include <hls_math.h>

inline void quantize_row_q8_1(const half src[QK8_1], block_q8_1 &res) {
    #pragma HLS INLINE

    // Logarithmic reduction tree for the maximum (5 steps vs 32 sequential ones)
    half max_tree[QK8_1];
    #pragma HLS ARRAY_PARTITION variable=max_tree complete dim=1

    for (int i = 0; i < QK8_1; i++) {
        #pragma HLS UNROLL
        max_tree[i] = hls::abs(src[i]);
    }

    for (int step = 1; step < QK8_1; step *= 2) {
        #pragma HLS UNROLL
        for (int i = 0; i < QK8_1; i += 2 * step) {
            #pragma HLS UNROLL
            if (max_tree[i + step] > max_tree[i]) {
                max_tree[i] = max_tree[i + step];
            }
        }
    }

    half amax = max_tree[0];
    half d = amax / (half)127.0;
    half id = (amax != (half)0.0) ? ((half)127.0 / amax) : (half)0.0;
    res.d = d;

    int32_t sum_qs = 0;

    quantize_loop: for (int i = 0; i < QK8_1; i++) {
        #pragma HLS UNROLL
        int8_t q = (int8_t)hls::round(src[i] * id);
        res.qs[i] = q;
        sum_qs += q;
    }

    res.s = d * (half)sum_qs;
}

inline half vec_dot_q5_1_q8_1(const block_q5_1 &w, const block_q8_1 &a) {
    #pragma HLS INLINE

    int32_t sum_dot = 0;
    int32_t sum_act = 0;

    dot_loop: for (int i = 0; i < QK8_1; i++) {
        #pragma HLS UNROLL

        // Rebuild the Q5_1 weight with the GGML layout:
        //   qs[j] = low nibble of element j | high nibble of element j+16
        //   qh bit i = fifth bit of element i (sequential order)
        uint8_t nibble = (i < QK8_1/2)
                       ? (w.qs[i]            & 0x0F)   // elements 0-15
                       : (w.qs[i - QK8_1/2] >> 4) & 0x0F; // elements 16-31
        uint8_t bit5   = (w.qh >> i) & 0x01;
        int32_t weight_int = (bit5 << 4) | nibble;

        int32_t act_int = a.qs[i];

        sum_dot += weight_int * act_int;
        sum_act += act_int;
    }

    // ggml Q5_1 * Q8_1 formula:
    // result = act_scale * (weight_scale * sum_dot + weight_min * sum_act)
    // IMPORTANT: accumulate in float32 to avoid FP16 overflow.
    // sum_dot can reach 32×31×127 = 125,984 which exceeds FP16 max (65,504).
    float math_w_f = (float)w.d * (float)sum_dot + (float)w.m * (float)sum_act;
    half result = a.d * (half)math_w_f;

    return result;
}
