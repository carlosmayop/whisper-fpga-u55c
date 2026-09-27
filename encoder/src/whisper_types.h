#pragma once

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_half.h> // Native Vitis HLS support for fp16

typedef ap_uint<512> bus_t;

//  Tiny model dimensions 
const int MEL_CHANNELS = 80;
const int TIME_FRAMES = 3000;
const int CONV1_OUT_CHANNELS = 384;
const int CONV2_IN_CHANNELS = 384;
const int CONV2_OUT_CHANNELS = 384;
const int D_MODEL = 384;
const int D_HEAD = 64;

const int QK8_1 = 32;
const int BLOCK_SIZE = 32; // Quantization block size

const int AUDIO_FRAMES = 1500;
const int PADDED_FRAMES = 1504; // 1500 + 4 padding frames
const int FRAMES = 1504;
const int TILES = PADDED_FRAMES / QK8_1;  //47 blocks
const int QKV_CHANNELS = 1152;  // 384 (Q) + 384 (K) + 384 (V)
const int D_FF          = 1536; // 4 * D_MODEL (Whisper Tiny FFN expansion)


//  Streaming structures 
struct MelFrameFP16 {
    half data[MEL_CHANNELS];
};

struct Conv1FrameFP16 {
    half data[CONV1_OUT_CHANNELS]; 
};

// --- GGML structures ---
typedef struct {
    half d;              // Scale factor (delta)
    half s;              // Precomputed sum: d * sum(qs)
    int8_t qs[QK8_1];    // The 32 quantized values
} block_q8_1;

typedef struct {
    half d;                // Scale
    half m;                // Minimum
    uint32_t qh;           // The 5th bit of the 32 weights
    uint8_t qs[QK8_1 / 2]; // The lower 4 bits (2 weights per byte)
} block_q5_1;