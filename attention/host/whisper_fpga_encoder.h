#pragma once
// whisper_fpga_encoder.h
//
// XRT host interface for the whisper_encoder_top HLS kernel.
// Replaces the CPU encoder in whisper.cpp (ggerganov) with the Alveo U55C
// FPGA kernel.
//
// Usage:
//   1. Build the xclbin with v++ (see connectivity.cfg).
//   2. whisper_fpga_init() once per process (loads xclbin + weights).
//   3. whisper_fpga_encode() for each audio segment.
//   4. whisper_fpga_free() to release resources.

#include <cstddef>
#include <cstdbool>

#ifdef __cplusplus
extern "C" {
#endif

struct whisper_fpga_context;

// Initialize the FPGA encoder.
//   xclbin_path : path to the compiled whisper_encoder_top.xclbin
//   weights_dir : directory containing test_vectors/*.bin weight files
//                 (same format as encoder_tb.cpp; can be the test_vectors/ dir)
//   device_idx  : XRT device index (0 for first Alveo card)
// Returns NULL on failure.
whisper_fpga_context* whisper_fpga_init(const char* xclbin_path,
                                        const char* weights_dir,
                                        unsigned int device_idx);

void whisper_fpga_free(whisper_fpga_context* ctx);

// Encode using the FPGA transformer kernel.
//
//   embd_conv   : float[n_audio_ctx × D_MODEL] in ggml conv output layout:
//                 element(t, c) = embd_conv[t + c * n_audio_ctx]
//                 Pass (const float*)wstate.embd_conv->data after ggml computes
//                 the conv graph. Positional embedding is added internally.
//
//   n_audio_ctx : number of time frames from ggml conv (wstate.embd_conv->ne[0],
//                 typically 1500). Padded to FRAMES=1504 internally.
//
//   out         : float[AUDIO_FRAMES × D_MODEL] = float[1500 × 384] output.
//                 Written in time-major layout: out[t * D_MODEL + c].
//                 In whisper.cpp, pass (float*)wstate.embd_enc->data.
//
// Returns true on success.
bool whisper_fpga_encode(whisper_fpga_context* ctx,
                         const float* embd_conv,
                         int          n_audio_ctx,
                         float*       out);

#ifdef __cplusplus
}
#endif
