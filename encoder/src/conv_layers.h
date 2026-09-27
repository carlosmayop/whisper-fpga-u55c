#pragma once
#include "whisper_types.h"

// Reads from HBM memory and feeds it into the stream
void read_mel_spectrogram_fp16(
    const half* mem_in, 
    hls::stream<MelFrameFP16>& stream_out
);

// Applies Conv1D (Kernel=3, Stride=1, Padding=1) and GELU
void conv1d_layer1(
    hls::stream<MelFrameFP16>& stream_in,
    hls::stream<half>& stream_out,
    const half weights[CONV1_OUT_CHANNELS][MEL_CHANNELS][3],
    const half biases[CONV1_OUT_CHANNELS]
);

// Applies Conv1D (Kernel=2, Stride=2, Padding=1) and GELU
void conv1d_layer2(
    hls::stream<half>& stream_in,
    hls::stream<half>& stream_out,
    const half weights[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][3],
    const half biases[CONV2_OUT_CHANNELS]  
);

void pad_audio_stream(hls::stream<half>& stream_in, hls::stream<half>& stream_out, int channels);