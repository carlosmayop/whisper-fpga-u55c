#include "conv_layers.h"
#include "math_utils.h"

void read_mel_spectrogram_fp16(
    const half* mem_in,
    hls::stream<MelFrameFP16>& stream_out
) {
    read_frames: for (int t = 0; t < TIME_FRAMES; t++) {
        #pragma HLS PIPELINE II=1
        MelFrameFP16 frame;
        for (int c = 0; c < MEL_CHANNELS; c++) {
            #pragma HLS UNROLL factor=4
            frame.data[c] = mem_in[t * MEL_CHANNELS + c];
        }
        stream_out.write(frame);
    }
}

void conv1d_layer1(
    hls::stream<MelFrameFP16>& stream_in,
    hls::stream<half>& stream_out,
    const half weights[CONV1_OUT_CHANNELS][MEL_CHANNELS][3],
    const half biases[CONV1_OUT_CHANNELS]
) {
    // Local BRAM buffer: loaded from single m_axi port, accessed in parallel.
    // No ARRAY_PARTITION on the 'weights' parameter -> single AXI port.
    half weights_local[CONV1_OUT_CHANNELS][MEL_CHANNELS][3];
    #pragma HLS ARRAY_PARTITION variable=weights_local cyclic factor=20 dim=2
    #pragma HLS ARRAY_PARTITION variable=weights_local complete dim=3
    #pragma HLS BIND_STORAGE variable=weights_local type=ram_2p impl=bram

    load_w1: for (int oc = 0; oc < CONV1_OUT_CHANNELS; oc++) {
        for (int ic = 0; ic < MEL_CHANNELS; ic++) {
            #pragma HLS PIPELINE II=1
            for (int k = 0; k < 3; k++) {
                #pragma HLS UNROLL
                weights_local[oc][ic][k] = weights[oc][ic][k];
            }
        }
    }

    MelFrameFP16 window[3];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=1

    // Left padding: initialise the window to zero
    init_window: for(int k = 0; k < 3; k++) {
        #pragma HLS UNROLL
        for(int c = 0; c < MEL_CHANNELS; c++) {
            #pragma HLS UNROLL
            window[k].data[c] = (half)0.0;
        }
    }

    // TIME_FRAMES + 1 iterations to cover the right padding
    process_frames: for (int t = 0; t < TIME_FRAMES + 1; t++) {

        window[0] = window[1];
        window[1] = window[2];

        if (t < TIME_FRAMES) {
            window[2] = stream_in.read();
        } else {
            for(int c = 0; c < MEL_CHANNELS; c++) {
                #pragma HLS UNROLL
                window[2].data[c] = (half)0.0;
            }
        }

        if (t >= 1) {
            calc_out: for (int oc = 0; oc < CONV1_OUT_CHANNELS; oc++) {
                #pragma HLS PIPELINE II=4
                half acc = biases[oc];
                mac_window: for (int k = 0; k < 3; k++) {
                    #pragma HLS UNROLL
                    mac_channels: for (int ic = 0; ic < MEL_CHANNELS; ic++) {
                        #pragma HLS UNROLL
                        acc += window[k].data[ic] * weights_local[oc][ic][k];
                    }
                }
                stream_out.write(apply_gelu(acc));
            }
        }
    }
}

void conv1d_layer2(
    hls::stream<half>& stream_in,
    hls::stream<half>& stream_out,
    const half weights[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][3],
    const half biases[CONV2_OUT_CHANNELS]
) {
    // Local URAM buffer: loaded from single m_axi port, accessed in parallel.
    // No ARRAY_PARTITION on the 'weights' parameter -> single AXI port.
    half weights2_local[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][3];
    #pragma HLS ARRAY_PARTITION variable=weights2_local cyclic factor=64 dim=2
    #pragma HLS ARRAY_PARTITION variable=weights2_local complete dim=3
    #pragma HLS BIND_STORAGE variable=weights2_local type=ram_2p impl=uram

    load_w2: for (int oc = 0; oc < CONV2_OUT_CHANNELS; oc++) {
        for (int ic = 0; ic < CONV2_IN_CHANNELS; ic++) {
            #pragma HLS PIPELINE II=1
            for (int k = 0; k < 3; k++) {
                #pragma HLS UNROLL
                weights2_local[oc][ic][k] = weights[oc][ic][k];
            }
        }
    }

    half window[3][CONV2_IN_CHANNELS];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=1
    #pragma HLS ARRAY_PARTITION variable=window complete dim=2  // registers: avoid LUTRAM concentration in SLR1

    init_pad: for(int k = 0; k < 3; k++) {
        for(int c = 0; c < CONV2_IN_CHANNELS; c++) {
            #pragma HLS UNROLL
            window[k][c] = 0;
        }
    }

    read_frames: for (int t = 0; t < TIME_FRAMES; t++) {

        shift_and_read: for (int c = 0; c < CONV2_IN_CHANNELS; c++) {
            #pragma HLS PIPELINE II=1
            window[0][c] = window[1][c];
            window[1][c] = window[2][c];
            window[2][c] = stream_in.read();
        }

        // Stride=2: output only on odd frames, reduces 3000 -> 1500 frames
        if (t % 2 == 1) {
            calc_out: for (int oc = 0; oc < CONV2_OUT_CHANNELS; oc++) {
                #pragma HLS PIPELINE II=6
                half acc = biases[oc];

                mac_window: for (int k = 0; k < 3; k++) {
                    #pragma HLS UNROLL
                    mac_channels: for (int ic = 0; ic < CONV2_IN_CHANNELS; ic++) {
                        #pragma HLS UNROLL
                        #pragma HLS BIND_OP variable=acc op=fmul impl=dsp
                        acc += window[k][ic] * weights2_local[oc][ic][k];
                    }
                }

                stream_out.write(apply_gelu(acc));
            }
        }
    }
}

void pad_audio_stream(hls::stream<half>& stream_in, hls::stream<half>& stream_out, int channels) {
    pad_loop: for (int f = 0; f < PADDED_FRAMES; f++) {
        for (int c = 0; c < channels; c++) {
            #pragma HLS PIPELINE II=1
            if (f < AUDIO_FRAMES) {
                stream_out.write(stream_in.read());
            } else {
                stream_out.write((half)0.0);
            }
        }
    }
}
