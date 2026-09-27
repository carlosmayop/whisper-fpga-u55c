#include "layer_norm.h"
#include <hls_math.h>
#include <hls_stream.h>

// --- Stream-architecture: separates the HBM access (movers, SLR0) from the compute
// (the layernorm itself, SLR1/2). layer_norm used to read x_in and write x_out
// directly from the compute logic; now the compute only sees streams and
// the movers (ln_load/ln_store) are the only ones touching HBM. ---

// SLR0: reads the tensor from HBM and pours it into the input stream.
static void ln_load(const half* x_in, hls::stream<half>& s_in, int total) {
    for (int i = 0; i < total; i++) {
        #pragma HLS PIPELINE II=1
        s_in.write(x_in[i]);
    }
}

// SLR0: drains the output stream into HBM.
static void ln_store(hls::stream<half>& s_out, half* x_out, int total) {
    for (int i = 0; i < total; i++) {
        #pragma HLS PIPELINE II=1
        x_out[i] = s_out.read();
    }
}

// SLR1/2: pure compute. Reads each frame from the stream into a local buffer, computes
// mean/variance/inv_std and emits the normalised frame on the output stream.
void ln_compute(
    hls::stream<half>& s_in,
    hls::stream<half>& s_out,
    const half gamma[D_MODEL],
    const half beta[D_MODEL]
) {
    half frame_buf[D_MODEL];
    #pragma HLS ARRAY_PARTITION variable=frame_buf cyclic factor=8 dim=1

    frame_loop: for (int f = 0; f < FRAMES; f++) {

        load: for (int c = 0; c < D_MODEL; c++) {
            #pragma HLS PIPELINE II=1
            frame_buf[c] = s_in.read();
        }

        float macc[8] = {0,0,0,0,0,0,0,0};
        #pragma HLS ARRAY_PARTITION variable=macc complete dim=1
        mean_sum: for (int c = 0; c < D_MODEL; c++) {
            #pragma HLS PIPELINE II=1
            macc[c & 7] += (float)frame_buf[c];
        }
        float mean = (macc[0]+macc[1]+macc[2]+macc[3]+
                      macc[4]+macc[5]+macc[6]+macc[7]) * (1.0f / (float)D_MODEL);

        float vacc[8] = {0,0,0,0,0,0,0,0};
        #pragma HLS ARRAY_PARTITION variable=vacc complete dim=1
        var_sum: for (int c = 0; c < D_MODEL; c++) {
            #pragma HLS PIPELINE II=1
            float d = (float)frame_buf[c] - mean;
            vacc[c & 7] += d * d;
        }
        half inv_std = (half)hls::rsqrt((vacc[0]+vacc[1]+vacc[2]+vacc[3]+
                                         vacc[4]+vacc[5]+vacc[6]+vacc[7]) * (1.0f/(float)D_MODEL) + 1e-5f);
        half mean_h  = (half)mean;

        norm_write: for (int c = 0; c < D_MODEL; c++) {
            #pragma HLS PIPELINE II=1
            half norm = (frame_buf[c] - mean_h) * inv_std;
            s_out.write(norm * gamma[c] + beta[c]);
        }
    }
}

void layer_norm(
    const half* x_in,
    half*       x_out,
    const half  gamma[D_MODEL],
    const half  beta[D_MODEL]
) {
    #pragma HLS DATAFLOW
    hls::stream<half> s_in("ln_in");
    hls::stream<half> s_out("ln_out");
    #pragma HLS STREAM variable=s_in  depth=512
    #pragma HLS STREAM variable=s_out depth=512

    ln_load(x_in, s_in, FRAMES * D_MODEL);          // SLR0 (mover)
    ln_compute(s_in, s_out, gamma, beta);           // SLR1/2 (compute)
    ln_store(s_out, x_out, FRAMES * D_MODEL);        // SLR0 (mover)
}
