#include "matmul.h"
#include "quantization.h"
#include "memory_interfaces.h"
#include "whisper_types.h"

#include <hls_stream.h>
#include <hls_half.h>


// Compute kernel. Processes T_OUT output rows per iteration to amortise
// the weight load and run T_OUT vec_dot in parallel each cycle.
// Requires out_features to be a multiple of T_OUT.
void compute_gemm_tile(
    block_q8_1 audio_cache[TILE_FRAMES][MAX_IN_BLOCKS],
    hls::stream<block_q5_1>& weight_stream,
    hls::stream<half>& audio_out_stream,
    int num_blocks,
    int out_features
) {
    compute_loop: for (int row = 0; row < out_features; row += T_OUT) {

        // T_OUT independent banks (dim=1 partitioned) for parallel reads
        // in the compute loop
        block_q5_1 w_rows[T_OUT][MAX_IN_BLOCKS];
        #pragma HLS ARRAY_PARTITION variable=w_rows complete dim=1

        load_weights: for (int r = 0; r < T_OUT; r++) {
            for (int b = 0; b < num_blocks; b++) {
                #pragma HLS PIPELINE II=1
                w_rows[r][b] = weight_stream.read();
            }
        }

        // All accumulators in registers (complete on both dims)
        half frame_acc[T_OUT][TILE_FRAMES];
        #pragma HLS ARRAY_PARTITION variable=frame_acc complete dim=1
        #pragma HLS ARRAY_PARTITION variable=frame_acc complete dim=2

        init_acc: for (int r = 0; r < T_OUT; r++) {
            #pragma HLS UNROLL
            for (int f = 0; f < TILE_FRAMES; f++) {
                #pragma HLS UNROLL
                frame_acc[r][f] = 0;
            }
        }

        // Pipeline on the inner loop (f changes every cycle): frame_acc[r][f] takes
        // TILE_FRAMES cycles before it is needed again, hiding the vec_dot latency.
        // unroll_rows instantiates T_OUT simultaneous vec_dot per cycle.
        compute_blocks: for (int b = 0; b < num_blocks; b++) {
            process_frames: for (int f = 0; f < TILE_FRAMES; f++) {
                #pragma HLS PIPELINE II=1
                unroll_rows: for (int r = 0; r < T_OUT; r++) {
                    #pragma HLS UNROLL
                    frame_acc[r][f] += vec_dot_q5_1_q8_1(w_rows[r][b], audio_cache[f][b]);
                }
            }
        }

        write_out: for (int r = 0; r < T_OUT; r++) {
            for (int f = 0; f < TILE_FRAMES; f++) {
                #pragma HLS PIPELINE II=1
                audio_out_stream.write(frame_acc[r][f]);
            }
        }
    }
}

// =====================================================================
// Stream-architecture split: separates the HBM access (loader, SLR0) from the
// compute (gemm, SLR1/2). Both used to live inside matmul_q5_1_q8_1_tile
// in a single DATAFLOW; once separated, the weight_stream (and the
// activation/output streams) cross the SLR boundary as narrow FIFOs.
// =====================================================================

// SLR0: sole owner of the weight m_axi port. For each of the TILES tiles it emits
// the out_features*num_blocks Q5.1 blocks (same order as before).
void weight_loader_all_tiles(
    const bus_t* weights_in_hbm,
    hls::stream<block_q5_1>& weight_stream,
    int in_features,
    int out_features
) {
    int num_blocks = in_features / QK8_1;
    wl_tiles: for (int tile = 0; tile < TILES; tile++) {
        hbm_to_q5_1_stream(weights_in_hbm, out_features * num_blocks, weight_stream);
    }
}

// SLR1/2: pure compute. Per tile: quantises the incoming activations (stream) into
// audio_cache and runs compute_gemm_tile consuming the weight_stream. No HBM.
void gemm_compute_all_tiles(
    hls::stream<half>& audio_in_stream,
    hls::stream<block_q5_1>& weight_stream,
    hls::stream<half>& audio_out_stream,
    int in_features,
    int out_features
) {
    int num_blocks = in_features / QK8_1;
    gc_tiles: for (int tile = 0; tile < TILES; tile++) {
        block_q8_1 audio_cache[TILE_FRAMES][MAX_IN_BLOCKS];
        #pragma HLS ARRAY_PARTITION variable=audio_cache complete dim=1

        for (int f = 0; f < TILE_FRAMES; f++) {
            for (int b = 0; b < num_blocks; b++) {
                half raw_audio[QK8_1];
                #pragma HLS ARRAY_PARTITION variable=raw_audio complete dim=1
                for (int i = 0; i < QK8_1; i++) {
                    #pragma HLS PIPELINE II=1
                    raw_audio[i] = audio_in_stream.read();
                }
                quantize_row_q8_1(raw_audio, audio_cache[f][b]);
            }
        }
        compute_gemm_tile(audio_cache, weight_stream, audio_out_stream, num_blocks, out_features);
    }
}

void matmul_q5_1_q8_1_tile(
    const bus_t* weights_in_hbm,
    hls::stream<half>& audio_in_stream,
    hls::stream<half>& audio_out_stream,
    int in_features,
    int out_features
) {
    int num_blocks = in_features / QK8_1;

    // Cache of quantised activations for the current tile.
    // dim=1 complete: the TILE_FRAMES banks are read in parallel in the compute loop.
    block_q8_1 audio_cache[TILE_FRAMES][MAX_IN_BLOCKS];
    #pragma HLS ARRAY_PARTITION variable=audio_cache complete dim=1

    for (int f = 0; f < TILE_FRAMES; f++) {
        for (int b = 0; b < num_blocks; b++) {
            half raw_audio[QK8_1];
            #pragma HLS ARRAY_PARTITION variable=raw_audio complete dim=1

            for(int i = 0; i < QK8_1; i++) {
                #pragma HLS PIPELINE II=1
                raw_audio[i] = audio_in_stream.read();
            }
            quantize_row_q8_1(raw_audio, audio_cache[f][b]);
        }
    }

    hls::stream<block_q5_1> weight_stream("weight_stream");
    #pragma HLS STREAM variable=weight_stream depth=32

    #pragma HLS DATAFLOW

    hbm_to_q5_1_stream(weights_in_hbm, out_features * num_blocks, weight_stream);
    compute_gemm_tile(audio_cache, weight_stream, audio_out_stream, num_blocks, out_features);
}
