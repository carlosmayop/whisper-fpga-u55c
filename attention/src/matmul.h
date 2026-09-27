#include "whisper_types.h"

#define TILE_FRAMES 32
#define MAX_IN_BLOCKS 48
#define T_OUT 4        // Output rows processed simultaneously

void compute_gemm_tile(
    block_q8_1 audio_cache[TILE_FRAMES][MAX_IN_BLOCKS], hls::stream<block_q5_1>& weight_stream,
    hls::stream<half>& audio_out_stream, int num_blocks, int out_features); 

void matmul_q5_1_q8_1_tile(const bus_t* weights_in_hbm, hls::stream<half>& audio_in_stream,
                        hls::stream<half>& audio_out_stream, int in_features, int out_features);

//  Stream-architecture split (data-mover on SLR0, compute on SLR1/2) 
// weight_loader_all_tiles: THE ONLY consumer of the weight m_axi port. Reads the
// weights for the TILES tiles from HBM and emits them on weight_stream. Floorplan -> SLR0.
void weight_loader_all_tiles(const bus_t* weights_in_hbm,
                             hls::stream<block_q5_1>& weight_stream,
                             int in_features, int out_features);

//  pure compute (no HBM). Consumes activations and weights over
// streams and produces output over a stream. Floorplan -> SLR1/2; its streams cross the
// SLR boundary as narrow FIFOs instead of wide AXI buses.
void gemm_compute_all_tiles(hls::stream<half>& audio_in_stream,
                            hls::stream<block_q5_1>& weight_stream,
                            hls::stream<half>& audio_out_stream,
                            int in_features, int out_features);