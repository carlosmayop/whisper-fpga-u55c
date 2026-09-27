#include <ap_int.h>
#include <hls_stream.h>
#include <hls_half.h>
#include "whisper_types.h"
#ifndef __SYNTHESIS__
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#endif
#include "conv_layers.h"
#include "matmul.h"
#include "math_utils.h"
#include "layer_norm.h"

void process_attention_head(const half* qkv_in, int head_id, hls::stream<half>& out_s);

#ifndef __SYNTHESIS__
// Dumps an FP16 buffer to disk if WHISPER_FPGA_DEBUG=1.
// The output directory is taken from WHISPER_FPGA_WEIGHTS (where the weights live),
// so the mel and the intermediate outputs end up next to the weights for the Python script.
static void save_debug_buf(const char* name, const half* buf, int n) {
    const char* en = std::getenv("WHISPER_FPGA_DEBUG");
    if (!en || en[0] != '1') return;
    const char* wdir = std::getenv("WHISPER_FPGA_WEIGHTS");
    std::string path = std::string(wdir ? wdir : ".") + "/" + name + ".bin";
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(buf), (std::streamsize)(n * sizeof(half)));
    float s = 0.f; int nans = 0;
    for (int i = 0; i < n; i++) { float v = (float)buf[i]; if (std::isnan(v)) nans++; else s += v; }
    fprintf(stderr, "[debug] saved %s  mean=%.4f  nan=%d\n", path.c_str(), s/(n-nans+1e-9f), nans);
}
#endif

void write_stream_to_hbm(hls::stream<half>& stream_in, half* mem_out, int total) {
    for (int i = 0; i < total; i++) {
        #pragma HLS PIPELINE II=1
        mem_out[i] = stream_in.read();
    }
}

void read_hbm_to_stream(const half* mem_in, hls::stream<half>& stream_out, int total) {
    for (int i = 0; i < total; i++) {
        #pragma HLS PIPELINE II=1
        stream_out.write(mem_in[i]);
    }
}

void run_matmul_tiles(
    const bus_t* weights_hbm,
    hls::stream<half>& in_stream,
    hls::stream<half>& out_stream,
    int in_ch, int out_ch
) {
    #pragma HLS DATAFLOW
    hls::stream<block_q5_1> weight_stream("weight_stream");
    // generous depth to absorb the latency of the SLR crossing (Laguna registers)
    #pragma HLS STREAM variable=weight_stream depth=128

    weight_loader_all_tiles(weights_hbm, weight_stream, in_ch, out_ch);   // SLR0
    gemm_compute_all_tiles(in_stream, weight_stream, out_stream, in_ch, out_ch);  // SLR1/2
}

void write_concat_to_hbm(hls::stream<half>& in_s, half* mha_own) {
    for (int f = 0; f < FRAMES; f++) {
        for (int c = 0; c < D_HEAD; c++) {
            #pragma HLS PIPELINE II=1
            mha_own[f * D_HEAD + c] = in_s.read();
        }
    }
}

void process_head_dual(
    const half* qkv_a, const half* qkv_b,
    int h0, int h1,
    half* mha_a, half* mha_b
) {
    #pragma HLS DATAFLOW

    // OPT #2 (streaming) + 6-out: each head writes to its OWN mha bank (mha_a/mha_b)
    // -> no m_axi arbitration -> the 2 heads DO run in parallel (they used to write
    // to the same bank and serialised, which is the 2x measured in HW).
    hls::stream<half> out_a("out_a");
    hls::stream<half> out_b("out_b");
    #pragma HLS STREAM variable=out_a depth=2048
    #pragma HLS STREAM variable=out_b depth=2048

    process_attention_head(qkv_a, h0, out_a);
    process_attention_head(qkv_b, h1, out_b);

    write_concat_to_hbm(out_a, mha_a);
    write_concat_to_hbm(out_b, mha_b);
}

void transpose_residual_compute(
    hls::stream<half>& matmul_out,
    hls::stream<half>& res_in,
    const half bias[D_MODEL],
    hls::stream<half>& out_stream
) {
    half tile_buf[D_MODEL][TILE_FRAMES];
    #pragma HLS ARRAY_PARTITION variable=tile_buf cyclic factor=4 dim=2

    for (int t = 0; t < TILES; t++) {
        load_tile: for (int ch = 0; ch < D_MODEL; ch++) {
            for (int f = 0; f < TILE_FRAMES; f++) {
                #pragma HLS PIPELINE II=1
                tile_buf[ch][f] = matmul_out.read();
            }
        }
        write_tile: for (int f = 0; f < TILE_FRAMES; f++) {
            for (int ch = 0; ch < D_MODEL; ch++) {
                #pragma HLS PIPELINE II=1
                out_stream.write(tile_buf[ch][f] + bias[ch] + res_in.read());
            }
        }
    }
}


void transpose_gelu_compute(
    hls::stream<half>& fc1_out,
    const half ffn1_bias[D_FF],
    hls::stream<half>& out_stream
) {
    // 1536 * 32 * 2 bytes = 96 KB -> URAM
    half tile_buf[D_FF][TILE_FRAMES];
    #pragma HLS BIND_STORAGE variable=tile_buf type=ram_2p impl=uram
    #pragma HLS ARRAY_PARTITION variable=tile_buf cyclic factor=4 dim=2

    for (int t = 0; t < TILES; t++) {
        load_tile: for (int ch = 0; ch < D_FF; ch++) {
            for (int f = 0; f < TILE_FRAMES; f++) {
                #pragma HLS PIPELINE II=1
                tile_buf[ch][f] = fc1_out.read();
            }
        }
        write_tile: for (int f = 0; f < TILE_FRAMES; f++) {
            for (int ch = 0; ch < D_FF; ch++) {
                #pragma HLS PIPELINE
                out_stream.write(apply_gelu(tile_buf[ch][f] + ffn1_bias[ch]));
            }
        }
    }
}

void add_posemb_and_write(
    hls::stream<half>& stream_in,
    const half* pos_emb,
    half* out_buf
) {
    for (int i = 0; i < FRAMES * D_MODEL; i++) {
        #pragma HLS PIPELINE II=1
        out_buf[i] = stream_in.read() + pos_emb[i];
    }
}

void phase1_frontend_and_posemb(
    const half* mel_spectrogram_in,
    const half* pos_emb,
    const half weights1[CONV1_OUT_CHANNELS][MEL_CHANNELS][3],
    const half biases1[CONV1_OUT_CHANNELS],
    const half weights2[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][3],
    const half biases2[CONV2_OUT_CHANNELS],
    half* out_buf
) {
    #pragma HLS DATAFLOW

    hls::stream<MelFrameFP16> mel_stream("mel_stream");
    hls::stream<half> conv1_to_conv2("conv1_to_conv2");
    hls::stream<half> conv2_to_padder("conv2_to_padder");
    hls::stream<half> padder_to_posemb("padder_to_posemb");

    #pragma HLS BIND_STORAGE variable=conv1_to_conv2 type=fifo impl=bram
    #pragma HLS BIND_STORAGE variable=padder_to_posemb type=fifo impl=bram

    read_mel_spectrogram_fp16(mel_spectrogram_in, mel_stream);
    conv1d_layer1(mel_stream, conv1_to_conv2, weights1, biases1);
    conv1d_layer2(conv1_to_conv2, conv2_to_padder, weights2, biases2);
    pad_audio_stream(conv2_to_padder, padder_to_posemb, CONV2_OUT_CHANNELS);
    add_posemb_and_write(padder_to_posemb, pos_emb, out_buf);
}

void write_qkv_with_bias(
    hls::stream<half>& stream_in,
    const half q_bias[D_MODEL],
    const half v_bias[D_MODEL],
    half* mem_out
) {
    for (int t = 0; t < TILES; t++) {
        for (int ch = 0; ch < QKV_CHANNELS; ch++) {
            for (int f = 0; f < BLOCK_SIZE; f++) {
                #pragma HLS PIPELINE II=1
                int idx = t * (QKV_CHANNELS * BLOCK_SIZE) + ch * BLOCK_SIZE + f;
                half val = stream_in.read();
                if      (ch < D_MODEL)          val += q_bias[ch];
                else if (ch >= 2 * D_MODEL)     val += v_bias[ch - 2 * D_MODEL];
                mem_out[idx] = val;
            }
        }
    }
}

void write_qkv_x3(
    hls::stream<half>& s,
    const half q_bias[D_MODEL], const half v_bias[D_MODEL],
    half* q_buf, half* q_buf_b, half* q_buf_c
) {
    for (int t = 0; t < TILES; t++)
        for (int ch = 0; ch < QKV_CHANNELS; ch++)
            for (int f = 0; f < BLOCK_SIZE; f++) {
                #pragma HLS PIPELINE II=1
                int idx = t * (QKV_CHANNELS * BLOCK_SIZE) + ch * BLOCK_SIZE + f;
                half val = s.read();
                if      (ch < D_MODEL)      val += q_bias[ch];
                else if (ch >= 2 * D_MODEL) val += v_bias[ch - 2 * D_MODEL];
                q_buf[idx] = val; q_buf_b[idx] = val; q_buf_c[idx] = val;
            }
}

void phase_prenorm_qkv(
    const half* x_in,
    const bus_t* qkv_weights_hbm,
    const half ln1_g[D_MODEL], const half ln1_b[D_MODEL],
    const half q_bias[D_MODEL], const half v_bias[D_MODEL],
    half* q_buf, half* q_buf_b, half* q_buf_c
) {
    #pragma HLS DATAFLOW

    hls::stream<half> stream_x("stream_x");
    hls::stream<half> stream_ln1("stream_ln1");
    hls::stream<half> stream_qkv("stream_qkv");
    #pragma HLS STREAM variable=stream_ln1 depth=1152
    #pragma HLS BIND_STORAGE variable=stream_ln1 type=fifo impl=bram

    read_hbm_to_stream(x_in, stream_x, FRAMES * D_MODEL);     // SLR0
    ln_compute(stream_x, stream_ln1, ln1_g, ln1_b);          // ln1 fusion (SLR1/2)
    run_matmul_tiles(qkv_weights_hbm, stream_ln1, stream_qkv, D_MODEL, QKV_CHANNELS);
    write_qkv_x3(stream_qkv, q_bias, v_bias, q_buf, q_buf_b, q_buf_c);  // SLR0, 3 banks
}

void phase_wo_and_residual(
    const half* mha_buffer,
    const bus_t* wo_weights_hbm,
    const half* residual1_buf,
    const half attn_out_bias[D_MODEL],
    half* out_buf
) {
    #pragma HLS DATAFLOW

    hls::stream<half> stream_mha("stream_mha");
    hls::stream<half> stream_wo("stream_wo");
    hls::stream<half> stream_res("stream_res");
    hls::stream<half> stream_out("stream_out");
    #pragma HLS STREAM variable=stream_mha depth=384
    #pragma HLS BIND_STORAGE variable=stream_mha type=fifo impl=bram
    #pragma HLS STREAM variable=stream_res depth=512
    #pragma HLS STREAM variable=stream_out depth=512

    read_hbm_to_stream(mha_buffer, stream_mha, FRAMES * D_MODEL);          // SLR0
    read_hbm_to_stream(residual1_buf, stream_res, FRAMES * D_MODEL);       // SLR0 (residual)
    run_matmul_tiles(wo_weights_hbm, stream_mha, stream_wo, D_MODEL, D_MODEL); // loader SLR0 + gemm SLR1/2
    transpose_residual_compute(stream_wo, stream_res, attn_out_bias, stream_out); // SLR1/2
    write_stream_to_hbm(stream_out, out_buf, FRAMES * D_MODEL);            // SLR0
}

void phase_ffn_fused(
    const half* ln2_out,
    const bus_t* fc1_weights_hbm,
    const bus_t* fc2_weights_hbm,
    const half ffn1_bias[D_FF],
    const half ffn2_bias[D_MODEL],
    const half* residual2_buf,
    half* out_buf
) {
    #pragma HLS DATAFLOW

    hls::stream<half> stream_ln2("stream_ln2");
    hls::stream<half> stream_fc1("stream_fc1");
    hls::stream<half> stream_mid("stream_mid");
    hls::stream<half> stream_fc2("stream_fc2");
    hls::stream<half> stream_res("stream_res2");
    hls::stream<half> stream_out("stream_out_ffn");
    #pragma HLS STREAM variable=stream_ln2 depth=1536
    #pragma HLS BIND_STORAGE variable=stream_ln2 type=fifo impl=bram
    #pragma HLS STREAM variable=stream_mid depth=512
    #pragma HLS STREAM variable=stream_res depth=512
    #pragma HLS STREAM variable=stream_out depth=512

    read_hbm_to_stream(ln2_out, stream_ln2, FRAMES * D_MODEL);                 // SLR0
    run_matmul_tiles(fc1_weights_hbm, stream_ln2, stream_fc1, D_MODEL, D_FF);  // fc1
    transpose_gelu_compute(stream_fc1, ffn1_bias, stream_mid);                 // gelu (ffn_mid on-chip)
    run_matmul_tiles(fc2_weights_hbm, stream_mid, stream_fc2, D_FF, D_MODEL);  // fc2
    read_hbm_to_stream(residual2_buf, stream_res, FRAMES * D_MODEL);           // SLR0 (residual)
    transpose_residual_compute(stream_fc2, stream_res, ffn2_bias, stream_out); // +bias+residual
    write_stream_to_hbm(stream_out, out_buf, FRAMES * D_MODEL);                // SLR0
}

// Full encoder
void run_encoder_layer(
    const half* x_in,           // [FRAMES * D_MODEL] frame-major
    half*       x_out,          // [FRAMES * D_MODEL] frame-major
    const bus_t* qkv_w, const bus_t* wo_w,
    const bus_t* fc1_w, const bus_t* fc2_w,
    const half ln1_gamma[D_MODEL], const half ln1_beta[D_MODEL],
    const half ln2_gamma[D_MODEL], const half ln2_beta[D_MODEL],
    const half q_bias[D_MODEL], const half v_bias[D_MODEL],
    const half attn_out_bias[D_MODEL],
    const half ffn1_bias[D_FF], const half ffn2_bias[D_MODEL],
    half* qkv_buf, half* qkv_buf_b, half* qkv_buf_c,
    half* mha_buf,
    half* residual2, half* ln2_out
) {
    #pragma HLS INLINE

    // ln1->qkv FUSION: pre-norm + QKV projection in a single dataflow region (no ln1_out in HBM)
    phase_prenorm_qkv(x_in, qkv_w, ln1_gamma, ln1_beta, q_bias, v_bias,
                      qkv_buf, qkv_buf_b, qkv_buf_c);

    // 3+3 heads (code not used by the attention_top top level; it only compiles).
    process_head_dual(qkv_buf,   qkv_buf_b, 0, 1, mha_buf, mha_buf);
    process_head_dual(qkv_buf_c, qkv_buf,   2, 3, mha_buf, mha_buf);
    process_head_dual(qkv_buf_b, qkv_buf_c, 4, 5, mha_buf, mha_buf);

    phase_wo_and_residual(mha_buf, wo_w, x_in, attn_out_bias, residual2);

    // FFN pre-norm (ln2 is kept separate) + ffn1->ffn2 FUSION (no ffn_mid in HBM)
    layer_norm(residual2, ln2_out, ln2_gamma, ln2_beta);
    phase_ffn_fused(ln2_out, fc1_w, fc2_w, ffn1_bias, ffn2_bias, residual2, x_out);
}

// Layout of the small_weights buffer (FP16 elements):
// For each layer L (L=0..3), base = L * SW_PER_LAYER:
//   +0*D_MODEL : ln1_g      [D_MODEL]
//   +1*D_MODEL : ln1_b      [D_MODEL]
//   +2*D_MODEL : ln2_g      [D_MODEL]
//   +3*D_MODEL : ln2_b      [D_MODEL]
//   +4*D_MODEL : q_bias     [D_MODEL]
//   +5*D_MODEL : v_bias     [D_MODEL]
//   +6*D_MODEL : attn_out_bias [D_MODEL]
//   +7*D_MODEL : ffn1_bias  [D_FF]
//   +7*D_MODEL+D_FF : ffn2_bias [D_MODEL]
// SW_PER_LAYER = 8*D_MODEL + D_FF = 3072 + 1536 = 4608
// After the 4 layers (offset SW_LNPOST = 4*SW_PER_LAYER = 18432):
//   +0 : ln_post_gamma [D_MODEL]
//   +D_MODEL : ln_post_beta [D_MODEL]
// Total: 19200 FP16 elements = 38400 bytes
static const int SW_PER_LAYER = 8 * D_MODEL + D_FF;
static const int SW_LNPOST    = 4 * SW_PER_LAYER;

void whisper_encoder_top(
    const bus_t* l0_qkv_w, const bus_t* l0_wo_w,
    const bus_t* l0_fc1_w, const bus_t* l0_fc2_w,

    const bus_t* l1_qkv_w, const bus_t* l1_wo_w,
    const bus_t* l1_fc1_w, const bus_t* l1_fc2_w,

    const bus_t* l2_qkv_w, const bus_t* l2_wo_w,
    const bus_t* l2_fc1_w, const bus_t* l2_fc2_w,

    const bus_t* l3_qkv_w, const bus_t* l3_wo_w,
    const bus_t* l3_fc1_w, const bus_t* l3_fc2_w,

    const half* small_weights,  // all the small arrays packed together (see layout above)

    half* buf_a,              // [FRAMES * D_MODEL]
    half* buf_b,              // [FRAMES * D_MODEL]

    half* qkv_buffer,         // [FRAMES * QKV_CHANNELS], AXI port head 0
    half* qkv_buffer_b,       // same physical buffer, AXI port head 1
    half* qkv_buffer_c,       // same physical buffer, AXI port head 2
    half* mha_buffer,         // [FRAMES * D_MODEL]
    half* residual2_buffer,   // [FRAMES * D_MODEL]
    half* ln2_out_buffer,     // [FRAMES * D_MODEL]

    half* layer_out            // [FRAMES * D_MODEL]
) {
    // Bundle shared by type: 1 AXI port per type, sequential access across layers
    #pragma HLS INTERFACE m_axi port=l0_qkv_w  depth=16384  bundle=gmem_qkv_w
    #pragma HLS INTERFACE m_axi port=l1_qkv_w  depth=16384  bundle=gmem_qkv_w
    #pragma HLS INTERFACE m_axi port=l2_qkv_w  depth=16384  bundle=gmem_qkv_w
    #pragma HLS INTERFACE m_axi port=l3_qkv_w  depth=16384  bundle=gmem_qkv_w

    #pragma HLS INTERFACE m_axi port=l0_wo_w   depth=5462   bundle=gmem_wo_w
    #pragma HLS INTERFACE m_axi port=l1_wo_w   depth=5462   bundle=gmem_wo_w
    #pragma HLS INTERFACE m_axi port=l2_wo_w   depth=5462   bundle=gmem_wo_w
    #pragma HLS INTERFACE m_axi port=l3_wo_w   depth=5462   bundle=gmem_wo_w

    #pragma HLS INTERFACE m_axi port=l0_fc1_w  depth=21845  bundle=gmem_fc1_w
    #pragma HLS INTERFACE m_axi port=l1_fc1_w  depth=21845  bundle=gmem_fc1_w
    #pragma HLS INTERFACE m_axi port=l2_fc1_w  depth=21845  bundle=gmem_fc1_w
    #pragma HLS INTERFACE m_axi port=l3_fc1_w  depth=21845  bundle=gmem_fc1_w

    #pragma HLS INTERFACE m_axi port=l0_fc2_w  depth=21845  bundle=gmem_fc2_w
    #pragma HLS INTERFACE m_axi port=l1_fc2_w  depth=21845  bundle=gmem_fc2_w
    #pragma HLS INTERFACE m_axi port=l2_fc2_w  depth=21845  bundle=gmem_fc2_w
    #pragma HLS INTERFACE m_axi port=l3_fc2_w  depth=21845  bundle=gmem_fc2_w

    // Small-weights buffer (biases + layer norms), 38400 bytes.
    // Shares a bundle with qkv_w: it is read only once at the start (so it does not compete for BW)
    // and avoids a 17th AXI port that used to cause routing congestion.
    #pragma HLS INTERFACE m_axi port=small_weights depth=19200 bundle=gmem_qkv_w

    // Ping-pong buffers between layers
    #pragma HLS INTERFACE m_axi port=buf_a     depth=578048  bundle=gmem_ba
    #pragma HLS INTERFACE m_axi port=buf_b     depth=578048  bundle=gmem_bb
    #pragma HLS INTERFACE m_axi port=qkv_buffer       depth=1732608  bundle=gmem_s2
    #pragma HLS INTERFACE m_axi port=qkv_buffer_b     depth=1732608  bundle=gmem_s2b
    #pragma HLS INTERFACE m_axi port=qkv_buffer_c     depth=1732608  bundle=gmem_s2c
    #pragma HLS INTERFACE m_axi port=mha_buffer       depth=578048   bundle=gmem_s3
    #pragma HLS INTERFACE m_axi port=residual2_buffer depth=578048   bundle=gmem_s4
    #pragma HLS INTERFACE m_axi port=ln2_out_buffer   depth=578048   bundle=gmem_s5
    #pragma HLS INTERFACE m_axi port=layer_out        depth=578048   bundle=gmem_out

    #pragma HLS INTERFACE s_axilite port=l0_qkv_w
    #pragma HLS INTERFACE s_axilite port=l0_wo_w
    #pragma HLS INTERFACE s_axilite port=l0_fc1_w
    #pragma HLS INTERFACE s_axilite port=l0_fc2_w
    #pragma HLS INTERFACE s_axilite port=l1_qkv_w
    #pragma HLS INTERFACE s_axilite port=l1_wo_w
    #pragma HLS INTERFACE s_axilite port=l1_fc1_w
    #pragma HLS INTERFACE s_axilite port=l1_fc2_w
    #pragma HLS INTERFACE s_axilite port=l2_qkv_w
    #pragma HLS INTERFACE s_axilite port=l2_wo_w
    #pragma HLS INTERFACE s_axilite port=l2_fc1_w
    #pragma HLS INTERFACE s_axilite port=l2_fc2_w
    #pragma HLS INTERFACE s_axilite port=l3_qkv_w
    #pragma HLS INTERFACE s_axilite port=l3_wo_w
    #pragma HLS INTERFACE s_axilite port=l3_fc1_w
    #pragma HLS INTERFACE s_axilite port=l3_fc2_w
    #pragma HLS INTERFACE s_axilite port=small_weights
    #pragma HLS INTERFACE s_axilite port=buf_a
    #pragma HLS INTERFACE s_axilite port=buf_b
    #pragma HLS INTERFACE s_axilite port=qkv_buffer
    #pragma HLS INTERFACE s_axilite port=qkv_buffer_b
    #pragma HLS INTERFACE s_axilite port=qkv_buffer_c
    #pragma HLS INTERFACE s_axilite port=mha_buffer
    #pragma HLS INTERFACE s_axilite port=residual2_buffer
    #pragma HLS INTERFACE s_axilite port=ln2_out_buffer
    #pragma HLS INTERFACE s_axilite port=layer_out
    #pragma HLS INTERFACE s_axilite port=return

    // -- Load the small arrays from HBM into local arrays (BRAM) --------------------
    // One sequential burst per array; the tool infers a burst read on the gmem_qkv_w port.
    half l0_ln1_g[D_MODEL], l0_ln1_b[D_MODEL];
    half l0_ln2_g[D_MODEL], l0_ln2_b[D_MODEL];
    half l0_q_bias[D_MODEL], l0_v_bias[D_MODEL], l0_attn_out_bias[D_MODEL];
    half l0_ffn1_bias[D_FF], l0_ffn2_bias[D_MODEL];

    half l1_ln1_g[D_MODEL], l1_ln1_b[D_MODEL];
    half l1_ln2_g[D_MODEL], l1_ln2_b[D_MODEL];
    half l1_q_bias[D_MODEL], l1_v_bias[D_MODEL], l1_attn_out_bias[D_MODEL];
    half l1_ffn1_bias[D_FF], l1_ffn2_bias[D_MODEL];

    half l2_ln1_g[D_MODEL], l2_ln1_b[D_MODEL];
    half l2_ln2_g[D_MODEL], l2_ln2_b[D_MODEL];
    half l2_q_bias[D_MODEL], l2_v_bias[D_MODEL], l2_attn_out_bias[D_MODEL];
    half l2_ffn1_bias[D_FF], l2_ffn2_bias[D_MODEL];

    half l3_ln1_g[D_MODEL], l3_ln1_b[D_MODEL];
    half l3_ln2_g[D_MODEL], l3_ln2_b[D_MODEL];
    half l3_q_bias[D_MODEL], l3_v_bias[D_MODEL], l3_attn_out_bias[D_MODEL];
    half l3_ffn1_bias[D_FF], l3_ffn2_bias[D_MODEL];

    half ln_post_gamma[D_MODEL], ln_post_beta[D_MODEL];

    // Force BRAM (ram_2p): by default HLS implemented them as ~100K flip-flops
    // that saturated SLR1 at 97% CLB and prevented routing. BRAM sits at 79%,
    // so there is headroom. Single write (LOAD_SW) + reads in later phases.
    #pragma HLS BIND_STORAGE variable=l0_ln1_g  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l0_ln1_b  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l0_ln2_g  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l0_ln2_b  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l0_q_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l0_v_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l0_attn_out_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l0_ffn1_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l0_ffn2_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l1_ln1_g  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l1_ln1_b  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l1_ln2_g  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l1_ln2_b  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l1_q_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l1_v_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l1_attn_out_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l1_ffn1_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l1_ffn2_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l2_ln1_g  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l2_ln1_b  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l2_ln2_g  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l2_ln2_b  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l2_q_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l2_v_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l2_attn_out_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l2_ffn1_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l2_ffn2_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l3_ln1_g  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l3_ln1_b  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l3_ln2_g  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l3_ln2_b  type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l3_q_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l3_v_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l3_attn_out_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l3_ffn1_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=l3_ffn2_bias type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=ln_post_gamma type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=ln_post_beta  type=ram_2p impl=bram

#define LOAD_SW(dst, base_elems, sz) \
    for (int _i = 0; _i < (sz); _i++) { \
        _Pragma("HLS PIPELINE II=1") \
        (dst)[_i] = small_weights[(base_elems) + _i]; \
    }

    LOAD_SW(l0_ln1_g,         0*SW_PER_LAYER + 0*D_MODEL,       D_MODEL)
    LOAD_SW(l0_ln1_b,         0*SW_PER_LAYER + 1*D_MODEL,       D_MODEL)
    LOAD_SW(l0_ln2_g,         0*SW_PER_LAYER + 2*D_MODEL,       D_MODEL)
    LOAD_SW(l0_ln2_b,         0*SW_PER_LAYER + 3*D_MODEL,       D_MODEL)
    LOAD_SW(l0_q_bias,        0*SW_PER_LAYER + 4*D_MODEL,       D_MODEL)
    LOAD_SW(l0_v_bias,        0*SW_PER_LAYER + 5*D_MODEL,       D_MODEL)
    LOAD_SW(l0_attn_out_bias, 0*SW_PER_LAYER + 6*D_MODEL,       D_MODEL)
    LOAD_SW(l0_ffn1_bias,     0*SW_PER_LAYER + 7*D_MODEL,       D_FF)
    LOAD_SW(l0_ffn2_bias,     0*SW_PER_LAYER + 7*D_MODEL+D_FF,  D_MODEL)

    LOAD_SW(l1_ln1_g,         1*SW_PER_LAYER + 0*D_MODEL,       D_MODEL)
    LOAD_SW(l1_ln1_b,         1*SW_PER_LAYER + 1*D_MODEL,       D_MODEL)
    LOAD_SW(l1_ln2_g,         1*SW_PER_LAYER + 2*D_MODEL,       D_MODEL)
    LOAD_SW(l1_ln2_b,         1*SW_PER_LAYER + 3*D_MODEL,       D_MODEL)
    LOAD_SW(l1_q_bias,        1*SW_PER_LAYER + 4*D_MODEL,       D_MODEL)
    LOAD_SW(l1_v_bias,        1*SW_PER_LAYER + 5*D_MODEL,       D_MODEL)
    LOAD_SW(l1_attn_out_bias, 1*SW_PER_LAYER + 6*D_MODEL,       D_MODEL)
    LOAD_SW(l1_ffn1_bias,     1*SW_PER_LAYER + 7*D_MODEL,       D_FF)
    LOAD_SW(l1_ffn2_bias,     1*SW_PER_LAYER + 7*D_MODEL+D_FF,  D_MODEL)

    LOAD_SW(l2_ln1_g,         2*SW_PER_LAYER + 0*D_MODEL,       D_MODEL)
    LOAD_SW(l2_ln1_b,         2*SW_PER_LAYER + 1*D_MODEL,       D_MODEL)
    LOAD_SW(l2_ln2_g,         2*SW_PER_LAYER + 2*D_MODEL,       D_MODEL)
    LOAD_SW(l2_ln2_b,         2*SW_PER_LAYER + 3*D_MODEL,       D_MODEL)
    LOAD_SW(l2_q_bias,        2*SW_PER_LAYER + 4*D_MODEL,       D_MODEL)
    LOAD_SW(l2_v_bias,        2*SW_PER_LAYER + 5*D_MODEL,       D_MODEL)
    LOAD_SW(l2_attn_out_bias, 2*SW_PER_LAYER + 6*D_MODEL,       D_MODEL)
    LOAD_SW(l2_ffn1_bias,     2*SW_PER_LAYER + 7*D_MODEL,       D_FF)
    LOAD_SW(l2_ffn2_bias,     2*SW_PER_LAYER + 7*D_MODEL+D_FF,  D_MODEL)

    LOAD_SW(l3_ln1_g,         3*SW_PER_LAYER + 0*D_MODEL,       D_MODEL)
    LOAD_SW(l3_ln1_b,         3*SW_PER_LAYER + 1*D_MODEL,       D_MODEL)
    LOAD_SW(l3_ln2_g,         3*SW_PER_LAYER + 2*D_MODEL,       D_MODEL)
    LOAD_SW(l3_ln2_b,         3*SW_PER_LAYER + 3*D_MODEL,       D_MODEL)
    LOAD_SW(l3_q_bias,        3*SW_PER_LAYER + 4*D_MODEL,       D_MODEL)
    LOAD_SW(l3_v_bias,        3*SW_PER_LAYER + 5*D_MODEL,       D_MODEL)
    LOAD_SW(l3_attn_out_bias, 3*SW_PER_LAYER + 6*D_MODEL,       D_MODEL)
    LOAD_SW(l3_ffn1_bias,     3*SW_PER_LAYER + 7*D_MODEL,       D_FF)
    LOAD_SW(l3_ffn2_bias,     3*SW_PER_LAYER + 7*D_MODEL+D_FF,  D_MODEL)

    LOAD_SW(ln_post_gamma, SW_LNPOST + 0*D_MODEL, D_MODEL)
    LOAD_SW(ln_post_beta,  SW_LNPOST + 1*D_MODEL, D_MODEL)

#undef LOAD_SW
    run_encoder_layer(
        buf_a, buf_b,
        l0_qkv_w, l0_wo_w, l0_fc1_w, l0_fc2_w,
        l0_ln1_g, l0_ln1_b, l0_ln2_g, l0_ln2_b,
        l0_q_bias, l0_v_bias, l0_attn_out_bias, l0_ffn1_bias, l0_ffn2_bias,
        qkv_buffer, qkv_buffer_b, qkv_buffer_c,
        mha_buffer,
        residual2_buffer, ln2_out_buffer
    );

#ifndef __SYNTHESIS__
    {
        bool nan=false, inf=false;
        for (int i=0; i<FRAMES*D_MODEL; i++) {
            float v=(float)buf_b[i];
            if (std::isnan(v)){nan=true;break;}
            if (std::isinf(v)){inf=true;break;}
        }
        fprintf(stderr,"[enc_top] after layer 0: buf_b[0..3]= %.4f %.4f %.4f %.4f  nan=%d inf=%d\n",
            (float)buf_b[0],(float)buf_b[1],(float)buf_b[2],(float)buf_b[3],nan,inf);
        save_debug_buf("l0_hls", buf_b, FRAMES*D_MODEL);
    }
#endif

    run_encoder_layer(
        buf_b, buf_a,
        l1_qkv_w, l1_wo_w, l1_fc1_w, l1_fc2_w,
        l1_ln1_g, l1_ln1_b, l1_ln2_g, l1_ln2_b,
        l1_q_bias, l1_v_bias, l1_attn_out_bias, l1_ffn1_bias, l1_ffn2_bias,
        qkv_buffer, qkv_buffer_b, qkv_buffer_c,
        mha_buffer,
        residual2_buffer, ln2_out_buffer
    );

#ifndef __SYNTHESIS__
    { fprintf(stderr,"[enc_top] after layer 1: buf_a[0..3]= %.4f %.4f %.4f %.4f\n",
        (float)buf_a[0],(float)buf_a[1],(float)buf_a[2],(float)buf_a[3]);
      save_debug_buf("l1_hls", buf_a, FRAMES*D_MODEL); }
#endif

    run_encoder_layer(
        buf_a, buf_b,
        l2_qkv_w, l2_wo_w, l2_fc1_w, l2_fc2_w,
        l2_ln1_g, l2_ln1_b, l2_ln2_g, l2_ln2_b,
        l2_q_bias, l2_v_bias, l2_attn_out_bias, l2_ffn1_bias, l2_ffn2_bias,
        qkv_buffer, qkv_buffer_b, qkv_buffer_c,
        mha_buffer,
        residual2_buffer, ln2_out_buffer
    );

#ifndef __SYNTHESIS__
    { fprintf(stderr,"[enc_top] after layer 2: buf_b[0..3]= %.4f %.4f %.4f %.4f\n",
        (float)buf_b[0],(float)buf_b[1],(float)buf_b[2],(float)buf_b[3]);
      save_debug_buf("l2_hls", buf_b, FRAMES*D_MODEL); }
#endif

    run_encoder_layer(
        buf_b, buf_a,
        l3_qkv_w, l3_wo_w, l3_fc1_w, l3_fc2_w,
        l3_ln1_g, l3_ln1_b, l3_ln2_g, l3_ln2_b,
        l3_q_bias, l3_v_bias, l3_attn_out_bias, l3_ffn1_bias, l3_ffn2_bias,
        qkv_buffer, qkv_buffer_b, qkv_buffer_c,
        mha_buffer,
        residual2_buffer, ln2_out_buffer
    );

#ifndef __SYNTHESIS__
    { fprintf(stderr,"[enc_top] after layer 3: buf_a[0..3]= %.4f %.4f %.4f %.4f\n",
        (float)buf_a[0],(float)buf_a[1],(float)buf_a[2],(float)buf_a[3]);
      save_debug_buf("l3_hls", buf_a, FRAMES*D_MODEL); }
#endif

    // Final LayerNorm (before the decoder cross-attention)
    layer_norm(buf_a, layer_out, ln_post_gamma, ln_post_beta);

#ifndef __SYNTHESIS__
    { fprintf(stderr,"[enc_top] after ln_post: layer_out[0..3]= %.4f %.4f %.4f %.4f\n",
        (float)layer_out[0],(float)layer_out[1],(float)layer_out[2],(float)layer_out[3]);
      save_debug_buf("final_hls", layer_out, FRAMES*D_MODEL); }
#endif
}

void attention_top(
    half* q0a, half* q0b,   // dual-pass 0: heads head_base+0,+1
    half* q1a, half* q1b,   // dual-pass 1: heads head_base+2,+3
    half* q2a, half* q2b,   // dual-pass 2: heads head_base+4,+5
    half* o0, half* o1, half* o2, half* o3, half* o4, half* o5, // 6 banks: 1 per head
    int head_base           // SMALL: 0 for heads 0-5, 6 for 6-11 (12 heads in 2 passes)
) {
    #pragma HLS INTERFACE m_axi port=q0a depth=3465216 bundle=gmem_q0
    #pragma HLS INTERFACE m_axi port=q0b depth=3465216 bundle=gmem_q1
    #pragma HLS INTERFACE m_axi port=q1a depth=3465216 bundle=gmem_q2
    #pragma HLS INTERFACE m_axi port=q1b depth=3465216 bundle=gmem_q3
    #pragma HLS INTERFACE m_axi port=q2a depth=3465216 bundle=gmem_q4
    #pragma HLS INTERFACE m_axi port=q2b depth=3465216 bundle=gmem_q5
    #pragma HLS INTERFACE m_axi port=o0 depth=96256 bundle=gmem_o0
    #pragma HLS INTERFACE m_axi port=o1 depth=96256 bundle=gmem_o1
    #pragma HLS INTERFACE m_axi port=o2 depth=96256 bundle=gmem_o2
    #pragma HLS INTERFACE m_axi port=o3 depth=96256 bundle=gmem_o3
    #pragma HLS INTERFACE m_axi port=o4 depth=96256 bundle=gmem_o4
    #pragma HLS INTERFACE m_axi port=o5 depth=96256 bundle=gmem_o5
    #pragma HLS INTERFACE s_axilite port=q0a
    #pragma HLS INTERFACE s_axilite port=q0b
    #pragma HLS INTERFACE s_axilite port=q1a
    #pragma HLS INTERFACE s_axilite port=q1b
    #pragma HLS INTERFACE s_axilite port=q2a
    #pragma HLS INTERFACE s_axilite port=q2b
    #pragma HLS INTERFACE s_axilite port=o0
    #pragma HLS INTERFACE s_axilite port=o1
    #pragma HLS INTERFACE s_axilite port=o2
    #pragma HLS INTERFACE s_axilite port=o3
    #pragma HLS INTERFACE s_axilite port=o4
    #pragma HLS INTERFACE s_axilite port=o5
    #pragma HLS INTERFACE s_axilite port=head_base
    #pragma HLS INTERFACE s_axilite port=return

    #pragma HLS DATAFLOW
    process_head_dual(q0a, q0b, head_base+0, head_base+1, o0, o1);  // 6 heads, each with its own bank
    process_head_dual(q1a, q1b, head_base+2, head_base+3, o2, o3);
    process_head_dual(q2a, q2b, head_base+4, head_base+5, o4, o5);
}
