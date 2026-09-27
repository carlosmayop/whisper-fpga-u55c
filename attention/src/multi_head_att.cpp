#include <hls_math.h>
#include <hls_stream.h>
#include <hls_half.h>
#include <ap_fixed.h>
#include "whisper_types.h"

typedef ap_fixed<16, 6, AP_RND, AP_SAT> qk_t;   // Q6.10, range +/-32, res ~1e-3
typedef ap_fixed<36, 20>                dot_t;  // generously sized accumulator

typedef ap_ufixed<18, 2, AP_RND>        e_t;    // exp() in [0,1), 16 frac
typedef ap_fixed<30, 16>                o_t;    // O accumulator (16 int, 14 frac), the sweet spot
#ifndef __SYNTHESIS__
#include <cmath>
#include <cstdio>
#endif

static void qkv_tile_loader(
    const half* qkv_in,
    int head_id,
    hls::stream<half>& q_s,
    hls::stream<half>& k_s,
    hls::stream<half>& v_s
) {
    const int q_ch_base = head_id * D_HEAD;
    const int k_ch_base = D_MODEL  + head_id * D_HEAD;
    const int v_ch_base = 2*D_MODEL + head_id * D_HEAD;

    // (stationary K,V): each Q/K/V tile is read from HBM EXACTLY once.
    // The compute stage caches them on-chip and reuses the 47 K/V across the 47 q-tiles
    // (K/V used to be re-read 47 times). Order: all K, then all V, then all Q.
    ql_K: for (int kt = 0; kt < TILES; kt++) {
        for (int c = 0; c < D_HEAD; c++) {
            int base = kt * QKV_CHANNELS * BLOCK_SIZE + (k_ch_base + c) * BLOCK_SIZE;
            for (int f = 0; f < BLOCK_SIZE; f++) {
                #pragma HLS PIPELINE II=1
                k_s.write(qkv_in[base + f]);
            }
        }
    }
    ql_V: for (int vt = 0; vt < TILES; vt++) {
        for (int c = 0; c < D_HEAD; c++) {
            int base = vt * QKV_CHANNELS * BLOCK_SIZE + (v_ch_base + c) * BLOCK_SIZE;
            for (int f = 0; f < BLOCK_SIZE; f++) {
                #pragma HLS PIPELINE II=1
                v_s.write(qkv_in[base + f]);
            }
        }
    }
    ql_Q: for (int qt = 0; qt < TILES; qt++) {
        for (int c = 0; c < D_HEAD; c++) {
            int base = qt * QKV_CHANNELS * BLOCK_SIZE + (q_ch_base + c) * BLOCK_SIZE;
            for (int f = 0; f < BLOCK_SIZE; f++) {
                #pragma HLS PIPELINE II=1
                q_s.write(qkv_in[base + f]);
            }
        }
    }
}

// SLR1/2 (compute): pure flash attention. Reads the Q/K/V tiles from the streams; it never
// touches HBM. Same FA-2 maths as before (S/max/corr/acc per tile).
static void flash_attention_compute(
    hls::stream<half>& q_s,
    hls::stream<half>& k_s,
    hls::stream<half>& v_s,
    hls::stream<half>& out_s
) {
    //  ONLY K,V are cached (they are the ones re-read 47 times in the inner loop).
    // Q is iterated in the OUTER loop, so each q-tile is used once; loading the
    // current tile per iteration is enough (no redundant re-read is possible). This avoids Q_all
    // and saves 1/3 of the URAM. K,V kept in half (lossless: they come from FP16 qkv) to halve
    // the URAM; cyclic factor=16 dim=3 to match the UNROLL factor=16 of the MAC.
    qk_t K_all[TILES][BLOCK_SIZE][D_HEAD];   // Q6.10 fixed point (16b, same URAM as half)
    #pragma HLS ARRAY_PARTITION variable=K_all cyclic factor=16 dim=3
    #pragma HLS BIND_STORAGE variable=K_all type=ram_2p impl=uram
    qk_t V_all[TILES][BLOCK_SIZE][D_HEAD];   // Q6.10 fixed point (16b, same URAM as half)
    #pragma HLS ARRAY_PARTITION variable=V_all cyclic factor=16 dim=3
    #pragma HLS BIND_STORAGE variable=V_all type=ram_2p impl=uram

    // Current Q tile (loaded per q-tile from the stream, as in the original design).
    qk_t Q_tile[BLOCK_SIZE][D_HEAD];         // Q6.10 fixed point (only used in S_phase)
    #pragma HLS ARRAY_PARTITION variable=Q_tile complete dim=2

    o_t O_tile[BLOCK_SIZE][D_HEAD];          // O accumulator in fixed point
    #pragma HLS ARRAY_PARTITION variable=O_tile complete dim=2

    // Whole-tile score matrix: 32 columns banked by ki, addressed by qi.
    // 1024 fp32 in LUTRAM, written by S_phase, read by max/acc phases.
    float S_tile[BLOCK_SIZE][BLOCK_SIZE];
    #pragma HLS ARRAY_PARTITION variable=S_tile complete dim=2

    float m_row[BLOCK_SIZE];            // running row-max (log-sum-exp stability)
    float l_row[BLOCK_SIZE];            // running denominator (partition function)
    float m_new[BLOCK_SIZE];            // per-tile updated max
    #pragma HLS ARRAY_PARTITION variable=m_row complete dim=1
    #pragma HLS ARRAY_PARTITION variable=l_row complete dim=1
    #pragma HLS ARRAY_PARTITION variable=m_new complete dim=1

    //     Load ONLY the head's K and V once (all 47 of their tiles).
    //     Q is loaded per q-tile inside the loop (it is not cached whole). ---
    load_K_all: for (int t = 0; t < TILES; t++)
        for (int c = 0; c < D_HEAD; c++)
            for (int f = 0; f < BLOCK_SIZE; f++) {
                #pragma HLS PIPELINE II=1
                K_all[t][f][c] = (qk_t)(float)k_s.read();   // half -> Q6.10
            }
    load_V_all: for (int t = 0; t < TILES; t++)
        for (int c = 0; c < D_HEAD; c++)
            for (int f = 0; f < BLOCK_SIZE; f++) {
                #pragma HLS PIPELINE II=1
                V_all[t][f][c] = (qk_t)(float)v_s.read();   // half -> Q6.10
            }


    q_tile_loop: for (int qt = 0; qt < TILES; qt++) {

        // --- Reset accumulators for this Q tile ---
        init_acc: for (int f = 0; f < BLOCK_SIZE; f++) {
            #pragma HLS PIPELINE II=1
            m_row[f] = -1.0e30f;
            l_row[f] = 0.0f;
            for (int c = 0; c < D_HEAD; c++) {
                #pragma HLS UNROLL
                O_tile[f][c] = 0.0f;
            }
        }

        // --- Load the current Q tile from the stream (once per q-tile) ---
        load_Q: for (int c = 0; c < D_HEAD; c++) {
            for (int f = 0; f < BLOCK_SIZE; f++) {
                #pragma HLS PIPELINE II=1
                Q_tile[f][c] = (qk_t)(float)q_s.read();   // half -> Q6.10 (AP_RND+AP_SAT)
            }
        }

        kv_tile_loop: for (int kt = 0; kt < TILES; kt++) {

            S_phase_ki: for (int ki = 0; ki < BLOCK_SIZE; ki++) {
                S_phase_qi: for (int qi = 0; qi < BLOCK_SIZE; qi++) {
                    #pragma HLS PIPELINE II=1
                    dot_t dot = 0;
                    for (int c = 0; c < D_HEAD; c++) {
                        #pragma HLS UNROLL factor=16
                        dot += (dot_t)(Q_tile[qi][c] * K_all[kt][ki][c]);
                    }
                    S_tile[qi][ki] = (float)dot * 0.125f;   // scale = 1/sqrt(D_HEAD=64)
                }
            }

            max_phase: for (int qi = 0; qi < BLOCK_SIZE; qi++) {
                #pragma HLS PIPELINE II=1
                float m = m_row[qi];
                for (int ki = 0; ki < BLOCK_SIZE; ki++) {
                    #pragma HLS UNROLL
                    if (S_tile[qi][ki] > m) m = S_tile[qi][ki];
                }
                m_new[qi] = m;
            }

            corr_phase: for (int qi = 0; qi < BLOCK_SIZE; qi++) {
                #pragma HLS PIPELINE II=1
#ifdef __SYNTHESIS__
                float corr = hls::expf(m_row[qi] - m_new[qi]);
#else
                float corr = expf(m_row[qi] - m_new[qi]);
#endif
                l_row[qi] *= corr;               // l in float
                m_row[qi] = m_new[qi];
                e_t corr_q = (e_t)corr;          // corr in (0,1] -> fixed point
                for (int c = 0; c < D_HEAD; c++) {
                    #pragma HLS UNROLL
                    O_tile[qi][c] = O_tile[qi][c] * corr_q;   // fixed-point rescale
                }
            }

            acc_phase_ki: for (int ki = 0; ki < BLOCK_SIZE; ki++) {
                acc_phase_qi: for (int qi = 0; qi < BLOCK_SIZE; qi++) {
                    #pragma HLS PIPELINE II=2
                    #pragma HLS DEPENDENCE variable=O_tile inter distance=32 true
                    #pragma HLS DEPENDENCE variable=l_row inter distance=32 true
#ifdef __SYNTHESIS__
                    float ef = hls::expf(S_tile[qi][ki] - m_new[qi]);
#else
                    float ef = expf(S_tile[qi][ki] - m_new[qi]);
#endif
                    l_row[qi] += ef;             // l in float (softmax denominator)
                    e_t e = (e_t)ef;             // exp in [0,1] -> fixed point
                    for (int c = 0; c < D_HEAD; c++) {
                        #pragma HLS UNROLL factor=16
                        O_tile[qi][c] += (o_t)(e * V_all[kt][ki][c]);   // fixed-point MAC (1 cycle)
                    }
                }
            }

        } // kv_tile_loop (kt)

        write_out: for (int f = 0; f < BLOCK_SIZE; f++) {
            float inv_l = 1.0f / l_row[f];
            write_c: for (int c = 0; c < D_HEAD; c++) {
                #pragma HLS PIPELINE II=1
                out_s.write((half)((float)O_tile[f][c] * inv_l));   // fixed O -> float -> half
            }
        }

    } // q_tile_loop (qt)
}

static void flash_attention_head(
    const half* qkv_in,
    int head_id,
    hls::stream<half>& out_s
) {
    #pragma HLS DATAFLOW
    hls::stream<half> q_s("q_s");
    hls::stream<half> k_s("k_s");
    hls::stream<half> v_s("v_s");
    #pragma HLS STREAM variable=q_s depth=2048
    #pragma HLS STREAM variable=k_s depth=2048
    #pragma HLS STREAM variable=v_s depth=2048

    qkv_tile_loader(qkv_in, head_id, q_s, k_s, v_s);        // SLR0 (HBM)
    flash_attention_compute(q_s, k_s, v_s, out_s);           // SLR1/2 (compute)
}

void process_attention_head(
    const half* qkv_in,
    int head_id,
    hls::stream<half>& out_s
) {
#ifndef __SYNTHESIS__
    fprintf(stderr, "  [FA head=%d] starting flash_attention_head\n", head_id);
#endif

    flash_attention_head(qkv_in, head_id, out_s);
}
