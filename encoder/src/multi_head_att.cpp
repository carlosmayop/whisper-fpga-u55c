#include <hls_math.h>
#include <hls_stream.h>
#include <hls_half.h>
#include "whisper_types.h"
#ifndef __SYNTHESIS__
#include <cmath>
#include <cstdio>
#endif

// Flash Attention for one head — FA-2 style phase structure.
//
// Replaces the per-row update (compute_S / exp / acc / update per qi — five
// short pipelines whose drain dominated: 1121 cycles per qi for ~190 of work)
// with whole-tile phases. Each phase is a single long pipeline whose drain
// is paid once per K/V tile instead of once per row:
//
//   S_phase    : S_tile[qi][ki] = Q[qi]·K[ki] * 0.125 — 1024 iters, II=1
//   max_phase  : m_new[qi] = max(m_row[qi], row max of S_tile) — 32 iters
//   corr_phase : O[qi] *= exp(m_old - m_new), l *= same — 32 iters
//   acc_phase  : O[qi][c] += exp(S[qi][ki]-m_new[qi]) * V[ki][c]
//                ki outer, qi inner → carried dep on O[qi] has distance 32 ≥
//                FP add latency, so II=1 with no extra accumulator banks.
//
// Memory layout of qkv_in (same as the old extract_qkv_head expected):
//   qkv_in[ tile * QKV_CHANNELS * BLOCK_SIZE + channel * BLOCK_SIZE + frame_in_tile ]
//
// Per (qt,kt) tile pair: ~4.2K cycles loads + ~2.6K compute, vs 40.1K for the
// per-row version (5.8x). All tile arrays partition dim=2 → LUTRAM columns
// with dynamic row addressing; no dynamic-index register MUXes.

// SLR0 (mover): the only point of the attention that touches HBM. Reads qkv_in in FA
// order (per qt: Q tile; per kt: K tile, V tile) and emits it on three streams. It re-reads
// K/V for every Q tile just as before (same HBM traffic), but now from a
// data-mover, so that the compute never accesses HBM.
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

    // OPT-A (stationary K,V): each Q/K/V tile is read from HBM EXACTLY once.
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
    half attention_out[FRAMES][D_HEAD]
) {
    // OPT-A: ONLY K,V are cached (they are the ones re-read 47 times in the inner loop).
    // Q is iterated in the OUTER loop, so each q-tile is used once; loading the
    // current tile per iteration is enough (no redundant re-read is possible). This avoids Q_all
    // and saves 1/3 of the URAM. K,V kept in half (lossless: they come from FP16 qkv) to halve
    // the URAM; cyclic factor=16 dim=3 to match the UNROLL factor=16 of the MAC.
    half K_all[TILES][BLOCK_SIZE][D_HEAD];
    #pragma HLS ARRAY_PARTITION variable=K_all cyclic factor=16 dim=3
    #pragma HLS BIND_STORAGE variable=K_all type=ram_2p impl=uram
    half V_all[TILES][BLOCK_SIZE][D_HEAD];
    #pragma HLS ARRAY_PARTITION variable=V_all cyclic factor=16 dim=3
    #pragma HLS BIND_STORAGE variable=V_all type=ram_2p impl=uram

    // Current Q tile (loaded per q-tile from the stream, as in the original design).
    float Q_tile[BLOCK_SIZE][D_HEAD];
    #pragma HLS ARRAY_PARTITION variable=Q_tile complete dim=2

    float O_tile[BLOCK_SIZE][D_HEAD];
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

    // --- OPT-A: load ONLY the head's K and V once (all 47 of their tiles).
    //     Q is loaded per q-tile inside the loop (it is not cached whole). ---
    load_K_all: for (int t = 0; t < TILES; t++)
        for (int c = 0; c < D_HEAD; c++)
            for (int f = 0; f < BLOCK_SIZE; f++) {
                #pragma HLS PIPELINE II=1
                K_all[t][f][c] = k_s.read();
            }
    load_V_all: for (int t = 0; t < TILES; t++)
        for (int c = 0; c < D_HEAD; c++)
            for (int f = 0; f < BLOCK_SIZE; f++) {
                #pragma HLS PIPELINE II=1
                V_all[t][f][c] = v_s.read();
            }

    // =========================================================================
    // Outer loop: iterate over Q tiles (each tile = BLOCK_SIZE consecutive frames)
    // =========================================================================
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
                Q_tile[f][c] = (float)q_s.read();
            }
        }

        // =====================================================================
        // Inner loop: iterate over K/V tiles (K,V already cached on-chip)
        // =====================================================================
        kv_tile_loop: for (int kt = 0; kt < TILES; kt++) {

            // --- Phase 1: whole-tile scores. ki outer / qi inner, II=1. ---
            // Q read at row qi, K row fixed per ki → both LUTRAM row reads.
            // S writes go to bank ki at address qi: one write port, no conflict.
            // II=2 with 32 MACs halves LUT/DSP vs full 64-wide unroll; the
            // extra 1024 cycles per tile are noise next to the 4.2K loads.
            // C1: UNROLL factor=16 / II=4 (was 32 / II=2) — quarter the parallel
            // MACs vs full unroll → fewer DSP/LUT/mux in the attention, lowering
            // SLR density to relieve the residual ~8K routing overlaps. Extra
            // cycles per tile are negligible next to the HBM loads, and the
            // 342 MHz csynth has huge margin over the 250 MHz target.
            S_phase_ki: for (int ki = 0; ki < BLOCK_SIZE; ki++) {
                S_phase_qi: for (int qi = 0; qi < BLOCK_SIZE; qi++) {
                    #pragma HLS PIPELINE II=4
                    float dot = 0.0f;
                    for (int c = 0; c < D_HEAD; c++) {
                        #pragma HLS UNROLL factor=16
                        dot += Q_tile[qi][c] * (float)K_all[kt][ki][c];
                    }
                    S_tile[qi][ki] = dot * 0.125f;   // scale = 1/sqrt(D_HEAD=64)
                }
            }

            // --- Phase 2: per-row max via comparator tree (one row per cycle) ---
            max_phase: for (int qi = 0; qi < BLOCK_SIZE; qi++) {
                #pragma HLS PIPELINE II=1
                float m = m_row[qi];
                for (int ki = 0; ki < BLOCK_SIZE; ki++) {
                    #pragma HLS UNROLL
                    if (S_tile[qi][ki] > m) m = S_tile[qi][ki];
                }
                m_new[qi] = m;
            }

            // --- Phase 3: rescale previous O and l by exp(m_old - m_new) ---
            corr_phase: for (int qi = 0; qi < BLOCK_SIZE; qi++) {
                #pragma HLS PIPELINE II=1
#ifdef __SYNTHESIS__
                float corr = hls::expf(m_row[qi] - m_new[qi]);
#else
                float corr = expf(m_row[qi] - m_new[qi]);
#endif
                l_row[qi] *= corr;
                m_row[qi] = m_new[qi];
                for (int c = 0; c < D_HEAD; c++) {
                    #pragma HLS UNROLL
                    O_tile[qi][c] *= corr;
                }
            }

            // --- Phase 4: accumulate. ki outer / qi inner → O[qi] and l[qi]
            // carried deps have distance 32 ≥ FP latency → II=1. V row fixed
            // per ki; exp computed inline (one expf per cycle). ---
            acc_phase_ki: for (int ki = 0; ki < BLOCK_SIZE; ki++) {
                acc_phase_qi: for (int qi = 0; qi < BLOCK_SIZE; qi++) {
                    #pragma HLS PIPELINE II=4
                    #pragma HLS DEPENDENCE variable=O_tile inter distance=32 true
                    #pragma HLS DEPENDENCE variable=l_row inter distance=32 true
#ifdef __SYNTHESIS__
                    float e = hls::expf(S_tile[qi][ki] - m_new[qi]);
#else
                    float e = expf(S_tile[qi][ki] - m_new[qi]);
#endif
                    l_row[qi] += e;
                    for (int c = 0; c < D_HEAD; c++) {
                        #pragma HLS UNROLL factor=16
                        O_tile[qi][c] += e * (float)V_all[kt][ki][c];
                    }
                }
            }

        } // kv_tile_loop (kt)

        // --- Normalize and write output tile to attention_out ---
        write_out: for (int f = 0; f < BLOCK_SIZE; f++) {
            float inv_l = 1.0f / l_row[f];
            write_c: for (int c = 0; c < D_HEAD; c++) {
                #pragma HLS PIPELINE II=1
                attention_out[qt * BLOCK_SIZE + f][c] = (half)(O_tile[f][c] * inv_l);
            }
        }

    } // q_tile_loop (qt)
}

// DATAFLOW wrapper: data-mover (SLR0) + flash compute (SLR1/2) connected by
// Q/K/V streams. The SLR crossing is now stream-only (narrow FIFOs), not the
// qkv_in AXI port crossing together with the compute logic.
static void flash_attention_head(
    const half* qkv_in,
    int head_id,
    half attention_out[FRAMES][D_HEAD]
) {
    #pragma HLS DATAFLOW
    hls::stream<half> q_s("q_s");
    hls::stream<half> k_s("k_s");
    hls::stream<half> v_s("v_s");
    #pragma HLS STREAM variable=q_s depth=2048
    #pragma HLS STREAM variable=k_s depth=2048
    #pragma HLS STREAM variable=v_s depth=2048

    qkv_tile_loader(qkv_in, head_id, q_s, k_s, v_s);        // SLR0 (HBM)
    flash_attention_compute(q_s, k_s, v_s, attention_out);   // SLR1/2 (compute)
}

// Public interface — same signature as the original process_attention_head so
// encoder_top.cpp and process_head_triple require no changes.
void process_attention_head(
    const half* qkv_in,
    int head_id,
    half attention_out[FRAMES][D_HEAD]
) {
    #pragma HLS ARRAY_PARTITION variable=attention_out type=complete dim=2

#ifndef __SYNTHESIS__
    fprintf(stderr, "  [FA head=%d] starting flash_attention_head\n", head_id);
#endif

    flash_attention_head(qkv_in, head_id, attention_out);

#ifndef __SYNTHESIS__
    if (head_id == 0) {
        bool anan = false; float amax = 0;
        for (int f = 0; f < FRAMES; f++)
            for (int c = 0; c < D_HEAD; c++) {
                float a = (float)attention_out[f][c];
                if (std::isnan(a) || std::isinf(a)) anan = true;
                if (std::abs(a) > amax) amax = std::abs(a);
            }
        fprintf(stderr, "  [FA h0] attn_out: bad=%d max=%.4f  [0][0..3]=%.4f %.4f %.4f %.4f\n",
            anan, amax,
            (float)attention_out[0][0], (float)attention_out[0][1],
            (float)attention_out[0][2], (float)attention_out[0][3]);
    }
#endif
}
