#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "whisper_types.h"

// ---- Top-level declaration ----
void whisper_encoder_top(
    //const half* mel_spectrogram_in,
    //const half  weights1[CONV1_OUT_CHANNELS][MEL_CHANNELS][3],
    //const half  biases1[CONV1_OUT_CHANNELS],
    //const half  weights2[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][3],
    //const half  biases2[CONV2_OUT_CHANNELS],
    //const half* pos_emb,
    const bus_t* l0_qkv_w, const bus_t* l0_wo_w,
    const bus_t* l0_fc1_w, const bus_t* l0_fc2_w,
    const half  l0_ln1_g[D_MODEL], const half l0_ln1_b[D_MODEL],
    const half  l0_ln2_g[D_MODEL], const half l0_ln2_b[D_MODEL],
    const half  l0_q_bias[D_MODEL], const half l0_v_bias[D_MODEL],
    const half  l0_attn_out_bias[D_MODEL],
    const half  l0_ffn1_bias[D_FF], const half l0_ffn2_bias[D_MODEL],
    const bus_t* l1_qkv_w, const bus_t* l1_wo_w,
    const bus_t* l1_fc1_w, const bus_t* l1_fc2_w,
    const half  l1_ln1_g[D_MODEL], const half l1_ln1_b[D_MODEL],
    const half  l1_ln2_g[D_MODEL], const half l1_ln2_b[D_MODEL],
    const half  l1_q_bias[D_MODEL], const half l1_v_bias[D_MODEL],
    const half  l1_attn_out_bias[D_MODEL],
    const half  l1_ffn1_bias[D_FF], const half l1_ffn2_bias[D_MODEL],
    const bus_t* l2_qkv_w, const bus_t* l2_wo_w,
    const bus_t* l2_fc1_w, const bus_t* l2_fc2_w,
    const half  l2_ln1_g[D_MODEL], const half l2_ln1_b[D_MODEL],
    const half  l2_ln2_g[D_MODEL], const half l2_ln2_b[D_MODEL],
    const half  l2_q_bias[D_MODEL], const half l2_v_bias[D_MODEL],
    const half  l2_attn_out_bias[D_MODEL],
    const half  l2_ffn1_bias[D_FF], const half l2_ffn2_bias[D_MODEL],
    const bus_t* l3_qkv_w, const bus_t* l3_wo_w,
    const bus_t* l3_fc1_w, const bus_t* l3_fc2_w,
    const half  l3_ln1_g[D_MODEL], const half l3_ln1_b[D_MODEL],
    const half  l3_ln2_g[D_MODEL], const half l3_ln2_b[D_MODEL],
    const half  l3_q_bias[D_MODEL], const half l3_v_bias[D_MODEL],
    const half  l3_attn_out_bias[D_MODEL],
    const half  l3_ffn1_bias[D_FF], const half l3_ffn2_bias[D_MODEL],
    const half  ln_post_gamma[D_MODEL],
    const half  ln_post_beta[D_MODEL],
    half* buf_a, half* buf_b,
    half* ln1_out_buffer,
    half* qkv_buffer, half* qkv_buffer_b, half* qkv_buffer_c,
    half* qkv_buffer_d, half* qkv_buffer_e, half* qkv_buffer_f,
    half* mha_buffer,
    half* residual2_buffer, half* ln2_out_buffer, half* ffn_mid_buffer,
    half* layer_out
);

// ---- Utilidades ----

union HalfBits { uint16_t i; half f; HalfBits() : i(0) {} };

static bool load_fp16(const char* path, half* dst, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "  [WARN] No se encontró: " << path << "\n"; return false; }
    for (size_t i = 0; i < count; i++) {
        uint16_t bits;
        f.read(reinterpret_cast<char*>(&bits), 2);
        HalfBits hb; hb.i = bits;
        dst[i] = hb.f;
    }
    return true;
}

static bool load_bus(const char* path, bus_t* dst, size_t n_bus_words) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "  [WARN] No se encontró: " << path << "\n"; return false; }
    for (size_t w = 0; w < n_bus_words; w++) {
        uint8_t bytes[64] = {};
        f.read(reinterpret_cast<char*>(bytes), 64);
        bus_t val = 0;
        for (int b = 63; b >= 0; b--) {
            val = (val << 8) | bytes[b];
        }
        dst[w] = val;
    }
    return true;
}

static size_t file_size(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return (size_t)f.tellg();
}

// ---- Sizes of the bus_t buffers (in 512-bit / 64-byte words) ----
// Bloques Q5.1: 24 bytes/bloque
// QKV: 1152 * (384/32) * 24 = 1152 * 12 * 24 = 331776 B -> 5184 words of 64 B
// WO:   384 * 12 * 24 =  110592 B → 1728 palabras
// FC1: 1536 * 12 * 24 =  442368 B → 6912 palabras
// FC2:  384 * 48 * 24 =  442368 B -> 6912 words  (D_FF/32=48 blocks per row)
static const size_t SZ_QKV = (3*D_MODEL * (D_MODEL/BLOCK_SIZE) * 24 + 63) / 64;
static const size_t SZ_WO  = (  D_MODEL * (D_MODEL/BLOCK_SIZE) * 24 + 63) / 64;
static const size_t SZ_FC1 = (  D_FF    * (D_MODEL/BLOCK_SIZE) * 24 + 63) / 64;
static const size_t SZ_FC2 = (  D_MODEL * (D_FF   /BLOCK_SIZE) * 24 + 63) / 64;

// ---- main ----
int main() {
    std::cout << "==========================================================\n";
    std::cout << " TESTBENCH: whisper_encoder_top (4 capas + ln_post)\n";
    std::cout << "==========================================================\n";

    // -- Inputs and small weights (FP16) -------------------------------
    half* mel_in   = new half[TIME_FRAMES * MEL_CHANNELS]();
    half* pos_emb  = new half[FRAMES * D_MODEL]();

    half (*w1)[MEL_CHANNELS][3]      = new half[CONV1_OUT_CHANNELS][MEL_CHANNELS][3]();
    half* b1                          = new half[CONV1_OUT_CHANNELS]();
    half (*w2)[CONV2_IN_CHANNELS][3] = new half[CONV2_OUT_CHANNELS][CONV2_IN_CHANNELS][3]();
    half* b2                          = new half[CONV2_OUT_CHANNELS]();

    half* ln1_g[4]; half* ln1_b[4];
    half* ln2_g[4]; half* ln2_b[4];
    bus_t* qkv_w[4]; bus_t* wo_w[4];
    bus_t* fc1_w[4]; bus_t* fc2_w[4];
    half* q_bias[4]; half* v_bias[4];
    half* attn_out_bias[4];
    half* ffn1_bias[4]; half* ffn2_bias[4];

    for (int i = 0; i < 4; i++) {
        ln1_g[i] = new half[D_MODEL]();  ln1_b[i] = new half[D_MODEL]();
        ln2_g[i] = new half[D_MODEL]();  ln2_b[i] = new half[D_MODEL]();
        qkv_w[i] = new bus_t[SZ_QKV](); wo_w[i]  = new bus_t[SZ_WO]();
        fc1_w[i] = new bus_t[SZ_FC1](); fc2_w[i] = new bus_t[SZ_FC2]();
        q_bias[i]        = new half[D_MODEL]();
        v_bias[i]        = new half[D_MODEL]();
        attn_out_bias[i] = new half[D_MODEL]();
        ffn1_bias[i]     = new half[D_FF]();
        ffn2_bias[i]     = new half[D_MODEL]();
    }

    half* ln_post_g = new half[D_MODEL]();
    half* ln_post_b = new half[D_MODEL]();

    // ── Buffers scratch HBM ────────────────────────────────────────────
    half* buf_a          = new half[(size_t)FRAMES * D_MODEL]();
    half* buf_b          = new half[(size_t)FRAMES * D_MODEL]();
    half* ln1_out        = new half[(size_t)FRAMES * D_MODEL]();
    half* qkv_buf        = new half[(size_t)FRAMES * QKV_CHANNELS]();
    half* qkv_buf_b      = qkv_buf;   // alias: mismo buffer, puerto AXI separado
    half* qkv_buf_c      = qkv_buf;
    half* qkv_buf_d      = qkv_buf;
    half* qkv_buf_e      = qkv_buf;
    half* qkv_buf_f      = qkv_buf;
    half* mha_buf        = new half[(size_t)FRAMES * D_MODEL]();
    half* residual2_buf  = new half[(size_t)FRAMES * D_MODEL]();
    half* ln2_out        = new half[(size_t)FRAMES * D_MODEL]();
    half* ffn_mid_buf    = new half[(size_t)FRAMES * D_FF]();
    half* layer_out      = new half[(size_t)FRAMES * D_MODEL]();
    half* expected       = new half[(size_t)FRAMES * D_MODEL]();

    // ── Intentar cargar vectores golden ────────────────────────────────
    // Absolute path to the test_vectors directory (valid from any cwd in csim)
    std::string src_dir;
    {
        std::string src_file = __FILE__;
        size_t p = src_file.find_last_of("/\\");
        src_dir = (p != std::string::npos) ? src_file.substr(0, p) : ".";
    }
    std::string tv_dir = src_dir + "/test_vectors";
    bool have_golden = true;

    auto tv = [&](const char* name) -> std::string {
        return tv_dir + "/" + name + ".bin";
    };

#define LOAD_FP16(name, dst, count) \
    have_golden &= load_fp16(tv(name).c_str(), dst, count)
#define LOAD_BUS(name, dst, count) \
    have_golden &= load_bus(tv(name).c_str(), dst, count)

    std::cout << "[1/3] Cargando vectores de test...\n";
    LOAD_FP16("mel_input",    mel_in,  (size_t)TIME_FRAMES * MEL_CHANNELS);
    LOAD_FP16("pos_emb",      pos_emb, (size_t)FRAMES * D_MODEL);
    LOAD_FP16("conv1_weights",(half*)w1, (size_t)CONV1_OUT_CHANNELS * MEL_CHANNELS * 3);
    LOAD_FP16("conv1_biases", b1,        CONV1_OUT_CHANNELS);
    LOAD_FP16("conv2_weights",(half*)w2, (size_t)CONV2_OUT_CHANNELS * CONV2_IN_CHANNELS * 3);
    LOAD_FP16("conv2_biases", b2,        CONV2_OUT_CHANNELS);

    for (int i = 0; i < 4; i++) {
        char n[64];
        sprintf(n, "l%d_qkv_w",        i); LOAD_BUS(n,  qkv_w[i],        SZ_QKV);
        sprintf(n, "l%d_wo_w",          i); LOAD_BUS(n,  wo_w[i],         SZ_WO);
        sprintf(n, "l%d_fc1_w",         i); LOAD_BUS(n,  fc1_w[i],        SZ_FC1);
        sprintf(n, "l%d_fc2_w",         i); LOAD_BUS(n,  fc2_w[i],        SZ_FC2);
        sprintf(n, "l%d_ln1_g",         i); LOAD_FP16(n, ln1_g[i],        D_MODEL);
        sprintf(n, "l%d_ln1_b",         i); LOAD_FP16(n, ln1_b[i],        D_MODEL);
        sprintf(n, "l%d_ln2_g",         i); LOAD_FP16(n, ln2_g[i],        D_MODEL);
        sprintf(n, "l%d_ln2_b",         i); LOAD_FP16(n, ln2_b[i],        D_MODEL);
        sprintf(n, "l%d_q_bias",        i); LOAD_FP16(n, q_bias[i],       D_MODEL);
        sprintf(n, "l%d_v_bias",        i); LOAD_FP16(n, v_bias[i],       D_MODEL);
        sprintf(n, "l%d_attn_out_bias", i); LOAD_FP16(n, attn_out_bias[i],D_MODEL);
        sprintf(n, "l%d_ffn1_bias",     i); LOAD_FP16(n, ffn1_bias[i],    D_FF);
        sprintf(n, "l%d_ffn2_bias",     i); LOAD_FP16(n, ffn2_bias[i],    D_MODEL);
    }
    LOAD_FP16("ln_post_g",      ln_post_g, D_MODEL);
    LOAD_FP16("ln_post_b",      ln_post_b, D_MODEL);
    LOAD_FP16("expected_output",expected,  (size_t)FRAMES * D_MODEL);

    if (!have_golden) {
        std::cout << "  Vectores golden no encontrados. Inicializando con valores triviales.\n";
        // Default values: LN with gamma=1, trivial weights (d=0 -> output 0)
        for (int i = 0; i < 4; i++) {
            for (int c = 0; c < D_MODEL; c++) {
                ln1_g[i][c] = (half)1.0f;
                ln2_g[i][c] = (half)1.0f;
            }
        }
        for (int c = 0; c < D_MODEL; c++) {
            ln_post_g[c] = (half)1.0f;
        }
        // Mel with a small alternating value
        for (int i = 0; i < TIME_FRAMES * MEL_CHANNELS; i++)
            mel_in[i] = (i % 2 == 0) ? (half)0.01f : (half)-0.01f;
    }

    std::cout << "[2/3] Ejecutando whisper_encoder_top...\n";
    whisper_encoder_top(
        //mel_in, w1, b1, w2, b2, pos_emb,
        qkv_w[0], wo_w[0], fc1_w[0], fc2_w[0],
        ln1_g[0], ln1_b[0], ln2_g[0], ln2_b[0],
        q_bias[0], v_bias[0], attn_out_bias[0], ffn1_bias[0], ffn2_bias[0],
        qkv_w[1], wo_w[1], fc1_w[1], fc2_w[1],
        ln1_g[1], ln1_b[1], ln2_g[1], ln2_b[1],
        q_bias[1], v_bias[1], attn_out_bias[1], ffn1_bias[1], ffn2_bias[1],
        qkv_w[2], wo_w[2], fc1_w[2], fc2_w[2],
        ln1_g[2], ln1_b[2], ln2_g[2], ln2_b[2],
        q_bias[2], v_bias[2], attn_out_bias[2], ffn1_bias[2], ffn2_bias[2],
        qkv_w[3], wo_w[3], fc1_w[3], fc2_w[3],
        ln1_g[3], ln1_b[3], ln2_g[3], ln2_b[3],
        q_bias[3], v_bias[3], attn_out_bias[3], ffn1_bias[3], ffn2_bias[3],
        ln_post_g, ln_post_b,
        buf_a, buf_b,
        ln1_out,
        qkv_buf, qkv_buf_b, qkv_buf_c,
        qkv_buf_d, qkv_buf_e, qkv_buf_f,
        mha_buf,
        residual2_buf, ln2_out, ffn_mid_buf,
        layer_out
    );
    std::cout << "  >>> Ejecución terminada <<<\n";

    std::cout << "[3/3] Verificando salida...\n";

    // -- NaN / Inf check ------------------------------------------------
    int nan_count = 0, inf_count = 0;
    for (int i = 0; i < FRAMES * D_MODEL; i++) {
        float v = (float)layer_out[i];
        if (std::isnan(v)) nan_count++;
        if (std::isinf(v)) inf_count++;
    }

    std::cout << "  NaNs: " << nan_count << "  Infs: " << inf_count << "\n";

    // -- Sample of the first values ------------------------------------
    std::cout << "  Primeros 8 valores del frame 0:\n";
    for (int c = 0; c < 8; c++)
        std::cout << "    out[0][" << c << "] = " << (float)layer_out[c] << "\n";

    // -- Comparison against golden (if available) ----------------------
    bool pass = (nan_count == 0 && inf_count == 0);

    if (have_golden) {
        double sum_ae = 0, max_ae = 0;
        int    n_total = FRAMES * D_MODEL;
        for (int i = 0; i < n_total; i++) {
            double ae = std::fabs((double)(float)layer_out[i]
                                - (double)(float)expected[i]);
            sum_ae += ae;
            if (ae > max_ae) max_ae = ae;
        }
        double mean_ae = sum_ae / n_total;

        std::cout << "  Error medio absoluto (MAE): " << mean_ae << "\n";
        std::cout << "  Error máximo absoluto:      " << max_ae  << "\n";

        // Tolerancia: Q5.1 introduce ~1-2% error; 4 capas acumulan ~0.3 MAE
        const double TOL_MAE = 0.5;
        const double TOL_MAX = 3.0;
        if (mean_ae < TOL_MAE && max_ae < TOL_MAX)
            std::cout << "  [PASS] Dentro de tolerancia (MAE<" << TOL_MAE
                      << " MAX<" << TOL_MAX << ")\n";
        else {
            std::cout << "  [FAIL] Fuera de tolerancia\n";
            pass = false;
        }
    } else {
        std::cout << "  [INFO] Sin golden — solo se verificó NaN/Inf\n";
    }

    std::cout << "==========================================================\n";
    std::cout << (pass ? " RESULTADO: PASS\n" : " RESULTADO: FAIL\n");
    std::cout << "==========================================================\n";

    // ── Limpieza ──────────────────────────────────────────────────────
    delete[] mel_in; delete[] pos_emb;
    delete[](half(*)[MEL_CHANNELS][3])w1; delete[] b1;
    delete[](half(*)[CONV2_IN_CHANNELS][3])w2; delete[] b2;
    for (int i = 0; i < 4; i++) {
        delete[] ln1_g[i]; delete[] ln1_b[i];
        delete[] ln2_g[i]; delete[] ln2_b[i];
        delete[] qkv_w[i]; delete[] wo_w[i];
        delete[] fc1_w[i]; delete[] fc2_w[i];
        delete[] q_bias[i]; delete[] v_bias[i];
        delete[] attn_out_bias[i];
        delete[] ffn1_bias[i]; delete[] ffn2_bias[i];
    }
    delete[] ln_post_g; delete[] ln_post_b;
    delete[] buf_a; delete[] buf_b;
    delete[] ln1_out; delete[] qkv_buf; delete[] mha_buf;
    delete[] residual2_buf; delete[] ln2_out; delete[] ffn_mid_buf;
    delete[] layer_out; delete[] expected;

    return pass ? 0 : 1;
}
