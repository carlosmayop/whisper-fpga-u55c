// whisper_fpga_encoder_stub.cpp
//
// CPU stub — same public API as whisper_fpga_encoder.cpp (XRT version) but
// calls whisper_encoder_top() directly as plain C++, without XRT or FPGA.
//
// Use this to test the full whisper.cpp ↔ encoder integration on a machine
// without an Alveo card.  When the real hardware is available, replace this
// file with whisper_fpga_encoder.cpp and recompile — no other changes needed.
//
// Input flow (mirrors the real path):
//   whisper.cpp: ggml runs conv1+conv2 → embd_conv [n_audio_ctx × D_MODEL]
//   stub: add pos_emb, convert to FP16 → buf_a → call whisper_encoder_top()
//
// Build (via CMake in xrt_host/build_stub/):
//   cmake .. -DFPGA_STUB=ON
//   make -j$(nproc)

#include "whisper_fpga_encoder.h"

// HLS kernel headers — work fine in plain C++ (same as csim)
#include "whisper_types.h"
#include "matmul.h"
#include "layer_norm.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Layout of the small_weights buffer (same as encoder_top.cpp):
// SW_PER_LAYER = 8*D_MODEL + D_FF = 4608 elements per layer
// Layer L base = L * SW_PER_LAYER
//   +0..6*D_MODEL : ln1_g, ln1_b, ln2_g, ln2_b, q_bias, v_bias, attn_out_bias
//   +7*D_MODEL    : ffn1_bias [D_FF]
//   +7*D_MODEL+D_FF : ffn2_bias [D_MODEL]
// SW_LNPOST = 4*SW_PER_LAYER : ln_post_gamma, ln_post_beta
static constexpr int STUB_SW_PER_LAYER = 8 * D_MODEL + D_FF;
static constexpr int STUB_SW_LNPOST    = 4 * STUB_SW_PER_LAYER;
static constexpr int STUB_SW_TOTAL     = STUB_SW_LNPOST + 2 * D_MODEL;

// Forward declaration, must match encoder_top.cpp exactly
void whisper_encoder_top(
    const bus_t* l0_qkv_w, const bus_t* l0_wo_w,
    const bus_t* l0_fc1_w, const bus_t* l0_fc2_w,

    const bus_t* l1_qkv_w, const bus_t* l1_wo_w,
    const bus_t* l1_fc1_w, const bus_t* l1_fc2_w,

    const bus_t* l2_qkv_w, const bus_t* l2_wo_w,
    const bus_t* l2_fc1_w, const bus_t* l2_fc2_w,

    const bus_t* l3_qkv_w, const bus_t* l3_wo_w,
    const bus_t* l3_fc1_w, const bus_t* l3_fc2_w,

    const half* small_weights,

    half* buf_a, half* buf_b,
    half* qkv_buffer, half* qkv_buffer_b, half* qkv_buffer_c,
    half* mha_buffer, half* residual2_buffer,
    half* ln2_out_buffer,
    half* layer_out
);

//  Constants 
static constexpr int N_FRAMES = FRAMES;       // 1504
static constexpr int N_AUDIO  = AUDIO_FRAMES; // 1500

// Quantized weight byte sizes per layer
static constexpr size_t BYTES_QKV = (size_t)3*D_MODEL * (D_MODEL/BLOCK_SIZE) * 24;
static constexpr size_t BYTES_WO  = (size_t)  D_MODEL * (D_MODEL/BLOCK_SIZE) * 24;
static constexpr size_t BYTES_FC1 = (size_t)  D_FF    * (D_MODEL/BLOCK_SIZE) * 24;
static constexpr size_t BYTES_FC2 = (size_t)  D_MODEL * (D_FF   /BLOCK_SIZE) * 24;

static size_t bytes_to_bus_count(size_t bytes) {
    return (bytes + sizeof(bus_t) - 1) / sizeof(bus_t);
}

//  File loading 
static std::vector<half> load_fp16(const std::string& dir,
                                    const std::string& name,
                                    size_t count) {
    std::string path = dir + "/" + name + ".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<half> buf(count);
    f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)(count * sizeof(half)));
    if (!f) throw std::runtime_error("Short read: " + path);
    return buf;
}

static std::vector<bus_t> load_quant(const std::string& dir,
                                      const std::string& name,
                                      size_t bytes) {
    std::string path = dir + "/" + name + ".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t n_bus = bytes_to_bus_count(bytes);
    std::vector<bus_t> buf(n_bus, bus_t(0));
    f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)bytes);
    if (!f) throw std::runtime_error("Short read: " + path);
    return buf;
}

//  Context 
struct whisper_fpga_context {
    // Positional embedding (FP16, time-major [N_FRAMES × D_MODEL])
    std::vector<half> pos_emb;

    // Transformer weights
    std::vector<bus_t> qkv_w[4], wo_w[4], fc1_w[4], fc2_w[4];
    std::vector<half>  ln1_g[4], ln1_b[4], ln2_g[4], ln2_b[4];
    std::vector<half>  q_bias[4], v_bias[4], attn_out_bias[4];
    std::vector<half>  ffn1_bias[4], ffn2_bias[4];
    std::vector<half>  ln_post_g, ln_post_b;

    // Scratch buffers (allocated once, reused every call)
    std::vector<half> buf_a, buf_b;
    std::vector<half> q_buf, q_buf_b, q_buf_c;
    std::vector<half> mha, res2, ln2_out;
    std::vector<half> out_buf;
};

//  Public API 

extern "C"
whisper_fpga_context* whisper_fpga_init(const char* xclbin_path,
                                        const char* weights_dir,
                                        unsigned int /*device_idx*/) {
    (void)xclbin_path; // ignored in CPU stub

    auto* ctx = new whisper_fpga_context();
    try {
        std::string wdir(weights_dir);
        char name[64];

        std::cout << "[whisper_fpga_stub] Loading weights from " << wdir << " ...\n";

        ctx->pos_emb   = load_fp16(wdir, "pos_emb",   (size_t)N_FRAMES * D_MODEL);
        ctx->ln_post_g = load_fp16(wdir, "ln_post_g", D_MODEL);
        ctx->ln_post_b = load_fp16(wdir, "ln_post_b", D_MODEL);

        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "l%d_qkv_w",         i); ctx->qkv_w[i]        = load_quant(wdir, name, BYTES_QKV);
            snprintf(name, sizeof(name), "l%d_wo_w",          i); ctx->wo_w[i]         = load_quant(wdir, name, BYTES_WO);
            snprintf(name, sizeof(name), "l%d_fc1_w",         i); ctx->fc1_w[i]        = load_quant(wdir, name, BYTES_FC1);
            snprintf(name, sizeof(name), "l%d_fc2_w",         i); ctx->fc2_w[i]        = load_quant(wdir, name, BYTES_FC2);
            snprintf(name, sizeof(name), "l%d_ln1_g",         i); ctx->ln1_g[i]        = load_fp16(wdir, name, D_MODEL);
            snprintf(name, sizeof(name), "l%d_ln1_b",         i); ctx->ln1_b[i]        = load_fp16(wdir, name, D_MODEL);
            snprintf(name, sizeof(name), "l%d_ln2_g",         i); ctx->ln2_g[i]        = load_fp16(wdir, name, D_MODEL);
            snprintf(name, sizeof(name), "l%d_ln2_b",         i); ctx->ln2_b[i]        = load_fp16(wdir, name, D_MODEL);
            snprintf(name, sizeof(name), "l%d_q_bias",        i); ctx->q_bias[i]       = load_fp16(wdir, name, D_MODEL);
            snprintf(name, sizeof(name), "l%d_v_bias",        i); ctx->v_bias[i]       = load_fp16(wdir, name, D_MODEL);
            snprintf(name, sizeof(name), "l%d_attn_out_bias", i); ctx->attn_out_bias[i]= load_fp16(wdir, name, D_MODEL);
            snprintf(name, sizeof(name), "l%d_ffn1_bias",     i); ctx->ffn1_bias[i]    = load_fp16(wdir, name, D_FF);
            snprintf(name, sizeof(name), "l%d_ffn2_bias",     i); ctx->ffn2_bias[i]    = load_fp16(wdir, name, D_MODEL);
        }

        ctx->buf_a  .resize((size_t)N_FRAMES * D_MODEL);
        ctx->buf_b  .resize((size_t)N_FRAMES * D_MODEL);
        ctx->q_buf.resize((size_t)N_FRAMES * QKV_CHANNELS);
        ctx->q_buf_b.resize((size_t)N_FRAMES * QKV_CHANNELS);
        ctx->q_buf_c.resize((size_t)N_FRAMES * QKV_CHANNELS);
        ctx->mha    .resize((size_t)N_FRAMES * D_MODEL);
        ctx->res2   .resize((size_t)N_FRAMES * D_MODEL);
        ctx->ln2_out.resize((size_t)N_FRAMES * D_MODEL);
        ctx->out_buf.resize((size_t)N_FRAMES * D_MODEL);

        std::cout << "[whisper_fpga_stub] Ready (CPU simulation mode).\n";
        return ctx;

    } catch (const std::exception& e) {
        std::cerr << "[whisper_fpga_stub] Init failed: " << e.what() << "\n";
        delete ctx;
        return nullptr;
    }
}

extern "C"
void whisper_fpga_free(whisper_fpga_context* ctx) {
    delete ctx;
}

extern "C"
bool whisper_fpga_encode(whisper_fpga_context* ctx,
                          const float* embd_conv,
                          int          n_audio_ctx,
                          float*       out) {
    try {
        //  embd_conv (ggml layout) + pos_emb → buf_a (FP16 time-major) 
        // embd_conv layout: element(t,c) = data[t + c * n_audio_ctx]
        // pos_emb layout:   element(t,c) = data[t * D_MODEL + c]  (time-major)
        // buf_a layout:     element(t,c) = data[t * D_MODEL + c]  (FP16)
        for (int t = 0; t < N_FRAMES; t++) {
            for (int c = 0; c < D_MODEL; c++) {
                float v = (t < n_audio_ctx) ? embd_conv[t + (size_t)c * n_audio_ctx] : 0.0f;
                v += (float)ctx->pos_emb[(size_t)t * D_MODEL + c];
                ctx->buf_a[(size_t)t * D_MODEL + c] = (half)v;
            }
        }

        //  Empaquetar small_weights en buffer plano 
        std::vector<half> sw(STUB_SW_TOTAL);
        for (int L = 0; L < 4; L++) {
            int base = L * STUB_SW_PER_LAYER;
            auto pack = [&](const std::vector<half>& arr, int off) {
                memcpy(sw.data() + base + off, arr.data(), arr.size() * sizeof(half));
            };
            pack(ctx->ln1_g[L],        0*D_MODEL);
            pack(ctx->ln1_b[L],        1*D_MODEL);
            pack(ctx->ln2_g[L],        2*D_MODEL);
            pack(ctx->ln2_b[L],        3*D_MODEL);
            pack(ctx->q_bias[L],       4*D_MODEL);
            pack(ctx->v_bias[L],       5*D_MODEL);
            pack(ctx->attn_out_bias[L],6*D_MODEL);
            pack(ctx->ffn1_bias[L],    7*D_MODEL);
            pack(ctx->ffn2_bias[L],    7*D_MODEL + D_FF);
        }
        memcpy(sw.data() + STUB_SW_LNPOST,           ctx->ln_post_g.data(), D_MODEL * sizeof(half));
        memcpy(sw.data() + STUB_SW_LNPOST + D_MODEL, ctx->ln_post_b.data(), D_MODEL * sizeof(half));

        //  Llamar al kernel HLS directamente 
        whisper_encoder_top(
            ctx->qkv_w[0].data(), ctx->wo_w[0].data(),
            ctx->fc1_w[0].data(), ctx->fc2_w[0].data(),

            ctx->qkv_w[1].data(), ctx->wo_w[1].data(),
            ctx->fc1_w[1].data(), ctx->fc2_w[1].data(),

            ctx->qkv_w[2].data(), ctx->wo_w[2].data(),
            ctx->fc1_w[2].data(), ctx->fc2_w[2].data(),

            ctx->qkv_w[3].data(), ctx->wo_w[3].data(),
            ctx->fc1_w[3].data(), ctx->fc2_w[3].data(),

            sw.data(),

            ctx->buf_a.data(), ctx->buf_b.data(),
            ctx->q_buf.data(), ctx->q_buf_b.data(), ctx->q_buf_c.data(),
            ctx->mha.data(), ctx->res2.data(),
            ctx->ln2_out.data(),
            ctx->out_buf.data()
        );

        //  Diagnostic 
        {
            bool has_nan = false, has_inf = false;
            for (size_t i = 0; i < (size_t)N_FRAMES * D_MODEL; i++) {
                float v = (float)ctx->out_buf[i];
                if (std::isnan(v)) { has_nan = true; break; }
                if (std::isinf(v)) { has_inf = true; break; }
            }
            fprintf(stderr, "[whisper_fpga_stub] out[0..7]:");
            for (int i = 0; i < 8; i++)
                fprintf(stderr, " %.4f", (float)ctx->out_buf[i]);
            fprintf(stderr, "\n[whisper_fpga_stub] has_nan=%d has_inf=%d\n",
                    (int)has_nan, (int)has_inf);
        }

        //  FP16 [1504×384] → FP32 [1500×384] 
        for (int t = 0; t < N_AUDIO; t++)
            for (int c = 0; c < D_MODEL; c++)
                out[(size_t)t * D_MODEL + c] = (float)ctx->out_buf[(size_t)t * D_MODEL + c];

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[whisper_fpga_stub] Encode failed: " << e.what() << "\n";
        return false;
    }
}
