// ─────────────────────────────────────────────────────────────────────────────
// whisper_fpga_encoder.cpp
//
// XRT host implementation for whisper_encoder_top (conv layers removed from HW).
//
// Flow:
//   1. CPU: mel (FP32, channel-major) → conv1d_layer1 → conv1d_layer2 → +pos_emb
//      Result written as FP16 to bo_buf_a [FRAMES × D_MODEL].
//   2. FPGA: 4× transformer encoder layers starting from buf_a.
//
// Argument index reference (0-based, 3-head kernel — qkv_buffer × 3 aliases):
//   0  : l0_qkv_w   1:l0_wo_w    2:l0_fc1_w   3:l0_fc2_w   (m_axi, layer 0)
//   4-12: l0 small arrays (s_axilite)
//   13 : l1_qkv_w  14:l1_wo_w   15:l1_fc1_w  16:l1_fc2_w
//   17-25: l1 small arrays
//   26 : l2_qkv_w  27:l2_wo_w   28:l2_fc1_w  29:l2_fc2_w
//   30-38: l2 small arrays
//   39 : l3_qkv_w  40:l3_wo_w   41:l3_fc1_w  42:l3_fc2_w
//   43-53: l3 small arrays + ln_post
//   54 : buf_a   55:buf_b   56:ln1_out_buffer           (m_axi scratch)
//   57-59: qkv_buffer × 3 aliases                       (m_axi scratch)
//   60 : mha_buffer  61:residual2_buffer  62:ln2_out    (m_axi scratch)
//   63 : ffn_mid_buffer  64:layer_out                   (m_axi scratch/output)
//
// Input flow: ggml runs conv1+conv2 (SIMD optimized) → whisper_fpga_encode
// receives embd_conv in ggml layout, adds pos_emb, sends FP16 to FPGA.
// ─────────────────────────────────────────────────────────────────────────────
#include <chrono>
#include "whisper_fpga_encoder.h"

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

//  Model constants (must match whisper_types.h)
static constexpr int D_MODEL      = 384;
static constexpr int D_FF         = 1536;
static constexpr int FRAMES       = 1504;   // PADDED_FRAMES
static constexpr int AUDIO_FRAMES = 1500;
static constexpr int QKV_CH       = 1152;
static constexpr int BLOCK_SIZE   = 32;

// Quantized weight sizes in bytes (Q5.1 blocks: 24 bytes each)
static constexpr size_t SZ_QKV = (size_t)3*D_MODEL * (D_MODEL/BLOCK_SIZE) * 24;
static constexpr size_t SZ_WO  = (size_t)  D_MODEL * (D_MODEL/BLOCK_SIZE) * 24;
static constexpr size_t SZ_FC1 = (size_t)  D_FF    * (D_MODEL/BLOCK_SIZE) * 24;
static constexpr size_t SZ_FC2 = (size_t)  D_MODEL * (D_FF   /BLOCK_SIZE) * 24;

// -- Layout of the small_weights buffer (FP16, elements) ---------------------
// SW_PER_LAYER = 8*D_MODEL + D_FF = 8*384 + 1536 = 4608 elements per layer
// Layer L base = L * SW_PER_LAYER
//   +0*D_MODEL : ln1_g      [D_MODEL]
//   +1*D_MODEL : ln1_b      [D_MODEL]
//   +2*D_MODEL : ln2_g      [D_MODEL]
//   +3*D_MODEL : ln2_b      [D_MODEL]
//   +4*D_MODEL : q_bias     [D_MODEL]
//   +5*D_MODEL : v_bias     [D_MODEL]
//   +6*D_MODEL : attn_out_bias [D_MODEL]
//   +7*D_MODEL : ffn1_bias  [D_FF]
//   +7*D_MODEL+D_FF : ffn2_bias [D_MODEL]
// After the 4 layers (SW_LNPOST = 4*SW_PER_LAYER = 18432):
//   +0       : ln_post_gamma [D_MODEL]
//   +D_MODEL : ln_post_beta  [D_MODEL]
// Total: 19200 elementos FP16 = 38400 bytes
static constexpr int SW_PER_LAYER = 8 * D_MODEL + D_FF;   // 4608
static constexpr int SW_LNPOST    = 4 * SW_PER_LAYER;      // 18432
static constexpr int SW_TOTAL     = SW_LNPOST + 2 * D_MODEL; // 19200

//  New kernel: argument indices
// 0-3:   l0 qkv/wo/fc1/fc2 weights
// 4-7:   l1 weights
// 8-11:  l2 weights
// 12-15: l3 weights
// 16:    small_weights (biases + layer norms)
// 17-18: buf_a, buf_b
// 19:    ln1_out_buffer
// 20-22: qkv_buffer × 3 aliases
// 23:    mha_buffer
// 24:    residual2_buffer
// 25:    ln2_out_buffer
// 26:    ffn_mid_buffer
// 27:    layer_out

//  FP32 ↔ FP16 conversion 
static uint16_t f32_to_f16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint16_t sign = (u >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = u & 0x7FFFFFu;
    if (exp <= 0)  return sign;
    if (exp >= 31) return sign | 0x7C00u;
    return sign | (uint16_t)((uint32_t)exp << 10) | (uint16_t)(mant >> 13);
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = (h & 0x3FFu) << 13;
    uint32_t u;
    if      (exp == 0u)  u = sign | mant;
    else if (exp == 31u) u = sign | 0x7F800000u | mant;
    else                 u = sign | ((exp - 15u + 127u) << 23) | mant;
    float f; memcpy(&f, &u, 4); return f;
}

//  File loading 
static std::vector<uint16_t> load_fp16(const std::string& dir,
                                        const std::string& name,
                                        size_t count) {
    std::string path = dir + "/" + name + ".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<uint16_t> buf(count);
    f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)(count * 2));
    if (!f) throw std::runtime_error("Short read from: " + path);
    return buf;
}

static std::vector<uint8_t> load_raw(const std::string& dir,
                                      const std::string& name,
                                      size_t bytes) {
    std::string path = dir + "/" + name + ".bin";
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<uint8_t> buf(bytes);
    f.read(reinterpret_cast<char*>(buf.data()), (std::streamsize)bytes);
    if (!f) throw std::runtime_error("Short read from: " + path);
    return buf;
}


//  FP16 weight vector → FP32 
static std::vector<float> fp16_vec_to_fp32(const std::vector<uint16_t>& h) {
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); i++) f[i] = f16_to_f32(h[i]);
    return f;
}

//  Context 
struct whisper_fpga_context {
    xrt::device device;
    xrt::uuid   uuid;
    xrt::kernel kernel;

    // Per-layer quantized weights (Q5.1)
    std::vector<xrt::bo> bo_qkv_w, bo_wo_w, bo_fc1_w, bo_fc2_w;

    // Small weights buffer (biases + layer norms, packed in HBM)
    xrt::bo bo_small_weights;

    // Scratch buffers
    xrt::bo bo_buf_a, bo_buf_b;
    xrt::bo bo_q, bo_qb, bo_qc;
    xrt::bo bo_mha;
    xrt::bo bo_res2, bo_ln2_out;
    xrt::bo bo_out;

    // Positional embedding (FP32, loaded once at init, time-major [FRAMES × D_MODEL])
    std::vector<float> pos_emb;

    // Host-mapped pointers
    uint16_t* p_buf_a = nullptr;   // write-only: CPU writes conv output here
    uint16_t* p_out   = nullptr;   // read-only:  CPU reads transformer output here
};

//  Public API 

extern "C"
whisper_fpga_context* whisper_fpga_init(const char* xclbin_path,
                                        const char* weights_dir,
                                        unsigned int device_idx) {
    auto* ctx = new whisper_fpga_context();
    try {
        std::string wdir(weights_dir);
        char name[64];

        std::cout << "[whisper_fpga] Opening device " << device_idx << "...\n";
        ctx->device = xrt::device(device_idx);
        ctx->uuid   = ctx->device.load_xclbin(xclbin_path);
        ctx->kernel = xrt::kernel(ctx->device, ctx->uuid, "whisper_encoder_top");
        std::cout << "[whisper_fpga] Kernel loaded from " << xclbin_path << "\n";

        auto alloc = [&](int arg_idx, size_t bytes) -> xrt::bo {
            return xrt::bo(ctx->device, bytes,
                           xrt::bo::flags::normal,
                           ctx->kernel.group_id(arg_idx));
        };
        auto upload_fp16 = [&](xrt::bo& bo, const std::vector<uint16_t>& data) {
            auto* p = bo.map<uint16_t*>();
            memcpy(p, data.data(), data.size() * 2);
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        };
        auto upload_raw = [&](xrt::bo& bo, const std::vector<uint8_t>& data) {
            auto* p = bo.map<uint8_t*>();
            memcpy(p, data.data(), data.size());
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        };

        //  Load positional embedding (FP16 on disk → FP32 in RAM) 
        std::cout << "[whisper_fpga] Loading positional embedding...\n";
        ctx->pos_emb = fp16_vec_to_fp32(load_fp16(wdir, "pos_emb", FRAMES * D_MODEL));

        //  Allocate HBM buffers 
        std::cout << "[whisper_fpga] Allocating HBM buffers...\n";

        // Arg indices: l0=0-3, l1=4-7, l2=8-11, l3=12-15
        static const int qkv_idx[4] = { 0,  4,  8, 12};
        static const int wo_idx[4]  = { 1,  5,  9, 13};
        static const int fc1_idx[4] = { 2,  6, 10, 14};
        static const int fc2_idx[4] = { 3,  7, 11, 15};
        ctx->bo_qkv_w.reserve(4); ctx->bo_wo_w.reserve(4);
        ctx->bo_fc1_w.reserve(4); ctx->bo_fc2_w.reserve(4);
        for (int i = 0; i < 4; i++) {
            ctx->bo_qkv_w.push_back(alloc(qkv_idx[i], SZ_QKV));
            ctx->bo_wo_w.push_back( alloc(wo_idx[i],  SZ_WO));
            ctx->bo_fc1_w.push_back(alloc(fc1_idx[i], SZ_FC1));
            ctx->bo_fc2_w.push_back(alloc(fc2_idx[i], SZ_FC2));
        }

        // arg 16: small_weights
        ctx->bo_small_weights = alloc(16, (size_t)SW_TOTAL * 2);

        const size_t sz_frame = (size_t)FRAMES * D_MODEL * 2;
        const size_t sz_qkv   = (size_t)FRAMES * QKV_CH  * 2;
        const size_t sz_ffn   = (size_t)FRAMES * D_FF    * 2;

        ctx->bo_buf_a   = alloc(17, sz_frame);
        ctx->bo_buf_b   = alloc(18, sz_frame);
        ctx->bo_q  = alloc(19, sz_qkv);   // qkv_buffer  (banco propio)
        ctx->bo_qb = alloc(20, sz_qkv);   // qkv_buffer_b (banco propio)
        ctx->bo_qc = alloc(21, sz_qkv);   // qkv_buffer_c (banco propio)
        ctx->bo_mha     = alloc(22, sz_frame);
        ctx->bo_res2    = alloc(23, sz_frame);
        ctx->bo_ln2_out = alloc(24, sz_frame);
        ctx->bo_out     = alloc(25, sz_frame);

        //  Empaquetar y subir small_weights al HBM 
        std::cout << "[whisper_fpga] Loading and uploading small weight arrays...\n";
        {
            auto* sw = ctx->bo_small_weights.map<uint16_t*>();
            auto pack = [&](int offset_elems, const char* fname, int count) {
                auto v = load_fp16(wdir, fname, count);
                memcpy(sw + offset_elems, v.data(), (size_t)count * 2);
            };
            for (int L = 0; L < 4; L++) {
                int base = L * SW_PER_LAYER;
                snprintf(name, sizeof(name), "l%d_ln1_g",         L); pack(base + 0*D_MODEL,         name, D_MODEL);
                snprintf(name, sizeof(name), "l%d_ln1_b",         L); pack(base + 1*D_MODEL,         name, D_MODEL);
                snprintf(name, sizeof(name), "l%d_ln2_g",         L); pack(base + 2*D_MODEL,         name, D_MODEL);
                snprintf(name, sizeof(name), "l%d_ln2_b",         L); pack(base + 3*D_MODEL,         name, D_MODEL);
                snprintf(name, sizeof(name), "l%d_q_bias",        L); pack(base + 4*D_MODEL,         name, D_MODEL);
                snprintf(name, sizeof(name), "l%d_v_bias",        L); pack(base + 5*D_MODEL,         name, D_MODEL);
                snprintf(name, sizeof(name), "l%d_attn_out_bias", L); pack(base + 6*D_MODEL,         name, D_MODEL);
                snprintf(name, sizeof(name), "l%d_ffn1_bias",     L); pack(base + 7*D_MODEL,         name, D_FF);
                snprintf(name, sizeof(name), "l%d_ffn2_bias",     L); pack(base + 7*D_MODEL + D_FF,  name, D_MODEL);
            }
            pack(SW_LNPOST + 0*D_MODEL, "ln_post_g", D_MODEL);
            pack(SW_LNPOST + 1*D_MODEL, "ln_post_b", D_MODEL);
            ctx->bo_small_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        //  Upload quantized weight matrices 
        std::cout << "[whisper_fpga] Uploading quantized weight matrices...\n";
        for (int i = 0; i < 4; i++) {
            snprintf(name, sizeof(name), "l%d_qkv_w", i);
            upload_raw(ctx->bo_qkv_w[i], load_raw(wdir, name, SZ_QKV));
            snprintf(name, sizeof(name), "l%d_wo_w",  i);
            upload_raw(ctx->bo_wo_w[i],  load_raw(wdir, name, SZ_WO));
            snprintf(name, sizeof(name), "l%d_fc1_w", i);
            upload_raw(ctx->bo_fc1_w[i], load_raw(wdir, name, SZ_FC1));
            snprintf(name, sizeof(name), "l%d_fc2_w", i);
            upload_raw(ctx->bo_fc2_w[i], load_raw(wdir, name, SZ_FC2));
        }

        //  Map host pointers 
        ctx->p_buf_a = ctx->bo_buf_a.map<uint16_t*>();
        ctx->p_out   = ctx->bo_out.map<uint16_t*>();

        std::cout << "[whisper_fpga] Ready.\n";
        return ctx;

    } catch (const std::exception& e) {
        std::cerr << "[whisper_fpga] Init failed: " << e.what() << "\n";
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
        // Add pos_emb to ggml conv output, convert to FP16 → bo_buf_a 
        // embd_conv layout (ggml [n_audio_ctx, D_MODEL]): element(t,c) = data[t + c*n_audio_ctx]
        // pos_emb layout (time-major [FRAMES × D_MODEL]):  element(t,c) = data[t*D_MODEL + c]
        // bo_buf_a layout (FP16, time-major [FRAMES × D_MODEL]): element(t,c) = data[t*D_MODEL + c]
        for (int t = 0; t < FRAMES; t++) {
            for (int c = 0; c < D_MODEL; c++) {
                float v = ctx->pos_emb[(size_t)t * D_MODEL + c];
                if (t < n_audio_ctx)
                    v += embd_conv[t + (size_t)c * n_audio_ctx];
                ctx->p_buf_a[(size_t)t * D_MODEL + c] = f32_to_f16(v);
            }
        }
        ctx->bo_buf_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        //  FPGA: 4× transformer encoder 
        xrt::run run(ctx->kernel);

        // Layer 0 (args 0-3)
        run.set_arg( 0, ctx->bo_qkv_w[0]);
        run.set_arg( 1, ctx->bo_wo_w[0]);
        run.set_arg( 2, ctx->bo_fc1_w[0]);
        run.set_arg( 3, ctx->bo_fc2_w[0]);
        // Layer 1 (args 4-7)
        run.set_arg( 4, ctx->bo_qkv_w[1]);
        run.set_arg( 5, ctx->bo_wo_w[1]);
        run.set_arg( 6, ctx->bo_fc1_w[1]);
        run.set_arg( 7, ctx->bo_fc2_w[1]);
        // Layer 2 (args 8-11)
        run.set_arg( 8, ctx->bo_qkv_w[2]);
        run.set_arg( 9, ctx->bo_wo_w[2]);
        run.set_arg(10, ctx->bo_fc1_w[2]);
        run.set_arg(11, ctx->bo_fc2_w[2]);
        // Layer 3 (args 12-15)
        run.set_arg(12, ctx->bo_qkv_w[3]);
        run.set_arg(13, ctx->bo_wo_w[3]);
        run.set_arg(14, ctx->bo_fc1_w[3]);
        run.set_arg(15, ctx->bo_fc2_w[3]);
        // arg 16: small_weights
        run.set_arg(16, ctx->bo_small_weights);
        // Scratch + output (args 17-27)
        run.set_arg(17, ctx->bo_buf_a);
        run.set_arg(18, ctx->bo_buf_b);
        run.set_arg(19, ctx->bo_q);    // qkv_buffer  (banco propio)
        run.set_arg(20, ctx->bo_qb);   // qkv_buffer_b (banco propio)
        run.set_arg(21, ctx->bo_qc);   // qkv_buffer_c (banco propio)
        run.set_arg(22, ctx->bo_mha);
        run.set_arg(23, ctx->bo_res2);
        run.set_arg(24, ctx->bo_ln2_out);
        run.set_arg(25, ctx->bo_out);

        auto _t0 = std::chrono::high_resolution_clock::now();
        run.start();
        run.wait();
        auto _t1 = std::chrono::high_resolution_clock::now();
        double _ms = std::chrono::duration<double, std::milli>(_t1 - _t0).count();
        std::cerr << "[whisper_fpga] kernel run.start->run.wait = " << _ms << " ms\n";

        //  FP16 → FP32, trim padding 
        ctx->bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        for (int t = 0; t < AUDIO_FRAMES; t++) {
            for (int c = 0; c < D_MODEL; c++) {
                out[t * D_MODEL + c] = f16_to_f32(ctx->p_out[t * D_MODEL + c]);
            }
        }
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[whisper_fpga] Encode failed: " << e.what() << "\n";
        return false;
    }
}
