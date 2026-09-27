// CPU attention benchmark using the SAME implementation as whisper.cpp: ggml.
//
// It literally replicates the encoder's attention subgraph (src/whisper.cpp, the non-flash branch,
// which is the one used on CPU because whisper_kv_cache_get_padding() disables flash_attn
// when use_gpu == false):
//
//     K  = cast<F16>(reshape_3d(Kcur, d_head, n_head, n_ctx)) permuted to (d_head, n_ctx, n_head)
//     KQ = ggml_mul_mat(K, Q)
//     SM = ggml_soft_max_ext(KQ, NULL, 1/sqrt(d_head), 0)
//     V  = cast<F16>(permute(reshape_3d(Vcur, d_head, n_head, n_ctx), 1,2,0,3))  -> (n_ctx, d_head, n_head)
//     KQV = ggml_mul_mat(V, SM)
//     out = cont_2d(permute(KQV, 0,2,1,3), n_state, n_ctx)
//
// Q, K and V are materialised ALREADY in their final layout, just as the FPGA benchmark uploads QKV
// to HBM before the loop: only the attention is measured, not the linear projections nor the traffic.
//
// Measured unit: 1 pass = N_HEADS heads x FRAMES frames, identical to attn_bench_small
// (FPGA) and to cpu_attn_mt (in-house implementation), so that the mJ/pass are comparable.
//
// Usage: ggml_attn_bench [iters] [threads]      (threads default: $OMP_NUM_THREADS, or all)

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <thread>
#include <vector>

#define FRAMES  1504
#define D_HEAD  64
#define N_HEADS 6
#define N_STATE (D_HEAD * N_HEADS)

// Own IEEE-754 binary32 -> binary16 conversion, to avoid depending on internal headers.
static uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFFu;
    if (exp <= 0)  return (uint16_t)sign;                       // subnormal -> 0, suficiente aqui
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);           // overflow -> inf
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t x;
    if (exp == 0)       x = sign;                                  // subnormal -> 0
    else if (exp == 31) x = sign | 0x7F800000u | (man << 13);
    else                x = sign | ((exp - 15 + 127) << 23) | (man << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}

int main(int argc, char ** argv) {
    const int iters = (argc > 1) ? atoi(argv[1]) : 20;
    int n_threads;
    if (argc > 2)                         n_threads = atoi(argv[2]);
    else if (const char * e = getenv("OMP_NUM_THREADS")) n_threads = atoi(e);
    else                                  n_threads = (int) std::thread::hardware_concurrency();
    if (n_threads < 1) n_threads = 1;

    // Context with data: ~120 MB of tensors + headroom.
    struct ggml_init_params ip = { (size_t)512*1024*1024, NULL, false };
    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) { fprintf(stderr, "ggml_init fallo\n"); return 1; }

    // Final layouts, exactly as they reach the mul_mat in the encoder graph.
    struct ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D_HEAD, FRAMES, N_HEADS);
    struct ggml_tensor * K = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D_HEAD, FRAMES, N_HEADS);
    struct ggml_tensor * V = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, FRAMES, D_HEAD, N_HEADS);

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    {
        float *    q = (float *)    Q->data;
        uint16_t * k = (uint16_t *) K->data;
        uint16_t * v = (uint16_t *) V->data;
        for (int64_t i = 0; i < ggml_nelements(Q); i++) q[i] = dist(rng);
        for (int64_t i = 0; i < ggml_nelements(K); i++) k[i] = f32_to_f16(dist(rng));
        for (int64_t i = 0; i < ggml_nelements(V); i++) v[i] = f32_to_f16(dist(rng));
    }

    const float KQscale = 1.0f / sqrtf((float) D_HEAD);

    struct ggml_tensor * KQ  = ggml_mul_mat(ctx, K, Q);
    struct ggml_tensor * SM  = ggml_soft_max_ext(ctx, KQ, NULL, KQscale, 0.0f);
    struct ggml_tensor * KQV = ggml_mul_mat(ctx, V, SM);
    struct ggml_tensor * out = ggml_cont_2d(ctx, ggml_permute(ctx, KQV, 0, 2, 1, 3), N_STATE, FRAMES);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, NULL);
    std::vector<uint8_t> work(plan.work_size ? plan.work_size : 1);
    plan.work_data = work.data();

    ggml_graph_compute(gf, &plan);   // calentamiento, no se contabiliza

    std::vector<double> t;
    t.reserve(iters);
    for (int it = 0; it < iters; it++) {
        auto a = std::chrono::high_resolution_clock::now();
        ggml_graph_compute(gf, &plan);
        auto b = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }

    // Optional verification: compares the output of the ggml graph against a reference
    // computed right here with plain loops and the SAME Q, K, V. It confirms that
    // the replicated subgraph (above all V's transposed layout) computes the real attention.
    if (getenv("CHECK")) {
        const float *    q = (const float *)    Q->data;
        const uint16_t * k = (const uint16_t *) K->data;
        const uint16_t * v = (const uint16_t *) V->data;
        const float *    o = (const float *)    out->data;   // out es (N_STATE, FRAMES)
        double worst = 0.0;
        std::mt19937 pick(7);
        for (int trial = 0; trial < 64; trial++) {
            int h = pick() % N_HEADS, i = pick() % FRAMES, c = pick() % D_HEAD;
            std::vector<float> S(FRAMES);
            float m = -1e30f;
            for (int j = 0; j < FRAMES; j++) {
                float dot = 0.f;
                for (int d = 0; d < D_HEAD; d++)
                    dot += q[(int64_t)h*FRAMES*D_HEAD + (int64_t)i*D_HEAD + d]
                         * f16_to_f32(k[(int64_t)h*FRAMES*D_HEAD + (int64_t)j*D_HEAD + d]);
                dot *= KQscale; S[j] = dot; if (dot > m) m = dot;
            }
            float sum = 0.f;
            for (int j = 0; j < FRAMES; j++) { S[j] = expf(S[j]-m); sum += S[j]; }
            float acc = 0.f;
            for (int j = 0; j < FRAMES; j++)
                acc += (S[j]/sum) * f16_to_f32(v[(int64_t)h*D_HEAD*FRAMES + (int64_t)c*FRAMES + j]);
            // out[(h*D_HEAD + c), i] with ne0 = N_STATE
            float got = o[(int64_t)i*N_STATE + h*D_HEAD + c];
            worst = std::max(worst, (double)fabsf(got - acc));
        }
        printf("  CHECK: error absoluto maximo sobre 64 muestras = %.3e\n", worst);
    }

    std::sort(t.begin(), t.end());
    double sum = 0; for (double x : t) sum += x;
    printf("=== atencion ggml (%d cabezas, FRAMES=%d, D_HEAD=%d, K/V en F16, %d hilos) ===\n",
           N_HEADS, FRAMES, D_HEAD, n_threads);
    printf("  min=%.3f  median=%.3f  mean=%.3f  max=%.3f ms\n",
           t.front(), t[t.size()/2], sum/t.size(), t.back());
    printf("  -> 1 capa small (12 cabezas = 2 pasadas) = %.1f ms\n", t[t.size()/2]*2);
    printf("  -> 12 capas small = %.1f ms\n", t[t.size()/2]*2*12);
    printf("  work_size del plan = %.1f MB\n", plan.work_size/1e6);

    ggml_free(ctx);
    return 0;
}
