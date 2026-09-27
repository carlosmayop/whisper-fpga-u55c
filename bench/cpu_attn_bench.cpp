// CPU attention benchmark, same computation as the FPGA kernel:
// 6 heads, FRAMES=1504, D_HEAD=64. Per head: S=Q*K^T*scale, softmax per row, O=softmax*V.
// Comparable to the per-layer latency of the FPGA kernel (baseline 213ms, noqi 81ms).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>

#define FRAMES  1504
#define D_HEAD  64
#define N_HEADS 6

static float Q[N_HEADS][FRAMES][D_HEAD];
static float K[N_HEADS][FRAMES][D_HEAD];
static float V[N_HEADS][FRAMES][D_HEAD];
static float O[N_HEADS][FRAMES][D_HEAD];

void attention_layer() {
    const float scale = 0.125f; // 1/sqrt(64)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < N_HEADS; h++) {
        for (int i = 0; i < FRAMES; i++) {
            float S[FRAMES];
            float m = -1e30f;
            for (int j = 0; j < FRAMES; j++) {
                float dot = 0.f;
                for (int c = 0; c < D_HEAD; c++) dot += Q[h][i][c] * K[h][j][c];
                dot *= scale;
                S[j] = dot;
                if (dot > m) m = dot;
            }
            float sum = 0.f;
            for (int j = 0; j < FRAMES; j++) { S[j] = expf(S[j] - m); sum += S[j]; }
            float inv = 1.f / sum;
            float acc[D_HEAD]; for (int c = 0; c < D_HEAD; c++) acc[c] = 0.f;
            for (int j = 0; j < FRAMES; j++) {
                float p = S[j] * inv;
                for (int c = 0; c < D_HEAD; c++) acc[c] += p * V[h][j][c];
            }
            for (int c = 0; c < D_HEAD; c++) O[h][i][c] = acc[c];
        }
    }
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 20;
    srand(1);
    for (int h = 0; h < N_HEADS; h++)
        for (int i = 0; i < FRAMES; i++)
            for (int c = 0; c < D_HEAD; c++) {
                Q[h][i][c] = (rand() % 1000) / 1000.f - 0.5f;
                K[h][i][c] = (rand() % 1000) / 1000.f - 0.5f;
                V[h][i][c] = (rand() % 1000) / 1000.f - 0.5f;
            }
    attention_layer(); // warm-up
    std::vector<double> t;
    for (int it = 0; it < iters; it++) {
        auto a = std::chrono::high_resolution_clock::now();
        attention_layer();
        auto b = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    double sum = 0; for (double x : t) sum += x;
    printf("=== atención CPU (1 capa, 6 cabezas, FRAMES=%d) ===\n", FRAMES);
    printf("  min=%.3f  median=%.3f  mean=%.3f  max=%.3f ms\n",
           t.front(), t[t.size()/2], sum/t.size(), t.back());
    printf("  -> 4 capas (mediana x4) = %.1f ms\n", t[t.size()/2] * 4);
    return 0;
}
