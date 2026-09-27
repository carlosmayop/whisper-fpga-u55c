// Host for the SMALL version: 6 qkv (args 0-5) + 6 mha (args 6-11) + head_base (arg 12).
// One invocation = 6 heads. With head_base=0 it measures heads 0-5; small has 12 -> 2 passes/layer.
// QKV_CHANNELS=2304 (small).
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>

// IEEE binary32 -> binary16 conversion, to generate realistic input data without
// depending on <half>. The values used here fit comfortably in the normal range.
static uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFFu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

#define FRAMES        1504
#define QKV_CHANNELS  2304   // SMALL (was 1152 on tiny)
#define D_HEAD        64

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "uso: %s <xclbin> [iters] [head_base] [patron]\n", argv[0]);
        fprintf(stderr, "  patron: const (por defecto, todo a 1.0) | rand (aleatorio en [-0.5,0.5))\n");
        return 1;
    }
    int iters     = (argc > 2) ? atoi(argv[2]) : 50;
    int head_base = (argc > 3) ? atoi(argv[3]) : 0;
    // With 'const' all 6.93 MB of QKV hold the same value: the multipliers and the buses
    // barely toggle, so the measured dynamic power is a FLOOR, not that of the kernel on real audio.
    // 'rand' uses the SAME distribution as cpu_attn_bench.cpp (seed 1, [-0.5,0.5)) so that the
    // switching activity is comparable between the two platforms.
    const bool aleatorio = (argc > 4) && (strcmp(argv[4], "rand") == 0);

    auto device = xrt::device(0);
    auto uuid   = device.load_xclbin(argv[1]);
    auto krnl   = xrt::kernel(device, uuid, "attention_top");

    const size_t qkv_bytes = (size_t)FRAMES * QKV_CHANNELS * 2;   // 6.93 MB
    const size_t mha_bytes = (size_t)FRAMES * D_HEAD * 2;

    srand(1);   // reproducible, same as the CPU benchmark
    std::vector<xrt::bo> qkv, mha;
    for (int i = 0; i < 6; i++) qkv.push_back(xrt::bo(device, qkv_bytes, krnl.group_id(i)));
    for (int i = 0; i < 6; i++) mha.push_back(xrt::bo(device, mha_bytes, krnl.group_id(6 + i)));

    for (int i = 0; i < 6; i++) {
        auto p = qkv[i].map<uint16_t*>();
        if (aleatorio) {
            for (size_t k = 0; k < qkv_bytes / 2; k++)
                p[k] = f32_to_f16((rand() % 1000) / 1000.f - 0.5f);
        } else {
            for (size_t k = 0; k < qkv_bytes / 2; k++) p[k] = 0x3C00;
        }
        qkv[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    printf("[attn_bench_small] qkv subido (%.1f MB/banco, patron=%s). head_base=%d. Ejecutando %d iters...\n",
           qkv_bytes / 1e6, aleatorio ? "rand" : "const", head_base, iters);

    auto run = xrt::run(krnl);
    for (int i = 0; i < 6; i++) run.set_arg(i, qkv[i]);
    for (int i = 0; i < 6; i++) run.set_arg(6 + i, mha[i]);
    run.set_arg(12, head_base);   // head_base scalar arg

    run.start(); run.wait();  // warm-up
    std::vector<double> t;
    for (int it = 0; it < iters; it++) {
        auto a = std::chrono::high_resolution_clock::now();
        run.start(); run.wait();
        auto b = std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    for (int i = 0; i < 6; i++) mha[i].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::sort(t.begin(), t.end());
    double sum = 0; for (double x : t) sum += x;
    printf("\n=== atencion SMALL (6 cabezas, head_base=%d, datos=%s) ===\n",
           head_base, aleatorio ? "aleatorios" : "constantes");
    printf("  min=%.3f  median=%.3f  mean=%.3f  max=%.3f ms\n",
           t.front(), t[t.size()/2], sum/t.size(), t.back());
    printf("  -> 1 capa small (12 cabezas = 2 pasadas) = %.1f ms\n", t[t.size()/2] * 2);
    printf("  -> 12 capas small = %.1f ms\n", t[t.size()/2] * 2 * 12);
    return 0;
}
