#pragma once
#include "whisper_types.h"

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Input/output: HBM buffers in frame-major layout [FRAMES * D_MODEL]
// gamma and beta: local arrays of D_MODEL elements
void layer_norm(
    const half* x_in,
    half*       x_out,
    const half  gamma[D_MODEL],
    const half  beta[D_MODEL]
);

#include <hls_stream.h>
// Pure stream->stream compute (exposed to fuse ln1->qkv and ln2->ffn).
void ln_compute(hls::stream<half>& s_in, hls::stream<half>& s_out,
                const half gamma[D_MODEL], const half beta[D_MODEL]);
