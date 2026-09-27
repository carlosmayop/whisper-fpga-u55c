#!/usr/bin/env python3
"""
Genera vectores de test para whisper_encoder_top (csim).

Implementa el mismo encoder Whisper Tiny en numpy (FP32) para obtener
una referencia golden. Los pesos se cuantizan a Q5.1 con el layout exacto
que espera hbm_to_q5_1_stream (192 bits/bloque, gearbox de 512 bits).

Uso:
    pip install numpy
    python3 generate_test_vectors.py [--seed 42] [--tiny-model PATH_GGUF]

Salida: directorio test_vectors/ con todos los .bin para encoder_tb.cpp
"""

import os, sys, struct, argparse
import numpy as np

#  Constants (must match whisper_types.h) 
MEL_CH   = 80
T_FRAMES = 3000        # TIME_FRAMES
D_MODEL  = 384
D_HEAD   = 64
N_HEADS  = 6
D_FF     = 1536
N_LAYERS = 4
A_FRAMES = 1500        # AUDIO_FRAMES
P_FRAMES = 1504        # PADDED_FRAMES (FRAMES)
BS       = 32          # BLOCK_SIZE / QK8_1

OUT_DIR  = "test_vectors"

#  Q5.1 quantization (must match memory_interfaces.h)
def pack_q5_1_block(vals):
    """vals: float32 array [32] → bytes [24]  (layout: d:2 m:2 qh:4 qs:16)"""
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        d, m = 0.0, float(vmin)
        q = np.zeros(BS, dtype=np.int32)
    else:
        d = (vmax - vmin) / 31.0
        m = float(vmin)
        q = np.clip(np.round((vals - m) / d), 0, 31).astype(np.int32)

    d_bits = np.float16(d).view(np.uint16).item()
    m_bits = np.float16(m).view(np.uint16).item()

    qh = np.uint32(0)
    for j in range(BS):
        if q[j] & 0x10:
            qh |= np.uint32(1 << j)

    qs = np.zeros(16, dtype=np.uint8)
    for j in range(16):
        qs[j] = (int(q[2*j]) & 0x0F) | ((int(q[2*j+1]) & 0x0F) << 4)

    return struct.pack('<HHI', d_bits, m_bits, int(qh)) + bytes(qs)

def matrix_to_q5_1_hbm(W):
    """W: [out_f, in_f] float32 → bytes alineados a 64 (bus_t de 512 bits)
    Layout: out_f * (in_f/32) bloques de 24 bytes, empaquetados en palabras de 512 bits.
    """
    assert W.shape[1] % BS == 0
    raw = bytearray()
    for row in W:
        for b in range(W.shape[1] // BS):
            raw += pack_q5_1_block(row[b*BS:(b+1)*BS].astype(np.float32))
    pad = (64 - len(raw) % 64) % 64
    raw += b'\x00' * pad
    return bytes(raw)

def dequant_q5_1_matrix(data, out_f, in_f):
    """Inverso exacto de pack_q5_1_block — para la referencia Python."""
    n_blocks = in_f // BS
    W = np.zeros((out_f, in_f), dtype=np.float32)
    off = 0
    for row in range(out_f):
        for b in range(n_blocks):
            d_bits, m_bits, qh = struct.unpack_from('<HHI', data, off)
            d = struct.unpack('<e', struct.pack('<H', d_bits))[0]
            m = struct.unpack('<e', struct.pack('<H', m_bits))[0]
            qs_bytes = list(data[off+8:off+24])
            off += 24
            for i in range(BS):
                nibble = (qs_bytes[i // 2] >> ((i % 2) * 4)) & 0x0F
                bit5   = (qh >> i) & 1
                w_int  = (bit5 << 4) | nibble
                W[row, b*BS + i] = d * w_int + m
    return W

#  GELU (tanh approximation, must match math_utils.h apply_gelu) -
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

#  LayerNorm (eps=1e-5, must match layer_norm.cpp) 
def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var  = ((x - mean)**2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * g + b

#  Encoder reference 

def conv1_ref(mel, W, b):
    """Conv1d stride=1, kernel=3, pad=1, GELU. mel:[T,80] W:[384,80,3] → [T,384]"""
    mel_pad = np.pad(mel, [(1,1),(0,0)])  # [T+2, 80]
    out = b.copy()[np.newaxis, :]         # [1, 384] broadcast
    for k in range(3):
        out = out + mel_pad[k:k+T_FRAMES] @ W[:,:,k].T
    return gelu(out)

def conv2_ref(x, W, b):
    """Conv1d stride=2, kernel=3, left-pad=1, GELU. x:[T,384] → [T//2,384]"""
    # Left-pad with one zero frame (window inicia a cero)
    x_pad = np.vstack([np.zeros((1, D_MODEL), np.float32), x])  # [T+1, 384]
    out = np.zeros((A_FRAMES, D_MODEL), np.float32)
    for k in range(3):
        out += x_pad[k::2][:A_FRAMES] @ W[:,:,k].T
    return gelu(out + b)

def mha_ref(x, W_qkv_data, W_o_data, q_bias, v_bias, attn_out_bias):
    """Multi-head self-attention con Q5.1 pesos dequantizados."""
    T = x.shape[0]
    W_qkv = dequant_q5_1_matrix(W_qkv_data, 3*D_MODEL, D_MODEL)  # [1152, 384]
    W_o   = dequant_q5_1_matrix(W_o_data,   D_MODEL,   D_MODEL)  # [384, 384]

    qkv = x @ W_qkv.T                                   # [T, 1152]
    Q = qkv[:, :D_MODEL]          + q_bias
    K = qkv[:, D_MODEL:2*D_MODEL]
    V = qkv[:, 2*D_MODEL:]        + v_bias

    out = np.zeros((T, D_MODEL), np.float32)
    for h in range(N_HEADS):
        s, e = h*D_HEAD, (h+1)*D_HEAD
        Qh = Q[:, s:e]; Kh = K[:, s:e]; Vh = V[:, s:e]
        scores = Qh @ Kh.T * 0.125                       # escala = 1/8
        scores -= scores.max(-1, keepdims=True)
        attn = np.exp(scores)
        attn /= attn.sum(-1, keepdims=True)
        out[:, s:e] = attn @ Vh

    return out @ W_o.T + attn_out_bias                   # W_O projection + bias

def ffn_ref(x, W_fc1_data, W_fc2_data, ffn1_bias, ffn2_bias):
    """FC1(GELU) + FC2 con bias."""
    W1 = dequant_q5_1_matrix(W_fc1_data, D_FF,    D_MODEL)
    W2 = dequant_q5_1_matrix(W_fc2_data, D_MODEL, D_FF)
    h  = gelu(x @ W1.T + ffn1_bias)
    return h @ W2.T + ffn2_bias

def encoder_layer_ref(x, qkv_d, wo_d, fc1_d, fc2_d, ln1g, ln1b, ln2g, ln2b,
                      q_bias, v_bias, attn_out_bias, ffn1_bias, ffn2_bias):
    x = x + mha_ref(layer_norm(x, ln1g, ln1b), qkv_d, wo_d, q_bias, v_bias, attn_out_bias)
    x = x + ffn_ref(layer_norm(x, ln2g, ln2b), fc1_d, fc2_d, ffn1_bias, ffn2_bias)
    return x

#  Weight generation 

def make_weights(rng, scale=0.02):
    """Pesos aleatorios numericamente estables."""
    def r(*sh): return rng.normal(0, scale, sh).astype(np.float32)
    def o(n):   return np.ones(n,  np.float32)
    def z(n):   return np.zeros(n, np.float32)

    w = {
        'mel':       rng.normal(0, 0.3, (T_FRAMES, MEL_CH)).astype(np.float32),
        'conv1_w':   r(D_MODEL, MEL_CH, 3),
        'conv1_b':   z(D_MODEL),
        'conv2_w':   r(D_MODEL, D_MODEL, 3),
        'conv2_b':   z(D_MODEL),
        'pos_emb':   r(P_FRAMES, D_MODEL),
        'ln_post_g': o(D_MODEL),
        'ln_post_b': z(D_MODEL),
    }
    for i in range(N_LAYERS):
        w[f'l{i}_qkv_w']        = r(3*D_MODEL, D_MODEL)
        w[f'l{i}_wo_w']         = r(D_MODEL, D_MODEL)
        w[f'l{i}_fc1_w']        = r(D_FF,    D_MODEL)
        w[f'l{i}_fc2_w']        = r(D_MODEL, D_FF)
        w[f'l{i}_ln1_g']        = o(D_MODEL)
        w[f'l{i}_ln1_b']        = z(D_MODEL)
        w[f'l{i}_ln2_g']        = o(D_MODEL)
        w[f'l{i}_ln2_b']        = z(D_MODEL)
        w[f'l{i}_q_bias']       = z(D_MODEL)
        w[f'l{i}_v_bias']       = z(D_MODEL)
        w[f'l{i}_attn_out_bias']= z(D_MODEL)
        w[f'l{i}_ffn1_bias']    = z(D_FF)
        w[f'l{i}_ffn2_bias']    = z(D_MODEL)
    return w

#  Main 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"[1/4] Generando pesos (seed={args.seed})...")
    W = make_weights(rng)

    print("[2/4] Cuantizando matrices a Q5.1...")
    Q5 = {}
    for name in [f'l{i}_{t}' for i in range(N_LAYERS) for t in ['qkv_w','wo_w','fc1_w','fc2_w']]:
        mat = W[name]
        Q5[name] = matrix_to_q5_1_hbm(mat)
        n_blocks = mat.shape[0] * (mat.shape[1] // BS)
        print(f"  {name}: {mat.shape}  →  {n_blocks} bloques, {len(Q5[name])} bytes")

    print("[3/4] Ejecutando referencia Python (FP32)...")
    x = conv1_ref(W['mel'], W['conv1_w'], W['conv1_b'])
    print(f"  conv1: {x.shape}  stats μ={x.mean():.3f} σ={x.std():.3f}")

    x = conv2_ref(x, W['conv2_w'], W['conv2_b'])
    print(f"  conv2: {x.shape}  stats μ={x.mean():.3f} σ={x.std():.3f}")

    # Pad to P_FRAMES=1504 and add the positional embedding
    x_full = np.zeros((P_FRAMES, D_MODEL), np.float32)
    x_full[:A_FRAMES] = x
    x_full += W['pos_emb']
    golden_frontend = x_full.copy()
    print(f"  frontend+posemb: μ={x_full.mean():.3f} σ={x_full.std():.3f}")

    goldens = {}
    for i in range(N_LAYERS):
        x_full = encoder_layer_ref(
            x_full,
            Q5[f'l{i}_qkv_w'], Q5[f'l{i}_wo_w'],
            Q5[f'l{i}_fc1_w'], Q5[f'l{i}_fc2_w'],
            W[f'l{i}_ln1_g'], W[f'l{i}_ln1_b'],
            W[f'l{i}_ln2_g'], W[f'l{i}_ln2_b'],
            W[f'l{i}_q_bias'], W[f'l{i}_v_bias'], W[f'l{i}_attn_out_bias'],
            W[f'l{i}_ffn1_bias'], W[f'l{i}_ffn2_bias'],
        )
        goldens[i] = x_full.copy()
        print(f"  layer {i}: μ={x_full.mean():.3f} σ={x_full.std():.3f}")

    x_full = layer_norm(x_full, W['ln_post_g'], W['ln_post_b'])
    print(f"  ln_post: μ={x_full.mean():.3f} σ={x_full.std():.3f}")
    expected = x_full.astype(np.float16)

    print("[4/4] Guardando archivos binarios en test_vectors/...")

    def save_fp16(name, arr):
        path = os.path.join(OUT_DIR, name + '.bin')
        arr.astype(np.float16).flatten().tofile(path)
        print(f"  {name}: {arr.shape}")

    def save_raw(name, data):
        path = os.path.join(OUT_DIR, name + '.bin')
        with open(path, 'wb') as f: f.write(data)
        print(f"  {name}: {len(data)} bytes")

    # Entradas
    save_fp16('mel_input',    W['mel'])
    save_fp16('pos_emb',      W['pos_emb'])
    save_fp16('conv1_weights',W['conv1_w'])
    save_fp16('conv1_biases', W['conv1_b'])
    save_fp16('conv2_weights',W['conv2_w'])
    save_fp16('conv2_biases', W['conv2_b'])

    # Per-layer weights + biases
    for i in range(N_LAYERS):
        save_raw(f'l{i}_qkv_w',         Q5[f'l{i}_qkv_w'])
        save_raw(f'l{i}_wo_w',           Q5[f'l{i}_wo_w'])
        save_raw(f'l{i}_fc1_w',          Q5[f'l{i}_fc1_w'])
        save_raw(f'l{i}_fc2_w',          Q5[f'l{i}_fc2_w'])
        save_fp16(f'l{i}_ln1_g',         W[f'l{i}_ln1_g'])
        save_fp16(f'l{i}_ln1_b',         W[f'l{i}_ln1_b'])
        save_fp16(f'l{i}_ln2_g',         W[f'l{i}_ln2_g'])
        save_fp16(f'l{i}_ln2_b',         W[f'l{i}_ln2_b'])
        save_fp16(f'l{i}_q_bias',        W[f'l{i}_q_bias'])
        save_fp16(f'l{i}_v_bias',        W[f'l{i}_v_bias'])
        save_fp16(f'l{i}_attn_out_bias', W[f'l{i}_attn_out_bias'])
        save_fp16(f'l{i}_ffn1_bias',     W[f'l{i}_ffn1_bias'])
        save_fp16(f'l{i}_ffn2_bias',     W[f'l{i}_ffn2_bias'])

    # Intermediate goldens (for layer-by-layer comparison)
    save_fp16('frontend_out', golden_frontend)
    for i in range(N_LAYERS):
        save_fp16(f'l{i}_out', goldens[i])

    # ln_post and expected output
    save_fp16('ln_post_g',       W['ln_post_g'])
    save_fp16('ln_post_b',       W['ln_post_b'])
    save_fp16('expected_output', expected)

    print(f"\n✓  {len(os.listdir(OUT_DIR))} archivos en {OUT_DIR}/")
    print("   Ahora ejecuta el csim: v++ --mode hls csim ...")

if __name__ == '__main__':
    main()
