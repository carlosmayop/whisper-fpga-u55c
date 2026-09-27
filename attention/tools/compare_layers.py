#!/usr/bin/env python3
"""
compare_layers.py - Compares the intermediate outputs of the HLS encoder (C++)
against the FP32 Python reference, layer by layer.

Flow:
1. Run whisper with WHISPER_FPGA_DEBUG=1 -> generates mel_input.bin + l{i}_hls.bin
2. Run this script -> Python reference + comparison

Usage:
    python3 compare_layers.py [--dir test_vectors_real]

Required in --dir:
    mel_input.bin          <- saved by the stub (WHISPER_FPGA_DEBUG=1)
    frontend_hls.bin       <- HLS conv+posemb output
    l0_hls.bin .. l3_hls.bin
    final_hls.bin          <- HLS output after ln_post
    conv1_weights.bin, conv1_biases.bin, conv2_weights.bin, conv2_biases.bin
    pos_emb.bin
    l{0-3}_qkv_w.bin, l{0-3}_wo_w.bin, l{0-3}_fc1_w.bin, l{0-3}_fc2_w.bin
    l{0-3}_ln1_g.bin, l{0-3}_ln1_b.bin, l{0-3}_ln2_g.bin, l{0-3}_ln2_b.bin
    l{0-3}_q_bias.bin, l{0-3}_v_bias.bin, l{0-3}_attn_out_bias.bin
    l{0-3}_ffn1_bias.bin, l{0-3}_ffn2_bias.bin
    ln_post_g.bin, ln_post_b.bin
"""

import os, sys, struct, argparse
import numpy as np

# Constantes 
MEL_CH   = 80
T_FRAMES = 3000
D_MODEL  = 384
D_HEAD   = 64
N_HEADS  = 6
D_FF     = 1536
N_LAYERS = 4
A_FRAMES = 1500
P_FRAMES = 1504
BS       = 32   # BLOCK_SIZE

# File loading

def load_fp16(path, count):
    arr = np.fromfile(path, dtype=np.float16, count=count)
    if len(arr) != count:
        raise ValueError(f"Lectura corta: {path}  ({len(arr)} != {count})")
    return arr.astype(np.float32)

def load_fp16_opt(path, count):
    """Devuelve el array o None si el fichero no existe."""
    if not os.path.exists(path):
        return None
    return load_fp16(path, count)

def dequant_q5_1_hbm(path, out_f, in_f):
    """Carga un fichero Q5.1 HBM y dequantiza → float32 [out_f, in_f]."""
    with open(path, 'rb') as f:
        data = f.read()
    W = np.zeros((out_f, in_f), np.float32)
    n_blocks = in_f // BS
    off = 0
    for row in range(out_f):
        for b in range(n_blocks):
            d_bits, m_bits, qh = struct.unpack_from('<HHI', data, off)
            d = float(np.frombuffer(struct.pack('<H', d_bits), dtype=np.float16)[0])
            m = float(np.frombuffer(struct.pack('<H', m_bits), dtype=np.float16)[0])
            qs = data[off+8 : off+24]
            off += 24
            for i in range(BS):
                # GGML Q5.1 layout: qs[j] = low nibble of elem j | high nibble of elem j+16
                if i < BS // 2:
                    nibble = qs[i] & 0x0F
                else:
                    nibble = (qs[i - BS // 2] >> 4) & 0x0F
                bit5   = (qh >> i) & 1
                W[row, b*BS + i] = d * ((bit5 << 4) | nibble) + m
    return W

# Referencia Python FP32

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, g, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var  = ((x - mean)**2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * g + b

def conv1_ref(mel, W, bias):
    """Conv1d stride=1 kernel=3 pad=1 GELU.  mel:[T,80] W:[384,80,3] → [T,384]"""
    mel_pad = np.pad(mel, [(1, 1), (0, 0)])
    out = bias.copy()[np.newaxis, :]
    for k in range(3):
        out = out + mel_pad[k:k+T_FRAMES] @ W[:, :, k].T
    return gelu(out)

def conv2_ref(x, W, bias):
    """Conv1d stride=2 kernel=3 left-pad=1 GELU.  x:[T,384] → [T//2,384]"""
    x_pad = np.vstack([np.zeros((1, D_MODEL), np.float32), x])
    out   = np.zeros((A_FRAMES, D_MODEL), np.float32)
    for k in range(3):
        out += x_pad[k::2][:A_FRAMES] @ W[:, :, k].T
    return gelu(out + bias)

def mha_ref(x, W_qkv, W_o, q_bias, v_bias, attn_out_bias):
    """Multi-head attention con pesos dequantizados."""
    Q = x @ W_qkv[:D_MODEL].T          + q_bias
    K = x @ W_qkv[D_MODEL:2*D_MODEL].T
    V = x @ W_qkv[2*D_MODEL:].T        + v_bias
    out = np.zeros_like(x)
    for h in range(N_HEADS):
        s, e = h * D_HEAD, (h+1) * D_HEAD
        scores = Q[:, s:e] @ K[:, s:e].T * 0.125
        scores -= scores.max(-1, keepdims=True)
        attn = np.exp(scores)
        attn /= attn.sum(-1, keepdims=True)
        out[:, s:e] = attn @ V[:, s:e]
    return out @ W_o.T + attn_out_bias

def ffn_ref(x, W1, W2, ffn1_bias, ffn2_bias):
    h = gelu(x @ W1.T + ffn1_bias)
    return h @ W2.T + ffn2_bias

def encoder_layer_ref(x, W_qkv, W_o, W1, W2,
                      ln1g, ln1b, ln2g, ln2b,
                      q_bias, v_bias, attn_out_bias,
                      ffn1_bias, ffn2_bias):
    x = x + mha_ref(layer_norm(x, ln1g, ln1b), W_qkv, W_o, q_bias, v_bias, attn_out_bias)
    x = x + ffn_ref(layer_norm(x, ln2g, ln2b), W1, W2, ffn1_bias, ffn2_bias)
    return x

# Comparison 

def compare(label, ref, hls):
    """Imprime MAE y max error entre ref (Python) y hls (C++).  Devuelve True si OK."""
    if hls is None:
        print(f"  [{label:20s}]  fichero HLS no encontrado, omitiendo")
        return True
    n   = min(len(ref.flat), len(hls.flat))
    ae  = np.abs(ref.flatten()[:n] - hls.flatten()[:n])
    mae = ae.mean()
    mx  = ae.max()
    idx = ae.argmax()
    ref_v = ref.flatten()[idx]
    hls_v = hls.flatten()[idx]
    ok  = mae < 0.5 and mx < 3.0
    tag = "OK      " if ok else "MISMATCH"
    print(f"  [{label:20s}]  MAE={mae:.5f}  MAX={mx:.5f}  {tag}"
          f"  (worst idx={idx}: ref={ref_v:.4f} hls={hls_v:.4f})")
    return ok

# Main 

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='test_vectors_real',
                    help='Directorio con pesos y salidas HLS (default: test_vectors_real)')
    args = ap.parse_args()
    d = args.dir

    # -- 1. Load weights -------------------------------------------------------
    print(f"\n[1/3] Cargando pesos desde {d}/")

    W1    = load_fp16(f"{d}/conv1_weights.bin", 384*80*3   ).reshape(384, 80,  3)
    b1    = load_fp16(f"{d}/conv1_biases.bin",  384        )
    W2    = load_fp16(f"{d}/conv2_weights.bin", 384*384*3  ).reshape(384, 384, 3)
    b2    = load_fp16(f"{d}/conv2_biases.bin",  384        )
    pos   = load_fp16(f"{d}/pos_emb.bin",       P_FRAMES*D_MODEL).reshape(P_FRAMES, D_MODEL)
    ln_g  = load_fp16(f"{d}/ln_post_g.bin",     D_MODEL    )
    ln_b  = load_fp16(f"{d}/ln_post_b.bin",     D_MODEL    )

    print("  Dequantizando matrices Q5.1 (puede tardar unos segundos)...")
    qkv_w, wo_w, fc1_w, fc2_w = [], [], [], []
    ln1_g, ln1_b, ln2_g, ln2_b = [], [], [], []
    q_bias, v_bias, aob, ffn1b, ffn2b = [], [], [], [], []

    for i in range(N_LAYERS):
        qkv_w.append(dequant_q5_1_hbm(f"{d}/l{i}_qkv_w.bin", 3*D_MODEL, D_MODEL))
        wo_w .append(dequant_q5_1_hbm(f"{d}/l{i}_wo_w.bin",    D_MODEL,  D_MODEL))
        fc1_w.append(dequant_q5_1_hbm(f"{d}/l{i}_fc1_w.bin",  D_FF,     D_MODEL))
        fc2_w.append(dequant_q5_1_hbm(f"{d}/l{i}_fc2_w.bin",  D_MODEL,  D_FF   ))
        ln1_g.append(load_fp16(f"{d}/l{i}_ln1_g.bin", D_MODEL))
        ln1_b.append(load_fp16(f"{d}/l{i}_ln1_b.bin", D_MODEL))
        ln2_g.append(load_fp16(f"{d}/l{i}_ln2_g.bin", D_MODEL))
        ln2_b.append(load_fp16(f"{d}/l{i}_ln2_b.bin", D_MODEL))
        q_bias.append(load_fp16(f"{d}/l{i}_q_bias.bin",        D_MODEL))
        v_bias.append(load_fp16(f"{d}/l{i}_v_bias.bin",        D_MODEL))
        aob   .append(load_fp16(f"{d}/l{i}_attn_out_bias.bin", D_MODEL))
        ffn1b .append(load_fp16(f"{d}/l{i}_ffn1_bias.bin",     D_FF   ))
        ffn2b .append(load_fp16(f"{d}/l{i}_ffn2_bias.bin",     D_MODEL))
        print(f"    capa {i} OK")

    # -- 2. Load input mel -----------------------------------------------------
    mel_path = f"{d}/mel_input.bin"
    if not os.path.exists(mel_path):
        print(f"\nERROR: {mel_path} no encontrado.")
        print("Lanza whisper con WHISPER_FPGA_DEBUG=1 primero para generarlo.")
        sys.exit(1)

    mel = load_fp16(mel_path, T_FRAMES * MEL_CH).reshape(T_FRAMES, MEL_CH)
    print(f"\n[2/3] Mel cargado: shape={mel.shape}  mean={mel.mean():.4f}  std={mel.std():.4f}")

    # ── 3. Referencia Python paso a paso ─────────────────────────────────────
    print("\n[3/3] Ejecutando referencia Python FP32 y comparando...")
    N = P_FRAMES * D_MODEL

    x = conv1_ref(mel, W1, b1)
    x = conv2_ref(x,  W2, b2)
    x_full = np.zeros((P_FRAMES, D_MODEL), np.float32)
    x_full[:A_FRAMES] = x
    x_full += pos
    ref_frontend = x_full.copy()

    hls_frontend = load_fp16_opt(f"{d}/frontend_hls.bin", N)
    print("\n  Checkpoint       MAE / MAX / estado")
    print("  " + "-"*70)
    all_ok = True
    all_ok &= compare("frontend+posemb", ref_frontend, hls_frontend)

    ref_layers = []
    layer_names = ["l0_hls", "l1_hls", "l2_hls", "l3_hls"]
    for i in range(N_LAYERS):
        x_full = encoder_layer_ref(
            x_full,
            qkv_w[i], wo_w[i], fc1_w[i], fc2_w[i],
            ln1_g[i], ln1_b[i], ln2_g[i], ln2_b[i],
            q_bias[i], v_bias[i], aob[i], ffn1b[i], ffn2b[i]
        )
        ref_layers.append(x_full.copy())
        hls_layer = load_fp16_opt(f"{d}/{layer_names[i]}.bin", N)
        ok = compare(f"layer {i}", x_full, hls_layer)
        all_ok &= ok
        if not ok and hls_layer is not None:
            print(f"    ^^^ PRIMERA DIVERGENCIA EN CAPA {i} ^^^")

    ref_final = layer_norm(x_full, ln_g, ln_b)
    hls_final = load_fp16_opt(f"{d}/final_hls.bin", N)
    all_ok &= compare("ln_post (final)", ref_final, hls_final)

    print("\n  " + "="*70)
    if all_ok:
        print("  RESULTADO: PASS — todas las capas dentro de tolerancia")
    else:
        print("  RESULTADO: MISMATCH — ver capas marcadas arriba")
    print()

if __name__ == '__main__':
    main()
