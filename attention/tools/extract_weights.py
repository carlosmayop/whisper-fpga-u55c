#!/usr/bin/env python3
"""
extract_weights.py — extrae pesos del modelo GGML whisper tiny y los guarda
en el formato binario que espera el kernel HLS (encoder_tb.cpp / whisper_fpga_encoder_stub).

Uso:
    python3 extract_weights.py \
        --model /path/to/ggml-tiny-q5_1.bin \
        --out   test_vectors_real/

Salida (en out_dir):
    conv1_weights.bin  conv1_biases.bin
    conv2_weights.bin  conv2_biases.bin
    pos_emb.bin                          ← padded 1500→1504 frames
    l{0-3}_qkv_w.bin                     ← Q+K+V concatenados, formato HBM Q5.1
    l{0-3}_wo_w.bin                      ← W_O, formato HBM Q5.1
    l{0-3}_fc1_w.bin  l{0-3}_fc2_w.bin  ← formato HBM Q5.1
    l{0-3}_ln1_g.bin  l{0-3}_ln1_b.bin
    l{0-3}_ln2_g.bin  l{0-3}_ln2_b.bin
    l{0-3}_q_bias.bin l{0-3}_v_bias.bin
    l{0-3}_attn_out_bias.bin
    l{0-3}_ffn1_bias.bin  l{0-3}_ffn2_bias.bin
    ln_post_g.bin  ln_post_b.bin
"""

import argparse, os, struct
import numpy as np

# Constantes 
D_MODEL  = 384
D_FF     = 1536
N_LAYERS = 4
A_FRAMES = 1500   # AUDIO_FRAMES
P_FRAMES = 1504   # PADDED_FRAMES
BS       = 32     # BLOCK_SIZE

# GGML per-tensor type ids
GGML_F32  = 0
GGML_F16  = 1
GGML_Q4_0 = 2   # 18 bytes / 32 elems
GGML_Q4_1 = 3   # 20 bytes / 32 elems
GGML_Q5_0 = 6   # 22 bytes / 32 elems
GGML_Q5_1 = 7   # 24 bytes / 32 elems  ← NOTE: type id 7, not 9
GGML_Q8_0 = 8   # 34 bytes / 32 elems
GGML_Q8_1 = 9   # 36 bytes / 32 elems

#  Q5.1 block size
Q5_1_BLOCK_BYTES = 24   # sizeof(block_q5_1) = 2+2+4+16

def block_bytes(n_elems, ftype):
    if   ftype == GGML_F32:  return n_elems * 4
    elif ftype == GGML_F16:  return n_elems * 2
    elif ftype == GGML_Q4_0: return (n_elems // BS) * 18
    elif ftype == GGML_Q4_1: return (n_elems // BS) * 20
    elif ftype == GGML_Q5_0: return (n_elems // BS) * 22
    elif ftype == GGML_Q5_1: return (n_elems // BS) * Q5_1_BLOCK_BYTES
    elif ftype == GGML_Q8_0: return (n_elems // BS) * 34
    elif ftype == GGML_Q8_1: return (n_elems // BS) * 36
    else: raise ValueError(f"Unknown ftype {ftype}")

# Parse GGML model file 
def load_ggml_tensors(model_path):
    """Returns dict: name → {'data': bytes, 'dims': tuple, 'ftype': int}"""
    tensors = {}
    with open(model_path, 'rb') as f:
        # magic (4) + hparams 11×int32 (44)
        f.read(4)
        hp = struct.unpack('<11i', f.read(44))
        n_mels = hp[9]

        # mel filters: n_mel, n_fft, data
        n_mel, n_fft = struct.unpack('<2i', f.read(8))
        f.read(n_mel * n_fft * 4)

        # vocab: count, then for each: len(int32) + token(bytes)  [no score field]
        n_vocab = struct.unpack('<i', f.read(4))[0]
        for _ in range(n_vocab):
            tlen = struct.unpack('<i', f.read(4))[0]
            f.read(tlen)

        # tensors until EOF
        while True:
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            n_dims, name_len, ftype = struct.unpack('<3i', hdr)
            dims = tuple(struct.unpack(f'<{n_dims}i', f.read(n_dims * 4)))
            name = f.read(name_len).decode('utf-8')

            # NOTE: this GGML whisper format stores data immediately after the
            # name with NO alignment padding.

            n_elems = 1
            for d in dims:
                n_elems *= d
            size = block_bytes(n_elems, ftype)
            data = f.read(size)
            tensors[name] = {'data': data, 'dims': dims, 'ftype': ftype}

    print(f"[load] {len(tensors)} tensors loaded from {model_path}")
    return tensors

# Tensor data helpers
def to_fp32(t):
    """Convert tensor data to numpy float32 array."""
    data, dims, ftype = t['data'], t['dims'], t['ftype']
    n_elems = 1
    for d in dims:
        n_elems *= d

    if ftype == GGML_F32:
        arr = np.frombuffer(data, dtype=np.float32).copy()
    elif ftype == GGML_F16:
        arr = np.frombuffer(data, dtype=np.float16).astype(np.float32)
    elif ftype == GGML_Q5_1:  # type id 7
        # Dequantize Q5_1 blocks
        n_blocks = n_elems // BS
        arr = np.zeros(n_elems, dtype=np.float32)
        for b in range(n_blocks):
            off = b * Q5_1_BLOCK_BYTES
            d_val, m_val, qh = struct.unpack_from('<hhI', data, off)
            d_f = np.float16(d_val).astype(np.float32) if isinstance(d_val, int) else float(np.frombuffer(struct.pack('<h', d_val), dtype=np.float16)[0])
            m_f = float(np.frombuffer(struct.pack('<h', m_val), dtype=np.float16)[0])
            qs = list(data[off+8:off+24])
            for i in range(BS):
                nibble = (qs[i // 2] >> ((i % 2) * 4)) & 0x0F
                bit5   = (qh >> i) & 1
                w_int  = (bit5 << 4) | nibble
                arr[b * BS + i] = d_f * w_int + m_f
    else:
        raise ValueError(f"Unsupported ftype {ftype}")

    return arr.reshape(dims[::-1])  # GGML dims are stored reversed

def to_fp16_bytes(t):
    """Return tensor as raw FP16 bytes (flat)."""
    arr = to_fp32(t).flatten().astype(np.float16)
    return arr.tobytes()

def q5_1_raw_hbm(t):
    """
    Return Q5.1 raw bytes in HBM layout (same as matrix_to_q5_1_hbm).
    If tensor is already Q5_1, copy blocks directly (no re-quantization).
    If tensor is F16/F32, re-quantize.
    Pads to 64-byte (bus_t) alignment.
    """
    data, dims, ftype = t['data'], t['dims'], t['ftype']
    # dims in GGML are stored innermost-first: dims[0]=cols, dims[1]=rows
    # For a 2D weight matrix [out, in]: dims=(in, out) in GGML
    n_elems = 1
    for d in dims: n_elems *= d

    if ftype == GGML_Q5_1:  # type id 7
        # Blocks are already in the right format; just copy raw bytes
        raw = bytearray(data)
    else:
        # Re-quantize from FP32
        arr = to_fp32(t).flatten()  # flat, row-major after reshape
        # Re-pack as Q5.1
        raw = bytearray()
        n_blocks = n_elems // BS
        for b in range(n_blocks):
            vals = arr[b*BS:(b+1)*BS]
            raw += pack_q5_1_block(vals)

    # Pad to 64-byte bus_t alignment
    pad = (64 - len(raw) % 64) % 64
    raw += b'\x00' * pad
    return bytes(raw)

def pack_q5_1_block(vals):
    """float32[32] → 24 bytes block_q5_1"""
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax == vmin:
        d, m = 0.0, vmin
        q = np.zeros(BS, dtype=np.int32)
    else:
        d = (vmax - vmin) / 31.0
        m = vmin
        q = np.clip(np.round((vals.astype(np.float32) - m) / d), 0, 31).astype(np.int32)

    d_bits = int(np.float16(d).view(np.uint16))
    m_bits = int(np.float16(m).view(np.uint16))

    qh = np.uint32(0)
    for j in range(BS):
        if q[j] & 0x10:
            qh |= np.uint32(1 << j)

    qs = np.zeros(16, dtype=np.uint8)
    for j in range(16):
        qs[j] = (int(q[2*j]) & 0x0F) | ((int(q[2*j+1]) & 0x0F) << 4)

    return struct.pack('<HHI', d_bits, m_bits, int(qh)) + bytes(qs)

def concat_q5_1_hbm(*tensors):
    """
    Concatenate multiple Q5_1 matrices row-wise into a single HBM buffer.
    Used to build QKV = [Q; K; V] from 3 separate matrices.
    Each tensor must have the same number of columns.
    """
    raw = bytearray()
    for t in tensors:
        data, dims, ftype = t['data'], t['dims'], t['ftype']
        n_elems = 1
        for d in dims: n_elems *= d

        if ftype == GGML_Q5_1:  # type id 7
            raw += data  # blocks already in row-major order
        else:
            arr = to_fp32(t).flatten()
            n_blocks = n_elems // BS
            for b in range(n_blocks):
                raw += pack_q5_1_block(arr[b*BS:(b+1)*BS])

    pad = (64 - len(raw) % 64) % 64
    raw += b'\x00' * pad
    return bytes(raw)

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/ggml-small.en.bin')
    ap.add_argument('--out',   default='test_vectors_real')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    T = load_ggml_tensors(args.model)

    # Show all encoder tensor names for verification
    enc_names = sorted(k for k in T if 'encoder' in k)
    print("\nEncoder tensors in model:")
    for n in enc_names:
        t = T[n]
        ft = {0:'F32',1:'F16',9:'Q5_1'}.get(t['ftype'], str(t['ftype']))
        print(f"  {n:60s} dims={t['dims']}  {ft}")

    def save(name, data):
        path = os.path.join(args.out, name + '.bin')
        with open(path, 'wb') as f:
            f.write(data)
        print(f"  saved {name}.bin  ({len(data)} bytes)")

    def get(name, default_zeros=None):
        if name in T:
            return T[name]
        if default_zeros is not None:
            print(f"  WARNING: {name} not found, using zeros ({default_zeros} fp16 elements)")
            zero_data = bytes(default_zeros * 2)
            return {'data': zero_data, 'dims': (default_zeros,), 'ftype': GGML_F16}
        raise KeyError(f"Tensor not found: {name}")

    print("\nExtracting weights...")

    #  Conv weights (stored as F16 or F32 in tiny model) 
    # GGML dims for conv1: (3, 80, 384) → reshape to [384, 80, 3]
    save('conv1_weights', to_fp16_bytes(get('encoder.conv1.weight')))
    save('conv1_biases',  to_fp16_bytes(get('encoder.conv1.bias')))
    save('conv2_weights', to_fp16_bytes(get('encoder.conv2.weight')))
    save('conv2_biases',  to_fp16_bytes(get('encoder.conv2.bias')))

    # Positional embedding: [1500, 384] → pad to [1504, 384] 
    pos_arr = to_fp32(get('encoder.positional_embedding'))  # [1500, 384]
    pos_full = np.zeros((P_FRAMES, D_MODEL), dtype=np.float32)
    pos_full[:A_FRAMES] = pos_arr.reshape(A_FRAMES, D_MODEL)
    save('pos_emb', pos_full.astype(np.float16).flatten().tobytes())

    # Per-layer weights 
    for i in range(N_LAYERS):
        print(f"\n  Layer {i}:")

        # QKV = concatenate Q, K, V matrices → [3×384, 384] Q5.1 HBM
        q = get(f'encoder.blocks.{i}.attn.query.weight')
        k = get(f'encoder.blocks.{i}.attn.key.weight')
        v = get(f'encoder.blocks.{i}.attn.value.weight')
        save(f'l{i}_qkv_w', concat_q5_1_hbm(q, k, v))

        # W_O, FC1, FC2
        save(f'l{i}_wo_w',  q5_1_raw_hbm(get(f'encoder.blocks.{i}.attn.out.weight')))
        save(f'l{i}_fc1_w', q5_1_raw_hbm(get(f'encoder.blocks.{i}.mlp.0.weight')))
        save(f'l{i}_fc2_w', q5_1_raw_hbm(get(f'encoder.blocks.{i}.mlp.2.weight')))

        # LayerNorm weights (F32 in model → FP16 output)
        save(f'l{i}_ln1_g', to_fp16_bytes(get(f'encoder.blocks.{i}.attn_ln.weight')))
        save(f'l{i}_ln1_b', to_fp16_bytes(get(f'encoder.blocks.{i}.attn_ln.bias')))
        save(f'l{i}_ln2_g', to_fp16_bytes(get(f'encoder.blocks.{i}.mlp_ln.weight')))
        save(f'l{i}_ln2_b', to_fp16_bytes(get(f'encoder.blocks.{i}.mlp_ln.bias')))

        # Attention biases (may be absent → zeros)
        save(f'l{i}_q_bias',        to_fp16_bytes(get(f'encoder.blocks.{i}.attn.query.bias', D_MODEL)))
        save(f'l{i}_v_bias',        to_fp16_bytes(get(f'encoder.blocks.{i}.attn.value.bias', D_MODEL)))
        save(f'l{i}_attn_out_bias', to_fp16_bytes(get(f'encoder.blocks.{i}.attn.out.bias',   D_MODEL)))

        # FFN biases (may be absent → zeros)
        save(f'l{i}_ffn1_bias', to_fp16_bytes(get(f'encoder.blocks.{i}.mlp.0.bias', D_FF)))
        save(f'l{i}_ffn2_bias', to_fp16_bytes(get(f'encoder.blocks.{i}.mlp.2.bias', D_MODEL)))

    #  Final LayerNorm 
    save('ln_post_g', to_fp16_bytes(get('encoder.ln_post.weight')))
    save('ln_post_b', to_fp16_bytes(get('encoder.ln_post.bias')))

    n_files = len(os.listdir(args.out))
    print(f"\n✓  {n_files} archivos guardados en {args.out}/")

if __name__ == '__main__':
    main()
