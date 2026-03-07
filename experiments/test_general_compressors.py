"""
Test general-purpose compressors (zstd, gzip, lzma) on BFloat16 weight data.
Compare raw compression and DFloat11-style byte-separated compression.
"""

import struct
import time
import zlib
import lzma
import io

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig
from dahuffman import HuffmanCodec

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


WEIGHT_TYPES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def extract_weights(model, num_layers):
    groups = {wt: [] for wt in WEIGHT_TYPES}
    for idx in range(num_layers):
        layer = model.model.layers[idx]
        for wt in WEIGHT_TYPES:
            parts = wt.split(".")
            mod = layer
            for p in parts:
                mod = getattr(mod, p)
            groups[wt].append(mod.weight.data.detach().cpu().to(torch.bfloat16))
    return groups


def extract_fields(w_bf16):
    W = w_bf16.contiguous().view(torch.int16)
    exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
    sm = ((W >> 8) & 0x80 | (W & 0x7F)).to(torch.uint8).numpy()
    return exp, sm


def compress_zstd(data: bytes, level=3) -> int:
    if not HAS_ZSTD:
        return len(data)
    cctx = zstd.ZstdCompressor(level=level)
    return len(cctx.compress(data))


def compress_gzip(data: bytes, level=9) -> int:
    return len(zlib.compress(data, level))


def compress_lzma(data: bytes) -> int:
    return len(lzma.compress(data))


def huffman_size(data_uint8: np.ndarray) -> int:
    vals, counts = np.unique(data_uint8, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    if len(freq) <= 1:
        return (len(data_uint8) + 7) // 8
    codec = HuffmanCodec.from_frequencies(freq)
    encoded = codec.encode(data_uint8.tolist())
    return len(encoded)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    num_layers = config.num_hidden_layers
    print(f"Model: {args.model_name_or_path}  ({num_layers} layers)")
    print(f"zstd available: {HAS_ZSTD}")
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("Extracting weights...", flush=True)
    groups = extract_weights(model, num_layers)
    del model

    total_raw = 0
    total_results = {}

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        all_w = torch.cat([t.flatten() for t in tensors])
        raw_bytes = all_w.view(torch.int16).numpy().tobytes()
        n = all_w.numel()
        raw_size = n * 2
        total_raw += raw_size

        exp, sm = extract_fields(all_w)
        exp_bytes_data = exp.tobytes()
        sm_bytes_data = sm.tobytes()

        print(f"\n{'='*90}")
        print(f"  {wt}  ({n:,} params, {raw_size/1e6:.1f} MB)")
        print(f"{'='*90}")
        print(f"  {'Method':<50} {'Size MB':>10} {'Ratio':>7}")
        print(f"  {'-'*50} {'-'*10} {'-'*7}")

        results = {}

        # DFloat11 baseline
        exp_huf = huffman_size(exp)
        df11_size = exp_huf + n
        results["DFloat11"] = df11_size
        print(f"  {'DFloat11: Huffman(exp) + raw(sm)':<50} {df11_size/1e6:>9.2f} {df11_size/raw_size*100:>6.2f}%")

        # Raw bf16 with general compressors
        for name, func in [("gzip-9", lambda d: compress_gzip(d, 9))]:
            t0 = time.time()
            sz = func(raw_bytes)
            elapsed = time.time() - t0
            results[f"raw_{name}"] = sz
            print(f"  {'Raw bf16 + ' + name:<50} {sz/1e6:>9.2f} {sz/raw_size*100:>6.2f}%  ({elapsed:.1f}s)")

        if HAS_ZSTD:
            for level in [3, 19]:
                name = f"zstd-{level}"
                t0 = time.time()
                sz = compress_zstd(raw_bytes, level)
                elapsed = time.time() - t0
                results[f"raw_{name}"] = sz
                print(f"  {'Raw bf16 + ' + name:<50} {sz/1e6:>9.2f} {sz/raw_size*100:>6.2f}%  ({elapsed:.1f}s)")

        # Separated channels with general compressors
        for name, func in [("gzip-9", lambda d: compress_gzip(d, 9))]:
            t0 = time.time()
            exp_sz = func(exp_bytes_data)
            sm_sz = func(sm_bytes_data)
            total_sz = exp_sz + sm_sz
            elapsed = time.time() - t0
            results[f"sep_{name}"] = total_sz
            print(f"  {'Sep channels + ' + name:<50} {total_sz/1e6:>9.2f} {total_sz/raw_size*100:>6.2f}%  (exp={exp_sz/1e6:.2f}, sm={sm_sz/1e6:.2f})")

        if HAS_ZSTD:
            for level in [3, 19]:
                name = f"zstd-{level}"
                t0 = time.time()
                exp_sz = compress_zstd(exp_bytes_data, level)
                sm_sz = compress_zstd(sm_bytes_data, level)
                total_sz = exp_sz + sm_sz
                elapsed = time.time() - t0
                results[f"sep_{name}"] = total_sz
                print(f"  {'Sep channels + ' + name:<50} {total_sz/1e6:>9.2f} {total_sz/raw_size*100:>6.2f}%  (exp={exp_sz/1e6:.2f}, sm={sm_sz/1e6:.2f})")

        # DFloat11 output (huffman bytes + raw sm) with zstd on top
        if HAS_ZSTD:
            # Simulate DFloat11 storage: huffman-encoded exp + raw sm
            # Just compress the sm bytes with zstd
            sm_zstd = compress_zstd(sm_bytes_data, 19)
            hybrid_size = exp_huf + sm_zstd
            results["hybrid_huf_exp_zstd_sm"] = hybrid_size
            print(f"  {'Huffman(exp) + zstd-19(sm)':<50} {hybrid_size/1e6:>9.2f} {hybrid_size/raw_size*100:>6.2f}%")

        # lzma on separated channels (slow but strong)
        t0 = time.time()
        exp_lzma = compress_lzma(exp_bytes_data)
        sm_lzma = compress_lzma(sm_bytes_data)
        total_lzma = exp_lzma + sm_lzma
        elapsed = time.time() - t0
        results["sep_lzma"] = total_lzma
        print(f"  {'Sep channels + lzma':<50} {total_lzma/1e6:>9.2f} {total_lzma/raw_size*100:>6.2f}%  (exp={exp_lzma/1e6:.2f}, sm={sm_lzma/1e6:.2f}, {elapsed:.1f}s)")

        # Huffman(exp) + lzma(sm)
        hybrid_lzma = exp_huf + sm_lzma
        results["hybrid_huf_exp_lzma_sm"] = hybrid_lzma
        print(f"  {'Huffman(exp) + lzma(sm)':<50} {hybrid_lzma/1e6:>9.2f} {hybrid_lzma/raw_size*100:>6.2f}%")

        for k, v in results.items():
            if k not in total_results:
                total_results[k] = 0
            total_results[k] += v

    # Summary
    print(f"\n\n{'='*90}")
    print("OVERALL SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Method':<50} {'Size MB':>10} {'Ratio':>7}")
    print(f"  {'-'*50} {'-'*10} {'-'*7}")

    for name, sz in sorted(total_results.items(), key=lambda x: x[1]):
        print(f"  {name:<50} {sz/1e6:>9.2f} {sz/total_raw*100:>6.2f}%")


if __name__ == "__main__":
    main()
