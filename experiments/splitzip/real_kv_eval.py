"""
Complete SplitZip evaluation using REAL KV caches from Qwen3-30B-A3B.

Generates KV caches from real prompts with semantic content, then measures:
1. Actual compression ratio on real KV data
2. Escape rate on real KV data
3. Encode/decode throughput on real KV tensors
4. Lossless correctness (bitwise exact reconstruction)
5. Transfer speedup simulation with real compressed sizes
6. Per-layer statistics
"""

import torch
import triton
import triton.language as tl
import time
import json
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.splitzip.lossless_fast import FastLosslessCodec


def collect_real_kv(model, tokenizer, prompts, device='cuda:0', max_new_tokens=32):
    """Run real prompts through the model and collect KV caches."""
    all_kv_layers = []  # list of (key_tensor, value_tensor) per layer, accumulated

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, use_cache=True)

        kv = out.past_key_values
        for i, layer in enumerate(kv.layers):
            if hasattr(layer, 'keys'):
                k = layer.keys.detach().cpu()
                v = layer.values.detach().cpu()
                if i >= len(all_kv_layers):
                    all_kv_layers.append({'keys': [k], 'values': [v]})
                else:
                    all_kv_layers[i]['keys'].append(k)
                    all_kv_layers[i]['values'].append(v)

    # Concatenate across prompts per layer
    result = []
    for layer_data in all_kv_layers:
        k_cat = torch.cat(layer_data['keys'], dim=2)  # concat along seq dim
        v_cat = torch.cat(layer_data['values'], dim=2)
        result.append((k_cat, v_cat))

    return result


def evaluate_splitzip_on_real_kv(kv_layers, device='cuda'):
    """Full SplitZip evaluation on real KV cache data."""

    # Flatten all KV data for calibration
    all_flat = []
    for k, v in kv_layers:
        all_flat.append(k.to(device).contiguous().view(-1))
        all_flat.append(v.to(device).contiguous().view(-1))
    all_flat = torch.cat(all_flat)
    total_bytes = all_flat.numel() * 2

    print(f"Total KV data: {total_bytes / 1e6:.1f} MB across {len(kv_layers)} layers")
    print(f"KV dtype: {all_flat.dtype}")
    print()

    # Calibrate codec on real data
    codec = FastLosslessCodec(device)
    codec.calibrate(all_flat.view(torch.bfloat16))

    # ---- 1. Exponent distribution ----
    exponents = ((all_flat.view(torch.int16) >> 7) & 0xFF).to(torch.uint8)
    vals, counts = torch.unique(exponents, return_counts=True)
    si = torch.argsort(counts, descending=True)
    total_n = counts.sum().item()
    probs = counts.float() / total_n
    entropy = -(probs * torch.log2(probs)).sum().item()

    top8 = counts[si[:min(8, vals.numel())]].sum().item() / total_n
    top15 = counts[si[:min(15, vals.numel())]].sum().item() / total_n
    top8_vals = [vals[si[i]].item() for i in range(min(8, vals.numel()))]

    print("=" * 80)
    print("1. EXPONENT DISTRIBUTION (real KV cache)")
    print("=" * 80)
    print(f"  Unique exponents: {vals.numel()}")
    print(f"  Entropy: {entropy:.3f} / 8 bits")
    print(f"  Top-8 coverage: {top8*100:.2f}%")
    print(f"  Top-15 coverage: {top15*100:.2f}%")
    print(f"  Top-8 values: {top8_vals}")
    print()

    # ---- 2. Per-layer encode/decode + correctness ----
    print("=" * 80)
    print("2. PER-LAYER ENCODE/DECODE (real KV, lossless)")
    print("=" * 80)
    print(f"{'Layer':>5} {'K MB':>7} {'V MB':>7} {'K esc%':>8} {'V esc%':>8} "
          f"{'K ratio':>8} {'V ratio':>8} {'K ok':>5} {'V ok':>5}")
    print("-" * 70)

    total_original = 0
    total_compressed = 0
    total_escapes = 0
    total_elements = 0
    all_correct = True
    layer_enc_times = []
    layer_dec_times = []

    for i, (k_tensor, v_tensor) in enumerate(kv_layers):
        for tag, tensor in [('K', k_tensor), ('V', v_tensor)]:
            flat = tensor.to(device).contiguous().view(-1).to(torch.bfloat16)
            n = flat.numel()
            nbytes = n * 2
            total_original += nbytes
            total_elements += n

            # Encode
            t0 = time.perf_counter()
            r = codec.encode(flat)
            torch.cuda.synchronize()
            enc_t = time.perf_counter() - t0

            pk, sm, esc_pos, esc_val, n_out, n_esc = r
            comp_bytes = pk.numel() + sm.numel() + esc_pos.numel() * 4 + esc_val.numel()
            total_compressed += comp_bytes
            total_escapes += n_esc

            # Decode
            t0 = time.perf_counter()
            decoded = codec.decode(*r)
            torch.cuda.synchronize()
            dec_t = time.perf_counter() - t0

            # Correctness
            correct = torch.equal(flat.view(torch.int16), decoded.view(torch.int16))
            all_correct = all_correct and correct

            ratio = nbytes / comp_bytes
            esc_rate = n_esc / n * 100

            if tag == 'K':
                k_mb = nbytes / 1e6
                k_esc = esc_rate
                k_ratio = ratio
                k_ok = correct
                layer_enc_times.append(enc_t)
                layer_dec_times.append(dec_t)
            else:
                v_mb = nbytes / 1e6
                v_esc = esc_rate
                v_ratio = ratio
                v_ok = correct
                layer_enc_times.append(enc_t)
                layer_dec_times.append(dec_t)

        if i < 5 or i >= len(kv_layers) - 3:
            print(f"{i:>5} {k_mb:>6.2f} {v_mb:>6.2f} {k_esc:>7.3f}% {v_esc:>7.3f}% "
                  f"{k_ratio:>7.3f}x {v_ratio:>7.3f}x {'✓' if k_ok else '✗':>5} {'✓' if v_ok else '✗':>5}")
        elif i == 5:
            print("  ...")

    actual_ratio = total_original / total_compressed
    overall_esc_rate = total_escapes / total_elements * 100

    print()
    print(f"  Overall: {total_original/1e6:.1f} MB → {total_compressed/1e6:.1f} MB")
    print(f"  Actual compression ratio: {actual_ratio:.4f}x")
    print(f"  Overall escape rate: {overall_esc_rate:.4f}% ({total_escapes:,} / {total_elements:,})")
    print(f"  All layers correct: {'YES ✓' if all_correct else 'NO ✗'}")
    print()

    # ---- 3. Throughput on real KV ----
    print("=" * 80)
    print("3. THROUGHPUT ON REAL KV DATA")
    print("=" * 80)

    # Benchmark on the largest layer
    largest_layer_idx = max(range(len(kv_layers)),
                           key=lambda i: kv_layers[i][0].numel() + kv_layers[i][1].numel())
    k_big, v_big = kv_layers[largest_layer_idx]
    big_flat = torch.cat([k_big.to(device).view(-1), v_big.to(device).view(-1)]).to(torch.bfloat16)
    big_bytes = big_flat.numel() * 2

    # Warmup
    for _ in range(20):
        r = codec.encode(big_flat)
        codec.decode(*r)

    # Encode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        r = codec.encode(big_flat)
    torch.cuda.synchronize()
    enc_t = (time.perf_counter() - t0) / 100

    # Decode
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        codec.decode(*r)
    torch.cuda.synchronize()
    dec_t = (time.perf_counter() - t0) / 100

    enc_gbs = big_bytes / enc_t / 1e9
    dec_gbs = big_bytes / dec_t / 1e9

    print(f"  Layer {largest_layer_idx} (K+V): {big_bytes/1e6:.1f} MB")
    print(f"  Encode: {enc_gbs:.0f} GB/s ({enc_t*1000:.3f} ms)")
    print(f"  Decode: {dec_gbs:.0f} GB/s ({dec_t*1000:.3f} ms)")
    print()

    # ---- 4. Transfer speedup simulation with REAL compressed sizes ----
    print("=" * 80)
    print("4. TRANSFER SPEEDUP (using real compressed sizes)")
    print("=" * 80)

    n_layers = len(kv_layers)
    per_layer_original = total_original / n_layers
    per_layer_compressed = total_compressed / n_layers
    per_layer_enc_ms = np.mean(layer_enc_times) * 1000
    per_layer_dec_ms = np.mean(layer_dec_times) * 1000

    print(f"  Per-layer: {per_layer_original/1e6:.2f} MB → {per_layer_compressed/1e6:.2f} MB "
          f"(ratio: {per_layer_original/per_layer_compressed:.4f}x)")
    print(f"  Per-layer encode: {per_layer_enc_ms:.3f} ms, decode: {per_layer_dec_ms:.3f} ms")
    print()

    # Scale to Llama-3-70B 64K context for paper comparison
    # 80 layers, 8 KV heads, 65536 seq, 128 dim = 21.5 GB
    scale_total = 21475 * 1024 * 1024  # 21.5 GB
    scale_compressed = scale_total / actual_ratio
    scale_layers = 80

    print(f"  Projected for Llama-3-70B 64K ({scale_total/1e9:.1f} GB, {scale_layers} layers):")
    print(f"  {'Network':>18} {'Raw ms':>9} {'Comp ms':>9} {'Speedup':>8}")
    print(f"  {'-'*48}")

    for name, bw in [("GPU-Direct(15)", 15), ("CPU-RDMA(47)", 47),
                      ("RoCE4x200(87)", 87), ("RoCE8x400(190)", 190)]:
        raw_ms = scale_total / (bw * 1e9) * 1000

        # Pipeline: per-layer times
        per_layer_bytes = scale_total / scale_layers
        per_layer_comp = per_layer_bytes / actual_ratio
        xfer_per_layer = per_layer_comp / (bw * 1e9) * 1000
        enc_per = per_layer_bytes / (enc_gbs * 1e9) * 1000
        dec_per = per_layer_bytes / (dec_gbs * 1e9) * 1000
        bottleneck = max(enc_per, xfer_per_layer, dec_per)
        pipe_ms = enc_per + bottleneck * scale_layers + dec_per
        speedup = raw_ms / pipe_ms

        print(f"  {name:>18} {raw_ms:>8.1f} {pipe_ms:>8.1f} {speedup:>7.3f}x")

    print()

    # ---- 5. Summary ----
    print("=" * 80)
    print("5. SUMMARY — REAL KV CACHE EVALUATION")
    print("=" * 80)
    print(f"  Model: Qwen3-30B-A3B (MoE, 128 experts)")
    print(f"  KV dtype: BF16")
    print(f"  Total KV evaluated: {total_original/1e6:.1f} MB")
    print(f"  Actual ratio on real KV: {actual_ratio:.4f}x")
    print(f"  Escape rate on real KV: {overall_esc_rate:.4f}%")
    print(f"  Lossless correctness: {'ALL PASS ✓' if all_correct else 'FAIL ✗'}")
    print(f"  Exponent entropy: {entropy:.3f} bits")
    print(f"  Encode throughput: {enc_gbs:.0f} GB/s")
    print(f"  Decode throughput: {dec_gbs:.0f} GB/s")
    print()

    return {
        'ratio': actual_ratio,
        'escape_rate': overall_esc_rate,
        'all_correct': all_correct,
        'entropy': entropy,
        'enc_gbs': enc_gbs,
        'dec_gbs': dec_gbs,
        'top8_coverage': top8 * 100,
        'top15_coverage': top15 * 100,
    }


def main():
    device = 'cuda:0'
    model_name = 'Qwen/Qwen3-30B-A3B'

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.eval()

    print(f"Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_experts} experts, "
          f"{model.config.num_key_value_heads} KV heads")
    print()

    # Real prompts with diverse semantic content
    prompts = [
        # Knowledge
        "Explain the theory of general relativity and how it differs from Newtonian gravity. "
        "Include examples of experimental confirmations such as gravitational lensing and "
        "the precession of Mercury's orbit.",

        # Reasoning
        "A farmer has 100 meters of fencing to enclose a rectangular field along a river. "
        "The river forms one side, so fencing is needed for only three sides. "
        "What dimensions maximize the enclosed area? Show your work step by step.",

        # Code
        "Write a Python implementation of a balanced binary search tree (AVL tree) with "
        "insert, delete, and search operations. Include proper rotation handling.",

        # Creative writing
        "Write a short story about a quantum physicist who discovers that the act of "
        "observation doesn't just collapse wave functions — it creates entirely new "
        "branches of reality that she can visit.",

        # Technical analysis
        "Compare and contrast the architectures of GPT-4, Llama 3, and Mixtral. "
        "Discuss their training strategies, parameter counts, and performance tradeoffs. "
        "What design choices make each model suitable for different deployment scenarios?",

        # Long factual
        "Provide a comprehensive overview of the history of computing from Charles Babbage's "
        "Analytical Engine through modern quantum computers. Cover key milestones including "
        "ENIAC, the transistor, integrated circuits, personal computers, the internet, "
        "GPUs for AI, and current quantum computing efforts.",

        # Multi-turn style
        "Question: What are the fundamental differences between TCP and UDP protocols? "
        "When should each be used? Provide specific examples of applications that use "
        "each protocol and explain why.",

        # Instruction following
        "You are a helpful assistant. A user asks: 'I need to deploy a large language model "
        "for real-time chat. I have 4 NVIDIA A100 GPUs. What serving framework should I use, "
        "and how should I configure tensor parallelism and batching?'",
    ]

    print(f"Collecting KV caches from {len(prompts)} real prompts...")
    kv_layers = collect_real_kv(model, tokenizer, prompts, device=device)
    print(f"Collected {len(kv_layers)} layers of KV cache")
    print()

    # Free model memory before running codec
    del model
    torch.cuda.empty_cache()

    # Run full evaluation
    results = evaluate_splitzip_on_real_kv(kv_layers, device=device)

    # Save results
    with open('experiments/splitzip/real_kv_eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to experiments/splitzip/real_kv_eval_results.json")


if __name__ == "__main__":
    main()
