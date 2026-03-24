"""
Test ANS-16bit compression on the FULL model including embeddings and lm_head.

DFloat11 only compresses attention/MLP weights via pattern matching.
ANS-16bit can compress ANY BF16 tensor. The embedding table alone is ~622 MB
on Qwen3-1.7B — compressing it would be a major advantage.

Also compare: single shared codebook vs per-weight-group codebooks.
"""

import torch
import numpy as np
import math
import sys
from collections import Counter
from transformers import AutoModelForCausalLM

sys.path.insert(0, '.')
from new_compression.codec_ans16 import compress_ans16, decompress_ans16, compute_ratio


def analyze_full_model(model_name):
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    # Categorize all parameters
    categories = {
        'embed_tokens': [],
        'lm_head': [],
        'self_attn.q_proj': [],
        'self_attn.k_proj': [],
        'self_attn.v_proj': [],
        'self_attn.o_proj': [],
        'mlp.gate_proj': [],
        'mlp.up_proj': [],
        'mlp.down_proj': [],
        'layernorm': [],
        'other': [],
    }

    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        categorized = False
        for cat in categories:
            if cat in name:
                categories[cat].append((name, param.data.cpu().to(torch.bfloat16)))
                categorized = True
                break
        if not categorized:
            categories['other'].append((name, param.data.cpu().to(torch.bfloat16)))

    total_bf16_bytes = total_params * 2
    print(f"\nTotal parameters: {total_params:,} ({total_bf16_bytes/1024/1024:.1f} MB BF16)")
    print(f"\nParameter breakdown:")
    for cat, params in categories.items():
        if params:
            n = sum(p.numel() for _, p in params)
            print(f"  {cat:25s}: {len(params):3d} tensors, {n:>12,} params ({n*2/1024/1024:>7.1f} MB, {n/total_params*100:>5.1f}%)")

    # Compress each category with ANS-16bit
    print(f"\n{'='*80}")
    print(f"ANS-16bit Compression Results")
    print(f"{'='*80}")

    total_compressed = 0
    total_original = 0

    for cat, params in categories.items():
        if not params:
            continue

        cat_original = sum(p.numel() * 2 for _, p in params)
        total_original += cat_original

        if cat == 'layernorm':
            # Layernorm params are tiny, just store raw
            total_compressed += cat_original
            print(f"  {cat:25s}: {cat_original/1024:.0f} KB (stored raw, too small to compress)")
            continue

        # Compress all tensors in this category together (shared codebook)
        all_vals = torch.cat([p.flatten() for _, p in params])
        int16 = all_vals.view(torch.int16).numpy().astype(np.int32)
        n = len(int16)

        # Entropy
        counts = Counter(int16.tolist())
        h = sum(-c/n * math.log2(c/n) for c in counts.values() if c > 0)
        n_unique = len(counts)

        # Actual ANS-16bit compressed size
        try:
            compressed = compress_ans16(all_vals)
            comp_bytes = len(compressed['compressed_words']) * 4
            table_bytes = len(compressed['symbol_table']) * 2 + len(compressed['probabilities']) * 4
            ans_total = comp_bytes + table_bytes
        except Exception as e:
            # Fallback: estimate from entropy
            ans_total = int(h * n / 8) + n_unique * 6
            print(f"  {cat:25s}: compression error: {e}")

        total_compressed += ans_total

        ratio = ans_total / cat_original * 100
        entropy_ratio = h / 16 * 100
        print(f"  {cat:25s}: {cat_original/1024/1024:>7.1f} MB → {ans_total/1024/1024:>7.1f} MB "
              f"({ratio:>5.2f}%) [H={h:.2f} bpw, {n_unique} unique, entropy={entropy_ratio:.2f}%]")

    overall_ratio = total_compressed / total_original * 100
    savings_mb = (total_original - total_compressed) / 1024 / 1024

    print(f"\n{'='*80}")
    print(f"TOTAL: {total_original/1024/1024:.1f} MB → {total_compressed/1024/1024:.1f} MB ({overall_ratio:.2f}%)")
    print(f"Savings: {savings_mb:.1f} MB")
    print(f"{'='*80}")

    # Compare with DFloat11 scope (attn + MLP only)
    df11_original = 0
    df11_compressed = 0
    for cat in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
        if categories[cat]:
            cat_orig = sum(p.numel() * 2 for _, p in categories[cat])
            df11_original += cat_orig
            # Re-compute compressed size for this subset
            all_vals = torch.cat([p.flatten() for _, p in categories[cat]])
            compressed = compress_ans16(all_vals)
            comp_bytes = len(compressed['compressed_words']) * 4
            table_bytes = len(compressed['symbol_table']) * 2 + len(compressed['probabilities']) * 4
            df11_compressed += comp_bytes + table_bytes

    print(f"\nScope comparison:")
    print(f"  DFloat11 scope (attn+MLP): {df11_original/1024/1024:.1f} MB → "
          f"~{df11_original*0.6662/1024/1024:.1f} MB (66.62% DFloat11) vs "
          f"{df11_compressed/1024/1024:.1f} MB ({df11_compressed/df11_original*100:.2f}% ANS-16)")
    print(f"  Full model scope: {total_original/1024/1024:.1f} MB → "
          f"{total_compressed/1024/1024:.1f} MB ({overall_ratio:.2f}% ANS-16)")

    # Check if embed_tokens and lm_head share weights
    if categories['embed_tokens'] and categories['lm_head']:
        embed = categories['embed_tokens'][0][1]
        lm = categories['lm_head'][0][1]
        if embed.shape == lm.shape:
            if torch.equal(embed, lm):
                print(f"\n  NOTE: embed_tokens and lm_head are TIED (identical weights)")
                print(f"  Only need to store one copy → saves {embed.numel()*2/1024/1024:.1f} MB")
            else:
                max_diff = (embed.float() - lm.float()).abs().max().item()
                print(f"\n  embed_tokens and lm_head: NOT tied (max diff={max_diff:.6f})")


def main():
    for model_name in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]:
        analyze_full_model(model_name)
        print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
