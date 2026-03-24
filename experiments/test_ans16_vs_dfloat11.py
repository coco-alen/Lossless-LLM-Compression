"""
ANS-16bit vs DFloat11: Full comparison on Qwen3-8B.

Measures:
1. Compression ratio (full model vs DFloat11 scope)
2. Compression/decompression speed
3. Actual compressed sizes including all overhead
"""

import torch
import numpy as np
import math
import time
import sys
from collections import Counter
from transformers import AutoModelForCausalLM

sys.path.insert(0, '.')
from new_compression.codec_ans16 import compress_ans16, decompress_ans16


def compress_full_model(model_name):
    print(f"Loading {model_name}...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Identify all weight tensors and their roles
    dfloat11_patterns = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                         'gate_proj', 'up_proj', 'down_proj']

    total_params = 0
    df11_params = 0
    embed_params = 0
    other_params = 0

    # Group tensors by weight type for shared codebook compression
    weight_groups = {}  # group_name -> list of flattened tensors
    raw_tensors = {}    # group_name -> raw byte count

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        tensor = param.data.cpu().to(torch.bfloat16)

        # Skip tiny params (layernorm, biases)
        if n < 1024:
            group = 'tiny_raw'
        elif 'embed_tokens' in name:
            group = 'embed_tokens'
            embed_params += n
        elif 'lm_head' in name:
            group = 'lm_head'
            embed_params += n
        elif any(p in name for p in dfloat11_patterns):
            # Find which pattern
            for p in dfloat11_patterns:
                if p in name:
                    group = p
                    df11_params += n
                    break
        elif 'norm' in name.lower():
            group = 'tiny_raw'
        else:
            group = 'other'
            other_params += n

        if group not in weight_groups:
            weight_groups[group] = []
            raw_tensors[group] = 0
        weight_groups[group].append(tensor.flatten())
        raw_tensors[group] += n * 2

    total_bytes = total_params * 2
    print(f"\nModel: {total_params:,} params ({total_bytes/1024/1024:.1f} MB BF16)")
    print(f"  DFloat11 scope: {df11_params:,} ({df11_params/total_params*100:.1f}%)")
    print(f"  Embeddings: {embed_params:,} ({embed_params/total_params*100:.1f}%)")

    # Compress each group
    print(f"\n{'='*80}")
    print(f"Compressing with ANS-16bit...")
    print(f"{'='*80}")

    total_compressed = 0
    total_df11_scope_compressed = 0
    total_df11_scope_original = 0
    compress_time = 0

    for group_name in sorted(weight_groups.keys()):
        tensors = weight_groups[group_name]
        raw_bytes = raw_tensors[group_name]

        if group_name == 'tiny_raw':
            # Store raw
            total_compressed += raw_bytes
            continue

        # Concatenate all tensors in group
        all_vals = torch.cat(tensors)
        n = all_vals.numel()

        # Entropy
        int16 = all_vals.view(torch.int16).numpy().astype(np.int32)
        counts = Counter(int16.tolist())
        h = sum(-c/n * math.log2(c/n) for c in counts.values() if c > 0)
        n_unique = len(counts)

        # Compress
        t0 = time.perf_counter()
        compressed = compress_ans16(all_vals)
        ct = time.perf_counter() - t0
        compress_time += ct

        comp_bytes = len(compressed['compressed_words']) * 4
        table_bytes = len(compressed['symbol_table']) * 2 + len(compressed['probabilities']) * 4
        ans_total = comp_bytes + table_bytes

        # Verify lossless
        t0 = time.perf_counter()
        decoded = decompress_ans16(compressed)
        dt = time.perf_counter() - t0

        if torch.equal(all_vals, decoded):
            status = "✓"
        else:
            status = "✗ MISMATCH"

        total_compressed += ans_total

        is_df11_scope = any(p in group_name for p in dfloat11_patterns)
        if is_df11_scope:
            total_df11_scope_compressed += ans_total
            total_df11_scope_original += raw_bytes

        ratio = ans_total / raw_bytes * 100
        speed = n * 2 / ct / 1024 / 1024  # MB/s compress
        dspeed = n * 2 / dt / 1024 / 1024  # MB/s decompress

        print(f"  {group_name:20s}: {raw_bytes/1024/1024:>7.1f} → {ans_total/1024/1024:>7.1f} MB "
              f"({ratio:>5.2f}%) [H={h:.2f}, {n_unique} uniq] "
              f"enc={speed:.0f} MB/s dec={dspeed:.0f} MB/s {status}")

    # Summary
    overall_ratio = total_compressed / total_bytes * 100
    df11_ans_ratio = total_df11_scope_compressed / total_df11_scope_original * 100 if total_df11_scope_original > 0 else 0

    print(f"\n{'='*80}")
    print(f"RESULTS for {model_name}")
    print(f"{'='*80}")
    print(f"  Total model:     {total_bytes/1024/1024:.1f} MB BF16")
    print(f"  ANS-16 full:     {total_compressed/1024/1024:.1f} MB ({overall_ratio:.2f}%)")
    print(f"  ANS-16 DFloat11 scope: {total_df11_scope_compressed/1024/1024:.1f} MB ({df11_ans_ratio:.2f}%)")
    print(f"  DFloat11 estimated*: {total_df11_scope_original*0.6662/1024/1024:.1f} MB (66.62%, DFloat11 scope only)")

    df11_full_size = total_df11_scope_original * 0.6662 + (total_bytes - total_df11_scope_original)
    print(f"  DFloat11 full model*: {df11_full_size/1024/1024:.1f} MB (uncompressed outside DFloat11 scope)")

    ans16_savings = total_bytes - total_compressed
    df11_savings = total_bytes - df11_full_size
    extra_savings = ans16_savings - df11_savings

    print(f"\n  ANS-16 savings:    {ans16_savings/1024/1024:.1f} MB")
    print(f"  DFloat11 savings*: {df11_savings/1024/1024:.1f} MB")
    print(f"  ANS-16 extra:      {extra_savings/1024/1024:.1f} MB MORE than DFloat11")
    print(f"\n  Compress speed: {total_bytes/compress_time/1024/1024:.0f} MB/s overall")
    print(f"\n  *DFloat11 ratio estimated from published results")


def main():
    # Test on available models
    for model_name in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]:
        compress_full_model(model_name)
        print("\n")


if __name__ == '__main__':
    main()
