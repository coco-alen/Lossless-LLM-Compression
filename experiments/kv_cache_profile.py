"""
KV Cache Compressibility Profiling
====================================
Profile Key and Value cache tensors from a real LLM to measure:
1. Exponent entropy per layer, per head
2. Zero fraction
3. Temporal redundancy (how much consecutive KV pages differ)
4. DFloat-style compression ratio
5. Page-level variation (for PagedAttention integration)
"""

import torch
import time
import numpy as np
import json
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_entropy_gpu(tensor_uint8):
    vals, counts = torch.unique(tensor_uint8, return_counts=True)
    probs = counts.float() / counts.sum()
    return -(probs * torch.log2(probs)).sum().item()


def analyze_kv(tensor, name):
    """Analyze a BF16 KV tensor's compressibility."""
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    int16 = flat.view(torch.int16)

    exponents = ((int16 >> 7) & 0xFF).to(torch.uint8)
    sign_mant = (((int16 >> 8) & 0x80) | (int16 & 0x7F)).to(torch.uint8)

    zero_frac = (int16 == 0).float().mean().item()
    exp_entropy = compute_entropy_gpu(exponents)
    sm_entropy = compute_entropy_gpu(sign_mant)
    full_entropy = compute_entropy_gpu(int16.to(torch.int32).view(torch.int16))

    ratio = 16.0 / (exp_entropy + 8.0)

    return {
        'name': name,
        'shape': list(tensor.shape),
        'numel': n,
        'bytes': n * 2,
        'zero_fraction': round(zero_frac, 4),
        'exponent_entropy': round(exp_entropy, 3),
        'mantissa_entropy': round(sm_entropy, 3),
        'full_entropy': round(full_entropy, 3),
        'dfloat_ratio': round(ratio, 3),
        'theoretical_max_ratio': round(16.0 / full_entropy, 3) if full_entropy > 0 else 999,
        'abs_mean': round(tensor.float().abs().mean().item(), 6),
        'abs_max': round(tensor.float().abs().max().item(), 4),
    }


def analyze_page_level(kv_tensor, page_size=16):
    """Analyze KV at page granularity (for PagedAttention).
    page_size = number of tokens per page (vLLM default = 16)."""
    # kv_tensor shape: [batch, num_heads, seq_len, head_dim]
    if kv_tensor.dim() == 4:
        b, h, s, d = kv_tensor.shape
    elif kv_tensor.dim() == 3:
        h, s, d = kv_tensor.shape
        b = 1
    else:
        return []

    results = []
    n_pages = (s + page_size - 1) // page_size

    for page_idx in range(min(n_pages, 20)):  # sample up to 20 pages
        start = page_idx * page_size
        end = min(start + page_size, s)
        page = kv_tensor[..., start:end, :].contiguous()
        r = analyze_kv(page, f"page_{page_idx}")
        r['page_idx'] = page_idx
        r['token_range'] = [start, end]
        results.append(r)

    return results


def profile_kv_cache(model_name='Qwen/Qwen2.5-7B', device='cuda',
                     prompt=None, max_new_tokens=64):
    """Profile KV cache from a real model."""
    if prompt is None:
        prompt = ("The theory of general relativity, proposed by Albert Einstein in 1915, "
                  "describes gravity as the curvature of spacetime caused by mass and energy. "
                  "This revolutionary insight fundamentally changed our understanding of the "
                  "universe, leading to predictions such as black holes, gravitational waves, "
                  "and the expansion of the cosmos. Einstein's field equations relate the "
                  "geometry of spacetime to the distribution of matter and energy within it.")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors='pt').to(device if device != 'auto' else 'cuda:0')
    seq_len = inputs['input_ids'].shape[1]
    print(f"Input: {seq_len} tokens")

    # Generate with KV cache returned
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            return_dict_in_generate=True, output_attentions=False,
            use_cache=True,
        )

    # Extract KV cache
    past_kv = outputs.past_key_values
    total_seq_len = outputs.sequences.shape[1]
    print(f"Generated: {total_seq_len - seq_len} new tokens, total seq_len={total_seq_len}")

    if past_kv is None:
        print("ERROR: No KV cache returned. Try a different model.")
        return []

    print(f"KV cache layers: {len(past_kv)}")

    all_results = []

    # Analyze each layer's K and V
    print(f"\n{'Layer':>5} {'Type':>5} {'Shape':<30} {'Zero%':>7} {'ExpEnt':>7} {'SmEnt':>7} "
          f"{'FullEnt':>8} {'DFloat':>7} {'MaxRatio':>9}")
    print("=" * 100)

    k_entropies = []
    v_entropies = []
    k_ratios = []
    v_ratios = []

    for layer_idx, (k, v) in enumerate(past_kv):
        if layer_idx >= 32:  # limit for speed
            break

        k_tensor = k.detach()
        v_tensor = v.detach()

        kr = analyze_kv(k_tensor, f"layer{layer_idx}_key")
        vr = analyze_kv(v_tensor, f"layer{layer_idx}_value")
        all_results.extend([kr, vr])

        k_entropies.append(kr['exponent_entropy'])
        v_entropies.append(vr['exponent_entropy'])
        k_ratios.append(kr['dfloat_ratio'])
        v_ratios.append(vr['dfloat_ratio'])

        print(f"{layer_idx:>5} {'K':>5} {str(kr['shape']):<30} {kr['zero_fraction']*100:>5.1f}% "
              f"{kr['exponent_entropy']:>6.2f} {kr['mantissa_entropy']:>6.2f} "
              f"{kr['full_entropy']:>7.2f} {kr['dfloat_ratio']:>6.3f}x {kr['theoretical_max_ratio']:>8.3f}x")
        print(f"{layer_idx:>5} {'V':>5} {str(vr['shape']):<30} {vr['zero_fraction']*100:>5.1f}% "
              f"{vr['exponent_entropy']:>6.2f} {vr['mantissa_entropy']:>6.2f} "
              f"{vr['full_entropy']:>7.2f} {vr['dfloat_ratio']:>6.3f}x {vr['theoretical_max_ratio']:>8.3f}x")

    # Page-level analysis (first few layers)
    print(f"\n{'='*80}")
    print("PAGE-LEVEL ANALYSIS (page_size=16 tokens, first 3 layers)")
    print(f"{'='*80}")
    print(f"{'Layer':>5} {'Type':>5} {'Page':>5} {'Tokens':<12} {'ExpEnt':>7} {'Ratio':>7}")
    print("-" * 50)

    for layer_idx in range(min(3, len(past_kv))):
        k, v = past_kv[layer_idx]
        k_pages = analyze_page_level(k.detach(), page_size=16)
        v_pages = analyze_page_level(v.detach(), page_size=16)
        for p in k_pages[:5]:
            print(f"{layer_idx:>5} {'K':>5} {p['page_idx']:>5} {str(p['token_range']):<12} "
                  f"{p['exponent_entropy']:>6.2f} {p['dfloat_ratio']:>6.3f}x")
        for p in v_pages[:5]:
            print(f"{layer_idx:>5} {'V':>5} {p['page_idx']:>5} {str(p['token_range']):<12} "
                  f"{p['exponent_entropy']:>6.2f} {p['dfloat_ratio']:>6.3f}x")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Key cache   — avg exp entropy: {np.mean(k_entropies):.3f} bits, avg ratio: {np.mean(k_ratios):.3f}x")
    print(f"Value cache — avg exp entropy: {np.mean(v_entropies):.3f} bits, avg ratio: {np.mean(v_ratios):.3f}x")
    print(f"Combined    — avg exp entropy: {np.mean(k_entropies + v_entropies):.3f} bits, "
          f"avg ratio: {np.mean(k_ratios + v_ratios):.3f}x")

    # Memory impact
    total_kv_bytes = sum(r['bytes'] for r in all_results)
    compressed_bytes = sum(r['bytes'] / r['dfloat_ratio'] for r in all_results)
    print(f"\nTotal KV cache: {total_kv_bytes / 1024 / 1024:.1f} MB")
    print(f"Compressed (DFloat-style): {compressed_bytes / 1024 / 1024:.1f} MB")
    print(f"Savings: {(1 - compressed_bytes/total_kv_bytes)*100:.1f}%")
    print(f"Effective ratio: {total_kv_bytes / compressed_bytes:.3f}x")

    # Save
    with open('experiments/kv_cache_profile.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    del model
    torch.cuda.empty_cache()
    return all_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
    p.add_argument('--device', default='cuda')
    p.add_argument('--max_new_tokens', type=int, default=64)
    args = p.parse_args()
    profile_kv_cache(args.model, args.device, max_new_tokens=args.max_new_tokens)
