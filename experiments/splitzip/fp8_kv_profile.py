"""
FP8 KV Cache Compressibility Profiling
========================================
Profile KV cache in FP8 E4M3 and E5M2 formats to measure:
1. Exponent entropy (how compressible?)
2. Byte-level entropy
3. Achievable lossless compression ratio
4. Comparison with BF16 KV cache compression

FP8 E4M3: [sign(1) | exponent(4) | mantissa(3)] = 8 bits
FP8 E5M2: [sign(1) | exponent(5) | mantissa(2)] = 8 bits

For FP8, the "DFloat-style" approach splits:
  - Exponent: 4 or 5 bits → Huffman-encode
  - Sign+mantissa: 4 or 3 bits → raw

Total compressed = entropy(exp) + (1+mantissa_bits) bits per element
"""

import torch
import numpy as np
import time


def compute_entropy_gpu(tensor):
    """Shannon entropy in bits."""
    vals, counts = torch.unique(tensor, return_counts=True)
    probs = counts.float() / counts.sum()
    return -(probs * torch.log2(probs)).sum().item()


def analyze_fp8_kv(tensor_bf16, fmt='e4m3'):
    """Analyze FP8 KV cache compressibility.

    Takes BF16 tensor (real KV cache), converts to FP8, then profiles.
    """
    n = tensor_bf16.numel()

    # Convert BF16 → FP8
    if fmt == 'e4m3':
        fp8 = tensor_bf16.to(torch.float8_e4m3fn)
        exp_bits = 4
        mant_bits = 3
    else:
        fp8 = tensor_bf16.to(torch.float8_e5m2)
        exp_bits = 5
        mant_bits = 2

    # View FP8 as uint8
    raw = fp8.view(torch.uint8)

    # Extract fields
    if fmt == 'e4m3':
        # E4M3: [S(1) | E(4) | M(3)]
        exponents = (raw >> 3) & 0x0F  # bits [6:3]
        sign_mant = ((raw >> 4) & 0x08) | (raw & 0x07)  # sign(1) + mantissa(3) = 4 bits
        sign_mant_bits = 4
    else:
        # E5M2: [S(1) | E(5) | M(2)]
        exponents = (raw >> 2) & 0x1F  # bits [6:2]
        sign_mant = ((raw >> 5) & 0x04) | (raw & 0x03)  # sign(1) + mantissa(2) = 3 bits
        sign_mant_bits = 3

    # Entropy
    exp_entropy = compute_entropy_gpu(exponents)
    sm_entropy = compute_entropy_gpu(sign_mant)
    full_entropy = compute_entropy_gpu(raw)

    # Zero fraction
    zero_frac = (raw == 0).float().mean().item()

    # Compression ratios
    # DFloat-style: Huffman(exponent) + raw(sign_mantissa)
    dfloat_bits = exp_entropy + sign_mant_bits
    dfloat_ratio = 8.0 / dfloat_bits

    # Theoretical max (full entropy coding)
    theoretical_ratio = 8.0 / full_entropy if full_entropy > 0 else float('inf')

    # Quantization error from BF16→FP8
    fp8_back = fp8.to(torch.bfloat16)
    abs_err = (tensor_bf16.float() - fp8_back.float()).abs()

    return {
        'format': fmt,
        'n_elements': n,
        'zero_fraction': round(zero_frac, 4),
        'exp_bits': exp_bits,
        'mant_bits': mant_bits,
        'exp_entropy': round(exp_entropy, 3),
        'sm_entropy': round(sm_entropy, 3),
        'full_entropy': round(full_entropy, 3),
        'dfloat_bits_per_elem': round(dfloat_bits, 2),
        'dfloat_ratio': round(dfloat_ratio, 3),
        'theoretical_ratio': round(theoretical_ratio, 3),
        'quant_max_err': round(abs_err.max().item(), 6),
        'quant_rmse': round(abs_err.pow(2).mean().sqrt().item(), 6),
    }


def profile_fp8_kv_from_model(model_name='Qwen/Qwen2.5-7B', device='cuda',
                                max_new_tokens=64):
    """Profile FP8 KV cache from a real model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt = ("The theory of general relativity describes gravity as the curvature of "
              "spacetime caused by mass and energy. Einstein's field equations relate "
              "the geometry of spacetime to the distribution of matter within it.")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors='pt').to(device if device != 'auto' else 'cuda:0')
    print(f"Input: {inputs['input_ids'].shape[1]} tokens")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            return_dict_in_generate=True, use_cache=True,
        )

    past_kv = outputs.past_key_values
    total_seq_len = outputs.sequences.shape[1]
    print(f"KV cache: {len(past_kv)} layers, seq_len={total_seq_len}")

    # Profile each format
    for fmt in ['e4m3', 'e5m2']:
        print(f"\n{'='*100}")
        print(f"FP8 {fmt.upper()} KV Cache Analysis")
        print(f"{'='*100}")
        print(f"{'Layer':>5} {'Type':>4} {'ExpEnt':>7} {'SmEnt':>6} {'FullEnt':>8} "
              f"{'DFloat':>7} {'MaxRatio':>8} {'Zero%':>6} {'QuantErr':>9}")
        print("-" * 75)

        k_ents = []
        v_ents = []
        k_ratios = []
        v_ratios = []

        for layer_idx, (k, v) in enumerate(past_kv):
            if layer_idx >= 28:
                break

            kr = analyze_fp8_kv(k.detach(), fmt)
            vr = analyze_fp8_kv(v.detach(), fmt)

            k_ents.append(kr['exp_entropy'])
            v_ents.append(vr['exp_entropy'])
            k_ratios.append(kr['dfloat_ratio'])
            v_ratios.append(vr['dfloat_ratio'])

            if layer_idx < 5 or layer_idx >= len(past_kv) - 3:  # show first 5 and last 3
                print(f"{layer_idx:>5} {'K':>4} {kr['exp_entropy']:>6.2f} {kr['sm_entropy']:>5.2f} "
                      f"{kr['full_entropy']:>7.2f} {kr['dfloat_ratio']:>6.3f}x "
                      f"{kr['theoretical_ratio']:>7.3f}x {kr['zero_fraction']*100:>5.1f}% "
                      f"{kr['quant_max_err']:>8.4f}")
                print(f"{layer_idx:>5} {'V':>4} {vr['exp_entropy']:>6.2f} {vr['sm_entropy']:>5.2f} "
                      f"{vr['full_entropy']:>7.2f} {vr['dfloat_ratio']:>6.3f}x "
                      f"{vr['theoretical_ratio']:>7.3f}x {vr['zero_fraction']*100:>5.1f}% "
                      f"{vr['quant_max_err']:>8.4f}")

        print(f"\nSummary ({fmt.upper()}):")
        print(f"  Key cache — avg exp entropy: {np.mean(k_ents):.3f}/{kr['exp_bits']} bits, "
              f"avg ratio: {np.mean(k_ratios):.3f}x")
        print(f"  Val cache — avg exp entropy: {np.mean(v_ents):.3f}/{vr['exp_bits']} bits, "
              f"avg ratio: {np.mean(v_ratios):.3f}x")
        print(f"  Combined  — avg ratio: {np.mean(k_ratios + v_ratios):.3f}x")

    # BF16 comparison
    print(f"\n{'='*100}")
    print("COMPARISON: BF16 vs FP8 KV Cache Compression")
    print(f"{'='*100}")

    # Collect BF16 stats
    bf16_ratios = []
    for layer_idx, (k, v) in enumerate(past_kv):
        if layer_idx >= 28:
            break
        for tensor in [k.detach(), v.detach()]:
            flat = tensor.view(-1)
            int16 = flat.view(torch.int16)
            exponents = ((int16 >> 7) & 0xFF).to(torch.uint8)
            ent = compute_entropy_gpu(exponents)
            bf16_ratios.append(16.0 / (ent + 8.0))

    avg_bf16_ratio = np.mean(bf16_ratios)

    # Summary table
    print(f"\n{'Format':<12} {'Size/elem':>10} {'Comp ratio':>11} {'Compressed':>11} {'Total vs BF16':>14}")
    print("-" * 60)

    # BF16 raw
    print(f"{'BF16 raw':<12} {'16 bits':>10} {'1.000x':>11} {'16.0 bits':>11} {'1.000x':>14}")

    # BF16 + SplitZip
    bf16_comp_bits = 16.0 / avg_bf16_ratio
    print(f"{'BF16+SplitZ':<12} {'16 bits':>10} {avg_bf16_ratio:>10.3f}x "
          f"{bf16_comp_bits:>9.1f} bits {avg_bf16_ratio:>13.3f}x")

    # FP8 E4M3 raw
    print(f"{'FP8 E4M3':<12} {'8 bits':>10} {'1.000x':>11} {'8.0 bits':>11} {'2.000x':>14}")

    # FP8 E4M3 + SplitZip
    e4m3_ratio = np.mean([r for r in k_ratios + v_ratios])  # from last fmt loop
    # Re-compute for e4m3
    e4m3_ents = []
    for layer_idx, (k, v) in enumerate(past_kv):
        if layer_idx >= 28:
            break
        for tensor in [k.detach(), v.detach()]:
            r = analyze_fp8_kv(tensor, 'e4m3')
            e4m3_ents.append(r['dfloat_ratio'])
    avg_e4m3_ratio = np.mean(e4m3_ents)
    e4m3_comp_bits = 8.0 / avg_e4m3_ratio
    total_vs_bf16 = 16.0 / e4m3_comp_bits

    print(f"{'FP8+SplitZ':<12} {'8 bits':>10} {avg_e4m3_ratio:>10.3f}x "
          f"{e4m3_comp_bits:>9.1f} bits {total_vs_bf16:>13.3f}x")

    # FP8 E5M2 + SplitZip
    e5m2_ents = []
    for layer_idx, (k, v) in enumerate(past_kv):
        if layer_idx >= 28:
            break
        for tensor in [k.detach(), v.detach()]:
            r = analyze_fp8_kv(tensor, 'e5m2')
            e5m2_ents.append(r['dfloat_ratio'])
    avg_e5m2_ratio = np.mean(e5m2_ents)
    e5m2_comp_bits = 8.0 / avg_e5m2_ratio
    total_vs_bf16_e5m2 = 16.0 / e5m2_comp_bits

    print(f"{'FP8E5+Split':<12} {'8 bits':>10} {avg_e5m2_ratio:>10.3f}x "
          f"{e5m2_comp_bits:>9.1f} bits {total_vs_bf16_e5m2:>13.3f}x")

    print(f"""
KEY INSIGHT:
  - BF16 + SplitZip: {avg_bf16_ratio:.2f}x compression (lossless)
  - FP8 alone: 2.00x compression (lossy)
  - FP8 + SplitZip: {total_vs_bf16:.2f}x compression vs BF16 (FP8 lossy + SplitZip lossless on FP8)
  - SplitZip is ORTHOGONAL to FP8: it works on top of FP8 for additional lossless savings
  - For native FP8 models, SplitZip gives an extra {avg_e4m3_ratio:.2f}x on top of FP8
""")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    profile_fp8_kv_from_model(args.model, args.device)
