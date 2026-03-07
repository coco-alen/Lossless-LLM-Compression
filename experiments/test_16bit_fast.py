"""
Fast test of 16-bit Huffman coding using code table analysis (no actual encoding).

Computes exact Huffman encoded size from frequency distribution and code table.
"""

import time
from argparse import ArgumentParser

import torch
import numpy as np
from dahuffman import HuffmanCodec
from transformers import AutoModelForCausalLM, AutoConfig

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


def huffman_size_from_freq(freq: dict) -> tuple:
    """Compute Huffman encoded size from frequency dict.
    Returns (encoded_bytes, huffman_bpw, entropy_bpw, max_code_len, n_unique).
    """
    total = sum(freq.values())
    if len(freq) <= 1:
        return (total + 7) // 8, 1.0, 0.0, 1, len(freq)

    codec = HuffmanCodec.from_frequencies(freq)
    table = codec.get_code_table()

    # Compute exact Huffman bit count
    huf_bits = 0
    max_code_len = 0
    for sym, (code_len, _) in table.items():
        if sym in freq:
            huf_bits += code_len * freq[sym]
            max_code_len = max(max_code_len, code_len)

    # Entropy
    probs = np.array(list(freq.values())) / total
    entropy = -np.sum(probs * np.log2(probs))

    encoded_bytes = (huf_bits + 7) // 8
    huf_bpw = huf_bits / total
    return encoded_bytes, huf_bpw, entropy, max_code_len, len(freq)


def analyze_weight_type(tensors, label):
    """Analyze all compression methods for one weight type."""
    all_w = torch.cat([t.flatten() for t in tensors])
    W = all_w.view(torch.int16)
    n = W.numel()
    original_bytes = n * 2

    # Extract fields
    exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
    sm = ((W >> 8) & 0x80 | (W & 0x7F)).to(torch.uint8).numpy()
    sign = ((W >> 15) & 1).to(torch.uint8).numpy()
    man7 = (W & 0x7F).to(torch.uint8).numpy()

    # Build frequency tables
    exp_freq = {}
    for v, c in zip(*np.unique(exp, return_counts=True)):
        exp_freq[int(v)] = int(c)

    sm_freq = {}
    for v, c in zip(*np.unique(sm, return_counts=True)):
        sm_freq[int(v)] = int(c)

    # Full 16-bit frequency table
    W_np = W.numpy()
    vals16, counts16 = np.unique(W_np, return_counts=True)
    freq16 = {int(v): int(c) for v, c in zip(vals16, counts16)}

    print(f"\n{'='*100}")
    print(f"  {label}  ({n:,} params, {original_bytes/1e6:.1f} MB)")
    print(f"{'='*100}")
    print(f"  {'Method':<45} {'Ratio':>7} {'bpw':>6} {'vs DF11':>8}  Detail")
    print(f"  {'-'*45} {'-'*7} {'-'*6} {'-'*8}  {'-'*30}")

    results = {}

    # ----- 1. DFloat11 baseline -----
    exp_enc_bytes, exp_huf_bpw, exp_ent, _, exp_nuniq = huffman_size_from_freq(exp_freq)
    df11_bytes = exp_enc_bytes + n  # huffman(exp) + raw(sm)
    df11_ratio = df11_bytes / original_bytes * 100
    df11_bpw = df11_ratio / 100 * 16
    print(f"  {'DFloat11: Huffman(exp) + raw(sm)':<45} {df11_ratio:>6.2f}% {df11_bpw:>5.3f} {'---':>8}  exp_huf={exp_huf_bpw:.3f}bpw, sm=8.000bpw")
    results["dfloat11"] = df11_bytes

    # ----- 2. Full 16-bit Huffman -----
    enc16_bytes, huf16_bpw, ent16, maxlen16, nuniq16 = huffman_size_from_freq(freq16)
    # Table overhead: 4 bytes per unique value (value + code description)
    tbl16 = nuniq16 * 4
    total16 = enc16_bytes + tbl16
    ratio16 = total16 / original_bytes * 100
    bpw16 = ratio16 / 100 * 16
    delta16 = ratio16 - df11_ratio
    marker = " ***" if delta16 < 0 else ""
    print(f"  {'16-bit Huffman (all layers together)':<45} {ratio16:>6.2f}% {bpw16:>5.3f} {delta16:>+7.2f}%  "
          f"H={ent16:.3f} huf={huf16_bpw:.3f} unique={nuniq16} maxlen={maxlen16} tbl={tbl16}{marker}")
    results["16bit_huf"] = total16

    # ----- 3. Per-layer 16-bit Huffman -----
    pl_total = 0
    pl_tbl = 0
    for t in tensors:
        W_l = t.contiguous().view(torch.int16).flatten().numpy()
        vals_l, counts_l = np.unique(W_l, return_counts=True)
        freq_l = {int(v): int(c) for v, c in zip(vals_l, counts_l)}
        enc_l, _, _, _, nuniq_l = huffman_size_from_freq(freq_l)
        pl_total += enc_l
        pl_tbl += nuniq_l * 4
    total_pl = pl_total + pl_tbl
    ratio_pl = total_pl / original_bytes * 100
    bpw_pl = ratio_pl / 100 * 16
    delta_pl = ratio_pl - df11_ratio
    marker = " ***" if delta_pl < 0 else ""
    print(f"  {'Per-layer 16-bit Huffman':<45} {ratio_pl:>6.2f}% {bpw_pl:>5.3f} {delta_pl:>+7.2f}%  "
          f"enc={pl_total/1e6:.2f}MB tbl={pl_tbl/1e3:.1f}KB{marker}")
    results["16bit_per_layer"] = total_pl

    # ----- 4. Exp-conditioned Huffman(sm) -----
    unique_exps = np.unique(exp)
    cond_sm_total = 0
    cond_tbl = 0
    for ev in unique_exps:
        mask = exp == ev
        sm_sub = sm[mask]
        vals_s, counts_s = np.unique(sm_sub, return_counts=True)
        freq_s = {int(v): int(c) for v, c in zip(vals_s, counts_s)}
        enc_s, _, _, _, nuniq_s = huffman_size_from_freq(freq_s)
        cond_sm_total += enc_s
        cond_tbl += nuniq_s * 3
    total_cond = exp_enc_bytes + cond_sm_total + cond_tbl
    ratio_cond = total_cond / original_bytes * 100
    bpw_cond = ratio_cond / 100 * 16
    delta_cond = ratio_cond - df11_ratio
    marker = " ***" if delta_cond < 0 else ""
    print(f"  {'Exp-conditioned Huffman(sm)':<45} {ratio_cond:>6.2f}% {bpw_cond:>5.3f} {delta_cond:>+7.2f}%  "
          f"exp={exp_enc_bytes/1e6:.2f}MB cond_sm={cond_sm_total/1e6:.2f}MB tbl={cond_tbl}{marker}")
    results["exp_cond"] = total_cond

    # ----- 5. 16-bit Huffman (no table overhead - theoretical) -----
    ratio16_nt = enc16_bytes / original_bytes * 100
    bpw16_nt = ratio16_nt / 100 * 16
    delta16_nt = ratio16_nt - df11_ratio
    print(f"  {'16-bit Huffman (no tbl overhead)':<45} {ratio16_nt:>6.2f}% {bpw16_nt:>5.3f} {delta16_nt:>+7.02f}%  "
          f"pure encoded bytes")
    results["16bit_no_tbl"] = enc16_bytes

    # ----- 6. 16-bit entropy lower bound -----
    ent_bytes = int(np.ceil(ent16 * n / 8))
    ratio_ent = ent_bytes / original_bytes * 100
    bpw_ent = ratio_ent / 100 * 16
    delta_ent = ratio_ent - df11_ratio
    print(f"  {'16-bit entropy lower bound':<45} {ratio_ent:>6.2f}% {bpw_ent:>5.3f} {delta_ent:>+7.2f}%  "
          f"H(bf16)={ent16:.4f} bpw")
    results["entropy_lb"] = ent_bytes

    # ----- 7. DFloat11 + Huffman(sm) (simplest improvement) -----
    sm_enc_bytes, sm_huf_bpw, sm_ent, _, sm_nuniq = huffman_size_from_freq(sm_freq)
    total_both = exp_enc_bytes + sm_enc_bytes
    ratio_both = total_both / original_bytes * 100
    bpw_both = ratio_both / 100 * 16
    delta_both = ratio_both - df11_ratio
    print(f"  {'Huffman(exp) + Huffman(sm)':<45} {ratio_both:>6.2f}% {bpw_both:>5.3f} {delta_both:>+7.2f}%  "
          f"sm_huf={sm_huf_bpw:.3f}bpw H(sm)={sm_ent:.3f}")
    results["huf_both"] = total_both

    # ----- 8. Exp-cond entropy lower bound -----
    cond_sm_entropy_bits = 0
    for ev in unique_exps:
        mask = exp == ev
        sm_sub = sm[mask]
        if len(sm_sub) == 0:
            continue
        vals_s, counts_s = np.unique(sm_sub, return_counts=True)
        probs_s = counts_s / len(sm_sub)
        h_s = -np.sum(counts_s * np.log2(probs_s))
        cond_sm_entropy_bits += h_s
    exp_entropy_bits = exp_ent * n
    total_cond_ent = int(np.ceil((exp_entropy_bits + cond_sm_entropy_bits) / 8))
    ratio_cond_ent = total_cond_ent / original_bytes * 100
    bpw_cond_ent = ratio_cond_ent / 100 * 16
    delta_cond_ent = ratio_cond_ent - df11_ratio
    print(f"  {'Exp-cond entropy LB (H(exp)+H(sm|exp))':<45} {ratio_cond_ent:>6.2f}% {bpw_cond_ent:>5.3f} {delta_cond_ent:>+7.2f}%  "
          f"H(sm|exp)={cond_sm_entropy_bits/n:.4f}bpw")

    return results


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    num_layers = config.num_hidden_layers
    print(f"Model: {args.model_name_or_path}  ({num_layers} layers)")
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("Extracting weights...", flush=True)
    groups = extract_weights(model, num_layers)
    del model

    all_results = {}
    totals = {}

    for wt in WEIGHT_TYPES:
        results = analyze_weight_type(groups[wt], wt)
        all_results[wt] = results
        for method, size in results.items():
            if method not in totals:
                totals[method] = 0
            totals[method] += size

    # Total original
    total_orig = sum(sum(t.numel() for t in groups[wt]) * 2 for wt in WEIGHT_TYPES)

    print(f"\n\n{'='*100}")
    print("OVERALL SUMMARY (all weight types combined)")
    print(f"{'='*100}")
    df11_ratio = totals["dfloat11"] / total_orig * 100

    methods = [
        ("DFloat11", "dfloat11"),
        ("16-bit Huffman", "16bit_huf"),
        ("Per-layer 16-bit Huffman", "16bit_per_layer"),
        ("Exp-conditioned Huffman(sm)", "exp_cond"),
        ("16-bit Huffman (no tbl)", "16bit_no_tbl"),
        ("Huffman(exp) + Huffman(sm)", "huf_both"),
        ("16-bit entropy LB", "entropy_lb"),
    ]

    for name, key in methods:
        if key not in totals:
            continue
        size = totals[key]
        ratio = size / total_orig * 100
        bpw = ratio / 100 * 16
        delta = ratio - df11_ratio
        marker = " ***BEATS DF11***" if delta < 0 else ""
        print(f"  {name:<45} {ratio:>6.2f}%  ({bpw:.3f} bpw)  {delta:>+.2f}%  "
              f"size={size/1e6:.1f}MB{marker}")


if __name__ == "__main__":
    main()
