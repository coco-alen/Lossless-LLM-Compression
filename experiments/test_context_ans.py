"""
Test context-dependent ANS coding.

Instead of using a single probability distribution for all weights,
use different distributions based on context:
1. Per-row probability tables
2. Exponent-conditioned probability tables (ANS version of exp-conditioned)
3. Previous-value conditioned (Markov model)
"""

import time
from argparse import ArgumentParser

import torch
import numpy as np
import constriction
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


def dfloat11_size(tensors):
    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    n = all_w.numel()
    exp = ((all_w >> 7) & 0xFF).to(torch.uint8).numpy()
    freq = {int(v): int(c) for v, c in zip(*np.unique(exp, return_counts=True))}
    codec = HuffmanCodec.from_frequencies(freq)
    table = codec.get_code_table()
    huf_bits = sum(code_len * freq.get(sym, 0) for sym, (code_len, _) in table.items() if sym in freq)
    return (huf_bits + 7) // 8 + n


def ans16_size(tensors):
    """Standard ANS-16bit (our best so far)."""
    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    W = all_w.numpy()
    n = len(W)

    vals, counts = np.unique(W, return_counts=True)
    n_unique = len(vals)
    probs = (counts / n).astype(np.float32)

    mapping = np.zeros(65536, dtype=np.int32)
    for i, v in enumerate(vals):
        mapping[int(v) + 32768] = i
    data_idx = mapping[(W.astype(np.int32) + 32768)].astype(np.int32)

    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(data_idx, model)
    compressed = encoder.get_compressed()

    data_bytes = len(compressed) * 4
    overhead = n_unique * 6  # table + probs
    return data_bytes + overhead


def per_layer_ans16_size(tensors):
    """Per-layer ANS-16bit."""
    total = 0
    for t in tensors:
        W = t.contiguous().view(torch.int16).flatten().numpy()
        n = len(W)

        vals, counts = np.unique(W, return_counts=True)
        n_unique = len(vals)
        probs = (counts / n).astype(np.float32)

        mapping = np.zeros(65536, dtype=np.int32)
        for i, v in enumerate(vals):
            mapping[int(v) + 32768] = i
        data_idx = mapping[(W.astype(np.int32) + 32768)].astype(np.int32)

        model = constriction.stream.model.Categorical(probs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(data_idx, model)
        compressed = encoder.get_compressed()

        total += len(compressed) * 4 + n_unique * 6
    return total


def exp_conditioned_ans_size(tensors):
    """Exponent-conditioned ANS: ANS(exp) + per-exp ANS(sign_mantissa).
    This should be close to H(exp) + H(sm|exp), the conditional entropy."""
    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    W_int16 = all_w.numpy()
    n = len(W_int16)

    exp = ((all_w >> 7) & 0xFF).to(torch.uint8).numpy()
    sm = ((all_w >> 8) & 0x80 | (all_w & 0x7F)).to(torch.uint8).numpy()

    # ANS encode exponent
    exp_vals, exp_counts = np.unique(exp, return_counts=True)
    exp_probs = (exp_counts / n).astype(np.float32)
    exp_mapping = np.zeros(256, dtype=np.int32)
    for i, v in enumerate(exp_vals):
        exp_mapping[v] = i
    exp_idx = exp_mapping[exp].astype(np.int32)

    model_exp = constriction.stream.model.Categorical(exp_probs, perfect=False)
    enc_exp = constriction.stream.stack.AnsCoder()
    enc_exp.encode_reverse(exp_idx, model_exp)
    exp_compressed = enc_exp.get_compressed()
    exp_bytes = len(exp_compressed) * 4 + len(exp_vals) * 5

    # Per-exponent ANS for sign_mantissa
    sm_bytes = 0
    for ev in exp_vals:
        mask = exp == ev
        sm_sub = sm[mask]
        n_sub = len(sm_sub)
        if n_sub == 0:
            continue

        sm_vals, sm_counts = np.unique(sm_sub, return_counts=True)
        sm_probs = (sm_counts / n_sub).astype(np.float64)
        # Ensure probabilities are valid float32 (no zeros after rounding)
        sm_probs = np.maximum(sm_probs, 1e-10).astype(np.float32)
        sm_probs = sm_probs / sm_probs.sum()  # renormalize
        sm_mapping = np.zeros(256, dtype=np.int32)
        for i, v in enumerate(sm_vals):
            sm_mapping[v] = i
        sm_idx = sm_mapping[sm_sub].astype(np.int32)

        model_sm = constriction.stream.model.Categorical(sm_probs, perfect=False)
        enc_sm = constriction.stream.stack.AnsCoder()
        enc_sm.encode_reverse(sm_idx, model_sm)
        compressed_sm = enc_sm.get_compressed()
        sm_bytes += len(compressed_sm) * 4 + len(sm_vals) * 5

    return exp_bytes + sm_bytes


def conditional_entropy_estimate(tensors):
    """Estimate the conditional entropy H(value | exponent) to see if
    conditioning on exponent gives better compression than unconditional."""
    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    W = all_w.numpy()
    n = len(W)

    exp = ((all_w >> 7) & 0xFF).to(torch.uint8).numpy()

    # H(full 16-bit)
    vals16, counts16 = np.unique(W, return_counts=True)
    p16 = counts16 / n
    h_full = -np.sum(p16 * np.log2(p16))

    # H(exp)
    exp_vals, exp_counts = np.unique(exp, return_counts=True)
    p_exp = exp_counts / n
    h_exp = -np.sum(p_exp * np.log2(p_exp))

    # H(full 16-bit | exp) = sum_e P(e) * H(W | exp=e)
    h_cond = 0
    for i, ev in enumerate(exp_vals):
        mask = exp == ev
        W_sub = W[mask]
        n_sub = len(W_sub)
        vals_sub, counts_sub = np.unique(W_sub, return_counts=True)
        p_sub = counts_sub / n_sub
        h_sub = -np.sum(p_sub * np.log2(p_sub))
        h_cond += (n_sub / n) * h_sub

    # H(exp) + H(W|exp) should be close to or equal to H(W) if exp is deterministic given W
    # Actually H(exp) + H(sm|exp) = H(W) when exp determines the high bits
    return h_full, h_exp, h_cond


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

    totals = {"df11": 0, "ans16": 0, "per_layer_ans16": 0, "exp_cond_ans": 0, "original": 0}

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)
        original = n * 2
        totals["original"] += original

        print(f"\n  {wt}  ({n:,} params)")

        # Conditional entropy analysis
        h_full, h_exp, h_cond = conditional_entropy_estimate(tensors)
        print(f"    H(W)={h_full:.4f}  H(exp)={h_exp:.4f}  H(W|exp)={h_cond:.4f}  "
              f"H(exp)+H(W|exp)={h_exp+h_cond:.4f}")

        # DFloat11
        df11 = dfloat11_size(tensors)
        totals["df11"] += df11

        # ANS-16bit
        t0 = time.time()
        ans = ans16_size(tensors)
        t1 = time.time()
        totals["ans16"] += ans

        # Per-layer ANS-16bit
        t2 = time.time()
        pl_ans = per_layer_ans16_size(tensors)
        t3 = time.time()
        totals["per_layer_ans16"] += pl_ans

        # Exp-conditioned ANS
        t4 = time.time()
        ec_ans = exp_conditioned_ans_size(tensors)
        t5 = time.time()
        totals["exp_cond_ans"] += ec_ans

        for name, size, elapsed in [
            ("DFloat11", df11, 0),
            ("ANS-16bit", ans, t1-t0),
            ("Per-layer ANS-16", pl_ans, t3-t2),
            ("Exp-cond ANS", ec_ans, t5-t4),
            ("Entropy LB", int(np.ceil(h_full * n / 8)), 0),
        ]:
            ratio = size / original * 100
            delta = ratio - df11 / original * 100
            print(f"    {name:<22} {ratio:6.2f}%  {delta:>+.2f}%  {size/1e6:.1f}MB  {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    df11_ratio = totals["df11"] / totals["original"] * 100
    for name, key in [("DFloat11", "df11"), ("ANS-16bit", "ans16"),
                       ("Per-layer ANS-16", "per_layer_ans16"),
                       ("Exp-cond ANS", "exp_cond_ans")]:
        size = totals[key]
        ratio = size / totals["original"] * 100
        delta = ratio - df11_ratio
        marker = " ***" if delta < 0 else ""
        print(f"  {name:<22} {ratio:6.2f}%  {delta:>+.2f}%  {size/1e6:.1f}MB{marker}")


if __name__ == "__main__":
    main()
