"""
Test prev-value-conditioned ANS coding.

Key insight from sequential entropy analysis:
  H(W[i]|W[i-1]) ≈ 9.9 bpw vs H(W) ≈ 10.55 bpw — 0.65 bits savings per symbol!

Approach: condition W[i]'s ANS table on features of W[i-1]:
1. Exponent of W[i-1] as context (~30 unique exponents → 30 ANS tables)
2. Quantized W[i-1] as context (bucket previous value into K bins)
3. Full previous value as context (only for high-frequency values)

Also measures the theoretical H(W[i] | feature(W[i-1])) for each approach.
"""

import time
from argparse import ArgumentParser

import torch
import numpy as np
import constriction
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


def iid_entropy(W):
    vals, counts = np.unique(W, return_counts=True)
    p = counts / len(W)
    return -np.sum(p * np.log2(p))


def conditional_entropy(W, contexts):
    """H(W | context) where context is a discrete variable."""
    n = len(W)
    unique_ctx = np.unique(contexts)
    total_h = 0
    for c in unique_ctx:
        mask = contexts == c
        n_c = mask.sum()
        if n_c == 0:
            continue
        W_c = W[mask]
        vals, counts = np.unique(W_c, return_counts=True)
        p = counts / n_c
        h = -np.sum(p * np.log2(p))
        total_h += (n_c / n) * h
    return total_h


def prev_exp_conditioned_ans_size(tensors, row_sequential=True):
    """
    Encode W[i] conditioned on exp(W[i-1]) using per-context ANS tables.

    For row_sequential=True: process each row left-to-right, first element uses global table.
    For row_sequential=False: process entire flattened array sequentially.

    Returns: (compressed_size_bytes, n_contexts, overhead_bytes)
    """
    # Collect all (prev_exp, current_value) pairs
    pairs_by_ctx = {}  # context -> list of current values
    first_values = []  # first elements (no context)

    for t in tensors:
        W = t.contiguous().view(torch.int16).numpy()
        if row_sequential:
            for row_idx in range(W.shape[0]):
                row = W[row_idx]
                first_values.append(row[0])
                for i in range(1, len(row)):
                    prev_exp = int((row[i-1] >> 7) & 0xFF)
                    if prev_exp not in pairs_by_ctx:
                        pairs_by_ctx[prev_exp] = []
                    pairs_by_ctx[prev_exp].append(int(row[i]))
        else:
            flat = W.flatten()
            first_values.append(flat[0])
            exps = ((flat >> 7) & 0xFF).astype(np.uint8)
            for i in range(1, len(flat)):
                prev_exp = int(exps[i-1])
                if prev_exp not in pairs_by_ctx:
                    pairs_by_ctx[prev_exp] = []
                pairs_by_ctx[prev_exp].append(int(flat[i]))

    total_data_bytes = 0
    total_overhead = 0
    n_contexts = len(pairs_by_ctx)

    # Encode first values with global ANS
    firsts = np.array(first_values, dtype=np.int16)
    n_first = len(firsts)
    if n_first > 0:
        vals, counts = np.unique(firsts, return_counts=True)
        probs = (counts / n_first).astype(np.float32)
        mapping = np.zeros(65536, dtype=np.int32)
        for i, v in enumerate(vals):
            mapping[int(v) + 32768] = i
        data_idx = mapping[(firsts.astype(np.int32) + 32768)].astype(np.int32)

        model = constriction.stream.model.Categorical(probs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(data_idx, model)
        compressed = encoder.get_compressed()
        total_data_bytes += len(compressed) * 4
        total_overhead += len(vals) * 6  # symbol table + probs

    # Encode per-context
    for ctx, values in sorted(pairs_by_ctx.items()):
        arr = np.array(values, dtype=np.int16)
        n_ctx = len(arr)

        vals, counts = np.unique(arr, return_counts=True)
        n_unique = len(vals)
        probs = (counts / n_ctx).astype(np.float32)

        mapping = np.zeros(65536, dtype=np.int32)
        for i, v in enumerate(vals):
            mapping[int(v) + 32768] = i
        data_idx = mapping[(arr.astype(np.int32) + 32768)].astype(np.int32)

        model = constriction.stream.model.Categorical(probs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(data_idx, model)
        compressed = encoder.get_compressed()

        total_data_bytes += len(compressed) * 4
        total_overhead += n_unique * 6  # symbol table + probs per context

    return total_data_bytes + total_overhead, n_contexts, total_overhead


def prev_exp_conditioned_ans_size_vectorized(tensors):
    """
    Faster version: vectorized grouping, row-sequential processing.
    Encode W[i] conditioned on exp(W[i-1]).
    """
    # Collect all data
    all_prev_exp = []
    all_curr_val = []
    first_values = []

    for t in tensors:
        W = t.contiguous().view(torch.int16).numpy()
        for row_idx in range(W.shape[0]):
            row = W[row_idx]
            first_values.append(row[0])
            if len(row) > 1:
                prev_exps = ((row[:-1] >> 7) & 0xFF).astype(np.uint8)
                all_prev_exp.append(prev_exps)
                all_curr_val.append(row[1:])

    all_prev_exp = np.concatenate(all_prev_exp)
    all_curr_val = np.concatenate(all_curr_val)

    total_data_bytes = 0
    total_overhead = 0

    # Encode first values
    firsts = np.array(first_values, dtype=np.int16)
    if len(firsts) > 0:
        vals, counts = np.unique(firsts, return_counts=True)
        probs = (counts / len(firsts)).astype(np.float32)
        mapping = np.zeros(65536, dtype=np.int32)
        for i, v in enumerate(vals):
            mapping[int(v) + 32768] = i
        data_idx = mapping[(firsts.astype(np.int32) + 32768)].astype(np.int32)

        model = constriction.stream.model.Categorical(probs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(data_idx, model)
        compressed = encoder.get_compressed()
        total_data_bytes += len(compressed) * 4
        total_overhead += len(vals) * 6

    # Group by prev exponent and encode each group
    unique_exps = np.unique(all_prev_exp)
    n_contexts = len(unique_exps)

    for exp_val in unique_exps:
        mask = all_prev_exp == exp_val
        curr = all_curr_val[mask]
        n_ctx = len(curr)

        vals, counts = np.unique(curr, return_counts=True)
        probs = (counts / n_ctx).astype(np.float32)

        mapping = np.zeros(65536, dtype=np.int32)
        for i, v in enumerate(vals):
            mapping[int(v) + 32768] = i
        data_idx = mapping[(curr.astype(np.int32) + 32768)].astype(np.int32)

        model = constriction.stream.model.Categorical(probs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(data_idx, model)
        compressed = encoder.get_compressed()

        total_data_bytes += len(compressed) * 4
        total_overhead += len(vals) * 6

    return total_data_bytes + total_overhead, n_contexts, total_overhead


def quantized_prev_conditioned_entropy(tensors, n_bins=256):
    """
    H(W[i] | quantized(W[i-1])) where we bucket W[i-1] into n_bins equal-frequency bins.
    """
    all_prev = []
    all_curr = []

    for t in tensors:
        W = t.contiguous().view(torch.int16).flatten().numpy()
        all_prev.append(W[:-1])
        all_curr.append(W[1:])

    prev = np.concatenate(all_prev)
    curr = np.concatenate(all_curr)
    n = len(curr)

    # Create quantized bins using percentiles
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(prev, percentiles)
    bin_edges[-1] += 1  # ensure last value is included
    bin_ids = np.digitize(prev, bin_edges[1:])  # 0 to n_bins-1

    return conditional_entropy(curr, bin_ids)


def prev_full16_conditioned_entropy(tensors, max_pairs=5_000_000):
    """H(W[i] | W[i-1]) with full 16-bit previous value as context."""
    all_prev = []
    all_curr = []

    for t in tensors:
        W = t.contiguous().view(torch.int16).flatten().numpy()
        if len(W) > 1:
            all_prev.append(W[:-1])
            all_curr.append(W[1:])

    prev = np.concatenate(all_prev)
    curr = np.concatenate(all_curr)

    if len(curr) > max_pairs:
        curr = curr[:max_pairs]
        prev = prev[:max_pairs]

    return conditional_entropy(curr, prev)


def ans16_size(tensors):
    """Standard ANS-16bit (baseline)."""
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
    overhead = n_unique * 6
    return data_bytes + overhead


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

    total_original = 0
    total_ans16 = 0
    total_prev_exp_ans = 0

    print(f"\n{'='*95}")
    print("PREV-CONDITIONED ANS ANALYSIS")
    print(f"{'='*95}")

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)
        original = n * 2
        total_original += original

        print(f"\n  {wt}  ({n:,} params)")

        # i.i.d. entropy
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
        W = all_w.numpy()
        h_iid = iid_entropy(W)
        print(f"    H(W) i.i.d.:              {h_iid:.6f} bpw  -> {h_iid*n/8/original*100:.3f}%")

        # H(W[i] | exp(W[i-1])) — theoretical
        all_prev = []
        all_curr = []
        for t in tensors:
            Wt = t.contiguous().view(torch.int16).numpy()
            for row_idx in range(Wt.shape[0]):
                row = Wt[row_idx]
                if len(row) > 1:
                    prev_exps = ((row[:-1] >> 7) & 0xFF).astype(np.uint8)
                    all_prev.append(prev_exps)
                    all_curr.append(row[1:])
        prev_exps_all = np.concatenate(all_prev)
        curr_vals_all = np.concatenate(all_curr)
        h_prev_exp = conditional_entropy(curr_vals_all, prev_exps_all)
        print(f"    H(W[i]|exp(W[i-1])):      {h_prev_exp:.6f} bpw  -> {h_prev_exp*n/8/original*100:.3f}%")

        # H(W[i] | W[i-1]) — full bigram (theoretical LB)
        t0 = time.time()
        h_bigram = prev_full16_conditioned_entropy(tensors)
        t1 = time.time()
        print(f"    H(W[i]|W[i-1]) full:      {h_bigram:.6f} bpw  -> {h_bigram*n/8/original*100:.3f}%  [{t1-t0:.1f}s]")

        # Standard ANS-16bit
        t0 = time.time()
        ans_size = ans16_size(tensors)
        t1 = time.time()
        total_ans16 += ans_size
        print(f"    ANS-16bit:                {ans_size/original*100:.3f}%  ({ans_size/1e6:.1f}MB)  [{t1-t0:.1f}s]")

        # Prev-exp-conditioned ANS (actual compression)
        t0 = time.time()
        pec_size, n_ctx, overhead = prev_exp_conditioned_ans_size_vectorized(tensors)
        t1 = time.time()
        total_prev_exp_ans += pec_size
        delta = pec_size/original*100 - ans_size/original*100
        print(f"    Prev-exp-cond ANS:        {pec_size/original*100:.3f}%  ({pec_size/1e6:.1f}MB)  "
              f"[{n_ctx} ctx, overhead={overhead/1e3:.0f}KB]  delta={delta:+.3f}%  [{t1-t0:.1f}s]")

    # Summary
    print(f"\n{'='*95}")
    print("OVERALL SUMMARY")
    print(f"{'='*95}")

    ans_ratio = total_ans16 / total_original * 100
    pec_ratio = total_prev_exp_ans / total_original * 100
    delta = pec_ratio - ans_ratio
    savings = (total_ans16 - total_prev_exp_ans) / 1e6

    print(f"  Original bf16:           {total_original/1e6:.1f}MB")
    print(f"  ANS-16bit:               {ans_ratio:.3f}%  ({total_ans16/1e6:.1f}MB)")
    print(f"  Prev-exp-cond ANS:       {pec_ratio:.3f}%  ({total_prev_exp_ans/1e6:.1f}MB)  delta={delta:+.3f}%")
    print(f"  Savings vs ANS-16bit:    {savings:.1f}MB")

    if delta < 0:
        print(f"\n  *** Prev-exp-conditioned ANS beats standard ANS-16bit by {-delta:.3f}% ***")


if __name__ == "__main__":
    main()
