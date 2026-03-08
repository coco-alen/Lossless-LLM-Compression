"""
Test row reordering + quantized context coding.

Approach 1: Sort rows by similarity to maximize sequential correlation,
then use sequential ANS. Permutation overhead: ~R*2 bytes (negligible).

Approach 2: Map previous 16-bit value to K buckets (quantized context),
use per-bucket ANS tables. Find optimal K that balances data savings vs overhead.

Approach 3: Interleave across layers at same (i,j) position.
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


def bigram_entropy(W, max_n=5_000_000):
    """H(W[i] | W[i-1]) estimated from bigram counts."""
    if len(W) > max_n + 1:
        W = W[:max_n + 1]
    n = len(W) - 1
    W_u = (W.astype(np.int32) + 32768).astype(np.uint16)
    prev = W_u[:-1]
    curr = W_u[1:]

    sort_idx = np.argsort(prev, kind='mergesort')
    prev_sorted = prev[sort_idx]
    curr_sorted = curr[sort_idx]

    boundaries = np.where(np.diff(prev_sorted) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [n]])

    total_h = 0.0
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        sz = end - start
        group = curr_sorted[start:end]
        vals, counts = np.unique(group, return_counts=True)
        p = counts / sz
        h = -np.sum(p * np.log2(p))
        total_h += (sz / n) * h

    return total_h


def sorted_rows_bigram_entropy(tensors, max_n=5_000_000):
    """Sort rows by first element, flatten, compute bigram entropy."""
    all_rows = []
    for t in tensors:
        W = t.contiguous().view(torch.int16).numpy()
        for row_idx in range(W.shape[0]):
            all_rows.append(W[row_idx])

    # Sort rows by mean value (as float)
    row_means = [r.astype(np.float32).mean() for r in all_rows]
    sort_order = np.argsort(row_means)
    sorted_flat = np.concatenate([all_rows[i] for i in sort_order])

    return bigram_entropy(sorted_flat, max_n)


def sorted_rows_iid_entropy(tensors):
    """i.i.d. entropy doesn't change with reordering (same values)."""
    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    return iid_entropy(all_w.numpy())


def row_sorted_quantized_context_entropy(tensors, K, max_n=5_000_000):
    """
    Sort rows, flatten, then compute H(W[i] | bucket(W[i-1])) with K buckets.
    Buckets are equal-frequency based on value rank.
    """
    all_rows = []
    for t in tensors:
        W = t.contiguous().view(torch.int16).numpy()
        for row_idx in range(W.shape[0]):
            all_rows.append(W[row_idx])

    row_means = [r.astype(np.float32).mean() for r in all_rows]
    sort_order = np.argsort(row_means)
    sorted_flat = np.concatenate([all_rows[i] for i in sort_order])

    if len(sorted_flat) > max_n + 1:
        sorted_flat = sorted_flat[:max_n + 1]

    prev = sorted_flat[:-1]
    curr = sorted_flat[1:]

    # Create rank-based buckets
    unique_vals = np.unique(prev)
    val_to_rank = np.zeros(65536, dtype=np.int32)
    for i, v in enumerate(unique_vals):
        val_to_rank[int(v) + 32768] = i
    ranks = val_to_rank[(prev.astype(np.int32) + 32768)]
    n_unique = len(unique_vals)
    bucket_ids = (ranks * K // n_unique).astype(np.int32)
    bucket_ids = np.minimum(bucket_ids, K - 1)

    # Conditional entropy
    n = len(curr)
    total_h = 0
    for b in range(K):
        mask = bucket_ids == b
        n_b = mask.sum()
        if n_b < 2:
            continue
        vals, counts = np.unique(curr[mask], return_counts=True)
        p = counts / n_b
        h = -np.sum(p * np.log2(p))
        total_h += (n_b / n) * h

    return total_h


def quantized_context_ans_size(tensors, K, row_sort=False):
    """
    Actual ANS compression with K quantized contexts.
    Optionally sort rows first.

    Returns (total_bytes, overhead_bytes)
    """
    # Prepare data
    if row_sort:
        all_rows = []
        for t in tensors:
            W = t.contiguous().view(torch.int16).numpy()
            for row_idx in range(W.shape[0]):
                all_rows.append(W[row_idx])
        row_means = [r.astype(np.float32).mean() for r in all_rows]
        sort_order = np.argsort(row_means)
        flat = np.concatenate([all_rows[i] for i in sort_order])
        permutation_overhead = len(all_rows) * 2  # store uint16 indices
    else:
        flat = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors]).numpy()
        permutation_overhead = 0

    n = len(flat)
    prev = flat[:-1]
    curr = flat[1:]

    # Create rank-based buckets for prev values
    unique_vals = np.unique(flat)
    n_unique = len(unique_vals)
    val_to_rank = np.zeros(65536, dtype=np.int32)
    for i, v in enumerate(unique_vals):
        val_to_rank[int(v) + 32768] = i
    ranks = val_to_rank[(prev.astype(np.int32) + 32768)]
    bucket_ids = np.minimum((ranks * K // n_unique).astype(np.int32), K - 1)

    total_data_bytes = 0
    total_overhead = 0

    # Encode first element with global table
    first_val = flat[:1]
    # (negligible, just count 4 bytes for it)
    total_data_bytes += 4

    # Per-bucket encoding
    for b in range(K):
        mask = bucket_ids == b
        n_b = mask.sum()
        if n_b == 0:
            continue

        data = curr[mask]
        vals, counts = np.unique(data, return_counts=True)
        if len(vals) == 1:
            total_overhead += 6  # just store the value
            continue

        probs = (counts / n_b).astype(np.float64)
        probs = np.maximum(probs, 1e-10)
        probs = (probs / probs.sum()).astype(np.float32)
        probs = np.maximum(probs, np.float32(1e-10))
        probs = probs / probs.sum()

        mapping = np.zeros(65536, dtype=np.int32)
        for i, v in enumerate(vals):
            mapping[int(v) + 32768] = i
        data_idx = mapping[(data.astype(np.int32) + 32768)].astype(np.int32)

        try:
            model = constriction.stream.model.Categorical(probs, perfect=False)
            encoder = constriction.stream.stack.AnsCoder()
            encoder.encode_reverse(data_idx, model)
            compressed = encoder.get_compressed()
            total_data_bytes += len(compressed) * 4
        except ValueError:
            # Fallback: estimate from entropy
            h = -np.sum(probs * np.log2(np.maximum(probs, 1e-30)))
            total_data_bytes += int(np.ceil(h * n_b / 8))

        total_overhead += len(vals) * 6

    total_overhead += permutation_overhead
    return total_data_bytes + total_overhead, total_overhead


def cross_layer_interleave_entropy(tensors, max_n=5_000_000):
    """
    Interleave weights at same (i,j) position across layers.
    Sequence: W_layer0[0,0], W_layer1[0,0], ..., W_layerN[0,0], W_layer0[0,1], ...
    """
    # All tensors should have same shape
    shapes = [t.shape for t in tensors]
    if len(set(shapes)) > 1:
        return None  # different shapes, can't interleave

    n_layers = len(tensors)
    shape = shapes[0]

    # Stack and transpose: [n_layers, rows, cols] -> [rows, cols, n_layers] -> flatten
    stacked = torch.stack([t.contiguous().view(torch.int16) for t in tensors])  # [L, R, C]
    interleaved = stacked.permute(1, 2, 0).contiguous().flatten().numpy()  # [R, C, L]

    if len(interleaved) > max_n + 1:
        interleaved = interleaved[:max_n + 1]

    h_iid = iid_entropy(interleaved)
    h_bigram = bigram_entropy(interleaved, max_n)

    return h_iid, h_bigram


def ans16_global_size(tensors):
    """Standard global ANS-16bit."""
    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    W = all_w.numpy()
    n = len(W)

    vals, counts = np.unique(W, return_counts=True)
    probs = (counts / n).astype(np.float32)

    mapping = np.zeros(65536, dtype=np.int32)
    for i, v in enumerate(vals):
        mapping[int(v) + 32768] = i
    data_idx = mapping[(W.astype(np.int32) + 32768)].astype(np.int32)

    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(data_idx, model)
    compressed = encoder.get_compressed()
    return len(compressed) * 4 + len(vals) * 6


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

    print(f"\n{'='*100}")
    print("ROW REORDERING + QUANTIZED CONTEXT ANALYSIS")
    print(f"{'='*100}")

    # Test on a subset first for speed
    test_types = ["self_attn.q_proj", "mlp.gate_proj"]

    for wt in test_types:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)
        original = n * 2

        print(f"\n  {wt}  ({n:,} params)")

        # Baseline
        h_iid = sorted_rows_iid_entropy(tensors)
        print(f"    i.i.d. H:                     {h_iid:.6f} bpw  -> {h_iid*n/8/original*100:.3f}%")

        # Row-major bigram
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors]).numpy()
        h_bigram_row = bigram_entropy(all_w)
        print(f"    Row-major bigram H:           {h_bigram_row:.6f} bpw  -> {h_bigram_row*n/8/original*100:.3f}%")

        # Sorted-rows bigram
        t0 = time.time()
        h_bigram_sorted = sorted_rows_bigram_entropy(tensors)
        t1 = time.time()
        print(f"    Sorted-rows bigram H:         {h_bigram_sorted:.6f} bpw  -> {h_bigram_sorted*n/8/original*100:.3f}%  [{t1-t0:.1f}s]")

        # Cross-layer interleave
        t0 = time.time()
        result = cross_layer_interleave_entropy(tensors)
        t1 = time.time()
        if result:
            h_il_iid, h_il_bigram = result
            print(f"    Cross-layer interleave iid:   {h_il_iid:.6f} bpw")
            print(f"    Cross-layer interleave bigram:{h_il_bigram:.6f} bpw  -> {h_il_bigram*n/8/original*100:.3f}%  [{t1-t0:.1f}s]")

        # ANS-16bit baseline
        t0 = time.time()
        ans16 = ans16_global_size(tensors)
        t1 = time.time()
        print(f"    ANS-16bit (baseline):         {ans16/original*100:.3f}%  [{t1-t0:.1f}s]")

        # Quantized context ANS with various K (no row sorting)
        print(f"\n    Quantized context ANS (no sort):")
        for K in [8, 16, 32, 64, 128, 256]:
            t0 = time.time()
            qc_size, qc_overhead = quantized_context_ans_size(tensors, K, row_sort=False)
            t1 = time.time()
            delta = qc_size/original*100 - ans16/original*100
            marker = " ***" if delta < -0.01 else ""
            print(f"      K={K:<4}  {qc_size/original*100:.3f}%  delta={delta:+.3f}%  "
                  f"overhead={qc_overhead/1e3:.0f}KB{marker}  [{t1-t0:.1f}s]")

        # Quantized context ANS with row sorting
        print(f"\n    Quantized context ANS (sorted rows):")
        for K in [8, 16, 32, 64, 128]:
            t0 = time.time()
            qc_size, qc_overhead = quantized_context_ans_size(tensors, K, row_sort=True)
            t1 = time.time()
            delta = qc_size/original*100 - ans16/original*100
            marker = " ***" if delta < -0.01 else ""
            print(f"      K={K:<4}  {qc_size/original*100:.3f}%  delta={delta:+.3f}%  "
                  f"overhead={qc_overhead/1e3:.0f}KB{marker}  [{t1-t0:.1f}s]")


if __name__ == "__main__":
    main()
