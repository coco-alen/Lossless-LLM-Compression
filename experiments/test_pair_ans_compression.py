"""
Test pair ANS coding: code consecutive BF16 value pairs as single symbols.

H(W[i]) = 10.55 bpw (i.i.d.)
H(W[i]|W[i-1]) = 9.91 bpw (conditional)
H(pair) = H(W[i]) + H(W[i+1]|W[i]) = 20.46 bits per pair = 10.23 bpw per value

If unique pairs ~ 500K, ANS pair coding should reach ~10.23 bpw = 64%.
Key question: How many unique pairs? What's the overhead?
"""

import torch
import numpy as np
import math
from collections import Counter
from transformers import AutoModelForCausalLM

try:
    import constriction
    HAS_CONSTRICTION = True
except ImportError:
    HAS_CONSTRICTION = False


def entropy_from_counter(counts, total):
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def analyze_pairs(model, weight_suffix):
    """Analyze pair statistics for a weight type."""
    weights = []
    for name, param in model.named_parameters():
        if name.endswith(weight_suffix):
            weights.append(param.data.cpu().to(torch.bfloat16))

    if not weights:
        return None

    print(f"\n{'='*70}")
    print(f"Weight type: {weight_suffix} ({len(weights)} layers)")
    print(f"{'='*70}")

    # Collect stats across all layers
    all_pair_counts = Counter()
    all_single_counts = Counter()
    total_values = 0
    total_pairs = 0

    for w in weights:
        flat = w.view(torch.int16).flatten().numpy().astype(np.int32)
        n = len(flat)
        total_values += n

        # Single value stats
        all_single_counts.update(flat.tolist())

        # Pair stats: (value[i], value[i+1]) coded as a single 32-bit key
        # Truncate to even length
        n_pairs = n // 2
        total_pairs += n_pairs
        pairs = flat[:n_pairs*2].reshape(-1, 2)
        pair_keys = (pairs[:, 0].astype(np.int64) << 16) | (pairs[:, 1].astype(np.int64) & 0xFFFF)
        all_pair_counts.update(pair_keys.tolist())

    # Also compute sequential pair stats (overlapping)
    seq_pair_counts = Counter()
    total_seq_pairs = 0
    for w in weights:
        flat = w.view(torch.int16).flatten().numpy().astype(np.int32)
        n = len(flat)
        total_seq_pairs += n - 1
        seq_keys = (flat[:-1].astype(np.int64) << 16) | (flat[1:].astype(np.int64) & 0xFFFF)
        seq_pair_counts.update(seq_keys.tolist())

    h_single = entropy_from_counter(all_single_counts, total_values)
    h_pair = entropy_from_counter(all_pair_counts, total_pairs)
    h_seq_pair = entropy_from_counter(seq_pair_counts, total_seq_pairs)

    n_unique_single = len(all_single_counts)
    n_unique_pair = len(all_pair_counts)
    n_unique_seq = len(seq_pair_counts)

    # Table overhead: need to store symbol_table (4 bytes per pair) + probabilities (4 bytes)
    pair_table_bytes = n_unique_pair * 8  # 4 bytes key + 4 bytes prob
    seq_table_bytes = n_unique_seq * 8

    # Compressed sizes
    iid_bits = h_single * total_values
    pair_bits = h_pair * total_pairs  # covers total_pairs*2 values
    pair_bpv = h_pair / 2  # bits per value
    seq_pair_bits = h_seq_pair * total_seq_pairs
    seq_pair_bpv = h_seq_pair / 2

    print(f"  Total values: {total_values} ({total_values*2/1024/1024:.1f} MB)")
    print(f"  Unique single values: {n_unique_single}")
    print(f"  Unique non-overlapping pairs: {n_unique_pair}")
    print(f"  Unique sequential (overlapping) pairs: {n_unique_seq}")
    print(f"\n  Entropy:")
    print(f"    Single i.i.d.: {h_single:.4f} bpw → {h_single/16*100:.2f}%")
    print(f"    Pair (non-overlap): {h_pair:.4f} bits/pair = {pair_bpv:.4f} bpw → {pair_bpv/16*100:.2f}%")
    print(f"    Pair (sequential): {h_seq_pair:.4f} bits/pair = {seq_pair_bpv:.4f} bpw → {seq_pair_bpv/16*100:.2f}%")
    print(f"\n  Table overhead:")
    print(f"    Pair table: {pair_table_bytes/1024:.1f} KB ({n_unique_pair} entries)")
    print(f"    Seq table:  {seq_table_bytes/1024:.1f} KB ({n_unique_seq} entries)")
    print(f"\n  Estimated compressed sizes (entropy + overhead):")
    data_bytes = total_values * 2
    iid_compressed = iid_bits / 8 + n_unique_single * 6
    pair_compressed = pair_bits / 8 + pair_table_bytes
    seq_compressed = seq_pair_bits / 8 + seq_table_bytes
    print(f"    i.i.d. ANS: {iid_compressed/1024/1024:.2f} MB ({iid_compressed/data_bytes*100:.2f}%)")
    print(f"    Pair ANS:   {pair_compressed/1024/1024:.2f} MB ({pair_compressed/data_bytes*100:.2f}%)")
    print(f"    Seq pair:   {seq_compressed/1024/1024:.2f} MB ({seq_compressed/data_bytes*100:.2f}%)")
    print(f"\n  Savings vs i.i.d.:")
    print(f"    Pair: {(iid_compressed - pair_compressed)/1024/1024:.2f} MB ({(1-pair_compressed/iid_compressed)*100:.2f}%)")
    print(f"    Seq:  {(iid_compressed - seq_compressed)/1024/1024:.2f} MB ({(1-seq_compressed/iid_compressed)*100:.2f}%)")

    # Actual ANS test on first layer (single values vs pairs)
    if HAS_CONSTRICTION and len(weights) > 0:
        w0 = weights[0]
        flat = w0.view(torch.int16).flatten().numpy().astype(np.int32)
        n = len(flat)

        # i.i.d. ANS
        counts = Counter(flat.tolist())
        symbols = sorted(counts.keys())
        sym_to_idx = {s: i for i, s in enumerate(symbols)}
        probs = np.array([counts[s] / n for s in symbols], dtype=np.float32)
        mapped = np.array([sym_to_idx[v] for v in flat], dtype=np.int32)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(mapped, constriction.stream.model.Categorical(probs, perfect=False))
        iid_bytes = len(encoder.get_compressed()) * 4
        iid_overhead = len(symbols) * 6
        iid_total = iid_bytes + iid_overhead

        # Pair ANS (non-overlapping)
        n_pairs = n // 2
        pairs = flat[:n_pairs*2].reshape(-1, 2)
        pair_keys = (pairs[:, 0].astype(np.int64) << 16) | (pairs[:, 1].astype(np.int64) & 0xFFFF)
        counts_p = Counter(pair_keys.tolist())
        symbols_p = sorted(counts_p.keys())
        sym_to_idx_p = {s: i for i, s in enumerate(symbols_p)}
        probs_p = np.array([counts_p[s] / n_pairs for s in symbols_p], dtype=np.float32)
        mapped_p = np.array([sym_to_idx_p[v] for v in pair_keys], dtype=np.int32)
        encoder2 = constriction.stream.stack.AnsCoder()
        encoder2.encode_reverse(mapped_p, constriction.stream.model.Categorical(probs_p, perfect=False))
        pair_bytes = len(encoder2.get_compressed()) * 4
        # If n is odd, we need to code the last value separately
        pair_overhead = len(symbols_p) * 8  # 4 bytes key + 4 bytes prob
        pair_total = pair_bytes + pair_overhead

        print(f"\n  Actual ANS (layer 0, {n} values):")
        print(f"    i.i.d.:  {iid_total} bytes ({iid_total/(n*2)*100:.2f}%)")
        print(f"    Pair:    {pair_total} bytes ({pair_total/(n*2)*100:.2f}%)")
        print(f"    Delta:   {pair_total - iid_total} bytes ({(pair_total - iid_total)/(n*2)*100:.3f}%)")

    return {
        'h_single': h_single,
        'h_pair_bpv': pair_bpv,
        'h_seq_bpv': seq_pair_bpv,
        'n_unique_single': n_unique_single,
        'n_unique_pair': n_unique_pair,
        'n_unique_seq': n_unique_seq,
        'n_values': total_values,
    }


def analyze_triplets(model, weight_suffix):
    """Quick check: do triplets help more?"""
    weights = []
    for name, param in model.named_parameters():
        if name.endswith(weight_suffix):
            weights.append(param.data.cpu().to(torch.bfloat16))
            break  # First layer only

    if not weights:
        return

    flat = weights[0].view(torch.int16).flatten().numpy().astype(np.int32)
    n = len(flat)

    print(f"\n--- Triplet analysis for {weight_suffix} (layer 0, {n} values) ---")

    for k in [2, 3, 4]:
        n_tuples = n // k
        tuples = flat[:n_tuples*k].reshape(-1, k)
        # Encode each tuple as a single key
        keys = np.zeros(n_tuples, dtype=np.int64)
        for j in range(k):
            keys |= (tuples[:, j].astype(np.int64) & 0xFFFF) << (16 * j)
        counts = Counter(keys.tolist())
        h = entropy_from_counter(counts, n_tuples)
        h_bpv = h / k
        overhead = len(counts) * (2 * k + 4)  # key + prob per entry
        print(f"  k={k}: {len(counts)} unique {k}-tuples, "
              f"H={h:.4f} bits/{k}-tuple = {h_bpv:.4f} bpw → {h_bpv/16*100:.2f}%, "
              f"overhead={overhead/1024:.0f} KB")


def main():
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    weight_types = [
        "self_attn.q_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
    ]

    results = {}
    for wt in weight_types:
        results[wt] = analyze_pairs(model, wt)

    # Triplet/tuple analysis
    for wt in weight_types[:1]:
        analyze_triplets(model, wt)

    # Overall summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total_n = sum(r['n_values'] for r in results.values())
    avg_single = sum(r['h_single'] * r['n_values'] for r in results.values()) / total_n
    avg_pair = sum(r['h_pair_bpv'] * r['n_values'] for r in results.values()) / total_n
    avg_seq = sum(r['h_seq_bpv'] * r['n_values'] for r in results.values()) / total_n

    print(f"  i.i.d.:     {avg_single:.4f} bpw → {avg_single/16*100:.2f}%")
    print(f"  Pair:       {avg_pair:.4f} bpw → {avg_pair/16*100:.2f}%")
    print(f"  Seq pair:   {avg_seq:.4f} bpw → {avg_seq/16*100:.2f}%")
    print(f"  Pair saves: {avg_single - avg_pair:.4f} bpw = {(avg_single - avg_pair)/avg_single*100:.2f}% relative")


if __name__ == '__main__':
    main()
