"""
Test sequential/positional entropy of BFloat16 weights.

If H(W[i] | W[i-1]) < H(W[i]), then Markov/sequential coding can beat i.i.d. ANS.
Also tests: per-row entropy, positional entropy, and general compressors on raw bytes.
"""

import time
from argparse import ArgumentParser
from collections import defaultdict

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
    """H(W) in bits per symbol."""
    vals, counts = np.unique(W, return_counts=True)
    p = counts / len(W)
    return -np.sum(p * np.log2(p))


def bigram_conditional_entropy(W, max_pairs=5_000_000):
    """Estimate H(W[i] | W[i-1]) using bigram counts.

    Uses a subsample if the sequence is too long to avoid memory issues.
    """
    n = len(W)
    if n < 2:
        return iid_entropy(W)

    # Subsample if too large
    if n > max_pairs + 1:
        # Take a contiguous block
        W = W[:max_pairs + 1]
        n = len(W)

    # Shift to unsigned for indexing
    W_u = (W.astype(np.int32) + 32768).astype(np.uint16)

    # Count bigrams using a dict (sparse)
    prev = W_u[:-1]
    curr = W_u[1:]

    # Use numpy to count: group by prev value
    sort_idx = np.argsort(prev)
    prev_sorted = prev[sort_idx]
    curr_sorted = curr[sort_idx]

    # Find boundaries where prev value changes
    boundaries = np.where(np.diff(prev_sorted) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(prev_sorted)]])

    total_cond_entropy = 0.0
    total_count = len(prev)

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        group_size = end - start
        group = curr_sorted[start:end]

        # Entropy of this conditional distribution
        vals, counts = np.unique(group, return_counts=True)
        p = counts / group_size
        h = -np.sum(p * np.log2(p))

        total_cond_entropy += (group_size / total_count) * h

    return total_cond_entropy


def per_row_entropy(tensors):
    """Average per-row entropy (each row has its own distribution)."""
    total_h = 0
    total_n = 0
    for t in tensors:
        W = t.contiguous().view(torch.int16).numpy()
        for row_idx in range(W.shape[0]):
            row = W[row_idx]
            n = len(row)
            vals, counts = np.unique(row, return_counts=True)
            p = counts / n
            h = -np.sum(p * np.log2(p))
            total_h += h * n
            total_n += n
    return total_h / total_n


def per_row_ans_size(tensors):
    """Estimate compressed size with per-row ANS coding."""
    total_bits = 0
    total_overhead = 0
    for t in tensors:
        W = t.contiguous().view(torch.int16).numpy()
        for row_idx in range(W.shape[0]):
            row = W[row_idx]
            n = len(row)
            vals, counts = np.unique(row, return_counts=True)
            p = counts / n
            h = -np.sum(p * np.log2(p))
            total_bits += h * n
            # Overhead: symbol table (2 bytes per unique) + probs (4 bytes per unique)
            total_overhead += len(vals) * 6
    return (total_bits / 8) + total_overhead


def per_layer_entropy(tensors):
    """Average per-layer entropy."""
    total_h = 0
    total_n = 0
    for t in tensors:
        W = t.contiguous().view(torch.int16).flatten().numpy()
        n = len(W)
        h = iid_entropy(W)
        total_h += h * n
        total_n += n
    return total_h / total_n


def byte_level_entropy(tensors):
    """Entropy of high byte and low byte separately."""
    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    W = all_w.numpy()

    high = ((W >> 8) & 0xFF).astype(np.uint8)
    low = (W & 0xFF).astype(np.uint8)

    h_high = iid_entropy(high)
    h_low = iid_entropy(low)

    return h_high, h_low


def test_zstd_compression(tensors):
    """Test zstd compression on raw bytes."""
    try:
        import zstandard as zstd
    except ImportError:
        return None

    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    raw_bytes = all_w.numpy().tobytes()

    results = {}
    for level in [1, 3, 10, 22]:
        cctx = zstd.ZstdCompressor(level=level)
        compressed = cctx.compress(raw_bytes)
        results[f"zstd-{level}"] = len(compressed)

    return results


def test_interleaved_bytes(tensors):
    """Test if interleaving high/low bytes improves zstd compression."""
    try:
        import zstandard as zstd
    except ImportError:
        return None

    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    W = all_w.numpy()

    high = ((W >> 8) & 0xFF).astype(np.uint8)
    low = (W & 0xFF).astype(np.uint8)

    # Separated: all high bytes then all low bytes
    separated = np.concatenate([high, low]).tobytes()

    # Original interleaved
    original = W.tobytes()

    cctx = zstd.ZstdCompressor(level=10)

    return {
        "interleaved": len(cctx.compress(original)),
        "separated": len(cctx.compress(separated)),
        "high_only": len(cctx.compress(high.tobytes())),
        "low_only": len(cctx.compress(low.tobytes())),
    }


def test_row_delta_coding(tensors):
    """Test delta coding within rows (adjacent elements)."""
    all_deltas = []
    all_firsts = []
    for t in tensors:
        W = t.contiguous().view(torch.int16).numpy()
        for row_idx in range(W.shape[0]):
            row = W[row_idx].astype(np.int32)
            first = row[0]
            delta = np.diff(row).astype(np.int16)
            all_firsts.append(first)
            all_deltas.append(delta)

    deltas = np.concatenate(all_deltas)
    h_delta = iid_entropy(deltas)

    # First elements
    firsts = np.array(all_firsts, dtype=np.int16)
    h_first = iid_entropy(firsts)

    # Total bits needed
    n_deltas = len(deltas)
    n_firsts = len(firsts)
    total_bits = h_delta * n_deltas + h_first * n_firsts
    total_elements = n_deltas + n_firsts

    return h_delta, h_first, total_bits / (total_elements * 16) * 100


def test_xor_coding(tensors):
    """Test XOR with previous element (captures patterns differently from delta)."""
    all_xors = []
    for t in tensors:
        W = t.contiguous().view(torch.int16).flatten().numpy()
        xored = np.bitwise_xor(W[1:], W[:-1])
        all_xors.append(xored)

    xors = np.concatenate(all_xors)
    return iid_entropy(xors)


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

    print(f"\n{'='*90}")
    print("SEQUENTIAL / CONTEXTUAL ENTROPY ANALYSIS")
    print(f"{'='*90}")

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
        W = all_w.numpy()
        n = len(W)
        original = n * 2

        print(f"\n  {wt}  ({n:,} params)")

        # 1. i.i.d. entropy
        h_iid = iid_entropy(W)
        iid_bytes = int(np.ceil(h_iid * n / 8))
        print(f"    H(W) i.i.d.:          {h_iid:.6f} bpw  -> {iid_bytes/original*100:.3f}%")

        # 2. Bigram conditional entropy
        t0 = time.time()
        h_bigram = bigram_conditional_entropy(W)
        t1 = time.time()
        bigram_bytes = int(np.ceil(h_bigram * n / 8))
        savings = (h_iid - h_bigram) * n / 8
        print(f"    H(W[i]|W[i-1]):       {h_bigram:.6f} bpw  -> {bigram_bytes/original*100:.3f}%  "
              f"(saves {savings/1e3:.1f}KB)  [{t1-t0:.1f}s]")

        # 3. Per-layer entropy
        h_layer = per_layer_entropy(tensors)
        layer_bytes = int(np.ceil(h_layer * n / 8))
        print(f"    H(W|layer):           {h_layer:.6f} bpw  -> {layer_bytes/original*100:.3f}%")

        # 4. Per-row entropy
        t0 = time.time()
        h_row = per_row_entropy(tensors)
        t1 = time.time()
        row_data_bytes = int(np.ceil(h_row * n / 8))
        row_overhead = per_row_ans_size(tensors) - row_data_bytes
        row_total = row_data_bytes + row_overhead
        print(f"    H(W|row):             {h_row:.6f} bpw  -> data={row_data_bytes/original*100:.3f}%  "
              f"+ overhead={row_overhead/1e6:.1f}MB  total={row_total/original*100:.2f}%  [{t1-t0:.1f}s]")

        # 5. Delta coding within rows
        h_delta, h_first, delta_ratio = test_row_delta_coding(tensors)
        print(f"    Row-delta H:          {h_delta:.6f} bpw  -> {delta_ratio:.3f}%")

        # 6. XOR coding
        h_xor = test_xor_coding(tensors)
        print(f"    XOR H(W[i]^W[i-1]):   {h_xor:.6f} bpw")

        # 7. Byte-level entropy
        h_high, h_low = byte_level_entropy(tensors)
        print(f"    Byte entropy:  high={h_high:.4f}  low={h_low:.4f}  sum={h_high+h_low:.4f}")

        # 8. zstd compression
        zstd_results = test_zstd_compression(tensors)
        if zstd_results:
            for name, size in zstd_results.items():
                ratio = size / original * 100
                print(f"    {name:<22} {ratio:.3f}%  ({size/1e6:.2f}MB)")

        # 9. Byte-separated zstd
        interleaved = test_interleaved_bytes(tensors)
        if interleaved:
            for name, size in interleaved.items():
                ratio = size / original * 100
                print(f"    zstd-10 {name:<14} {ratio:.3f}%  ({size/1e6:.2f}MB)")

    # Overall summary
    print(f"\n{'='*90}")
    print("OVERALL SUMMARY - ENTROPY BOUNDS")
    print(f"{'='*90}")

    total_n = 0
    total_iid_bits = 0
    total_bigram_bits = 0
    total_layer_bits = 0
    total_row_bits = 0
    total_original = 0

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
        W = all_w.numpy()
        n = len(W)
        total_n += n
        total_original += n * 2

        h_iid = iid_entropy(W)
        total_iid_bits += h_iid * n

        h_bigram = bigram_conditional_entropy(W)
        total_bigram_bits += h_bigram * n

        h_layer = per_layer_entropy(tensors)
        total_layer_bits += h_layer * n

    for name, bits in [("i.i.d. H(W)", total_iid_bits),
                        ("Bigram H(W[i]|W[i-1])", total_bigram_bits),
                        ("Per-layer H(W|layer)", total_layer_bits)]:
        nbytes = int(np.ceil(bits / 8))
        ratio = nbytes / total_original * 100
        print(f"  {name:<28} {ratio:.3f}%  ({nbytes/1e6:.1f}MB)")

    print(f"  Original bf16:               {total_original/1e6:.1f}MB")
    print(f"  Current ANS-16bit:           ~65.96%")


if __name__ == "__main__":
    main()
