"""
Test practical improvements over ANS-16bit.

Focus on approaches with actual compression gains:
1. Per-layer ANS-16bit (each layer has own probability table)
2. Layer-group ANS-16bit (cluster similar layers)
3. Sequential exponent + per-exp sm coding
4. Hybrid approaches combining the above

Goal: beat ANS-16bit's 65.96% on Qwen3-1.7B.
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


def ans_encode_int16(data):
    """ANS encode int16 data, return compressed bytes."""
    vals, counts = np.unique(data, return_counts=True)
    n = len(data)
    probs = (counts / n).astype(np.float32)

    mapping = np.zeros(65536, dtype=np.int32)
    for i, v in enumerate(vals):
        mapping[int(v) + 32768] = i
    data_idx = mapping[(data.astype(np.int32) + 32768)].astype(np.int32)

    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(data_idx, model)
    compressed = encoder.get_compressed()
    return len(compressed) * 4 + len(vals) * 6  # data + table overhead


def ans_encode_uint8_safe(data):
    """ANS encode uint8 data with safety for small/degenerate groups."""
    if len(data) == 0:
        return 0
    vals, counts = np.unique(data, return_counts=True)
    if len(vals) == 1:
        return 5  # just store the single value + count
    n = len(data)
    probs = (counts / n).astype(np.float64)
    probs = np.maximum(probs, 1e-10)
    probs = (probs / probs.sum()).astype(np.float32)
    # Ensure no zeros after float32 conversion
    probs = np.maximum(probs, np.float32(1e-10))
    probs = probs / probs.sum()

    mapping = np.zeros(256, dtype=np.int32)
    for i, v in enumerate(vals):
        mapping[v] = i
    data_idx = mapping[data].astype(np.int32)

    try:
        model = constriction.stream.model.Categorical(probs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(data_idx, model)
        compressed = encoder.get_compressed()
        return len(compressed) * 4 + len(vals) * 5
    except ValueError:
        # Fallback: estimate from entropy
        h = -np.sum(probs * np.log2(np.maximum(probs, 1e-30)))
        return int(np.ceil(h * n / 8)) + len(vals) * 5


def global_ans16_size(tensors):
    """Standard ANS-16bit with global probability table."""
    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    W = all_w.numpy()
    return ans_encode_int16(W)


def per_layer_ans16_size(tensors):
    """Per-layer ANS-16bit: each layer has its own probability table."""
    total = 0
    for t in tensors:
        W = t.contiguous().view(torch.int16).flatten().numpy()
        total += ans_encode_int16(W)
    return total


def per_layer_group_ans16_size(tensors, group_size=4):
    """Group layers into clusters of group_size, each cluster has shared table."""
    total = 0
    n_layers = len(tensors)
    for start in range(0, n_layers, group_size):
        end = min(start + group_size, n_layers)
        group = tensors[start:end]
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in group])
        W = all_w.numpy()
        total += ans_encode_int16(W)
    return total


def sequential_exp_per_exp_sm_size(tensors):
    """
    Sequential exponent coding + per-exponent sm coding.
    Process row by row: exp[i] coded with prev-exp context, sm coded with current-exp context.
    """
    # Collect sequential exponent pairs and sm by exponent
    all_exp_pairs = {}  # prev_exp -> list of curr_exp
    all_sm_by_exp = {}  # curr_exp -> list of sm
    first_exps = []

    for t in tensors:
        W = t.contiguous().view(torch.int16)
        exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
        sm = ((W >> 8) & 0x80 | (W & 0x7F)).to(torch.uint8).numpy()

        for row_idx in range(exp.shape[0]):
            exp_row = exp[row_idx]
            sm_row = sm[row_idx]

            first_exps.append(exp_row[0])

            for i in range(1, len(exp_row)):
                pe = int(exp_row[i-1])
                ce = int(exp_row[i])
                if pe not in all_exp_pairs:
                    all_exp_pairs[pe] = []
                all_exp_pairs[pe].append(ce)

            for i in range(len(exp_row)):
                e = int(exp_row[i])
                if e not in all_sm_by_exp:
                    all_sm_by_exp[e] = []
                all_sm_by_exp[e].append(int(sm_row[i]))

    total_bytes = 0

    # First exponents
    first_exps = np.array(first_exps, dtype=np.uint8)
    total_bytes += ans_encode_uint8_safe(first_exps)

    # Sequential exponents per context
    for prev_e, curr_list in sorted(all_exp_pairs.items()):
        curr = np.array(curr_list, dtype=np.uint8)
        total_bytes += ans_encode_uint8_safe(curr)

    # SM per exponent
    for e, sm_list in sorted(all_sm_by_exp.items()):
        sm_arr = np.array(sm_list, dtype=np.uint8)
        total_bytes += ans_encode_uint8_safe(sm_arr)

    return total_bytes


def per_layer_sequential_exp_per_exp_sm_size(tensors):
    """
    Per-layer version of sequential exp + per-exp sm.
    Each layer has its own set of context tables.
    """
    total = 0
    for t in tensors:
        total += sequential_exp_per_exp_sm_size([t])
    return total


def hybrid_per_layer_ans16_with_sharing(tensors, min_unique_for_own_table=100):
    """
    Hybrid: layers with enough unique values get their own table,
    rare layers share a common table.
    """
    # Analyze each layer
    layer_data = []
    for t in tensors:
        W = t.contiguous().view(torch.int16).flatten().numpy()
        vals = np.unique(W)
        layer_data.append((W, len(vals)))

    # Layers with many unique values get own table
    own_table_total = 0
    shared_layers = []

    for W, n_unique in layer_data:
        if n_unique >= min_unique_for_own_table:
            own_table_total += ans_encode_int16(W)
        else:
            shared_layers.append(W)

    # Shared table for remaining layers
    if shared_layers:
        all_shared = np.concatenate(shared_layers)
        own_table_total += ans_encode_int16(all_shared)

    return own_table_total


def iid_entropy_bytes(tensors):
    """Theoretical i.i.d. entropy lower bound in bytes."""
    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    W = all_w.numpy()
    n = len(W)
    vals, counts = np.unique(W, return_counts=True)
    p = counts / n
    h = -np.sum(p * np.log2(p))
    return int(np.ceil(h * n / 8))


def per_layer_entropy_bytes(tensors):
    """Per-layer entropy lower bound."""
    total = 0
    for t in tensors:
        W = t.contiguous().view(torch.int16).flatten().numpy()
        n = len(W)
        vals, counts = np.unique(W, return_counts=True)
        p = counts / n
        h = -np.sum(p * np.log2(p))
        total += int(np.ceil(h * n / 8))
    return total


def zstd_separated_size(tensors, level=19):
    """zstd on byte-separated (high/low bytes) data."""
    try:
        import zstandard as zstd
    except ImportError:
        return None

    all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
    W = all_w.numpy()

    high = ((W >> 8) & 0xFF).astype(np.uint8)
    low = (W & 0xFF).astype(np.uint8)

    separated = np.concatenate([high, low]).tobytes()
    cctx = zstd.ZstdCompressor(level=level)
    return len(cctx.compress(separated))


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
    method_totals = {}

    methods = [
        ("ANS-16bit (global)", "ans16", global_ans16_size),
        ("Per-layer ANS-16", "per_layer", per_layer_ans16_size),
        ("Per-2-layer ANS-16", "per2_layer", lambda t: per_layer_group_ans16_size(t, 2)),
        ("Per-4-layer ANS-16", "per4_layer", lambda t: per_layer_group_ans16_size(t, 4)),
        ("Per-7-layer ANS-16", "per7_layer", lambda t: per_layer_group_ans16_size(t, 7)),
        ("Seq-exp + per-exp(sm)", "seq_exp_sm", sequential_exp_per_exp_sm_size),
        ("Per-layer seq-exp+sm", "perlayer_seq", per_layer_sequential_exp_per_exp_sm_size),
    ]

    print(f"\n{'='*100}")
    print("PRACTICAL COMPRESSION COMPARISON")
    print(f"{'='*100}")

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)
        original = n * 2
        total_original += original

        print(f"\n  {wt}  ({n:,} params)")

        # Entropy bounds
        h_global = iid_entropy_bytes(tensors)
        h_perlayer = per_layer_entropy_bytes(tensors)
        print(f"    Entropy LB:  global={h_global/original*100:.3f}%  per-layer={h_perlayer/original*100:.3f}%")

        baseline = None
        for name, key, fn in methods:
            t0 = time.time()
            size = fn(tensors)
            t1 = time.time()

            method_totals.setdefault(key, 0)
            method_totals[key] += size

            ratio = size / original * 100
            if baseline is None:
                baseline = size
                print(f"    {name:<28} {ratio:.3f}%  ({size/1e6:.1f}MB)  [{t1-t0:.1f}s]")
            else:
                delta = ratio - baseline / original * 100
                marker = " ***" if delta < -0.01 else ""
                print(f"    {name:<28} {ratio:.3f}%  delta={delta:+.3f}%{marker}  [{t1-t0:.1f}s]")

    # Summary
    print(f"\n{'='*100}")
    print("OVERALL SUMMARY")
    print(f"{'='*100}")

    baseline_total = method_totals.get("ans16", 0)
    baseline_ratio = baseline_total / total_original * 100

    for name, key, _ in methods:
        if key in method_totals:
            size = method_totals[key]
            ratio = size / total_original * 100
            delta = ratio - baseline_ratio
            savings = (baseline_total - size) / 1e6
            marker = " ***" if delta < -0.01 else ""
            print(f"  {name:<28} {ratio:.3f}%  delta={delta:+.3f}%  ({savings:+.1f}MB){marker}")

    print(f"\n  Original bf16:               {total_original/1e6:.1f}MB")
    print(f"  ANS-16bit baseline:          {baseline_ratio:.3f}%  ({baseline_total/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
