"""
Test single-stream context-conditioned ANS for BFloat16 weight compression.

Previous attempts at context-dependent coding used SEPARATE ANS streams per context,
causing bitstream fragmentation overhead. This test uses a SINGLE ANS stream where
each symbol is coded with a context-dependent probability table.

Context = exponent of previous value (~30 unique values).
Table overhead: 30 contexts × ~6000 symbols × 6 bytes ≈ 1.1 MB.
No bitstream fragmentation — just the entropy + table storage.

Expected: H(W[i] | exp[i-1]) should be measurably lower than H(W[i]).
"""

import torch
import numpy as np
import math
from collections import Counter, defaultdict
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


def analyze_context_entropy(model, weight_suffix):
    """Measure H(W[i] | context) for various context definitions."""
    weights = []
    for name, param in model.named_parameters():
        if name.endswith(weight_suffix):
            weights.append(param.data.cpu().to(torch.bfloat16))

    if not weights:
        return None

    print(f"\n{'='*70}")
    print(f"Weight type: {weight_suffix} ({len(weights)} layers)")
    print(f"{'='*70}")

    # Collect all values and contexts across layers
    total_n = sum(w.numel() for w in weights)

    # Global distribution
    all_vals = []
    for w in weights:
        all_vals.append(w.view(torch.int16).flatten().numpy().astype(np.int32))
    all_vals = np.concatenate(all_vals)
    global_counts = Counter(all_vals.tolist())
    h_global = entropy_from_counter(global_counts, total_n)
    n_unique = len(global_counts)

    # Context: prev exponent
    # BF16: bit 15 = sign, bits 14-7 = exponent, bits 6-0 = mantissa
    exponents = (all_vals >> 7) & 0xFF

    # Conditional entropy H(W[i] | exp[i-1]) across all layers
    # Build per-context distributions
    context_dists = defaultdict(Counter)
    context_counts = Counter()
    for w in weights:
        flat = w.view(torch.int16).flatten().numpy().astype(np.int32)
        exps = (flat >> 7) & 0xFF
        for i in range(1, len(flat)):
            ctx = int(exps[i-1])
            context_dists[ctx][int(flat[i])] += 1
            context_counts[ctx] += 1

    total_cond = sum(context_counts.values())
    h_cond = 0.0
    for ctx, dist in context_dists.items():
        n_ctx = context_counts[ctx]
        h_ctx = entropy_from_counter(dist, n_ctx)
        h_cond += h_ctx * n_ctx / total_cond

    n_contexts = len(context_dists)

    # Context: prev full exponent+sign (more specific)
    context_dists2 = defaultdict(Counter)
    context_counts2 = Counter()
    for w in weights:
        flat = w.view(torch.int16).flatten().numpy().astype(np.int32)
        hi_bytes = (flat >> 8) & 0xFF  # sign + top 7 bits of exponent
        for i in range(1, len(flat)):
            ctx = int(hi_bytes[i-1])
            context_dists2[ctx][int(flat[i])] += 1
            context_counts2[ctx] += 1

    total_cond2 = sum(context_counts2.values())
    h_cond2 = 0.0
    for ctx, dist in context_dists2.items():
        n_ctx = context_counts2[ctx]
        h_ctx = entropy_from_counter(dist, n_ctx)
        h_cond2 += h_ctx * n_ctx / total_cond2

    n_contexts2 = len(context_dists2)

    # Context: current exponent (not previous — conditioning on own exponent)
    context_dists3 = defaultdict(Counter)
    context_counts3 = Counter()
    for w in weights:
        flat = w.view(torch.int16).flatten().numpy().astype(np.int32)
        exps = (flat >> 7) & 0xFF
        for i in range(len(flat)):
            ctx = int(exps[i])
            context_dists3[ctx][int(flat[i])] += 1
            context_counts3[ctx] += 1

    total_cond3 = sum(context_counts3.values())
    h_cond3 = 0.0
    for ctx, dist in context_dists3.items():
        n_ctx = context_counts3[ctx]
        h_ctx = entropy_from_counter(dist, n_ctx)
        h_cond3 += h_ctx * n_ctx / total_cond3

    n_contexts3 = len(context_dists3)

    print(f"  Values: {total_n}, unique: {n_unique}")
    print(f"\n  Entropy analysis:")
    print(f"    H(W) global:           {h_global:.4f} bpw → {h_global/16*100:.2f}%")
    print(f"    H(W|own_exp):          {h_cond3:.4f} bpw → {h_cond3/16*100:.2f}% ({n_contexts3} contexts)")
    print(f"    H(W|prev_exp):         {h_cond:.4f} bpw → {h_cond/16*100:.2f}% ({n_contexts} contexts)")
    print(f"    H(W|prev_hi_byte):     {h_cond2:.4f} bpw → {h_cond2/16*100:.2f}% ({n_contexts2} contexts)")
    print(f"\n  Savings vs global:")
    print(f"    own_exp context:   {h_global - h_cond3:.4f} bpw = {(h_global-h_cond3)/h_global*100:.2f}%")
    print(f"    prev_exp context:  {h_global - h_cond:.4f} bpw = {(h_global-h_cond)/h_global*100:.2f}%")
    print(f"    prev_hi context:   {h_global - h_cond2:.4f} bpw = {(h_global-h_cond2)/h_global*100:.2f}%")

    # Table overhead analysis
    for name, n_ctx, h_c in [
        ("prev_exp", n_contexts, h_cond),
        ("prev_hi", n_contexts2, h_cond2),
        ("own_exp", n_contexts3, h_cond3),
    ]:
        table_bytes = n_ctx * n_unique * 6  # symbol (2B) + prob (4B) per entry
        data_bytes = total_n * 2
        savings_bytes = (h_global - h_c) / 8 * total_n
        net_bytes = savings_bytes - table_bytes
        print(f"    {name:12s}: table={table_bytes/1024:.0f} KB, savings={savings_bytes/1024:.0f} KB, net={net_bytes/1024:.0f} KB")

    return {
        'h_global': h_global,
        'h_cond_prev_exp': h_cond,
        'h_cond_prev_hi': h_cond2,
        'h_cond_own_exp': h_cond3,
        'n_unique': n_unique,
        'n_values': total_n,
        'context_dists': context_dists,  # for ANS testing
        'context_counts': context_counts,
    }


def test_actual_ans(model, weight_suffix):
    """Actually compress one layer with context-conditioned ANS vs standard ANS."""
    if not HAS_CONSTRICTION:
        print("  [constriction not available, skipping actual ANS test]")
        return

    weights = []
    for name, param in model.named_parameters():
        if name.endswith(weight_suffix):
            weights.append(param.data.cpu().to(torch.bfloat16))
            break

    if not weights:
        return

    flat = weights[0].view(torch.int16).flatten().numpy().astype(np.int32)
    n = len(flat)
    exps = (flat >> 7) & 0xFF

    print(f"\n--- Actual ANS test: {weight_suffix} layer 0 ({n} values) ---")

    # Standard ANS-16bit
    counts = Counter(flat.tolist())
    symbols = sorted(counts.keys())
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    probs = np.array([counts[s] / n for s in symbols], dtype=np.float64)
    probs = (probs / probs.sum()).astype(np.float32)
    mapped = np.array([sym_to_idx[v] for v in flat], dtype=np.int32)

    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(mapped, constriction.stream.model.Categorical(probs, perfect=False))
    std_compressed = encoder.get_compressed()
    std_bytes = len(std_compressed) * 4
    std_overhead = len(symbols) * 6
    std_total = std_bytes + std_overhead

    print(f"  Standard ANS-16: {std_total} bytes ({std_total/(n*2)*100:.2f}%)")

    # Context-conditioned ANS: prev_exp context
    # Build per-context probability tables
    context_tables = {}
    unique_exps = sorted(set(exps.tolist()))

    for exp_ctx in unique_exps:
        # Values that follow this exponent
        mask = np.zeros(n, dtype=bool)
        mask[1:] = (exps[:-1] == exp_ctx)
        vals_after = flat[mask]
        if len(vals_after) == 0:
            continue
        ctx_counts = Counter(vals_after.tolist())
        # Use the global symbol table but with context-specific probabilities
        ctx_probs = np.zeros(len(symbols), dtype=np.float64)
        for s, idx in sym_to_idx.items():
            ctx_probs[idx] = ctx_counts.get(s, 0)
        # Smoothing: add 1 to all (Laplace smoothing) to avoid zero probabilities
        ctx_probs += 1
        ctx_probs = (ctx_probs / ctx_probs.sum()).astype(np.float32)
        context_tables[exp_ctx] = ctx_probs

    # Default table for first value and rare contexts
    default_probs = probs.copy()

    # Encode: first value with global table, rest with context table
    # Create per-position probability array
    all_probs = np.zeros((n, len(symbols)), dtype=np.float32)
    all_probs[0] = default_probs
    for i in range(1, n):
        ctx = int(exps[i-1])
        if ctx in context_tables:
            all_probs[i] = context_tables[ctx]
        else:
            all_probs[i] = default_probs

    # Encode with per-position models
    # constriction doesn't support per-position Categorical directly in batch
    # We need to encode one at a time
    encoder2 = constriction.stream.stack.AnsCoder()
    for i in range(n - 1, -1, -1):
        model = constriction.stream.model.Categorical(all_probs[i], perfect=False)
        encoder2.encode_reverse(np.array([mapped[i]], dtype=np.int32), model)
    ctx_compressed = encoder2.get_compressed()
    ctx_bytes = len(ctx_compressed) * 4

    # Overhead: per-context probability tables
    ctx_overhead = len(context_tables) * len(symbols) * 4 + len(symbols) * 2  # probs + symbol table
    ctx_total = ctx_bytes + ctx_overhead

    print(f"  Context ANS:     {ctx_total} bytes ({ctx_total/(n*2)*100:.2f}%)")
    print(f"    Data: {ctx_bytes} bytes, Tables: {ctx_overhead/1024:.0f} KB")
    print(f"  Improvement: {std_total - ctx_total} bytes ({(std_total - ctx_total)/(n*2)*100:.3f}%)")

    # Verify lossless decode
    decoder = constriction.stream.stack.AnsCoder(ctx_compressed)
    decoded = np.zeros(n, dtype=np.int32)
    for i in range(n):
        model = constriction.stream.model.Categorical(all_probs[i], perfect=False)
        decoded[i] = decoder.decode(model, 1)[0]

    if np.array_equal(mapped, decoded):
        print(f"  Decode verified: LOSSLESS ✓")
    else:
        mismatches = np.sum(mapped != decoded)
        print(f"  Decode FAILED: {mismatches} mismatches ✗")


def main():
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

    weight_types = [
        "self_attn.q_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
    ]

    for wt in weight_types:
        analyze_context_entropy(model, wt)

    # Actual ANS test on one weight type
    test_actual_ans(model, "self_attn.q_proj.weight")

    print("\nDone.")


if __name__ == '__main__':
    main()
