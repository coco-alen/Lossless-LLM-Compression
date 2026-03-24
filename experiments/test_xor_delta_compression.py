"""
Test XOR delta coding for BFloat16 weight compression.

Key idea: XOR adjacent int16 values. If adjacent weights are numerically
similar (same exponent, close mantissa), XOR produces values with many
leading zeros → concentrated distribution → low entropy → compressible.

This captures BOTH exponent AND mantissa correlation in a single transform
without needing explicit bigram/context tables.
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


def entropy(counts_dict, total):
    """Shannon entropy in bits."""
    h = 0.0
    for c in counts_dict.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def ans_compressed_size(values_np, symbol_table, probabilities):
    """Actual ANS compressed size in bytes using constriction."""
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(values_np, constriction.stream.model.Categorical(probabilities))
    compressed = encoder.get_compressed()
    return len(compressed) * 4  # uint32 words → bytes


def analyze_weight_type(model, weight_suffix, model_name):
    """Analyze XOR delta coding for a specific weight type across all layers."""
    # Collect all weights of this type
    weights = []
    for name, param in model.named_parameters():
        if name.endswith(weight_suffix):
            weights.append(param.data.cpu())

    if not weights:
        return None

    print(f"\n{'='*70}")
    print(f"Weight type: {weight_suffix} ({len(weights)} layers)")
    print(f"{'='*70}")

    total_original = 0
    total_iid_bits = 0
    total_xor_bits = 0
    total_xor_hi_bits = 0
    total_xor_lo_bits = 0
    total_delta_bits = 0

    # Also track ANS compressed sizes
    total_iid_ans = 0
    total_xor_ans = 0

    for li, w in enumerate(weights):
        flat = w.flatten().to(torch.bfloat16)
        int16 = flat.view(torch.int16).numpy().astype(np.int32)
        n = len(int16)
        total_original += n * 2  # 2 bytes per bf16

        # 1. i.i.d. entropy (baseline)
        counts_iid = Counter(int16.tolist())
        h_iid = entropy(counts_iid, n)
        total_iid_bits += h_iid * n

        # 2. XOR delta: xor[i] = int16[i] XOR int16[i-1]
        xor_vals = np.bitwise_xor(int16[1:], int16[:-1])
        counts_xor = Counter(xor_vals.tolist())
        h_xor = entropy(counts_xor, n - 1)
        # First value coded with global table
        total_xor_bits += h_iid / len(counts_iid) + h_xor * (n - 1)  # rough

        # 3. XOR byte-separated
        xor_hi = (xor_vals >> 8) & 0xFF
        xor_lo = xor_vals & 0xFF
        h_xor_hi = entropy(Counter(xor_hi.tolist()), n - 1)
        h_xor_lo = entropy(Counter(xor_lo.tolist()), n - 1)
        total_xor_hi_bits += h_xor_hi * (n - 1)
        total_xor_lo_bits += h_xor_lo * (n - 1)

        # 4. Arithmetic delta: delta[i] = int16[i] - int16[i-1] (mod 65536)
        delta_vals = (int16[1:] - int16[:-1]) & 0xFFFF
        counts_delta = Counter(delta_vals.tolist())
        h_delta = entropy(counts_delta, n - 1)
        total_delta_bits += h_delta * (n - 1)

        # 5. Actual ANS compressed sizes (first layer only for speed)
        if li == 0 and HAS_CONSTRICTION:
            # i.i.d. ANS
            symbols_iid = sorted(counts_iid.keys())
            sym_to_idx = {s: i for i, s in enumerate(symbols_iid)}
            probs_iid = np.array([counts_iid[s] / n for s in symbols_iid], dtype=np.float32)
            mapped_iid = np.array([sym_to_idx[v] for v in int16], dtype=np.int32)
            iid_bytes = ans_compressed_size(mapped_iid, symbols_iid, probs_iid)
            overhead_iid = len(symbols_iid) * 2 + len(probs_iid) * 4  # symbol table + probs
            total_iid_ans += iid_bytes + overhead_iid

            # XOR ANS
            symbols_xor = sorted(counts_xor.keys())
            sym_to_idx_xor = {s: i for i, s in enumerate(symbols_xor)}
            probs_xor = np.array([counts_xor[s] / (n - 1) for s in symbols_xor], dtype=np.float32)
            mapped_xor = np.array([sym_to_idx_xor[v] for v in xor_vals], dtype=np.int32)
            xor_bytes = ans_compressed_size(mapped_xor, symbols_xor, probs_xor)
            # Need to also store first value (2 bytes) + XOR codebook
            overhead_xor = 2 + len(symbols_xor) * 2 + len(probs_xor) * 4
            total_xor_ans += xor_bytes + overhead_xor

        if li == 0:
            # Print detailed stats for first layer
            print(f"\n  Layer 0 ({n} values):")
            print(f"    i.i.d.:     H={h_iid:.4f} bpw, unique={len(counts_iid)}")
            print(f"    XOR delta:  H={h_xor:.4f} bpw, unique={len(counts_xor)}, saved={h_iid-h_xor:.4f} bpw")
            print(f"    XOR hi:     H={h_xor_hi:.4f} bpw/byte, lo: H={h_xor_lo:.4f} bpw/byte")
            print(f"    Arith delta: H={h_delta:.4f} bpw, unique={len(counts_delta)}")

            # Distribution of XOR values
            xor_zero_pct = counts_xor.get(0, 0) / (n - 1) * 100
            xor_small = sum(v for k, v in counts_xor.items() if abs(k) < 256) / (n - 1) * 100
            print(f"    XOR==0: {xor_zero_pct:.1f}%, |XOR|<256: {xor_small:.1f}%")

            if HAS_CONSTRICTION:
                print(f"    ANS i.i.d.: {iid_bytes + overhead_iid} bytes ({(iid_bytes + overhead_iid) / (n*2) * 100:.2f}%)")
                print(f"    ANS XOR:    {xor_bytes + overhead_xor} bytes ({(xor_bytes + overhead_xor) / (n*2) * 100:.2f}%)")

    # Summary across all layers
    total_values = sum(w.numel() for w in weights)
    h_iid_avg = total_iid_bits / total_values
    h_xor_avg = total_xor_bits / total_values
    h_delta_avg = total_delta_bits / total_values

    print(f"\n  All {len(weights)} layers combined ({total_values} values, {total_original/1024/1024:.1f} MB):")
    print(f"    i.i.d. entropy:  {h_iid_avg:.4f} bpw → {h_iid_avg/16*100:.2f}%")
    print(f"    XOR entropy:     {total_xor_bits/total_values:.4f} bpw → {total_xor_bits/total_values/16*100:.2f}%")
    print(f"    XOR hi entropy:  {total_xor_hi_bits/(total_values):.4f} bpw")
    print(f"    XOR lo entropy:  {total_xor_lo_bits/(total_values):.4f} bpw")
    print(f"    Delta entropy:   {h_delta_avg:.4f} bpw → {h_delta_avg/16*100:.2f}%")

    return {
        'h_iid': h_iid_avg,
        'h_xor': total_xor_bits / total_values,
        'h_delta': h_delta_avg,
        'n_values': total_values,
    }


def analyze_scan_orders(model, weight_suffix):
    """Compare different scan orders for XOR delta coding."""
    weights = []
    for name, param in model.named_parameters():
        if name.endswith(weight_suffix):
            weights.append(param.data.cpu().to(torch.bfloat16))
            break  # Just first layer

    if not weights:
        return

    w = weights[0]
    print(f"\n--- Scan order comparison for {weight_suffix} (shape {w.shape}) ---")

    int16 = w.view(torch.int16)

    orders = {
        'row-major': int16.flatten().numpy().astype(np.int32),
        'col-major': int16.t().flatten().numpy().astype(np.int32),
    }

    # Zigzag scan for 2D
    if w.dim() == 2:
        rows, cols = w.shape
        zigzag = []
        for r in range(rows):
            row_data = int16[r].numpy().astype(np.int32)
            if r % 2 == 1:
                row_data = row_data[::-1].copy()
            zigzag.extend(row_data.tolist())
        orders['zigzag'] = np.array(zigzag, dtype=np.int32)

    for order_name, flat in orders.items():
        n = len(flat)
        xor_vals = np.bitwise_xor(flat[1:], flat[:-1])
        h_xor = entropy(Counter(xor_vals.tolist()), n - 1)
        h_iid = entropy(Counter(flat.tolist()), n)
        print(f"  {order_name:12s}: H_iid={h_iid:.4f}, H_xor={h_xor:.4f}, saved={h_iid-h_xor:.4f} bpw")


def analyze_cross_layer_xor(model, weight_suffix):
    """XOR between corresponding positions across layers."""
    weights = []
    for name, param in model.named_parameters():
        if name.endswith(weight_suffix):
            weights.append(param.data.cpu().to(torch.bfloat16))

    if len(weights) < 2:
        return

    print(f"\n--- Cross-layer XOR for {weight_suffix} ({len(weights)} layers) ---")

    # XOR layer[i] with layer[i-1] (element-wise)
    total_bits_iid = 0
    total_bits_xor = 0
    total_n = 0

    for i in range(1, len(weights)):
        curr = weights[i].view(torch.int16).flatten().numpy().astype(np.int32)
        prev = weights[i-1].view(torch.int16).flatten().numpy().astype(np.int32)
        n = len(curr)
        total_n += n

        xor_vals = np.bitwise_xor(curr, prev)
        h_iid = entropy(Counter(curr.tolist()), n)
        h_xor = entropy(Counter(xor_vals.tolist()), n)
        total_bits_iid += h_iid * n
        total_bits_xor += h_xor * n

    avg_iid = total_bits_iid / total_n
    avg_xor = total_bits_xor / total_n
    print(f"  i.i.d.: {avg_iid:.4f} bpw, cross-layer XOR: {avg_xor:.4f} bpw, saved={avg_iid-avg_xor:.4f} bpw")


def main():
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    weight_types = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]

    all_results = {}
    for wt in weight_types:
        result = analyze_weight_type(model, wt, model_name)
        if result:
            all_results[wt] = result

    # Scan order comparison
    analyze_scan_orders(model, "self_attn.q_proj.weight")
    analyze_scan_orders(model, "mlp.gate_proj.weight")

    # Cross-layer XOR
    analyze_cross_layer_xor(model, "self_attn.q_proj.weight")
    analyze_cross_layer_xor(model, "mlp.gate_proj.weight")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total_values = sum(r['n_values'] for r in all_results.values())
    avg_iid = sum(r['h_iid'] * r['n_values'] for r in all_results.values()) / total_values
    avg_xor = sum(r['h_xor'] * r['n_values'] for r in all_results.values()) / total_values
    avg_delta = sum(r['h_delta'] * r['n_values'] for r in all_results.values()) / total_values

    print(f"Total values: {total_values} ({total_values*2/1024/1024:.1f} MB BF16)")
    print(f"i.i.d. entropy:    {avg_iid:.4f} bpw → {avg_iid/16*100:.2f}%")
    print(f"XOR delta entropy: {avg_xor:.4f} bpw → {avg_xor/16*100:.2f}%")
    print(f"Arith delta entropy: {avg_delta:.4f} bpw → {avg_delta/16*100:.2f}%")
    print(f"XOR savings: {avg_iid - avg_xor:.4f} bpw = {(avg_iid - avg_xor)/avg_iid*100:.2f}% relative")
    print(f"Potential ratio: {avg_xor/16*100:.2f}% (vs ANS-16bit 65.96%)")


if __name__ == '__main__':
    main()
