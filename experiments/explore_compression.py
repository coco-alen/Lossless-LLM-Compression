"""
Systematic exploration of lossless BFloat16 compression methods.

Tests multiple approaches head-to-head on actual model weights to find
methods that beat DFloat11's ~68.75% compression ratio.

Approaches tested:
  1. DFloat11 baseline: Huffman(exp) + raw(sign_mantissa)
  2. Huffman both channels: Huffman(exp) + Huffman(sign_mantissa)
  3. Cross-layer delta on exp only: Huffman(delta_exp) + raw(sm)
  4. Cross-layer delta on both channels: Huffman(delta_exp) + Huffman(delta_sm)
  5. Cross-layer delta + left-predictor on both: Huffman(pred_delta_exp) + Huffman(pred_delta_sm)
  6. Exponent-conditioned mantissa: Huffman(exp) + per-exp-Huffman(sm)
  7. Full int16 cross-layer delta + predictor + Huffman both bytes
  8. Row-reorder by exponent then delta-code mantissa
  9. ANS-style lower bound (entropy estimation for both channels)
  10. Cross-layer delta on exp + left-predictor on sm (mixed)
  11. Cross-layer mean-delta on both channels (delta from mean instead of prev layer)

Usage:
    python experiments/explore_compression.py
    python experiments/explore_compression.py --model_name_or_path Qwen/Qwen3-1.7B
"""

import sys
import time
import json
import os
from argparse import ArgumentParser
from collections import defaultdict

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


# =========================================================================
# Helpers
# =========================================================================

def huffman_encoded_size(data_uint8: np.ndarray) -> int:
    """Return the size in bytes of Huffman-encoded uint8 data."""
    vals, counts = np.unique(data_uint8, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    if len(freq) <= 1:
        return (len(data_uint8) + 7) // 8  # 1 bit per symbol minimum
    codec = HuffmanCodec.from_frequencies(freq)
    encoded = codec.encode(data_uint8.tolist())
    return len(encoded)


def huffman_bits_from_freq(freq: dict) -> int:
    """Total Huffman-encoded bits from {symbol: count}."""
    if not freq or sum(freq.values()) == 0:
        return 0
    if len(freq) == 1:
        return sum(freq.values())
    codec = HuffmanCodec.from_frequencies(freq)
    table = codec.get_code_table()
    return sum(code_len * freq.get(sym, 0) for sym, (code_len, _) in table.items() if sym in freq)


def entropy_bits(data_uint8: np.ndarray) -> float:
    """Shannon entropy in total bits."""
    vals, counts = np.unique(data_uint8, return_counts=True)
    n = len(data_uint8)
    probs = counts / n
    return -np.sum(counts * np.log2(probs))


def arr_freq(arr: np.ndarray) -> dict:
    vals, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def extract_fields(w_bf16: torch.Tensor):
    """Extract exponent and sign_mantissa bytes from bf16 tensor."""
    W = w_bf16.contiguous().view(torch.int16)
    exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
    sm = ((W >> 8) & 0x80 | (W & 0x7F)).to(torch.uint8).numpy()
    return exp, sm


def extract_fields_tensors(w_bf16: torch.Tensor):
    """Extract exponent and sign_mantissa as torch int16 tensors (for delta ops)."""
    W = w_bf16.contiguous().view(torch.int16)
    exp = ((W >> 7) & 0xFF).to(torch.int16)
    sm = ((W >> 8) & 0x80 | (W & 0x7F)).to(torch.int16)
    return exp, sm


def int16_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert int16 tensor to uint8 numpy (mod 256)."""
    return (t & 0xFF).to(torch.uint8).numpy()


# =========================================================================
# Weight extraction
# =========================================================================

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


# =========================================================================
# Compression methods
# =========================================================================

def method_dfloat11(tensors: list[torch.Tensor]) -> dict:
    """DFloat11: Huffman(exponent) + raw(sign_mantissa)."""
    all_w = torch.cat([t.flatten() for t in tensors])
    exp, sm = extract_fields(all_w)
    n = len(exp)

    exp_bytes = huffman_encoded_size(exp)
    sm_bytes = n  # raw

    return {
        "compressed_bytes": exp_bytes + sm_bytes,
        "original_bytes": n * 2,
        "detail": f"exp_huf={exp_bytes}, sm_raw={sm_bytes}",
    }


def method_huffman_both(tensors: list[torch.Tensor]) -> dict:
    """Huffman(exponent) + Huffman(sign_mantissa) — no grouping."""
    all_w = torch.cat([t.flatten() for t in tensors])
    exp, sm = extract_fields(all_w)
    n = len(exp)

    exp_bytes = huffman_encoded_size(exp)
    sm_bytes = huffman_encoded_size(sm)

    return {
        "compressed_bytes": exp_bytes + sm_bytes,
        "original_bytes": n * 2,
        "detail": f"exp_huf={exp_bytes}, sm_huf={sm_bytes}",
    }


def method_crosslayer_delta_exp(tensors: list[torch.Tensor]) -> dict:
    """Cross-layer delta on exponent only + Huffman(delta_exp) + raw(sm)."""
    exps, sms = [], []
    for t in tensors:
        e, s = extract_fields_tensors(t.flatten())
        exps.append(e)
        sms.append(s)

    # Delta encode exponents
    delta_exps = [exps[0]]
    for i in range(1, len(exps)):
        delta_exps.append(exps[i] - exps[i - 1])  # int16 wrapping

    all_delta_exp = int16_to_uint8(torch.cat(delta_exps))
    all_sm = np.concatenate([s.numpy().astype(np.uint8) for s in sms])
    n = len(all_delta_exp)

    exp_bytes = huffman_encoded_size(all_delta_exp)
    sm_bytes = n  # raw

    return {
        "compressed_bytes": exp_bytes + sm_bytes,
        "original_bytes": n * 2,
        "detail": f"delta_exp_huf={exp_bytes}, sm_raw={sm_bytes}",
    }


def method_crosslayer_delta_both(tensors: list[torch.Tensor]) -> dict:
    """Cross-layer delta on BOTH exponent and sign_mantissa, Huffman each."""
    exps, sms = [], []
    for t in tensors:
        e, s = extract_fields_tensors(t.flatten())
        exps.append(e)
        sms.append(s)

    # Delta encode both
    delta_exps = [exps[0]]
    delta_sms = [sms[0]]
    for i in range(1, len(exps)):
        delta_exps.append(exps[i] - exps[i - 1])
        delta_sms.append(sms[i] - sms[i - 1])

    all_delta_exp = int16_to_uint8(torch.cat(delta_exps))
    all_delta_sm = int16_to_uint8(torch.cat(delta_sms))
    n = len(all_delta_exp)

    exp_bytes = huffman_encoded_size(all_delta_exp)
    sm_bytes = huffman_encoded_size(all_delta_sm)

    return {
        "compressed_bytes": exp_bytes + sm_bytes,
        "original_bytes": n * 2,
        "detail": f"delta_exp_huf={exp_bytes}, delta_sm_huf={sm_bytes}",
    }


def _left_predict_uint8(data: np.ndarray, shape) -> np.ndarray:
    """Left-predictor residual on 2D reshaped data, mod 256."""
    if len(shape) == 2:
        mat = data.reshape(shape).astype(np.int16)
        res = np.zeros_like(mat)
        res[:, 0] = mat[:, 0]
        res[:, 1:] = mat[:, 1:] - mat[:, :-1]
        return (res & 0xFF).astype(np.uint8).flatten()
    else:
        arr = data.astype(np.int16)
        res = np.zeros_like(arr)
        res[0] = arr[0]
        res[1:] = arr[1:] - arr[:-1]
        return (res & 0xFF).astype(np.uint8)


def method_crosslayer_delta_predict_both(tensors: list[torch.Tensor]) -> dict:
    """Cross-layer delta + left-predictor on both channels, Huffman each."""
    exps, sms = [], []
    shapes = []
    for t in tensors:
        shapes.append(t.shape)
        e, s = extract_fields_tensors(t.flatten())
        exps.append(e)
        sms.append(s)

    # Delta encode both
    delta_exps = [exps[0]]
    delta_sms = [sms[0]]
    for i in range(1, len(exps)):
        delta_exps.append(exps[i] - exps[i - 1])
        delta_sms.append(sms[i] - sms[i - 1])

    # Left-predictor on each layer's delta (reshaped to original 2D)
    pred_exps, pred_sms = [], []
    for i, (de, ds, shape) in enumerate(zip(delta_exps, delta_sms, shapes)):
        de_u8 = int16_to_uint8(de)
        ds_u8 = int16_to_uint8(ds)
        pred_exps.append(_left_predict_uint8(de_u8, shape))
        pred_sms.append(_left_predict_uint8(ds_u8, shape))

    all_pred_exp = np.concatenate(pred_exps)
    all_pred_sm = np.concatenate(pred_sms)
    n = len(all_pred_exp)

    exp_bytes = huffman_encoded_size(all_pred_exp)
    sm_bytes = huffman_encoded_size(all_pred_sm)

    return {
        "compressed_bytes": exp_bytes + sm_bytes,
        "original_bytes": n * 2,
        "detail": f"pred_delta_exp_huf={exp_bytes}, pred_delta_sm_huf={sm_bytes}",
    }


def method_exp_conditioned_mantissa(tensors: list[torch.Tensor]) -> dict:
    """Exponent-conditioned mantissa: Huffman(exp) + per-exp-value Huffman(sm)."""
    all_w = torch.cat([t.flatten() for t in tensors])
    exp, sm = extract_fields(all_w)
    n = len(exp)

    # Huffman encode exponent (same as DFloat11)
    exp_bytes = huffman_encoded_size(exp)

    # Per-exponent-value Huffman on sign_mantissa
    # We need to store which Huffman table to use (the exp value tells us),
    # so no extra index needed — the decoder knows the exp value.
    unique_exps = np.unique(exp)
    sm_total_bytes = 0
    table_overhead = 0

    for ev in unique_exps:
        mask = exp == ev
        sm_subset = sm[mask]
        if len(sm_subset) == 0:
            continue

        # Huffman encode this subset
        sm_encoded_bytes = huffman_encoded_size(sm_subset)
        sm_total_bytes += sm_encoded_bytes

        # Table overhead: need to store the Huffman table for this exponent value
        # Estimate: ~2 bytes per unique symbol in the table
        n_unique = len(np.unique(sm_subset))
        table_overhead += n_unique * 2  # conservative estimate

    return {
        "compressed_bytes": exp_bytes + sm_total_bytes + table_overhead,
        "original_bytes": n * 2,
        "detail": f"exp_huf={exp_bytes}, cond_sm={sm_total_bytes}, tbl_overhead={table_overhead}, n_exp_vals={len(unique_exps)}",
    }


def method_full_int16_delta_predict(tensors: list[torch.Tensor]) -> dict:
    """Full int16 cross-layer delta + left-predictor + Huffman both bytes."""
    int16_tensors = [t.contiguous().view(torch.int16).flatten() for t in tensors]
    shapes = [t.shape for t in tensors]

    # Cross-layer delta
    deltas = [int16_tensors[0]]
    for i in range(1, len(int16_tensors)):
        deltas.append(int16_tensors[i] - int16_tensors[i - 1])

    # Left-predictor on each layer
    predicted = []
    for delta, shape in zip(deltas, shapes):
        mat = delta.view(shape) if len(shape) == 2 else delta
        if len(shape) == 2 and shape[1] > 1:
            res = mat.clone()
            res[:, 1:] = mat[:, 1:] - mat[:, :-1]
        elif len(shape) == 1 and shape[0] > 1:
            res = mat.clone()
            res[1:] = mat[1:] - mat[:-1]
        else:
            res = mat.clone()
        predicted.append(res.flatten())

    all_pred = torch.cat(predicted)

    # Split into high/low bytes
    raw_bytes = all_pred.contiguous().view(torch.uint8)
    low_bytes = raw_bytes[0::2].numpy()  # little-endian: low first
    high_bytes = raw_bytes[1::2].numpy()
    n = len(high_bytes)

    high_enc = huffman_encoded_size(high_bytes)
    low_enc = huffman_encoded_size(low_bytes)

    return {
        "compressed_bytes": high_enc + low_enc,
        "original_bytes": n * 2,
        "detail": f"high_huf={high_enc}, low_huf={low_enc}",
    }


def method_crosslayer_meandelta_both(tensors: list[torch.Tensor]) -> dict:
    """Cross-layer delta from MEAN (not previous layer) on both channels."""
    exps, sms = [], []
    for t in tensors:
        e, s = extract_fields_tensors(t.flatten())
        exps.append(e)
        sms.append(s)

    # Compute mean across layers (element-wise)
    exp_stack = torch.stack(exps)  # (L, N)
    sm_stack = torch.stack(sms)

    exp_mean = exp_stack.float().mean(dim=0).round().to(torch.int16)
    sm_mean = sm_stack.float().mean(dim=0).round().to(torch.int16)

    # Delta from mean
    delta_exps = [e - exp_mean for e in exps]
    delta_sms = [s - sm_mean for s in sms]

    # Need to store the mean as well (one copy)
    mean_exp_bytes = len(int16_to_uint8(exp_mean))  # raw
    mean_sm_bytes = len(int16_to_uint8(sm_mean))  # raw

    all_delta_exp = int16_to_uint8(torch.cat(delta_exps))
    all_delta_sm = int16_to_uint8(torch.cat(delta_sms))
    n_per_layer = len(exps[0])
    n = len(all_delta_exp)

    exp_bytes = huffman_encoded_size(all_delta_exp)
    sm_bytes = huffman_encoded_size(all_delta_sm)

    return {
        "compressed_bytes": exp_bytes + sm_bytes + mean_exp_bytes + mean_sm_bytes,
        "original_bytes": n * 2,
        "detail": f"mean_delta_exp_huf={exp_bytes}, mean_delta_sm_huf={sm_bytes}, mean_store={mean_exp_bytes + mean_sm_bytes}",
    }


def method_predict_sm_only(tensors: list[torch.Tensor]) -> dict:
    """DFloat11 exp + left-predictor on sign_mantissa (per-layer 2D prediction)."""
    all_w = torch.cat([t.flatten() for t in tensors])
    exp, _ = extract_fields(all_w)

    # Exp: standard Huffman
    exp_bytes = huffman_encoded_size(exp)

    # SM: left-predictor per layer
    pred_sms = []
    for t in tensors:
        _, sm = extract_fields(t.flatten())
        pred_sms.append(_left_predict_uint8(sm, t.shape))

    all_pred_sm = np.concatenate(pred_sms)
    n = len(exp)
    sm_bytes = huffman_encoded_size(all_pred_sm)

    return {
        "compressed_bytes": exp_bytes + sm_bytes,
        "original_bytes": n * 2,
        "detail": f"exp_huf={exp_bytes}, pred_sm_huf={sm_bytes}",
    }


def method_crosslayer_delta_exp_predict_sm(tensors: list[torch.Tensor]) -> dict:
    """Cross-layer delta on exp + left-predictor on sm (mixed best-of-both)."""
    exps, sms = [], []
    shapes = [t.shape for t in tensors]
    for t in tensors:
        e, s = extract_fields_tensors(t.flatten())
        exps.append(e)
        sms.append(s)

    # Delta exponents
    delta_exps = [exps[0]]
    for i in range(1, len(exps)):
        delta_exps.append(exps[i] - exps[i - 1])
    all_delta_exp = int16_to_uint8(torch.cat(delta_exps))

    # Left-predict sign_mantissa per layer (no cross-layer delta)
    pred_sms = []
    for s, shape in zip(sms, shapes):
        s_u8 = int16_to_uint8(s)
        pred_sms.append(_left_predict_uint8(s_u8, shape))
    all_pred_sm = np.concatenate(pred_sms)

    n = len(all_delta_exp)
    exp_bytes = huffman_encoded_size(all_delta_exp)
    sm_bytes = huffman_encoded_size(all_pred_sm)

    return {
        "compressed_bytes": exp_bytes + sm_bytes,
        "original_bytes": n * 2,
        "detail": f"delta_exp_huf={exp_bytes}, pred_sm_huf={sm_bytes}",
    }


def method_entropy_lower_bound(tensors: list[torch.Tensor]) -> dict:
    """Theoretical entropy lower bound for separate-channel coding."""
    all_w = torch.cat([t.flatten() for t in tensors])
    exp, sm = extract_fields(all_w)
    n = len(exp)

    exp_entropy_total = entropy_bits(exp)
    sm_entropy_total = entropy_bits(sm)

    # Convert bits to bytes (ceil)
    exp_bytes = int(np.ceil(exp_entropy_total / 8))
    sm_bytes = int(np.ceil(sm_entropy_total / 8))

    return {
        "compressed_bytes": exp_bytes + sm_bytes,
        "original_bytes": n * 2,
        "detail": f"exp_entropy={exp_entropy_total/n:.3f}bpw, sm_entropy={sm_entropy_total/n:.3f}bpw",
    }


def method_crosslayer_delta_both_predict_both(tensors: list[torch.Tensor]) -> dict:
    """Cross-layer delta on both + left-predictor on both + Huffman each.
    Entropy lower bound version for comparison."""
    exps, sms = [], []
    shapes = [t.shape for t in tensors]
    for t in tensors:
        e, s = extract_fields_tensors(t.flatten())
        exps.append(e)
        sms.append(s)

    # Delta + predict on exponent
    delta_exps = [exps[0]]
    delta_sms = [sms[0]]
    for i in range(1, len(exps)):
        delta_exps.append(exps[i] - exps[i - 1])
        delta_sms.append(sms[i] - sms[i - 1])

    pred_exps, pred_sms = [], []
    for de, ds, shape in zip(delta_exps, delta_sms, shapes):
        de_u8 = int16_to_uint8(de)
        ds_u8 = int16_to_uint8(ds)
        pred_exps.append(_left_predict_uint8(de_u8, shape))
        pred_sms.append(_left_predict_uint8(ds_u8, shape))

    all_pe = np.concatenate(pred_exps)
    all_ps = np.concatenate(pred_sms)
    n = len(all_pe)

    # Entropy lower bound
    exp_entropy = entropy_bits(all_pe)
    sm_entropy = entropy_bits(all_ps)

    return {
        "compressed_bytes": int(np.ceil(exp_entropy / 8)) + int(np.ceil(sm_entropy / 8)),
        "original_bytes": n * 2,
        "detail": f"entropy: pred_delta_exp={exp_entropy/n:.3f}bpw, pred_delta_sm={sm_entropy/n:.3f}bpw",
    }


def method_exp_conditioned_no_overhead(tensors: list[torch.Tensor]) -> dict:
    """Exponent-conditioned mantissa WITHOUT table overhead (best-case estimate)."""
    all_w = torch.cat([t.flatten() for t in tensors])
    exp, sm = extract_fields(all_w)
    n = len(exp)

    exp_bytes = huffman_encoded_size(exp)

    unique_exps = np.unique(exp)
    sm_total_bits = 0
    for ev in unique_exps:
        mask = exp == ev
        sm_subset = sm[mask]
        if len(sm_subset) == 0:
            continue
        sm_total_bits += entropy_bits(sm_subset)

    sm_bytes = int(np.ceil(sm_total_bits / 8))

    return {
        "compressed_bytes": exp_bytes + sm_bytes,
        "original_bytes": n * 2,
        "detail": f"exp_huf={exp_bytes}, cond_sm_entropy={sm_bytes} ({sm_total_bits/n:.3f}bpw)",
    }


def method_crosslayer_delta_both_predict_exp(tensors: list[torch.Tensor]) -> dict:
    """Cross-layer delta both + left-predictor on exp ONLY + Huffman both."""
    exps, sms = [], []
    shapes = [t.shape for t in tensors]
    for t in tensors:
        e, s = extract_fields_tensors(t.flatten())
        exps.append(e)
        sms.append(s)

    delta_exps = [exps[0]]
    delta_sms = [sms[0]]
    for i in range(1, len(exps)):
        delta_exps.append(exps[i] - exps[i - 1])
        delta_sms.append(sms[i] - sms[i - 1])

    # Left-predict exp only
    pred_exps = []
    for de, shape in zip(delta_exps, shapes):
        de_u8 = int16_to_uint8(de)
        pred_exps.append(_left_predict_uint8(de_u8, shape))

    all_pe = np.concatenate(pred_exps)
    all_ds = int16_to_uint8(torch.cat(delta_sms))
    n = len(all_pe)

    exp_bytes = huffman_encoded_size(all_pe)
    sm_bytes = huffman_encoded_size(all_ds)

    return {
        "compressed_bytes": exp_bytes + sm_bytes,
        "original_bytes": n * 2,
        "detail": f"pred_delta_exp_huf={exp_bytes}, delta_sm_huf={sm_bytes}",
    }


# =========================================================================
# Main
# =========================================================================

METHODS = [
    ("1. DFloat11 baseline", method_dfloat11),
    ("2. Huffman(exp)+Huffman(sm)", method_huffman_both),
    ("3. XL-delta(exp)+raw(sm)", method_crosslayer_delta_exp),
    ("4. XL-delta(both)+Huffman", method_crosslayer_delta_both),
    ("5. XL-delta+predict(both)", method_crosslayer_delta_predict_both),
    ("6. Exp-conditioned mantissa", method_exp_conditioned_mantissa),
    ("6b. Exp-cond (no overhead)", method_exp_conditioned_no_overhead),
    ("7. Full int16 delta+pred+Huf", method_full_int16_delta_predict),
    ("8. DFloat11 exp+predict(sm)", method_predict_sm_only),
    ("9. XL-delta(exp)+predict(sm)", method_crosslayer_delta_exp_predict_sm),
    ("10. XL-delta(both)+pred(exp)", method_crosslayer_delta_both_predict_exp),
    ("11. XL-mean-delta(both)+Huf", method_crosslayer_meandelta_both),
    ("12. XL-delta+pred entropy LB", method_crosslayer_delta_both_predict_both),
    ("13. Entropy lower bound", method_entropy_lower_bound),
]


def main():
    parser = ArgumentParser("Explore compression methods")
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Per weight-type results ----
    all_results = {}
    totals = defaultdict(lambda: {"compressed": 0, "original": 0})

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        print(f"\n{'='*100}")
        print(f"  {wt}  ({len(tensors)} layers, {sum(t.numel() for t in tensors):,} params)")
        print(f"{'='*100}")
        print(f"  {'Method':<40} {'Ratio':>7} {'Comp MB':>10} {'Orig MB':>10} {'bpw':>6}  Detail")
        print(f"  {'-'*40} {'-'*7} {'-'*10} {'-'*10} {'-'*6}  {'-'*30}")

        wt_results = {}
        for name, method in METHODS:
            t0 = time.time()
            try:
                result = method(tensors)
            except Exception as e:
                print(f"  {name:<40} ERROR: {e}")
                continue
            elapsed = time.time() - t0

            ratio = result["compressed_bytes"] / result["original_bytes"] * 100
            bpw = ratio / 100 * 16
            comp_mb = result["compressed_bytes"] / 1e6
            orig_mb = result["original_bytes"] / 1e6

            print(f"  {name:<40} {ratio:>6.2f}% {comp_mb:>9.2f} {orig_mb:>9.2f} {bpw:>5.2f}  {result['detail']}")

            wt_results[name] = result
            totals[name]["compressed"] += result["compressed_bytes"]
            totals[name]["original"] += result["original_bytes"]

        all_results[wt] = wt_results

    # ---- Overall summary ----
    print(f"\n\n{'='*100}")
    print("OVERALL SUMMARY (all weight types combined)")
    print(f"{'='*100}")
    print(f"  {'Method':<40} {'Ratio':>7} {'Comp MB':>10} {'Orig MB':>10} {'bpw':>6}  {'vs DF11':>8}")
    print(f"  {'-'*40} {'-'*7} {'-'*10} {'-'*10} {'-'*6}  {'-'*8}")

    df11_ratio = None
    summary_data = []

    for name, _ in METHODS:
        t = totals[name]
        if t["original"] == 0:
            continue
        ratio = t["compressed"] / t["original"] * 100
        bpw = ratio / 100 * 16
        comp_mb = t["compressed"] / 1e6
        orig_mb = t["original"] / 1e6

        if "DFloat11" in name:
            df11_ratio = ratio
            delta = "---"
        else:
            delta = f"{ratio - df11_ratio:+.2f}%" if df11_ratio else "---"

        print(f"  {name:<40} {ratio:>6.2f}% {comp_mb:>9.2f} {orig_mb:>9.2f} {bpw:>5.2f}  {delta:>8}")
        summary_data.append({"method": name, "ratio": ratio, "bpw": bpw})

    # ---- Save results ----
    output_path = os.path.join(os.path.dirname(__file__), "exploration_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model_name_or_path,
            "num_layers": num_layers,
            "summary": summary_data,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Find best method that beats DFloat11
    if df11_ratio:
        better = [s for s in summary_data if s["ratio"] < df11_ratio and "entropy" not in s["method"].lower() and "no overhead" not in s["method"].lower()]
        if better:
            best = min(better, key=lambda x: x["ratio"])
            print(f"\n*** BEST METHOD BEATING DFloat11: {best['method']} at {best['ratio']:.2f}% (vs {df11_ratio:.2f}%) ***")
        else:
            print(f"\n*** No practical method beats DFloat11 ({df11_ratio:.2f}%) yet ***")
            # Show theoretical bounds
            theoretical = [s for s in summary_data if "entropy" in s["method"].lower() or "no overhead" in s["method"].lower()]
            if theoretical:
                best_th = min(theoretical, key=lambda x: x["ratio"])
                print(f"    Theoretical best: {best_th['method']} at {best_th['ratio']:.2f}%")
                print(f"    Gap to close: {df11_ratio - best_th['ratio']:.2f}%")


if __name__ == "__main__":
    main()
