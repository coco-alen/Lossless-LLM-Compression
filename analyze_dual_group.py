"""
Dual-Group Huffman: compress BOTH exponent AND mantissa of BFloat16 weights.

Core idea:
    DFloat11 only compresses exponent (8→~3 bits), mantissa stored raw (8 bits).
    → ~11 bits/weight, ~68.75% of bf16.

    Our proposal: split weights into 2 groups via 1-bit mask.
    Within each group, Huffman-encode BOTH exponent AND mantissa.
    If grouping makes per-group distributions more concentrated,
    the mantissa (currently 8 raw bits) becomes compressible,
    more than compensating the 1-bit mask overhead.

    Target: < 11 bits/weight → better than DFloat11.

What this script analyzes:
    1. How many bits mantissa actually needs (globally and per-group)
    2. Multiple grouping strategies, all focused on making mantissa compressible
    3. Detailed breakdown: mask cost, exp savings, mantissa savings
    4. Theoretical best: what if we had perfect grouping?

Usage:
    python analyze_dual_group.py
    python analyze_dual_group.py --model_name_or_path Qwen/Qwen3-1.7B
    python analyze_dual_group.py --model_name_or_path Qwen/Qwen3-1.7B --per_layer
"""

import sys
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


# =========================================================================
# Huffman helpers
# =========================================================================

def huffman_bits_from_freq(freq: dict) -> int:
    """Total Huffman-encoded bits from {symbol: count}."""
    if not freq:
        return 0
    total = sum(freq.values())
    if total == 0:
        return 0
    if len(freq) == 1:
        return total  # Huffman minimum: 1 bit per symbol
    codec = HuffmanCodec.from_frequencies(freq)
    table = codec.get_code_table()
    return sum(code_len * freq[sym] for sym, (code_len, _) in table.items() if sym in freq)


def arr_to_freq(arr: np.ndarray) -> dict:
    """uint8 array → {value: count} dict via bincount."""
    c = np.bincount(arr.ravel(), minlength=256)
    return {i: int(c[i]) for i in range(256) if c[i] > 0}


def huffman_bits(arr: np.ndarray) -> int:
    return huffman_bits_from_freq(arr_to_freq(arr))


def shannon_entropy_bits(freq: dict) -> float:
    """Shannon entropy in bits (theoretical minimum, lower bound for Huffman)."""
    total = sum(freq.values())
    if total == 0:
        return 0.0
    return sum(-c * np.log2(c / total) for c in freq.values() if c > 0)


# =========================================================================
# BFloat16 field extraction
# =========================================================================

def extract_fields(w_bf16: torch.Tensor):
    """
    BFloat16: [sign(1) | exponent(8) | mantissa(7)]

    Returns:
        exp:  exponent 8-bit (uint8 numpy)
        man:  mantissa 7-bit as uint8 (0..127)
        sign: sign bit as uint8 (0 or 1)
        sm:   sign_mantissa 8-bit = sign(1)|mantissa(7), same as DFloat11's "other byte"
    """
    W = w_bf16.contiguous().view(torch.int16)
    exp  = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
    sign = ((W >> 15) & 0x1).to(torch.uint8).numpy()
    man  = (W & 0x7F).to(torch.uint8).numpy()
    sm   = (sign.astype(np.uint8) << 7) | man  # sign_mantissa byte
    return exp, man, sign, sm


# =========================================================================
# Pre-compute joint frequency matrix (exponent × mantissa)
# =========================================================================

def build_joint_matrix(exp: np.ndarray, sm: np.ndarray):
    """
    Returns (256, 256) matrix: joint[exp_val, sm_val] = count.
    Also returns marginals.
    """
    joint_idx = exp.astype(np.int32) * 256 + sm.astype(np.int32)
    joint = np.bincount(joint_idx, minlength=65536).reshape(256, 256)
    exp_marginal = joint.sum(axis=1)  # (256,)
    sm_marginal  = joint.sum(axis=0)  # (256,)
    return joint, exp_marginal, sm_marginal


def counts_to_freq(c: np.ndarray) -> dict:
    nz = c.nonzero()[0]
    return {int(i): int(c[i]) for i in nz}


# =========================================================================
# Compression analysis functions
# =========================================================================

def analyze_baseline(exp, sm, n):
    """DFloat11: Huffman(exp) + raw(sign_mantissa)."""
    exp_b = huffman_bits(exp)
    return {
        "total_bits": exp_b + n * 8,
        "bpw": (exp_b + n * 8) / n,
        "exp_bpw": exp_b / n,
        "man_bpw": 8.0,
        "mask_bpw": 0.0,
    }


def analyze_single_huffman_both(exp, sm, n):
    """Single-table Huffman on exp + single-table Huffman on sign_mantissa (no grouping)."""
    exp_b = huffman_bits(exp)
    man_b = huffman_bits(sm)
    return {
        "total_bits": exp_b + man_b,
        "bpw": (exp_b + man_b) / n,
        "exp_bpw": exp_b / n,
        "man_bpw": man_b / n,
        "mask_bpw": 0.0,
    }


def analyze_dual_group(exp, sm, mask, n):
    """
    Dual-group: 1-bit mask + per-group Huffman(exp) + per-group Huffman(sign_mantissa).
    This is the core of the proposal: compress mantissa via grouping.
    """
    a, b = (mask == 0), (mask == 1)
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 or nb == 0:
        return None

    exp_b = huffman_bits(exp[a]) + huffman_bits(exp[b])
    man_b = huffman_bits(sm[a]) + huffman_bits(sm[b])
    total = n + exp_b + man_b  # n = mask cost (1 bit/weight)

    return {
        "total_bits": total,
        "bpw": total / n,
        "exp_bpw": exp_b / n,
        "man_bpw": man_b / n,
        "mask_bpw": 1.0,
        "na": na, "nb": nb,
        "frac_a": na / n,
        # Per-group mantissa detail
        "man_bpw_a": huffman_bits(sm[a]) / na,
        "man_bpw_b": huffman_bits(sm[b]) / nb,
    }


def analyze_dual_group_from_freq(n, exp_fa, exp_fb, man_fa, man_fb, na, nb):
    """Same as analyze_dual_group but from pre-computed frequency dicts."""
    if na == 0 or nb == 0:
        return None
    exp_b = huffman_bits_from_freq(exp_fa) + huffman_bits_from_freq(exp_fb)
    man_b = huffman_bits_from_freq(man_fa) + huffman_bits_from_freq(man_fb)
    total = n + exp_b + man_b
    return {
        "total_bits": total,
        "bpw": total / n,
        "exp_bpw": exp_b / n,
        "man_bpw": man_b / n,
        "mask_bpw": 1.0,
        "na": na, "nb": nb,
        "frac_a": na / n,
    }


# =========================================================================
# Grouping strategies
# =========================================================================

def strategy_exp_median(exp, sm, n):
    """Split by exponent median."""
    mask = (exp > np.median(exp)).astype(np.uint8)
    return analyze_dual_group(exp, sm, mask, n)


def strategy_sign(exp, sm, sign, n):
    """Split by sign bit."""
    return analyze_dual_group(exp, sm, sign, n)


def strategy_exp_optimal_threshold(exp, sm, n):
    """
    Sweep all exponent thresholds incrementally.
    Optimizes: 1-bit mask + per-group Huffman(exp) + per-group Huffman(mantissa).
    """
    joint, exp_marginal, sm_marginal = build_joint_matrix(exp, sm)

    active_exps = np.where(exp_marginal > 0)[0]
    if len(active_exps) <= 1:
        return None, None

    # Incremental: start with A=empty, B=all; move exp values low→high into A
    exp_ca = np.zeros(256, dtype=np.int64)
    exp_cb = exp_marginal.copy().astype(np.int64)
    man_ca = np.zeros(256, dtype=np.int64)
    man_cb = sm_marginal.copy().astype(np.int64)
    na, nb = 0, n

    best_bits = float("inf")
    best_thr = None

    for ev in active_exps[:-1]:
        cnt = int(exp_marginal[ev])
        exp_ca[ev] = cnt
        exp_cb[ev] = 0
        man_ca += joint[ev]
        man_cb -= joint[ev]
        na += cnt
        nb -= cnt

        r = analyze_dual_group_from_freq(
            n,
            counts_to_freq(exp_ca), counts_to_freq(exp_cb),
            counts_to_freq(man_ca), counts_to_freq(man_cb),
            na, nb,
        )
        if r and r["total_bits"] < best_bits:
            best_bits = r["total_bits"]
            best_thr = int(ev)

    if best_thr is None:
        return None, None

    mask = (exp > best_thr).astype(np.uint8)
    result = analyze_dual_group(exp, sm, mask, n)
    return result, best_thr


def strategy_kmeans(exp, sm, n, n_sample=200000):
    """K-means (k=2) on exponent values."""
    sample = exp if len(exp) <= n_sample else exp[np.random.choice(len(exp), n_sample, replace=False)]
    sf = sample.astype(np.float32)
    c0, c1 = float(sf.min()), float(sf.max())
    for _ in range(30):
        lab = np.abs(sf - c1) < np.abs(sf - c0)
        if lab.sum() == 0 or (~lab).sum() == 0:
            break
        c0, c1 = sf[~lab].mean(), sf[lab].mean()
    af = exp.astype(np.float32)
    mask = (np.abs(af - c1) < np.abs(af - c0)).astype(np.uint8)
    return analyze_dual_group(exp, sm, mask, n)


def strategy_mantissa_median(exp, sm, man, n):
    """Split by mantissa median (group weights with similar mantissa patterns)."""
    mask = (man > np.median(man)).astype(np.uint8)
    return analyze_dual_group(exp, sm, mask, n)


def strategy_joint_optimal(exp, sm, n):
    """
    Sweep exponent threshold, but also try splitting sign_mantissa
    into (sign, mantissa_7bit) and Huffman-coding each separately per group.
    This tests whether separating sign from mantissa helps.
    """
    joint, exp_marginal, sm_marginal = build_joint_matrix(exp, sm)

    sign = sm >> 7          # shape (n,) uint8 0/1
    man7 = sm & 0x7F        # shape (n,) uint8 0..127

    # Build joint matrices for sign and man7 separately
    joint_sign = np.zeros((256, 2), dtype=np.int64)
    joint_man7 = np.zeros((256, 128), dtype=np.int64)

    for ev in range(256):
        if exp_marginal[ev] == 0:
            continue
        mask_ev = exp == ev
        s_vals, s_counts = np.unique(sign[mask_ev], return_counts=True)
        for v, c in zip(s_vals, s_counts):
            joint_sign[ev, v] = c
        m_vals, m_counts = np.unique(man7[mask_ev], return_counts=True)
        for v, c in zip(m_vals, m_counts):
            joint_man7[ev, v] = c

    active_exps = np.where(exp_marginal > 0)[0]
    if len(active_exps) <= 1:
        return None, None

    exp_ca = np.zeros(256, dtype=np.int64)
    exp_cb = exp_marginal.copy().astype(np.int64)
    sign_ca = np.zeros(2, dtype=np.int64)
    sign_cb = joint_sign.sum(axis=0).astype(np.int64)
    man7_ca = np.zeros(128, dtype=np.int64)
    man7_cb = joint_man7.sum(axis=0).astype(np.int64)
    na, nb = 0, n

    best_bits = float("inf")
    best_thr = None

    for ev in active_exps[:-1]:
        cnt = int(exp_marginal[ev])
        exp_ca[ev] = cnt
        exp_cb[ev] = 0
        sign_ca += joint_sign[ev]
        sign_cb -= joint_sign[ev]
        man7_ca += joint_man7[ev]
        man7_cb -= joint_man7[ev]
        na += cnt
        nb -= cnt

        # per-group: Huffman(exp) + Huffman(sign_2val) + Huffman(man_7bit)
        exp_b = huffman_bits_from_freq(counts_to_freq(exp_ca)) + huffman_bits_from_freq(counts_to_freq(exp_cb))

        def sign_bits(sc):
            f = {i: int(sc[i]) for i in range(2) if sc[i] > 0}
            return huffman_bits_from_freq(f)

        def man7_bits(mc):
            f = {i: int(mc[i]) for i in range(128) if mc[i] > 0}
            return huffman_bits_from_freq(f)

        s_b = sign_bits(sign_ca) + sign_bits(sign_cb)
        m_b = man7_bits(man7_ca) + man7_bits(man7_cb)
        total = n + exp_b + s_b + m_b  # 1-bit mask + exp + sign + mantissa

        if total < best_bits:
            best_bits = total
            best_thr = int(ev)

    if best_thr is None:
        return None, None

    mask = (exp > best_thr).astype(np.uint8)
    # Recompute with detail
    a, b = (mask == 0), (mask == 1)
    na_f, nb_f = int(a.sum()), int(b.sum())
    exp_b = huffman_bits(exp[a]) + huffman_bits(exp[b])

    def sign_man_bits(idx):
        s = sign[idx]
        m = man7[idx]
        sb = huffman_bits_from_freq({i: int(c) for i, c in enumerate(np.bincount(s, minlength=2)) if c > 0})
        mb = huffman_bits_from_freq({i: int(c) for i, c in enumerate(np.bincount(m, minlength=128)) if c > 0})
        return sb, mb

    sa_b, ma_b = sign_man_bits(a)
    sb_b, mb_b = sign_man_bits(b)
    total = n + exp_b + sa_b + sb_b + ma_b + mb_b

    return {
        "total_bits": total,
        "bpw": total / n,
        "exp_bpw": exp_b / n,
        "sign_bpw": (sa_b + sb_b) / n,
        "man7_bpw": (ma_b + mb_b) / n,
        "man_bpw": (sa_b + sb_b + ma_b + mb_b) / n,  # sign + man7 combined
        "mask_bpw": 1.0,
        "na": na_f, "nb": nb_f,
        "frac_a": na_f / n,
    }, best_thr


# =========================================================================
# Full analysis
# =========================================================================

def analyze_weights(w_bf16: torch.Tensor, label: str = "") -> dict:
    exp, man, sign, sm = extract_fields(w_bf16)
    n = len(exp)
    t0 = time.time()

    results = {"n": n}

    # 1. DFloat11 baseline
    results["dfloat11"] = analyze_baseline(exp, sm, n)

    # 2. Single-table Huffman(exp) + Huffman(sign_mantissa) — no group, just compress mantissa
    results["single_both"] = analyze_single_huffman_both(exp, sm, n)

    # 3. Single-table: Huffman(exp) + Huffman(sign) + Huffman(man7) — split sign from mantissa
    exp_b = huffman_bits(exp)
    sign_b = huffman_bits(sign)
    man7_b = huffman_bits(man)
    results["single_split_sm"] = {
        "total_bits": exp_b + sign_b + man7_b,
        "bpw": (exp_b + sign_b + man7_b) / n,
        "exp_bpw": exp_b / n,
        "sign_bpw": sign_b / n,
        "man7_bpw": man7_b / n,
        "man_bpw": (sign_b + man7_b) / n,
        "mask_bpw": 0.0,
    }

    # 4. Dual-group strategies (all compress mantissa)
    r = strategy_exp_median(exp, sm, n)
    if r: results["dual_exp_median"] = r

    r = strategy_sign(exp, sm, sign, n)
    if r: results["dual_sign"] = r

    r, thr = strategy_exp_optimal_threshold(exp, sm, n)
    if r:
        r["threshold"] = thr
        results["dual_optimal_thr"] = r

    r = strategy_kmeans(exp, sm, n)
    if r: results["dual_kmeans"] = r

    r = strategy_mantissa_median(exp, sm, man, n)
    if r: results["dual_man_median"] = r

    # 5. Joint optimal: split sign and mantissa7 separately per group
    r, thr = strategy_joint_optimal(exp, sm, n)
    if r:
        r["threshold"] = thr
        results["dual_joint_opt"] = r

    elapsed = time.time() - t0
    print(f"  [{label}] {n:,} weights, {elapsed:.1f}s", file=sys.stderr, flush=True)
    return results


# =========================================================================
# Pretty print
# =========================================================================

def print_results(results: dict, label: str):
    n = results["n"]
    bl = results["dfloat11"]

    print(f"\n{'='*95}")
    print(f"  {label}  ({n:,} weights)")
    print(f"{'='*95}")

    # Mantissa focus header
    print(f"  {'Method':<38} {'bpw':>6} {'ratio':>7} "
          f"{'mask':>5} {'exp':>5} {'man':>5} {'vs DF11':>8} {'man_detail'}")
    print(f"  {'-'*38} {'-'*6} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*8} {'-'*25}")

    def row(name, r, is_bl=False):
        bpw   = r["bpw"]
        ratio = bpw / 16 * 100
        mask  = r.get("mask_bpw", 0)
        eb    = r.get("exp_bpw", 0)
        mb    = r.get("man_bpw", 0)
        delta = "---" if is_bl else f"{bpw - bl['bpw']:+.3f}"

        detail_parts = []
        if "man_bpw_a" in r:
            detail_parts.append(f"A:{r['man_bpw_a']:.2f} B:{r['man_bpw_b']:.2f}")
        if "sign_bpw" in r:
            detail_parts.append(f"sign:{r['sign_bpw']:.2f} man7:{r.get('man7_bpw',0):.2f}")
        if "threshold" in r:
            detail_parts.append(f"thr={r['threshold']}")
        if "frac_a" in r:
            detail_parts.append(f"A:{r['frac_a']:.1%}")
        detail = "  ".join(detail_parts)

        print(f"  {name:<38} {bpw:>6.3f} {ratio:>6.2f}% "
              f"{mask:>5.2f} {eb:>5.2f} {mb:>5.2f} {delta:>8} {detail}")

    row("DFloat11 (Huf_exp + raw_man)", bl, is_bl=True)
    print()

    # No-grouping references
    print("  -- No grouping (compress mantissa with single table) --")
    row("Huf(exp) + Huf(sign_mantissa)", results["single_both"])
    if "single_split_sm" in results:
        row("Huf(exp) + Huf(sign) + Huf(man7)", results["single_split_sm"])
    print()

    # Dual-group strategies
    print("  -- Dual-group: 1-bit mask + per-group Huffman(exp) + per-group Huffman(man) --")
    strats = [
        ("dual_exp_median",   "Group by exponent median"),
        ("dual_sign",         "Group by sign"),
        ("dual_optimal_thr",  "Group by optimal exp threshold"),
        ("dual_kmeans",       "Group by k-means(exp)"),
        ("dual_man_median",   "Group by mantissa median"),
        ("dual_joint_opt",    "Optimal thr + split sign/man7"),
    ]
    for key, name in strats:
        if key in results:
            row(name, results[key])

    # Mantissa compression potential
    man_raw = 8.0
    man_single = results["single_both"]["man_bpw"]
    man_saved = man_raw - man_single
    print(f"\n  Mantissa insight:")
    print(f"    Global mantissa entropy:  {man_single:.3f} bits (saves {man_saved:.3f} vs raw 8-bit)")
    best_dual_key = None
    best_dual_man = man_single
    for key, _ in strats:
        if key in results and results[key]["man_bpw"] < best_dual_man:
            best_dual_man = results[key]["man_bpw"]
            best_dual_key = key
    if best_dual_key:
        print(f"    Best dual-group mantissa: {best_dual_man:.3f} bits ({best_dual_key})")
        print(f"    Mantissa gain from grouping: {man_single - best_dual_man:.3f} bits/weight")
        net = results[best_dual_key]["bpw"] - bl["bpw"]
        print(f"    Net vs DFloat11:  {net:+.3f} bits/weight ({net/16*100:+.2f}% ratio)")


# =========================================================================
# Extract weights
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
            groups[wt].append(mod.weight.data.detach().cpu().to(torch.bfloat16).flatten())
    return groups


# =========================================================================
# Main
# =========================================================================

def main():
    parser = ArgumentParser("Dual-group Huffman analysis — mantissa compression focus")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--per_layer", action="store_true")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    num_layers = config.num_hidden_layers
    print(f"Model: {args.model_name_or_path}  ({num_layers} layers)")
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    model.eval()

    print("Extracting weights...", flush=True)
    groups = extract_weights(model, num_layers)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = []

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        if args.per_layer:
            for li, t in enumerate(tensors):
                label = f"L{li}/{wt}"
                r = analyze_weights(t, label)
                print_results(r, label)
                summary.append((label, r))
        else:
            label = f"all/{wt}"
            r = analyze_weights(torch.cat(tensors), label)
            print_results(r, label)
            summary.append((label, r))

    # All combined
    all_w = torch.cat([torch.cat(groups[wt]) for wt in WEIGHT_TYPES])
    label = "ALL_COMBINED"
    r = analyze_weights(all_w, label)
    print_results(r, label)
    summary.append((label, r))

    # ================================================================
    # Final summary
    # ================================================================
    print(f"\n\n{'='*115}")
    print("FINAL SUMMARY")
    print(f"{'='*115}")
    print(f"  {'Weight':<25} {'DF11':>7} {'Single':>7} {'BestDual':>9} "
          f"{'delta':>7} {'man_bpw':>8} {'man_save':>9} {'Strategy':<28}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*9} "
          f"{'-'*7} {'-'*8} {'-'*9} {'-'*28}")

    for label, res in summary:
        bl  = res["dfloat11"]["bpw"]
        sh  = res["single_both"]["bpw"]
        man_single = res["single_both"]["man_bpw"]

        best_key, best_bpw = "single_both", sh
        for key in ["dual_exp_median", "dual_sign", "dual_optimal_thr",
                     "dual_kmeans", "dual_man_median", "dual_joint_opt"]:
            if key in res and res[key]["bpw"] < best_bpw:
                best_bpw = res[key]["bpw"]
                best_key = key

        if sh < best_bpw:
            best_bpw = sh
            best_key = "single_both"

        delta = best_bpw - bl
        best_man = res[best_key]["man_bpw"]
        man_save = 8.0 - best_man

        print(f"  {label:<25} {bl:>7.3f} {sh:>7.3f} {best_bpw:>9.3f} "
              f"{delta:>+6.3f} {best_man:>8.3f} {man_save:>+8.3f} {best_key:<28}")

    print(f"""
Legend:
  DF11      = Huffman(exp) + raw(sign_mantissa 8-bit)         [current DFloat11]
  Single    = Huffman(exp) + Huffman(sign_mantissa)            [compress mantissa, no group]
  BestDual  = 1-bit mask + per-group Huffman(exp+mantissa)     [your proposal]
  delta     = BestDual - DF11  (negative = beats DFloat11)
  man_bpw   = mantissa bits/weight in best method  (DFloat11 = 8.000)
  man_save  = 8.0 - man_bpw  (mantissa bits saved per weight)
  1 bit/weight ≈ 6.25% compression ratio improvement
""")


if __name__ == "__main__":
    main()
