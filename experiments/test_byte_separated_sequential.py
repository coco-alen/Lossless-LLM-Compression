"""
Byte-separated sequential ANS coding.

Key insights from previous experiments:
1. Full 16-bit ANS is at i.i.d. entropy limit (65.96%)
2. Adjacent weights are correlated (bigram H = 9.91 bpw vs 10.55 bpw)
3. Prev-exp conditioning barely helps (10.614 vs 10.630)
4. Exponent has low entropy (2.7 bpw) and might be sequentially predictable
5. Sign+mantissa is ~7.97 bpw (nearly random)

New approach: separate into exponent and sign+mantissa streams, then:
- Sequentially encode exponents with prev-exp context (exp has only ~30 unique values)
- Encode sign+mantissa conditioned on current exponent (slight correlation)
- Try column-major ordering to increase correlation

Also try: zstd on the exponent stream (LZ77 can capture long-range exp patterns)
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


def conditional_entropy(values, contexts):
    """H(values | contexts)"""
    n = len(values)
    unique_ctx = np.unique(contexts)
    total_h = 0
    for c in unique_ctx:
        mask = contexts == c
        n_c = mask.sum()
        if n_c < 2:
            continue
        v_c = values[mask]
        vals, counts = np.unique(v_c, return_counts=True)
        p = counts / n_c
        h = -np.sum(p * np.log2(p))
        total_h += (n_c / n) * h
    return total_h


def ans_encode_size(data, probs, mapping, vals):
    """Encode data with pre-computed probability table, return compressed bytes."""
    data_idx = mapping[(data.astype(np.int32) + 32768)].astype(np.int32)
    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(data_idx, model)
    compressed = encoder.get_compressed()
    return len(compressed) * 4 + len(vals) * 6


def ans_encode_uint8(data):
    """ANS encode uint8 data, return compressed bytes."""
    vals, counts = np.unique(data, return_counts=True)
    n = len(data)
    probs = (counts / n).astype(np.float32)
    probs = np.maximum(probs, 1e-10).astype(np.float32)
    probs /= probs.sum()

    mapping = np.zeros(256, dtype=np.int32)
    for i, v in enumerate(vals):
        mapping[v] = i
    data_idx = mapping[data].astype(np.int32)

    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(data_idx, model)
    compressed = encoder.get_compressed()
    return len(compressed) * 4 + len(vals) * 5


def sequential_exp_ans_size(tensors, row_sequential=True):
    """
    Encode exponents sequentially: ANS(exp[i] | exp[i-1]).
    With ~30 unique exponents, we have ~30 context tables, each with ~30 symbols.
    Overhead: 30 × 30 × 5 = 4.5KB — negligible!

    Then encode sign+mantissa conditioned on current exponent.
    """
    # Collect sequences
    all_exp_pairs = {}  # prev_exp -> list of curr_exp
    all_sm_by_exp = {}  # curr_exp -> list of sm
    first_exps = []
    first_sms = []

    for t in tensors:
        W = t.contiguous().view(torch.int16)
        exp = ((W >> 7) & 0xFF).to(torch.uint8).numpy()
        sm = ((W >> 8) & 0x80 | (W & 0x7F)).to(torch.uint8).numpy()

        if row_sequential:
            for row_idx in range(exp.shape[0]):
                exp_row = exp[row_idx]
                sm_row = sm[row_idx]

                first_exps.append(exp_row[0])
                first_sms.append(sm_row[0])

                for i in range(1, len(exp_row)):
                    pe = int(exp_row[i-1])
                    ce = int(exp_row[i])
                    if pe not in all_exp_pairs:
                        all_exp_pairs[pe] = []
                    all_exp_pairs[pe].append(ce)

                # sm conditioned on current exp
                for i in range(len(exp_row)):
                    e = int(exp_row[i])
                    if e not in all_sm_by_exp:
                        all_sm_by_exp[e] = []
                    all_sm_by_exp[e].append(int(sm_row[i]))

    total_bytes = 0

    # 1. Encode first exponents (global distribution)
    first_exps = np.array(first_exps, dtype=np.uint8)
    total_bytes += ans_encode_uint8(first_exps)

    # 2. Encode sequential exponents per context
    for prev_e, curr_list in sorted(all_exp_pairs.items()):
        curr = np.array(curr_list, dtype=np.uint8)
        total_bytes += ans_encode_uint8(curr)

    # 3. Encode sign+mantissa per exponent
    for e, sm_list in sorted(all_sm_by_exp.items()):
        sm_arr = np.array(sm_list, dtype=np.uint8)
        total_bytes += ans_encode_uint8(sm_arr)

    return total_bytes


def sequential_exp_then_16bit_sm_size(tensors):
    """
    Hybrid approach:
    - Sequential ANS for exponents (with prev-exp context)
    - 16-bit ANS for full value (but grouped by exponent)

    This separates the easy part (exp) from the hard part (sm),
    using context for the easy part and i.i.d. for the hard part.

    Wait, this doesn't make sense as decomposed. Let's try:
    - Sequential coding of exponent with prev-exp context
    - Standard ANS for sign+mantissa (globally, no context)
    """
    # Sequential exponent coding
    all_exp_pairs = {}
    first_exps = []

    for t in tensors:
        exp = ((t.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).numpy()
        for row_idx in range(exp.shape[0]):
            row = exp[row_idx]
            first_exps.append(row[0])
            for i in range(1, len(row)):
                pe = int(row[i-1])
                if pe not in all_exp_pairs:
                    all_exp_pairs[pe] = []
                all_exp_pairs[pe].append(int(row[i]))

    exp_bytes = 0
    first_exps = np.array(first_exps, dtype=np.uint8)
    exp_bytes += ans_encode_uint8(first_exps)
    for prev_e, curr_list in sorted(all_exp_pairs.items()):
        curr = np.array(curr_list, dtype=np.uint8)
        exp_bytes += ans_encode_uint8(curr)

    # Standard ANS for sign+mantissa
    all_sm = torch.cat([((t.contiguous().view(torch.int16) >> 8) & 0x80 |
                          (t.contiguous().view(torch.int16) & 0x7F)).to(torch.uint8).flatten()
                         for t in tensors])
    sm = all_sm.numpy()
    sm_bytes = ans_encode_uint8(sm)

    return exp_bytes, sm_bytes, exp_bytes + sm_bytes


def zstd_exp_plus_ans_sm_size(tensors):
    """Use zstd for exponent stream (captures LZ77 patterns) + ANS for sm."""
    try:
        import zstandard as zstd
    except ImportError:
        return None

    # Exponent stream
    all_exp = torch.cat([((t.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).flatten()
                          for t in tensors])
    exp_bytes_raw = all_exp.numpy().tobytes()

    cctx = zstd.ZstdCompressor(level=19)
    exp_compressed = cctx.compress(exp_bytes_raw)
    exp_bytes = len(exp_compressed)

    # SM stream with ANS
    all_sm = torch.cat([((t.contiguous().view(torch.int16) >> 8) & 0x80 |
                          (t.contiguous().view(torch.int16) & 0x7F)).to(torch.uint8).flatten()
                         for t in tensors])
    sm = all_sm.numpy()
    sm_bytes = ans_encode_uint8(sm)

    return exp_bytes, sm_bytes, exp_bytes + sm_bytes


def column_major_ans16_size(tensors):
    """Reorder weights to column-major (transpose) then do ANS-16bit."""
    all_w = torch.cat([t.T.contiguous().view(torch.int16).flatten() for t in tensors])
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


def column_major_entropy(tensors):
    """Entropy with column-major ordering."""
    all_w = torch.cat([t.T.contiguous().view(torch.int16).flatten() for t in tensors])
    W = all_w.numpy()
    return iid_entropy(W)


def column_major_bigram_entropy(tensors, max_pairs=5_000_000):
    """Bigram entropy with column-major ordering."""
    all_w = torch.cat([t.T.contiguous().view(torch.int16).flatten() for t in tensors])
    W = all_w.numpy()
    if len(W) > max_pairs + 1:
        W = W[:max_pairs + 1]

    W_u = (W.astype(np.int32) + 32768).astype(np.uint16)
    prev = W_u[:-1]
    curr = W_u[1:]

    sort_idx = np.argsort(prev)
    prev_sorted = prev[sort_idx]
    curr_sorted = curr[sort_idx]

    boundaries = np.where(np.diff(prev_sorted) != 0)[0] + 1
    boundaries = np.concatenate([[0], boundaries, [len(prev_sorted)]])

    total_cond_entropy = 0.0
    total_count = len(prev)

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        group_size = end - start
        group = curr_sorted[start:end]
        vals, counts = np.unique(group, return_counts=True)
        p = counts / group_size
        h = -np.sum(p * np.log2(p))
        total_cond_entropy += (group_size / total_count) * h

    return total_cond_entropy


def zstd_on_ans16_residuals(tensors):
    """
    Try: ANS-16bit encodes well, but can zstd further compress the ANS output?
    (Usually no, because ANS output is near-entropy, but worth checking.)
    """
    try:
        import zstandard as zstd
    except ImportError:
        return None

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

    # Try zstd on the ANS output
    ans_bytes = np.array(compressed, dtype=np.uint32).tobytes()
    cctx = zstd.ZstdCompressor(level=19)
    zstd_of_ans = cctx.compress(ans_bytes)

    return len(ans_bytes), len(zstd_of_ans)


def per_layer_sequential_exp_entropy(tensors):
    """H(exp[i] | exp[i-1]) computed per-layer (within each layer's weight matrix)."""
    total_h_seq = 0
    total_h_iid = 0
    total_n = 0

    for t in tensors:
        exp = ((t.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).numpy()
        flat_exp = exp.flatten()
        n = len(flat_exp)
        total_n += n

        # i.i.d. entropy of exp
        h_iid = iid_entropy(flat_exp)
        total_h_iid += h_iid * n

        # Sequential entropy of exp (row by row)
        for row_idx in range(exp.shape[0]):
            row = exp[row_idx]
            if len(row) < 2:
                continue
            prev = row[:-1]
            curr = row[1:]

            # Group by prev value
            for pe in np.unique(prev):
                mask = prev == pe
                curr_sub = curr[mask]
                n_sub = len(curr_sub)
                vals, counts = np.unique(curr_sub, return_counts=True)
                p = counts / n_sub
                h = -np.sum(p * np.log2(p))
                total_h_seq += h * n_sub

    return total_h_iid / total_n, total_h_seq / total_n


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
    totals = {}

    print(f"\n{'='*95}")
    print("BYTE-SEPARATED SEQUENTIAL ANS + ORDERING ANALYSIS")
    print(f"{'='*95}")

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)
        original = n * 2
        total_original += original

        print(f"\n  {wt}  ({n:,} params)")

        # Standard ANS-16bit (baseline)
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
        W = all_w.numpy()
        h_iid = iid_entropy(W)

        vals, counts = np.unique(W, return_counts=True)
        probs = (counts / n).astype(np.float32)
        mapping = np.zeros(65536, dtype=np.int32)
        for i, v in enumerate(vals):
            mapping[int(v) + 32768] = i
        data_idx = mapping[(W.astype(np.int32) + 32768)].astype(np.int32)
        model_ans = constriction.stream.model.Categorical(probs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(data_idx, model_ans)
        compressed = encoder.get_compressed()
        ans16 = len(compressed) * 4 + len(vals) * 6
        totals.setdefault("ans16", 0)
        totals["ans16"] += ans16
        print(f"    ANS-16bit:                {ans16/original*100:.3f}%")

        # Sequential exponent entropy
        h_exp_iid, h_exp_seq = per_layer_sequential_exp_entropy(tensors)
        print(f"    Exp entropy:  iid={h_exp_iid:.4f}  seq={h_exp_seq:.4f}  savings={h_exp_iid-h_exp_seq:.4f} bpw")

        # Sequential exp + conditioned sm (actual ANS)
        t0 = time.time()
        seq_exp_bytes, sm_bytes, seq_total = sequential_exp_then_16bit_sm_size(tensors)
        t1 = time.time()
        totals.setdefault("seq_exp_ans_sm", 0)
        totals["seq_exp_ans_sm"] += seq_total
        delta = seq_total/original*100 - ans16/original*100
        print(f"    Seq-exp + ANS(sm):        {seq_total/original*100:.3f}%  "
              f"(exp={seq_exp_bytes/original*100:.2f}% sm={sm_bytes/original*100:.2f}%)  "
              f"delta={delta:+.3f}%  [{t1-t0:.1f}s]")

        # Full sequential exp + per-exp sm
        t0 = time.time()
        seq_full = sequential_exp_ans_size(tensors)
        t1 = time.time()
        totals.setdefault("seq_exp_percond_sm", 0)
        totals["seq_exp_percond_sm"] += seq_full
        delta = seq_full/original*100 - ans16/original*100
        print(f"    Seq-exp + per-exp(sm):    {seq_full/original*100:.3f}%  "
              f"delta={delta:+.3f}%  [{t1-t0:.1f}s]")

        # zstd(exp) + ANS(sm)
        result = zstd_exp_plus_ans_sm_size(tensors)
        if result:
            zstd_exp, zstd_sm, zstd_total = result
            totals.setdefault("zstd_exp_ans_sm", 0)
            totals["zstd_exp_ans_sm"] += zstd_total
            delta = zstd_total/original*100 - ans16/original*100
            print(f"    zstd-19(exp) + ANS(sm):   {zstd_total/original*100:.3f}%  "
                  f"(exp={zstd_exp/original*100:.2f}% sm={zstd_sm/original*100:.2f}%)  "
                  f"delta={delta:+.3f}%  ")

        # Column-major ordering
        t0 = time.time()
        col_ans16 = column_major_ans16_size(tensors)
        t1 = time.time()
        totals.setdefault("col_ans16", 0)
        totals["col_ans16"] += col_ans16
        delta = col_ans16/original*100 - ans16/original*100
        print(f"    Column-major ANS-16:      {col_ans16/original*100:.3f}%  "
              f"delta={delta:+.3f}%  [{t1-t0:.1f}s]")

        # Column-major bigram entropy
        t0 = time.time()
        col_bigram_h = column_major_bigram_entropy(tensors)
        t1 = time.time()
        row_bigram_bytes = int(np.ceil(col_bigram_h * n / 8))
        print(f"    Col-major bigram H:       {col_bigram_h:.4f} bpw  -> {row_bigram_bytes/original*100:.3f}%  [{t1-t0:.1f}s]")

        # zstd on ANS output
        result2 = zstd_on_ans16_residuals(tensors)
        if result2:
            ans_raw, zstd_of_ans = result2
            print(f"    zstd(ANS output):         {zstd_of_ans/original*100:.3f}%  (ANS raw={ans_raw/original*100:.3f}%)")

    # Summary
    print(f"\n{'='*95}")
    print("OVERALL SUMMARY")
    print(f"{'='*95}")

    for name, key in [("ANS-16bit", "ans16"),
                       ("Seq-exp + ANS(sm)", "seq_exp_ans_sm"),
                       ("Seq-exp + per-exp(sm)", "seq_exp_percond_sm"),
                       ("zstd-19(exp) + ANS(sm)", "zstd_exp_ans_sm"),
                       ("Column-major ANS-16", "col_ans16")]:
        if key in totals:
            size = totals[key]
            ratio = size / total_original * 100
            delta = ratio - totals["ans16"] / total_original * 100
            marker = " ***" if delta < -0.01 else ""
            print(f"  {name:<28} {ratio:.3f}%  ({size/1e6:.1f}MB)  delta={delta:+.3f}%{marker}")

    print(f"  Original bf16:               {total_original/1e6:.1f}MB")


if __name__ == "__main__":
    main()
