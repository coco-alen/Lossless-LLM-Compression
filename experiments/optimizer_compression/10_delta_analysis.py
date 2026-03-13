"""
Experiment 10: Delta Coding Analysis for Optimizer States

Key idea: Adam's moments are exponential moving averages:
  m_t = β₁ * m_{t-1} + (1-β₁) * g_t     (β₁=0.9)
  v_t = β₂ * v_{t-1} + (1-β₂) * g_t²    (β₂=0.999)

Since β₂=0.999, v changes very slowly. In BF16, many elements may stay
the same between steps. If we store XOR deltas, zero-run-length encoding
could compress them significantly.

This analysis measures:
1. What fraction of BF16 values change between consecutive steps?
2. What is the entropy of the XOR delta?
3. How does this change over training?
4. Is sparse storage of changed values practical?
"""

import torch
import numpy as np
import time
import gc
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer


def analyze_delta(before, after, name):
    """Analyze XOR delta between two BF16 tensors."""
    n = before.numel()

    # BF16 → int16 for bitwise ops
    b_int = before.view(torch.int16)
    a_int = after.view(torch.int16)

    xor = b_int ^ a_int
    changed = (xor != 0).sum().item()
    pct_changed = 100 * changed / n

    # Sample for entropy analysis
    sample_size = min(10_000_000, n)
    if n > sample_size:
        idx = torch.randperm(n, device=before.device)[:sample_size]
        xor_sample = xor[idx]
    else:
        xor_sample = xor

    # Entropy of XOR delta (as 16-bit symbol)
    xor_np = xor_sample.cpu().numpy().astype(np.uint16)
    vals, counts = np.unique(xor_np, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log2(probs))

    # Sparse representation cost: for each changed element, store (index, new_value)
    # index: 4 bytes (uint32), value: 2 bytes (bf16) = 6 bytes per change
    sparse_cost = changed * 6
    dense_cost = n * 2  # original bf16

    # XOR delta compression: store XOR values
    # If most are 0, we can use run-length encoding
    xor_cost_entropy = entropy / 16 * n * 2  # entropy-coded XOR values

    print(f"  {name}: {changed:,}/{n:,} changed ({pct_changed:.1f}%), "
          f"XOR entropy={entropy:.4f} bits, "
          f"sparse={sparse_cost/dense_cost*100:.1f}%, "
          f"xor_coded={xor_cost_entropy/dense_cost*100:.1f}%")

    return {
        'changed': changed,
        'total': n,
        'pct_changed': pct_changed,
        'xor_entropy': entropy,
        'sparse_ratio': sparse_cost / dense_cost,
        'xor_coded_ratio': xor_cost_entropy / dense_cost,
    }


def main():
    print("="*80)
    print("Delta Coding Analysis for Optimizer States")
    print("="*80)

    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Train for some steps first
    print("\nWarming up optimizer (50 steps)...")
    for _ in range(50):
        ids = torch.randint(100, 10000, (2, 128), device='cuda')
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Now analyze deltas over several steps
    steps_to_analyze = [1, 5, 10, 20, 50]
    max_steps = max(steps_to_analyze)

    print(f"\nAnalyzing deltas over {max_steps} steps...")

    # Capture initial state
    def capture_states():
        all_m = []
        all_v = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    all_m.append(optimizer.state[p]['exp_avg'].flatten().clone())
                    all_v.append(optimizer.state[p]['exp_avg_sq'].flatten().clone())
        return torch.cat(all_m), torch.cat(all_v)

    prev_m, prev_v = capture_states()
    base_m, base_v = prev_m.clone(), prev_v.clone()

    results = {}
    step_counter = 0

    for target_step in sorted(steps_to_analyze):
        while step_counter < target_step:
            ids = torch.randint(100, 10000, (2, 128), device='cuda')
            model(input_ids=ids, labels=ids).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step_counter += 1

        curr_m, curr_v = capture_states()

        print(f"\n--- After {target_step} additional step(s) ---")

        # Delta from previous step
        if target_step == sorted(steps_to_analyze)[0]:
            # Delta from 1 step before
            r_m = analyze_delta(prev_m, curr_m, f"m (Δ1 step)")
            r_v = analyze_delta(prev_v, curr_v, f"v (Δ1 step)")
        else:
            r_m = analyze_delta(prev_m, curr_m, f"m (Δ from prev checkpoint)")
            r_v = analyze_delta(prev_v, curr_v, f"v (Δ from prev checkpoint)")

        # Delta from base (cumulative)
        r_m_base = analyze_delta(base_m, curr_m, f"m (Δ from base)")
        r_v_base = analyze_delta(base_v, curr_v, f"v (Δ from base)")

        results[target_step] = {
            'm_incr': r_m, 'v_incr': r_v,
            'm_base': r_m_base, 'v_base': r_v_base,
        }

        prev_m, prev_v = curr_m.clone(), curr_v.clone()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Delta Coding Potential")
    print(f"{'='*80}")

    print(f"\n{'State':>5} {'Steps':>6} {'Changed%':>10} {'Sparse%':>10} {'XOR-coded%':>12}")
    print("-" * 50)
    for step in sorted(steps_to_analyze):
        r = results[step]
        for state, key in [('m', 'm_incr'), ('v', 'v_incr')]:
            d = r[key]
            print(f"{state:>5} {step:>6} {d['pct_changed']:>9.1f}% {d['sparse_ratio']*100:>9.1f}% "
                  f"{d['xor_coded_ratio']*100:>11.1f}%")

    print(f"\nCumulative delta from base:")
    print(f"{'State':>5} {'Steps':>6} {'Changed%':>10} {'Sparse%':>10} {'XOR-coded%':>12}")
    print("-" * 50)
    for step in sorted(steps_to_analyze):
        r = results[step]
        for state, key in [('m', 'm_base'), ('v', 'v_base')]:
            d = r[key]
            print(f"{state:>5} {step:>6} {d['pct_changed']:>9.1f}% {d['sparse_ratio']*100:>9.1f}% "
                  f"{d['xor_coded_ratio']*100:>11.1f}%")

    # Practical analysis: can sparse delta storage beat dense transfer?
    print(f"\n--- Practical Analysis ---")
    total_bytes = n_params * 2  # bf16
    for step in [1]:
        r = results[step]
        m_changed = r['m_incr']['pct_changed']
        v_changed = r['v_incr']['pct_changed']
        m_sparse = r['m_incr']['sparse_ratio']
        v_sparse = r['v_incr']['sparse_ratio']

        print(f"After 1 step:")
        print(f"  m: {m_changed:.1f}% elements change → sparse transfer = {m_sparse*100:.1f}% of dense")
        print(f"  v: {v_changed:.1f}% elements change → sparse transfer = {v_sparse*100:.1f}% of dense")

        # For CPU offload: instead of transferring full state, transfer only changes
        # Dense: 2274 MB bidirectional
        # Sparse m: 2274/2 * m_sparse MB + index overhead
        # Sparse v: 2274/2 * v_sparse MB + index overhead
        dense_mb = n_params * 2 / 1024 / 1024
        sparse_m_mb = dense_mb * m_sparse
        sparse_v_mb = dense_mb * v_sparse
        print(f"  Dense transfer: {dense_mb*2:.0f} MB (m+v bidirectional)")
        print(f"  Sparse transfer: {(sparse_m_mb + sparse_v_mb)*2:.0f} MB "
              f"({(sparse_m_mb + sparse_v_mb)*2 / (dense_mb*2) * 100:.1f}% of dense)")


if __name__ == '__main__':
    main()
