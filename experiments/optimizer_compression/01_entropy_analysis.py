"""
Experiment 1: Entropy Analysis of Optimizer States

Train a model, then analyze byte-level entropy of AdamW's m, v states.
Uses np.unique for fast entropy computation and samples if tensors are too large.

Key question: PyTorch stores optimizer states in the same dtype as params.
For bf16 model → bf16 optimizer states (2 bytes each).
For mixed-precision with fp32 master weights → fp32 optimizer states (4 bytes each).
We analyze BOTH scenarios.
"""

import torch
import torch.nn as nn
import numpy as np
import math
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def entropy_from_array(arr: np.ndarray) -> float:
    """Fast entropy computation using np.unique."""
    vals, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def analyze_tensor_entropy(tensor: torch.Tensor, name: str, max_elements=10_000_000):
    """Analyze byte-level and symbol-level entropy of a tensor."""
    # Sample if too large
    t = tensor.detach().cpu().flatten()
    if t.numel() > max_elements:
        indices = torch.randperm(t.numel())[:max_elements]
        t = t[indices]

    n = t.numel()
    dtype = t.dtype
    bytes_per_elem = t.element_size()

    print(f"\n  {name} ({tensor.numel():,} elements, dtype={dtype}, sampled={n:,})")
    print(f"    Range: [{t.min().item():.6e}, {t.max().item():.6e}]")
    print(f"    Mean: {t.mean().float().item():.6e}, Std: {t.std().float().item():.6e}")
    print(f"    Zeros: {(t == 0).sum().item()} ({100*(t == 0).sum().item()/n:.1f}%)")

    # Convert to raw bytes (bf16 needs special handling)
    if dtype == torch.bfloat16:
        # View bf16 as uint16 to get raw bytes
        raw = t.view(torch.uint16).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    byte_arr = np.frombuffer(raw, dtype=np.uint8).reshape(n, bytes_per_elem)

    total_entropy = 0
    for i in range(bytes_per_elem):
        col = byte_arr[:, i]
        e = entropy_from_array(col)
        n_unique = len(np.unique(col))
        total_entropy += e
        print(f"    byte{i}: entropy={e:.4f} bits, {n_unique} unique values")

    print(f"    Sum of byte entropies: {total_entropy:.4f} / {bytes_per_elem*8} bits")
    print(f"    Byte-wise compression ratio: {total_entropy/(bytes_per_elem*8)*100:.2f}%")

    # Full symbol entropy
    if bytes_per_elem == 2:
        uint_data = np.frombuffer(raw, dtype=np.uint16)
    else:
        uint_data = np.frombuffer(raw, dtype=np.uint32)

    full_entropy = entropy_from_array(uint_data)
    n_unique_symbols = len(np.unique(uint_data))
    print(f"    Full {bytes_per_elem*8}-bit entropy: {full_entropy:.4f} bits, {n_unique_symbols:,} unique symbols")
    print(f"    Symbol-wise compression ratio: {full_entropy/(bytes_per_elem*8)*100:.2f}%")

    # Exponent analysis
    if bytes_per_elem == 4:  # FP32
        exponents = (uint_data >> 23) & 0xFF
    elif bytes_per_elem == 2:  # BF16
        exponents = (uint_data >> 7) & 0xFF
    else:
        exponents = None

    if exponents is not None:
        exp_entropy = entropy_from_array(exponents)
        n_unique_exp = len(np.unique(exponents))
        print(f"    Exponent (8-bit): entropy={exp_entropy:.4f}, {n_unique_exp} unique")

    return {
        'byte_entropy_sum': total_entropy,
        'full_entropy': full_entropy,
        'compression_ratio_bytes': total_entropy / (bytes_per_elem * 8),
        'compression_ratio_symbol': full_entropy / (bytes_per_elem * 8),
    }


def run_analysis(model_name="Qwen/Qwen3-0.6B"):
    print("=" * 80)
    print(f"Entropy Analysis of Optimizer States: {model_name}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ====== Scenario 1: BF16 model → BF16 optimizer states ======
    print("\n\n" + "=" * 80)
    print("SCENARIO 1: BF16 model with BF16 optimizer states (default PyTorch)")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    steps_to_analyze = [10, 100, 500]
    results = {}

    for step in range(1, max(steps_to_analyze) + 1):
        input_ids = torch.randint(100, 10000, (2, 128), device='cuda')
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step in steps_to_analyze:
            print(f"\n{'='*60}")
            print(f"Step {step} (loss={outputs.loss.item():.4f})")
            print(f"{'='*60}")

            all_m = []
            all_v = []
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p in optimizer.state:
                        state = optimizer.state[p]
                        if 'exp_avg' in state:
                            all_m.append(state['exp_avg'].flatten())
                        if 'exp_avg_sq' in state:
                            all_v.append(state['exp_avg_sq'].flatten())

            if all_m:
                m_cat = torch.cat(all_m)
                v_cat = torch.cat(all_v)
                r_m = analyze_tensor_entropy(m_cat, f"m (first moment)")
                r_v = analyze_tensor_entropy(v_cat, f"v (second moment)")
                results[f'bf16_step{step}'] = {'m': r_m, 'v': r_v}
                del m_cat, v_cat
            del all_m, all_v

    # Check actual optimizer state dtypes
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                print(f"\nOptimizer state dtypes: exp_avg={state['exp_avg'].dtype}, exp_avg_sq={state['exp_avg_sq'].dtype}")
                break
        break

    del model, optimizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ====== Scenario 2: FP32 master weights + FP32 optimizer states ======
    print("\n\n" + "=" * 80)
    print("SCENARIO 2: FP32 master weights + FP32 optimizer states (mixed-precision)")
    print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    model.train()
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M) in FP32")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    for step in range(1, max(steps_to_analyze) + 1):
        input_ids = torch.randint(100, 10000, (2, 128), device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step in steps_to_analyze:
            print(f"\n{'='*60}")
            print(f"Step {step} (loss={outputs.loss.item():.4f})")
            print(f"{'='*60}")

            all_m = []
            all_v = []
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p in optimizer.state:
                        state = optimizer.state[p]
                        if 'exp_avg' in state:
                            all_m.append(state['exp_avg'].flatten())
                        if 'exp_avg_sq' in state:
                            all_v.append(state['exp_avg_sq'].flatten())

            if all_m:
                m_cat = torch.cat(all_m)
                v_cat = torch.cat(all_v)
                r_m = analyze_tensor_entropy(m_cat, f"m (first moment)")
                r_v = analyze_tensor_entropy(v_cat, f"v (second moment)")
                results[f'fp32_step{step}'] = {'m': r_m, 'v': r_v}

                # Also analyze master weights
                all_w = [p.data.flatten() for p in model.parameters()]
                w_cat = torch.cat(all_w)
                r_w = analyze_tensor_entropy(w_cat, "master weights (FP32)")
                results[f'fp32_step{step}']['w'] = r_w
                del m_cat, v_cat, w_cat, all_w
            del all_m, all_v

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for key, val in sorted(results.items()):
        print(f"\n{key}:")
        for state_name, r in val.items():
            print(f"  {state_name}: byte-wise={r['compression_ratio_bytes']*100:.2f}%, "
                  f"symbol-wise={r['compression_ratio_symbol']*100:.2f}%")


if __name__ == '__main__':
    run_analysis()
