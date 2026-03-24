"""
Pilot Experiment: Temporal Prediction Coding for Adam v state.

Idea: v[t+1] = β₂·v[t] + (1-β₂)·grad²
If we can predict v[t+1] exactly from v[t] and grad², we can eliminate v storage.

This script measures:
1. What fraction of v_pred values exactly match actual v_new?
2. Distribution of residuals (v_new - v_pred)
3. Entropy of residuals (compressibility)
"""

import torch
import numpy as np
from collections import Counter
import struct

def bytes_entropy(data_bytes):
    """Compute per-byte entropy of raw bytes."""
    counts = Counter(data_bytes)
    total = len(data_bytes)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * np.log2(p)
    return entropy

def fp32_byte_entropies(tensor):
    """Compute entropy of each byte plane of FP32 tensor."""
    raw = tensor.detach().cpu().numpy().tobytes()
    n = len(raw) // 4
    planes = [bytearray() for _ in range(4)]
    for i in range(n):
        for b in range(4):
            planes[b].append(raw[i*4 + b])
    return [bytes_entropy(p) for p in planes]

def analyze_residuals(residuals, label=""):
    """Analyze residual tensor."""
    r = residuals.detach().cpu()
    total = r.numel()

    # Exact zeros
    exact_zero = (r == 0).sum().item()
    exact_zero_frac = exact_zero / total

    # Statistics
    abs_r = r.abs()

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total elements:     {total:,}")
    print(f"  Exact zero:         {exact_zero:,} ({exact_zero_frac*100:.4f}%)")
    print(f"  Max |residual|:     {abs_r.max().item():.6e}")
    print(f"  Mean |residual|:    {abs_r.mean().item():.6e}")
    print(f"  Median |residual|:  {abs_r.median().item():.6e}")

    # Check relative error
    print(f"\n  Residual magnitude distribution:")
    for threshold in [0, 1e-45, 1e-40, 1e-35, 1e-30, 1e-20, 1e-10]:
        frac = (abs_r <= threshold).sum().item() / total
        print(f"    |r| <= {threshold:.0e}: {frac*100:.4f}%")

    # Byte-plane entropy of residuals
    if total > 0 and not (exact_zero_frac == 1.0):
        entropies = fp32_byte_entropies(r)
        print(f"\n  Byte-plane entropies (FP32 residual):")
        for i, e in enumerate(entropies):
            print(f"    Byte {i}: {e:.4f} bits")
        total_bits = sum(entropies)
        print(f"    Total: {total_bits:.4f} bits/value (vs 32 raw)")
        print(f"    Compression ratio: {total_bits/32*100:.1f}%")

    # Check unique values in residual
    unique = r.unique().numel()
    print(f"\n  Unique residual values: {unique:,} (out of {total:,})")

    # ULP analysis: how many ULPs off?
    if not (exact_zero_frac == 1.0):
        # View as int32 to measure ULP distance
        nonzero_mask = r != 0
        if nonzero_mask.any():
            nonzero_r = r[nonzero_mask]
            r_int = nonzero_r.view(torch.int32)
            ulp_magnitudes = r_int.abs()
            # For small residuals near zero, the int32 view IS the ULP count
            print(f"\n  ULP analysis (nonzero residuals):")
            print(f"    Min ULP magnitude:  {ulp_magnitudes.min().item()}")
            print(f"    Max ULP magnitude:  {ulp_magnitudes.max().item()}")
            print(f"    Mean ULP magnitude: {ulp_magnitudes.float().mean().item():.1f}")


def main():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load GPT-2 small ----
    from transformers import GPT2LMHeadModel, GPT2Config
    print("Loading GPT-2 small...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # ---- Setup optimizer ----
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)

    # ---- Training loop with v prediction analysis ----
    num_steps = 5
    vocab_size = model.config.vocab_size
    seq_len = 128
    batch_size = 4

    for step in range(1, num_steps + 1):
        print(f"\n{'#'*70}")
        print(f"# STEP {step}")
        print(f"{'#'*70}")

        # Save v_old for ALL parameters before step
        v_old_dict = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            state = optimizer.state.get(param)
            if state and "exp_avg_sq" in state:
                v_old_dict[name] = state["exp_avg_sq"].clone()

        # Forward pass with random data
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Save grad² for all parameters
        grad_sq_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sq_dict[name] = (param.grad.detach().clone() ** 2)

        # Optimizer step
        optimizer.step()

        # Now compare predicted v vs actual v
        all_residuals = []
        total_exact = 0
        total_elements = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            state = optimizer.state.get(param)
            if state is None or "exp_avg_sq" not in state:
                continue

            v_new = state["exp_avg_sq"]

            if name in v_old_dict and name in grad_sq_dict:
                v_old = v_old_dict[name]
                g_sq = grad_sq_dict[name]

                # Predict: v_pred = β₂ * v_old + (1-β₂) * grad²
                v_pred = beta2 * v_old + (1.0 - beta2) * g_sq

                residual = v_new - v_pred
                n = residual.numel()
                exact = (residual == 0).sum().item()
                total_exact += exact
                total_elements += n
                all_residuals.append(residual.flatten())
            elif name in grad_sq_dict:
                # First step: v_old = 0, so v_pred = (1-β₂) * grad²
                g_sq = grad_sq_dict[name]
                v_pred = (1.0 - beta2) * g_sq
                v_new_val = state["exp_avg_sq"]

                residual = v_new_val - v_pred
                n = residual.numel()
                exact = (residual == 0).sum().item()
                total_exact += exact
                total_elements += n
                all_residuals.append(residual.flatten())

        print(f"\n  Overall exact match: {total_exact:,} / {total_elements:,} "
              f"({total_exact/total_elements*100:.4f}%)")

        if all_residuals:
            all_res = torch.cat(all_residuals)
            analyze_residuals(all_res, f"Step {step} - All parameters combined")

            # Also check: what if we look at residuals in BF16?
            # (since model might be doing mixed precision internally)
            all_res_bf16 = all_res.to(torch.bfloat16).to(torch.float32)
            bf16_zero = (all_res_bf16 == 0).sum().item()
            print(f"\n  Residuals that round to zero in BF16: {bf16_zero}/{all_res.numel()} "
                  f"({bf16_zero/all_res.numel()*100:.2f}%)")

    # ---- Additional analysis: per-layer breakdown for last step ----
    print(f"\n{'#'*70}")
    print(f"# PER-LAYER BREAKDOWN (last step)")
    print(f"{'#'*70}")

    # Show a few representative layers
    shown = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        state = optimizer.state.get(param)
        if state is None:
            continue
        if name in v_old_dict and name in grad_sq_dict:
            v_old = v_old_dict[name]
            g_sq = grad_sq_dict[name]
            v_pred = beta2 * v_old + (1.0 - beta2) * g_sq
            v_new = state["exp_avg_sq"]
            residual = v_new - v_pred

            n = residual.numel()
            exact = (residual == 0).sum().item()
            max_r = residual.abs().max().item()
            print(f"  {name}: {exact}/{n} exact ({exact/n*100:.2f}%), max|r|={max_r:.3e}")
            shown += 1
            if shown >= 15:
                print("  ... (truncated)")
                break

    # ---- Check: is the residual pattern deterministic? ----
    # Run the same prediction twice to see if residuals are consistent
    print(f"\n{'#'*70}")
    print(f"# DETERMINISM CHECK")
    print(f"{'#'*70}")
    print("If prediction is off by floating-point rounding, residuals should be")
    print("exactly +/- 1 ULP. Let's check the sign pattern of residuals...")

    if all_residuals:
        all_res = torch.cat(all_residuals)
        nonzero = all_res[all_res != 0]
        if nonzero.numel() > 0:
            pos = (nonzero > 0).sum().item()
            neg = (nonzero < 0).sum().item()
            print(f"  Positive residuals: {pos} ({pos/(pos+neg)*100:.1f}%)")
            print(f"  Negative residuals: {neg} ({neg/(pos+neg)*100:.1f}%)")

            # Check if residuals are all 1 ULP
            # For FP32, 1 ULP at different magnitudes is different absolute values
            # But we can check by looking at int32 representation
            nonzero_int = nonzero.view(torch.int32)
            unique_ints = nonzero_int.unique()
            print(f"  Unique int32 representations of nonzero residuals: {unique_ints.numel()}")
            if unique_ints.numel() <= 20:
                print(f"  Values: {unique_ints.tolist()}")


if __name__ == "__main__":
    main()
