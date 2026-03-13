"""
Experiment 11: FP32 Mixed-Precision Baseline + In-Memory Compression

CORRECT SETUP: Standard mixed-precision training stores:
- FP32 master weights (4Ψ bytes)
- FP32 m first moment (4Ψ bytes)
- FP32 v second moment (4Ψ bytes)
- Total optimizer memory: 12Ψ bytes
- BF16 working copy: 2Ψ bytes (not stored separately if we use autocast)

The model params ARE fp32, forward pass uses autocast to bf16.

Goal: Losslessly compress m and v (8Ψ bytes) in GPU memory.
From entropy analysis: FP32 states have ~23.2 bits/32 = 72.5% theoretical limit.

Approach: Split FP32 into (high16, low16). Compress high16 using ANS/Huffman
since it contains the exponent (low entropy). Store low16 raw.
Also measure: byte3-only compression as simpler alternative.
"""

import torch
import time
import gc
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def measure_fp32_baseline(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    """Measure standard FP32 mixed-precision training baseline."""
    print("=" * 80)
    print("FP32 Mixed-Precision Training Baseline")
    print("=" * 80)

    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_before = torch.cuda.memory_allocated() / 1024**2

    # Load model in FP32 (as in real mixed-precision training)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_model = torch.cuda.memory_allocated() / 1024**2

    print(f"Model: {model_name} ({n_params/1e6:.1f}M params)")
    print(f"Model memory (FP32): {mem_model - mem_before:.1f} MB")
    print(f"Expected: {n_params * 4 / 1024**2:.1f} MB")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Warmup step with autocast
    ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(input_ids=ids, labels=ids)
    outputs.loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    gc.collect(); torch.cuda.empty_cache()
    mem_after_step = torch.cuda.memory_allocated() / 1024**2

    # Verify optimizer state dtypes
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                print(f"\nOptimizer state dtypes: exp_avg={state['exp_avg'].dtype}, "
                      f"exp_avg_sq={state['exp_avg_sq'].dtype}")
                print(f"Parameter dtype: {p.dtype}")
                break
        break

    # Count optimizer state memory
    state_elements = 0
    state_bytes = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state_bytes += v.numel() * v.element_size()
                        state_elements += v.numel()

    print(f"\nOptimizer state memory: {state_bytes/1024**2:.1f} MB")
    print(f"Optimizer state elements: {state_elements:,}")
    print(f"Expected (m+v, FP32): {n_trainable * 4 * 2 / 1024**2:.1f} MB")
    print(f"Model (FP32): {mem_model - mem_before:.1f} MB")
    print(f"Total GPU (model+optimizer): {mem_after_step - mem_before:.1f} MB")

    # Speed benchmark
    print(f"\n--- Speed Benchmark ---")
    n_warmup, n_measure = 5, 30

    for _ in range(n_warmup):
        ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step(); optimizer.zero_grad()

    torch.cuda.synchronize()
    times = []
    for _ in range(n_measure):
        ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step(); optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    # Optimizer-only timing
    opt_times = []
    for _ in range(20):
        ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(input_ids=ids, labels=ids).loss.backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        opt_times.append(time.perf_counter() - t0)
        optimizer.zero_grad()

    avg_step = sum(times) / len(times)
    avg_opt = sum(opt_times) / len(opt_times)
    peak = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Step: {avg_step*1000:.1f} ms")
    print(f"Optimizer step: {avg_opt*1000:.1f} ms ({avg_opt/avg_step*100:.1f}%)")
    print(f"Tokens/sec: {batch_size * seq_len / avg_step:.0f}")
    print(f"Peak memory: {peak:.1f} MB")

    # Detailed entropy of actual FP32 states
    print(f"\n--- FP32 State Entropy (at step 10) ---")
    all_m, all_v = [], []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                all_m.append(optimizer.state[p]['exp_avg'].flatten())
                all_v.append(optimizer.state[p]['exp_avg_sq'].flatten())

    for name, tensors in [("m", all_m), ("v", all_v)]:
        cat = torch.cat(tensors)
        n = min(10_000_000, cat.numel())
        sample = cat[:n].cpu()

        # View as bytes
        raw = sample.numpy().tobytes()
        barr = np.frombuffer(raw, dtype=np.uint8).reshape(n, 4)

        byte_entropies = []
        for i in range(4):
            col = barr[:, i]
            vals, counts = np.unique(col, return_counts=True)
            probs = counts / counts.sum()
            e = -np.sum(probs * np.log2(probs))
            byte_entropies.append(e)

        # High-16 entropy
        u32 = np.frombuffer(raw, dtype=np.uint32)
        high16 = (u32 >> 16).astype(np.uint16)
        low16 = (u32 & 0xFFFF).astype(np.uint16)

        vals_h, counts_h = np.unique(high16, return_counts=True)
        h16_entropy = -np.sum(counts_h / counts_h.sum() * np.log2(counts_h / counts_h.sum()))
        n_unique_h16 = len(vals_h)

        vals_l, counts_l = np.unique(low16, return_counts=True)
        l16_entropy = -np.sum(counts_l / counts_l.sum() * np.log2(counts_l / counts_l.sum()))

        # Full 32-bit
        vals32, counts32 = np.unique(u32, return_counts=True)
        full_entropy = -np.sum(counts32 / counts32.sum() * np.log2(counts32 / counts32.sum()))

        print(f"\n  {name} (FP32):")
        print(f"    Byte entropies: [{byte_entropies[0]:.3f}, {byte_entropies[1]:.3f}, "
              f"{byte_entropies[2]:.3f}, {byte_entropies[3]:.3f}]")
        print(f"    Byte sum: {sum(byte_entropies):.3f}/32 = {sum(byte_entropies)/32*100:.1f}%")
        print(f"    High-16 entropy: {h16_entropy:.3f}/16 ({n_unique_h16:,} unique) = {h16_entropy/16*100:.1f}%")
        print(f"    Low-16 entropy: {l16_entropy:.3f}/16 = {l16_entropy/16*100:.1f}%")
        print(f"    High16+Low16: {h16_entropy+l16_entropy:.3f}/32 = {(h16_entropy+l16_entropy)/32*100:.1f}%")
        print(f"    Full 32-bit: {full_entropy:.3f}/32 = {full_entropy/32*100:.1f}%")
        print(f"    Savings from high16 ANS: {(1 - (h16_entropy/16*2 + 2)/4)*100:.1f}%")

    del model, optimizer
    gc.collect(); torch.cuda.empty_cache()

    return {
        'n_params': n_params,
        'model_mem_mb': mem_model - mem_before,
        'state_mem_mb': state_bytes / 1024**2,
        'total_mem_mb': mem_after_step - mem_before,
        'step_ms': avg_step * 1000,
        'opt_ms': avg_opt * 1000,
        'peak_mb': peak,
    }


if __name__ == '__main__':
    measure_fp32_baseline()
