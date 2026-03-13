"""
Experiment 6: Investigating Fused Compressed Optimizer Step

Key question: Can we avoid the full decompress→compute→recompress cycle?

Observation: In a standard AdamW step, the compute is:
  m = β₁ * m + (1-β₁) * g
  v = β₂ * v + (1-β₂) * g²
  p = p - lr * m / (√v + ε)

The m/v updates are simple EMA operations. Can we do EMA directly on compressed data?

No — EMA requires full-precision arithmetic. But we CAN:
1. Keep m/v in a compressed format that's cheap to convert
2. Convert only when needed (during optimizer step)
3. The optimizer step is only ~8.5% of total time

This experiment measures the actual compress/decompress overhead to understand
what compression methods are practical given the time budget.

Time budget: optimizer step = ~14ms
If compression adds <5ms (35% overhead on step, <3% on total), it's acceptable.
"""

import torch
import time
import gc
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_compress_decompress_ops(n_elements=100_000_000):
    """Benchmark various compress/decompress operations on GPU."""
    print(f"\nBenchmarking compress/decompress on {n_elements/1e6:.0f}M elements")
    print("=" * 80)

    device = 'cuda'

    # Create realistic BF16 optimizer state data
    data_bf16 = torch.randn(n_elements, device=device, dtype=torch.bfloat16) * 0.001

    # ---- Operation 1: BF16 ↔ uint16 view (zero cost) ----
    torch.cuda.synchronize()
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        uint16_view = data_bf16.view(torch.int16)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"  BF16→uint16 view: {min(times)*1000:.3f} ms (zero-copy)")

    # ---- Operation 2: Extract high/low bytes ----
    int16_data = data_bf16.view(torch.int16)
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        low = (int16_data & 0xFF).to(torch.uint8)
        high = ((int16_data >> 8) & 0xFF).to(torch.uint8)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"  Extract high+low bytes: {min(times)*1000:.3f} ms")
    del low, high

    # ---- Operation 3: Reconstruct from high/low bytes ----
    low = (int16_data & 0xFF).to(torch.uint8)
    high = ((int16_data >> 8) & 0xFF).to(torch.uint8)
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        reconstructed = (high.to(torch.int16) << 8) | low.to(torch.int16)
        result = reconstructed.view(torch.bfloat16)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"  Reconstruct from bytes: {min(times)*1000:.3f} ms")

    # Verify lossless
    assert torch.all(data_bf16 == result), "Reconstruction failed!"
    del low, high, reconstructed, result

    # ---- Operation 4: High-byte codebook lookup + pack ----
    high = ((int16_data >> 8) & 0xFF).to(torch.uint8)
    unique = torch.unique(high)
    n_unique = len(unique)
    print(f"\n  High byte: {n_unique} unique values")

    # Build lookup table (always, for later use)
    lut = torch.zeros(256, dtype=torch.uint8, device=device)
    for idx, val in enumerate(unique):
        lut[val] = idx

    if n_unique <= 16:

        torch.cuda.synchronize()
        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            indices = lut[high.long()]
            # Pack 2 per byte
            n_padded = (n_elements + 1) // 2 * 2
            if n_elements % 2 != 0:
                padded = torch.cat([indices, torch.zeros(1, dtype=torch.uint8, device=device)])
            else:
                padded = indices
            packed = (padded[0::2] << 4) | padded[1::2]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        print(f"  Pack high bytes (codebook + 4-bit): {min(times)*1000:.3f} ms")

        # Unpack
        torch.cuda.synchronize()
        times = []
        for _ in range(50):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            hi = (packed >> 4) & 0x0F
            lo_idx = packed & 0x0F
            unpacked = torch.zeros(len(packed) * 2, dtype=torch.uint8, device=device)
            unpacked[0::2] = hi
            unpacked[1::2] = lo_idx
            unpacked = unpacked[:n_elements]
            high_restored = unique[unpacked.long()]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        print(f"  Unpack high bytes: {min(times)*1000:.3f} ms")

        assert torch.all(high == high_restored), "Pack/unpack failed!"
        del indices, packed, hi, lo_idx, unpacked, high_restored

    # ---- Operation 5: Full round-trip (extract + pack + unpack + reconstruct) ----
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Compress
        u16 = data_bf16.view(torch.int16)
        low = (u16 & 0xFF).to(torch.uint8)
        high = ((u16 >> 8) & 0xFF).to(torch.uint8)
        idx = lut[high.long()]
        if n_elements % 2 != 0:
            padded_idx = torch.cat([idx, torch.zeros(1, dtype=torch.uint8, device=device)])
        else:
            padded_idx = idx
        pk = (padded_idx[0::2] << 4) | padded_idx[1::2]

        # Decompress
        hi = (pk >> 4) & 0x0F
        lo_idx = pk & 0x0F
        unpacked = torch.zeros(len(pk) * 2, dtype=torch.uint8, device=device)
        unpacked[0::2] = hi
        unpacked[1::2] = lo_idx
        unpacked = unpacked[:n_elements]
        h_restored = unique[unpacked.long()]
        recon = (h_restored.to(torch.int16) << 8) | low.to(torch.int16)
        result = recon.view(torch.bfloat16)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    print(f"\n  Full compress+decompress round-trip: {min(times)*1000:.3f} ms")
    print(f"  Memory saved: {n_elements * 0.5 / 1024 / 1024:.1f} MB "
          f"({n_elements * 2 / 1024 / 1024:.1f} → {n_elements * 1.5 / 1024 / 1024:.1f} MB)")

    assert torch.all(data_bf16 == result), "Full round-trip failed!"
    del low, high, pk, result

    # ---- Operation 6: BF16 → FP32 → BF16 round trip (for comparison) ----
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fp32 = data_bf16.float()
        back = fp32.bfloat16()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    print(f"\n  BF16→FP32→BF16 (reference): {min(times)*1000:.3f} ms")

    # ---- Operation 7: Simple bitwise truncation idea ----
    # What if we just store the exponent separately?
    # BF16 = sign(1) + exp(8) + mantissa(7)
    # Exponent has ~3.8 bits entropy out of 8
    # We could store exponent as variable-length codes...
    # But variable-length on GPU is exactly what DFloat11 does!

    print(f"\n--- Summary for {n_elements/1e6:.0f}M elements ---")
    print(f"  Optimizer step budget: ~14 ms")
    print(f"  For m+v (2x), multiply times by 2")

    gc.collect()
    torch.cuda.empty_cache()

    # ---- Operation 8: Simple compression - store deltas from previous step ----
    print(f"\n\n--- Delta Compression Potential ---")
    # Simulate: how much do m/v change between steps?
    model_name = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Train a few steps to build up states
    for _ in range(50):
        ids = torch.randint(100, 10000, (2, 128), device='cuda')
        model(input_ids=ids, labels=ids).loss.backward()
        opt.step()
        opt.zero_grad()

    # Capture states before and after one more step
    m_before = []
    v_before = []
    for group in opt.param_groups:
        for p in group['params']:
            if p in opt.state:
                m_before.append(opt.state[p]['exp_avg'].clone())
                v_before.append(opt.state[p]['exp_avg_sq'].clone())

    ids = torch.randint(100, 10000, (2, 128), device='cuda')
    model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    opt.zero_grad()

    m_after = []
    v_after = []
    for group in opt.param_groups:
        for p in group['params']:
            if p in opt.state:
                m_after.append(opt.state[p]['exp_avg'].clone())
                v_after.append(opt.state[p]['exp_avg_sq'].clone())

    # Analyze deltas
    m_before_cat = torch.cat([m.flatten() for m in m_before])
    m_after_cat = torch.cat([m.flatten() for m in m_after])
    v_before_cat = torch.cat([v.flatten() for v in v_before])
    v_after_cat = torch.cat([v.flatten() for v in v_after])

    m_delta = m_after_cat.float() - m_before_cat.float()
    v_delta = v_after_cat.float() - v_before_cat.float()

    # In BF16, check how many values actually changed
    m_changed = (m_after_cat != m_before_cat).sum().item()
    v_changed = (v_after_cat != v_before_cat).sum().item()
    total = m_before_cat.numel()

    print(f"  m values changed: {m_changed:,} / {total:,} ({100*m_changed/total:.1f}%)")
    print(f"  v values changed: {v_changed:,} / {total:,} ({100*v_changed/total:.1f}%)")

    # XOR delta (for BF16, how many bits change?)
    m_xor = m_after_cat.view(torch.int16) ^ m_before_cat.view(torch.int16)
    v_xor = v_after_cat.view(torch.int16) ^ v_before_cat.view(torch.int16)

    m_zero_xor = (m_xor == 0).sum().item()
    v_zero_xor = (v_xor == 0).sum().item()
    print(f"  m zero XOR (unchanged bf16): {m_zero_xor:,} / {total:,} ({100*m_zero_xor/total:.1f}%)")
    print(f"  v zero XOR (unchanged bf16): {v_zero_xor:,} / {total:,} ({100*v_zero_xor/total:.1f}%)")

    # Entropy of XOR delta
    m_xor_sample = m_xor[:min(10_000_000, total)].cpu().numpy()
    v_xor_sample = v_xor[:min(10_000_000, total)].cpu().numpy()

    for name, xor_data in [('m XOR delta', m_xor_sample), ('v XOR delta', v_xor_sample)]:
        vals, counts = np.unique(xor_data, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        print(f"  {name}: entropy={entropy:.4f} bits, {len(vals):,} unique, "
              f"compression={entropy/16*100:.2f}%")


if __name__ == '__main__':
    benchmark_compress_decompress_ops()
