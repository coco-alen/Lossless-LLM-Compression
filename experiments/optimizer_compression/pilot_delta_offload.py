"""
Pilot: Delta-Compressed CPU Offload for v (second moment)

Prior findings:
  - v changes only ~22% of BF16 values per step (β₂=0.999)
  - Hooked CPU offload saves 2274 MB with 13% slowdown (exp 09)

Idea: Instead of transferring full v between CPU↔GPU each step,
compute XOR delta on GPU, transfer only changed values.

Measures:
  a. Full v transfer size
  b. XOR delta as sparse COO (index + value per changed element)
  c. Bitmask + changed-values format
  d. GPU delta compute time
  e. Transfer time: full v vs each delta format
  f. End-to-end: offload-with-delta vs offload-without-delta
"""

import torch
import time
import gc
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer


def fmt_mb(nbytes):
    return f"{nbytes / 1024**2:.1f} MB"


def fmt_us(seconds):
    return f"{seconds * 1e6:.0f} us"


def fmt_ms(seconds):
    return f"{seconds * 1e3:.2f} ms"


@torch.no_grad()
def benchmark_delta_formats(prev_v_flat, curr_v_flat):
    """Benchmark different delta representations on GPU tensors (both bf16)."""
    n = prev_v_flat.numel()
    dense_bytes = n * 2  # bf16 = 2 bytes

    # --- XOR delta ---
    prev_int = prev_v_flat.view(torch.int16)
    curr_int = curr_v_flat.view(torch.int16)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    xor = prev_int ^ curr_int
    changed_mask = xor != 0
    n_changed = changed_mask.sum().item()
    torch.cuda.synchronize()
    delta_compute_time = time.perf_counter() - t0

    pct_changed = 100.0 * n_changed / n

    # --- Format A: Sparse COO (int32 index + int16 value per change) ---
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    changed_indices = changed_mask.nonzero(as_tuple=False).squeeze(1).to(torch.int32)
    changed_values = curr_int[changed_mask]
    torch.cuda.synchronize()
    sparse_build_time = time.perf_counter() - t0

    sparse_coo_bytes = n_changed * (4 + 2)  # int32 idx + int16 val

    # --- Format B: Bitmask + changed values ---
    # bitmask: ceil(n/8) bytes, then changed values as int16
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Pack bitmask into uint8 (every 8 bools -> 1 byte)
    # Use the bool mask directly for transfer size estimate
    bitmask_bytes = (n + 7) // 8
    # changed_values already computed above
    torch.cuda.synchronize()
    bitmask_build_time = time.perf_counter() - t0

    bitmask_total_bytes = bitmask_bytes + n_changed * 2

    # --- Format C: Run-length encoded XOR ---
    # Just estimate: for 22% changed, RLE won't beat bitmask much
    # Skip actual RLE for now

    return {
        'n_total': n,
        'n_changed': n_changed,
        'pct_changed': pct_changed,
        'dense_bytes': dense_bytes,
        'sparse_coo_bytes': sparse_coo_bytes,
        'bitmask_bytes': bitmask_total_bytes,
        'delta_compute_time': delta_compute_time,
        'sparse_build_time': sparse_build_time,
        'bitmask_build_time': bitmask_build_time,
        # Keep tensors for transfer benchmarks
        '_changed_indices': changed_indices,
        '_changed_values': changed_values,
        '_changed_mask': changed_mask,
        '_curr_int': curr_int,
        '_full_tensor': curr_v_flat,
    }


def benchmark_transfers(info, n_iters=20):
    """Benchmark GPU->CPU transfer for full vs delta formats."""
    results = {}

    # Warmup
    dummy = torch.empty(1024 * 1024, dtype=torch.bfloat16, pin_memory=True)

    # --- Full transfer ---
    full_cpu = torch.empty(info['n_total'], dtype=torch.bfloat16, pin_memory=True)
    full_gpu = info['_full_tensor']
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        full_cpu.copy_(full_gpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['full_gpu2cpu'] = min(times)
    results['full_gpu2cpu_median'] = sorted(times)[len(times)//2]

    # --- Full transfer CPU->GPU ---
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        full_gpu.copy_(full_cpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['full_cpu2gpu'] = min(times)
    results['full_cpu2gpu_median'] = sorted(times)[len(times)//2]

    # --- Sparse COO transfer (indices + values) ---
    idx_gpu = info['_changed_indices']
    val_gpu = info['_changed_values']
    idx_cpu = torch.empty_like(idx_gpu, pin_memory=True, device='cpu')
    val_cpu = torch.empty_like(val_gpu, pin_memory=True, device='cpu')
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx_cpu.copy_(idx_gpu)
        val_cpu.copy_(val_gpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['sparse_gpu2cpu'] = min(times)
    results['sparse_gpu2cpu_median'] = sorted(times)[len(times)//2]

    # --- Sparse COO transfer CPU->GPU (restore path) ---
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx_gpu.copy_(idx_cpu)
        val_gpu.copy_(val_cpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['sparse_cpu2gpu'] = min(times)
    results['sparse_cpu2gpu_median'] = sorted(times)[len(times)//2]

    # --- Bitmask transfer (mask + values) ---
    mask_gpu = info['_changed_mask']  # bool tensor
    # Pack to uint8 for realistic transfer
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # Reshape to groups of 8, pack bits
    n = mask_gpu.numel()
    padded_n = ((n + 7) // 8) * 8
    if padded_n > n:
        mask_padded = torch.zeros(padded_n, dtype=torch.bool, device='cuda')
        mask_padded[:n] = mask_gpu
    else:
        mask_padded = mask_gpu
    # Manual bit packing: view as uint8 groups
    mask_bytes_gpu = mask_padded.view(-1, 8)
    # Convert 8 bools to 1 byte using bit shifts
    weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device='cuda')
    packed_mask_gpu = (mask_bytes_gpu.to(torch.uint8) * weights).sum(dim=1).to(torch.uint8)
    torch.cuda.synchronize()
    pack_time = time.perf_counter() - t0

    packed_cpu = torch.empty_like(packed_mask_gpu, pin_memory=True, device='cpu')
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        packed_cpu.copy_(packed_mask_gpu)
        val_cpu.copy_(val_gpu)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    results['bitmask_gpu2cpu'] = min(times)
    results['bitmask_gpu2cpu_median'] = sorted(times)[len(times)//2]
    results['bitmask_pack_time'] = pack_time

    return results


def benchmark_end_to_end_offload(model, optimizer, param_info, total_n):
    """Benchmark full offload vs delta offload for a few training steps."""
    dtype = torch.bfloat16

    # Pre-allocate CPU buffers
    cpu_v_full = torch.empty(total_n, dtype=dtype, pin_memory=True)
    cpu_v_prev = torch.empty(total_n, dtype=dtype, pin_memory=True)  # previous snapshot

    # Flatten current v into a contiguous GPU buffer
    def flatten_v():
        parts = []
        for p, offset, n, shape in param_info:
            parts.append(optimizer.state[p]['exp_avg_sq'].flatten())
        return torch.cat(parts)

    def scatter_v(flat):
        for p, offset, n, shape in param_info:
            optimizer.state[p]['exp_avg_sq'] = flat[offset:offset+n].view(shape)

    # === Baseline: full offload ===
    n_steps = 10
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Warmup
    for _ in range(3):
        ids = torch.randint(100, 10000, (2, 128), device='cuda')
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Full offload benchmark
    full_times = []
    for i in range(n_steps):
        ids = torch.randint(100, 10000, (2, 128), device='cuda')

        # Restore v from CPU (simulate)
        v_flat = flatten_v()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # GPU -> CPU offload
        cpu_v_full.copy_(v_flat)
        torch.cuda.synchronize()
        offload_t = time.perf_counter() - t0

        # CPU -> GPU restore
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v_flat.copy_(cpu_v_full)
        torch.cuda.synchronize()
        restore_t = time.perf_counter() - t0

        full_times.append(offload_t + restore_t)

    # === Delta offload benchmark ===
    # First, take a snapshot
    v_snapshot = flatten_v().clone()
    cpu_v_prev.copy_(v_snapshot)

    # Do one step to create a delta
    ids = torch.randint(100, 10000, (2, 128), device='cuda')
    model(input_ids=ids, labels=ids).loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    delta_times = []
    delta_compute_times = []
    delta_transfer_times = []
    for i in range(n_steps):
        ids = torch.randint(100, 10000, (2, 128), device='cuda')

        v_curr = flatten_v()
        torch.cuda.synchronize()

        # Compute delta
        t0 = time.perf_counter()
        prev_int = v_snapshot.view(torch.int16)
        curr_int = v_curr.view(torch.int16)
        xor = prev_int ^ curr_int
        changed_mask = xor != 0
        changed_indices = changed_mask.nonzero(as_tuple=False).squeeze(1).to(torch.int32)
        changed_values = curr_int[changed_mask]
        torch.cuda.synchronize()
        compute_t = time.perf_counter() - t0

        # Transfer delta to CPU
        idx_cpu = torch.empty_like(changed_indices, pin_memory=True, device='cpu')
        val_cpu = torch.empty_like(changed_values, pin_memory=True, device='cpu')
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        idx_cpu.copy_(changed_indices)
        val_cpu.copy_(changed_values)
        torch.cuda.synchronize()
        transfer_t = time.perf_counter() - t0

        # Apply delta on CPU side (update cpu_v_prev)
        t0 = time.perf_counter()
        cpu_v_prev.view(torch.int16)[idx_cpu.long()] = val_cpu
        cpu_apply_t = time.perf_counter() - t0

        # Restore: transfer delta back to GPU (simulate)
        t0 = time.perf_counter()
        changed_indices.copy_(idx_cpu)
        changed_values.copy_(val_cpu)
        torch.cuda.synchronize()
        restore_transfer_t = time.perf_counter() - t0

        # Apply on GPU
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v_snapshot.view(torch.int16)[changed_indices.long()] = changed_values
        torch.cuda.synchronize()
        gpu_apply_t = time.perf_counter() - t0

        total_delta = compute_t + transfer_t + cpu_apply_t + restore_transfer_t + gpu_apply_t
        delta_times.append(total_delta)
        delta_compute_times.append(compute_t)
        delta_transfer_times.append(transfer_t + restore_transfer_t)

        # Do a training step to create new delta
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return {
        'full_roundtrip_ms': [t * 1000 for t in full_times],
        'delta_total_ms': [t * 1000 for t in delta_times],
        'delta_compute_ms': [t * 1000 for t in delta_compute_times],
        'delta_transfer_ms': [t * 1000 for t in delta_transfer_times],
    }


def main():
    print("=" * 80)
    print("Pilot: Delta-Compressed CPU Offload for v (second moment)")
    print("=" * 80)

    model_name = "Qwen/Qwen3-0.6B"
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).cuda()
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params, {n_params*2/1024**2:.1f} MB (bf16)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Warmup: 20 steps to initialize optimizer states
    print("\nWarming up optimizer (20 steps)...")
    for step in range(20):
        ids = torch.randint(100, 10000, (2, 128), device='cuda')
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Build param info
    param_info = []
    offset = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state and 'exp_avg_sq' in optimizer.state[p]:
                n = p.numel()
                param_info.append((p, offset, n, p.shape))
                offset += n
    total_n = offset
    print(f"Optimizer v total: {total_n:,} elements = {fmt_mb(total_n * 2)}")

    # =========================================================================
    # Part 1: Capture v before and after one step, measure delta
    # =========================================================================
    print("\n" + "=" * 80)
    print("Part 1: Delta analysis after 1 optimizer step")
    print("=" * 80)

    # Snapshot before
    v_before_parts = []
    for p, off, n, shape in param_info:
        v_before_parts.append(optimizer.state[p]['exp_avg_sq'].flatten().clone())
    v_before = torch.cat(v_before_parts)

    # One training step
    ids = torch.randint(100, 10000, (2, 128), device='cuda')
    model(input_ids=ids, labels=ids).loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Snapshot after
    v_after_parts = []
    for p, off, n, shape in param_info:
        v_after_parts.append(optimizer.state[p]['exp_avg_sq'].flatten().clone())
    v_after = torch.cat(v_after_parts)

    # Analyze delta formats
    info = benchmark_delta_formats(v_before, v_after)

    print(f"\n  Total elements:    {info['n_total']:>15,}")
    print(f"  Changed elements:  {info['n_changed']:>15,} ({info['pct_changed']:.1f}%)")
    print(f"  Full v size:       {fmt_mb(info['dense_bytes']):>15}")
    print(f"  Sparse COO size:   {fmt_mb(info['sparse_coo_bytes']):>15} "
          f"({100*info['sparse_coo_bytes']/info['dense_bytes']:.1f}% of full)")
    print(f"  Bitmask+vals size: {fmt_mb(info['bitmask_bytes']):>15} "
          f"({100*info['bitmask_bytes']/info['dense_bytes']:.1f}% of full)")
    print(f"  Delta compute:     {fmt_ms(info['delta_compute_time']):>15}")
    print(f"  Sparse build:      {fmt_ms(info['sparse_build_time']):>15}")

    # =========================================================================
    # Part 2: Transfer benchmarks
    # =========================================================================
    print("\n" + "=" * 80)
    print("Part 2: Transfer time benchmarks (GPU <-> CPU)")
    print("=" * 80)

    xfer = benchmark_transfers(info)

    print(f"\n  Full GPU->CPU:      {fmt_ms(xfer['full_gpu2cpu']):>12} (min), "
          f"{fmt_ms(xfer['full_gpu2cpu_median'])} (median)")
    print(f"  Full CPU->GPU:      {fmt_ms(xfer['full_cpu2gpu']):>12} (min), "
          f"{fmt_ms(xfer['full_cpu2gpu_median'])} (median)")
    full_roundtrip = xfer['full_gpu2cpu'] + xfer['full_cpu2gpu']
    print(f"  Full roundtrip:     {fmt_ms(full_roundtrip):>12}")

    print(f"\n  Sparse GPU->CPU:    {fmt_ms(xfer['sparse_gpu2cpu']):>12} (min), "
          f"{fmt_ms(xfer['sparse_gpu2cpu_median'])} (median)")
    print(f"  Sparse CPU->GPU:    {fmt_ms(xfer['sparse_cpu2gpu']):>12} (min), "
          f"{fmt_ms(xfer['sparse_cpu2gpu_median'])} (median)")
    sparse_roundtrip = xfer['sparse_gpu2cpu'] + xfer['sparse_cpu2gpu']
    print(f"  Sparse roundtrip:   {fmt_ms(sparse_roundtrip):>12}")

    print(f"\n  Bitmask GPU->CPU:   {fmt_ms(xfer['bitmask_gpu2cpu']):>12} (min), "
          f"{fmt_ms(xfer['bitmask_gpu2cpu_median'])} (median)")
    print(f"  Bitmask pack time:  {fmt_ms(xfer['bitmask_pack_time']):>12}")

    print(f"\n  Transfer speedup (sparse vs full roundtrip): "
          f"{full_roundtrip/sparse_roundtrip:.2f}x")

    # =========================================================================
    # Part 3: End-to-end comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("Part 3: End-to-end offload benchmark (full vs delta)")
    print("=" * 80)

    e2e = benchmark_end_to_end_offload(model, optimizer, param_info, total_n)

    full_med = sorted(e2e['full_roundtrip_ms'])[len(e2e['full_roundtrip_ms'])//2]
    delta_med = sorted(e2e['delta_total_ms'])[len(e2e['delta_total_ms'])//2]
    delta_compute_med = sorted(e2e['delta_compute_ms'])[len(e2e['delta_compute_ms'])//2]
    delta_xfer_med = sorted(e2e['delta_transfer_ms'])[len(e2e['delta_transfer_ms'])//2]

    print(f"\n  Full offload roundtrip (median):   {full_med:.2f} ms")
    print(f"  Delta offload total (median):      {delta_med:.2f} ms")
    print(f"    - Delta compute (median):        {delta_compute_med:.2f} ms")
    print(f"    - Delta transfer (median):       {delta_xfer_med:.2f} ms")
    print(f"  Speedup:                           {full_med/delta_med:.2f}x")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
  v size:             {fmt_mb(info['dense_bytes'])}
  Changed per step:   {info['pct_changed']:.1f}% ({info['n_changed']:,} elements)

  Format sizes:
    Full:             {fmt_mb(info['dense_bytes'])}
    Sparse COO:       {fmt_mb(info['sparse_coo_bytes'])} ({100*info['sparse_coo_bytes']/info['dense_bytes']:.1f}%)
    Bitmask+values:   {fmt_mb(info['bitmask_bytes'])} ({100*info['bitmask_bytes']/info['dense_bytes']:.1f}%)

  Transfer roundtrip (GPU<->CPU):
    Full:             {fmt_ms(full_roundtrip)}
    Sparse:           {fmt_ms(sparse_roundtrip)}
    Speedup:          {full_roundtrip/sparse_roundtrip:.2f}x

  End-to-end offload:
    Full:             {full_med:.2f} ms
    Delta:            {delta_med:.2f} ms
    Speedup:          {full_med/delta_med:.2f}x

  Note: Sparse scatter on GPU/CPU adds overhead. The net benefit depends
  on whether transfer savings outweigh scatter costs.
""")


if __name__ == '__main__':
    main()
