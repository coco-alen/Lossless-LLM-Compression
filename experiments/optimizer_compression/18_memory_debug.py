"""
Experiment 18: Memory debugging + optimized compression

Diagnose where memory overhead comes from, then fix it.
"""

import torch
import time
import gc
import math
from transformers import AutoModelForCausalLM


def mem_mb():
    return torch.cuda.memory_allocated() / 1024**2

def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2

def mem_report(label):
    print(f"  [{label}] allocated={mem_mb():.0f} MB, peak={peak_mb():.0f} MB")


def pack_bits_int64(indices: torch.Tensor, bits: int) -> torch.Tensor:
    device = indices.device
    n = indices.numel()
    remainder = n % 8
    if remainder:
        indices = torch.cat([indices, torch.zeros(8 - remainder, dtype=torch.uint8, device=device)])
    groups = indices.reshape(-1, 8).to(torch.int64)
    shifts = torch.arange(8, device=device, dtype=torch.int64) * bits
    combined = (groups << shifts.unsqueeze(0)).sum(dim=1)
    byte_shifts = torch.arange(bits, device=device, dtype=torch.int64) * 8
    packed = ((combined.unsqueeze(1) >> byte_shifts.unsqueeze(0)) & 0xFF).to(torch.uint8)
    return packed.reshape(-1)


def unpack_bits_int64(packed: torch.Tensor, bits: int, n: int) -> torch.Tensor:
    device = packed.device
    n_groups = ((n + 7) // 8)
    groups = packed[:n_groups * bits].reshape(n_groups, bits).to(torch.int64)
    byte_shifts = torch.arange(bits, device=device, dtype=torch.int64) * 8
    combined = (groups << byte_shifts.unsqueeze(0)).sum(dim=1)
    bit_shifts = torch.arange(8, device=device, dtype=torch.int64) * bits
    mask = (1 << bits) - 1
    result = ((combined.unsqueeze(1) >> bit_shifts.unsqueeze(0)) & mask).to(torch.uint8)
    return result.reshape(-1)[:n]


def debug_memory_flow(model_name="Qwen/Qwen3-0.6B"):
    """Trace memory at each stage of compress/decompress."""
    print("=" * 80)
    print("Memory Flow Debug")
    print("=" * 80)

    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_report("start")

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    model.train()
    mem_report("model loaded")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # One step to initialize states
    ids = torch.randint(100, 10000, (2, 128), device='cuda')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        model(input_ids=ids, labels=ids).loss.backward()
    optimizer.step(); optimizer.zero_grad()
    gc.collect(); torch.cuda.empty_cache()
    mem_report("after first step (states initialized)")

    # Collect param info
    params = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state and 'exp_avg' in optimizer.state[p]:
                params.append(p)

    total_n = sum(p.numel() for p in params)
    print(f"\n  Total params: {len(params)}, elements: {total_n:,}")
    print(f"  Expected state memory: {total_n * 4 * 2 / 1024**2:.0f} MB")

    # ---- Test 1: Simple gather + free ----
    print(f"\n--- Test 1: Gather m into flat tensor, free per-param ---")
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_report("before gather")

    tensors = [optimizer.state[p]['exp_avg'].flatten() for p in params]
    flat_m = torch.cat(tensors)
    del tensors
    mem_report("after gather (flat_m allocated)")

    # Free per-param m states
    for p in params:
        optimizer.state[p]['exp_avg'] = torch.empty(0, dtype=torch.float32, device='cuda')
    gc.collect(); torch.cuda.empty_cache()
    mem_report("after freeing per-param m")

    # Now we have flat_m instead of per-param m. Memory should be same.
    print(f"  flat_m size: {flat_m.numel() * 4 / 1024**2:.0f} MB")

    # ---- Test 2: Compress flat_m into byte planes ----
    print(f"\n--- Test 2: Compress flat_m into byte planes ---")
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_report("before compress")

    int32_view = flat_m.view(torch.int32)

    # Extract byte3
    byte3 = ((int32_view >> 24) & 0xFF).to(torch.uint8)
    mem_report("after byte3 extraction")

    codebook = torch.unique(byte3)
    n_unique = len(codebook)
    bits = max(1, math.ceil(math.log2(max(n_unique, 2))))
    print(f"  byte3: {n_unique} unique → {bits} bits")

    lut = torch.zeros(256, dtype=torch.uint8, device='cuda')
    lut[codebook.long()] = torch.arange(n_unique, device='cuda', dtype=torch.uint8)
    indices = lut[byte3.long()]
    del byte3, lut
    gc.collect(); torch.cuda.empty_cache()
    mem_report("after indices (byte3 freed)")

    packed_byte3 = pack_bits_int64(indices, bits)
    del indices
    gc.collect(); torch.cuda.empty_cache()
    mem_report("after bitpack (indices freed)")

    # Extract byte planes
    byte0 = (int32_view & 0xFF).to(torch.uint8)
    byte1 = ((int32_view >> 8) & 0xFF).to(torch.uint8)
    byte2 = ((int32_view >> 16) & 0xFF).to(torch.uint8)
    mem_report("after byte012 extraction")

    # Stack into byte012
    byte012 = torch.stack([byte0, byte1, byte2], dim=1).reshape(-1)
    del byte0, byte1, byte2
    gc.collect(); torch.cuda.empty_cache()
    mem_report("after byte012 stack (individuals freed)")

    # Free flat_m
    del flat_m, int32_view
    gc.collect(); torch.cuda.empty_cache()
    mem_report("after freeing flat_m")

    print(f"\n  byte012 size: {byte012.numel() / 1024**2:.0f} MB")
    print(f"  packed_byte3 size: {packed_byte3.numel() / 1024**2:.0f} MB")
    print(f"  Total compressed m: {(byte012.numel() + packed_byte3.numel()) / 1024**2:.0f} MB")
    print(f"  Original m: {total_n * 4 / 1024**2:.0f} MB")
    print(f"  Savings: {(total_n * 4 - byte012.numel() - packed_byte3.numel()) / 1024**2:.0f} MB")

    # ---- Test 3: Alternative - just use raw bytes ----
    print(f"\n--- Test 3: Alternative - view as uint8 and slice ---")
    # Restore m from compressed
    restored_indices = unpack_bits_int64(packed_byte3, bits, total_n)
    restored_byte3 = codebook[restored_indices.long()]
    byte012_3col = byte012.reshape(total_n, 3)
    result = (byte012_3col[:, 0].to(torch.int32) |
              (byte012_3col[:, 1].to(torch.int32) << 8) |
              (byte012_3col[:, 2].to(torch.int32) << 16) |
              (restored_byte3.to(torch.int32) << 24))
    flat_m_restored = result.view(torch.float32)
    del restored_indices, restored_byte3, byte012_3col, result

    # Scatter back to per-param states
    offset = 0
    for p in params:
        n = p.numel()
        optimizer.state[p]['exp_avg'] = flat_m_restored[offset:offset+n].view(p.shape)
        offset += n
    # Note: per-param states are VIEWS into flat_m_restored
    del flat_m_restored  # won't free because views exist
    gc.collect(); torch.cuda.empty_cache()
    mem_report("after restore to per-param (views into flat)")

    del byte012, packed_byte3, codebook
    gc.collect(); torch.cuda.empty_cache()
    mem_report("after cleanup compressed data")

    print(f"\n--- Test 4: Measure overhead of having N tensors vs 1 flat ---")
    # How much overhead does 310 per-param tensors add vs 1 flat tensor?
    mem_before = mem_mb()
    dummy_tensors = [torch.empty(s, dtype=torch.float32, device='cuda') for s in [p.numel() for p in params]]
    mem_after = mem_mb()
    total_expected = sum(p.numel() for p in params) * 4 / 1024**2
    print(f"  310 tensors: allocated {mem_after - mem_before:.0f} MB, expected {total_expected:.0f} MB")
    print(f"  Overhead: {(mem_after - mem_before) - total_expected:.0f} MB")
    del dummy_tensors
    gc.collect(); torch.cuda.empty_cache()

    del model, optimizer
    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    debug_memory_flow()
