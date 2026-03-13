"""
Experiment 20: Detailed memory tracing to find the overhead source.
"""

import torch
import time
import gc
import math
from transformers import AutoModelForCausalLM


def mem():
    return torch.cuda.memory_allocated() / 1024**2


def pack_6bit(indices: torch.Tensor) -> torch.Tensor:
    device = indices.device
    n = indices.numel()
    remainder = n % 4
    if remainder:
        indices = torch.cat([indices, torch.zeros(4 - remainder, dtype=torch.uint8, device=device)])
    groups = indices.reshape(-1, 4)
    combined = groups[:, 0].to(torch.int32)
    combined = combined | (groups[:, 1].to(torch.int32) << 6)
    combined = combined | (groups[:, 2].to(torch.int32) << 12)
    combined = combined | (groups[:, 3].to(torch.int32) << 18)
    packed = torch.stack([
        (combined & 0xFF).to(torch.uint8),
        ((combined >> 8) & 0xFF).to(torch.uint8),
        ((combined >> 16) & 0xFF).to(torch.uint8),
    ], dim=1).reshape(-1)
    return packed


def unpack_6bit(packed: torch.Tensor, n: int) -> torch.Tensor:
    device = packed.device
    n_groups = (n + 3) // 4
    groups = packed[:n_groups * 3].reshape(n_groups, 3)
    combined = (groups[:, 0].to(torch.int32) |
                (groups[:, 1].to(torch.int32) << 8) |
                (groups[:, 2].to(torch.int32) << 16))
    result = torch.stack([
        (combined & 0x3F).to(torch.uint8),
        ((combined >> 6) & 0x3F).to(torch.uint8),
        ((combined >> 12) & 0x3F).to(torch.uint8),
        ((combined >> 18) & 0x3F).to(torch.uint8),
    ], dim=1).reshape(-1)[:n]
    return result


def main():
    model_name = "Qwen/Qwen3-0.6B"

    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Init states
    ids = torch.randint(100, 10000, (4, 256), device='cuda')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        model(input_ids=ids, labels=ids).loss.backward()
    optimizer.step(); optimizer.zero_grad()
    del ids
    gc.collect(); torch.cuda.empty_cache()

    print(f"Baseline memory: {mem():.0f} MB")

    # Collect params
    params = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state and 'exp_avg' in optimizer.state[p]:
                params.append(p)
    total_n = sum(p.numel() for p in params)
    print(f"Params: {len(params)}, elements: {total_n:,}, expected state: {total_n*4/1024**2:.0f} MB each")

    # ============ Manual compress of m ============
    print(f"\n=== Compressing m ===")
    m0 = mem()
    print(f"  Before: {m0:.0f} MB")

    # Step 1: Gather into flat tensor
    flat = torch.empty(total_n, dtype=torch.float32, device='cuda')
    offset = 0
    for p in params:
        n = p.numel()
        flat[offset:offset+n] = optimizer.state[p]['exp_avg'].flatten()
        offset += n
    print(f"  After gather (flat allocated): {mem():.0f} MB (+{mem()-m0:.0f})")

    # Step 2: Free per-param m states
    for p in params:
        optimizer.state[p]['exp_avg'] = torch.empty(0, dtype=torch.float32, device='cuda')
    gc.collect(); torch.cuda.empty_cache()
    m1 = mem()
    print(f"  After free per-param m: {m1:.0f} MB (delta from start: {m1-m0:+.0f})")

    # Step 3: Extract byte3, build codebook
    int32_view = flat.view(torch.int32)
    byte3 = ((int32_view >> 24) & 0xFF).to(torch.uint8)
    codebook = torch.unique(byte3)
    lut = torch.zeros(256, dtype=torch.uint8, device='cuda')
    lut[codebook.long()] = torch.arange(len(codebook), device='cuda', dtype=torch.uint8)
    print(f"  Byte3: {len(codebook)} unique → {math.ceil(math.log2(max(len(codebook), 2)))} bits")

    # Step 4: Map to indices
    indices = lut[byte3.long()]
    del byte3, lut
    gc.collect(); torch.cuda.empty_cache()
    m2 = mem()
    print(f"  After indices: {m2:.0f} MB (+{m2-m1:.0f} from prev)")

    # Step 5: Pack 6-bit
    packed3 = pack_6bit(indices)
    del indices
    gc.collect(); torch.cuda.empty_cache()
    m3 = mem()
    print(f"  After pack+free indices: {m3:.0f} MB (+{m3-m1:.0f} from flat-only)")

    # Step 6: Extract byte012
    uint8_view = int32_view.view(torch.uint8).reshape(total_n, 4)
    byte012 = uint8_view[:, :3].contiguous().reshape(-1)
    del uint8_view
    m4 = mem()
    print(f"  After byte012: {m4:.0f} MB (+{m4-m1:.0f} from flat-only)")
    print(f"  byte012: {byte012.numel()/1024**2:.0f} MB, packed3: {packed3.numel()/1024**2:.0f} MB")

    # Step 7: Free flat tensor
    del flat, int32_view
    gc.collect(); torch.cuda.empty_cache()
    m5 = mem()
    print(f"  After free flat: {m5:.0f} MB (delta from start: {m5-m0:+.0f})")
    print(f"  Expected delta: {(byte012.numel() + packed3.numel() - total_n*4)/1024**2:+.0f} MB")

    # ============ What does memory look like now? ============
    print(f"\n=== Memory accounting ===")
    # Count all tracked tensors
    tracked_mem = 0
    # Model params
    for p in model.parameters():
        tracked_mem += p.numel() * p.element_size()
    model_mem = tracked_mem / 1024**2
    # v states
    for p in params:
        s = optimizer.state[p]['exp_avg_sq']
        tracked_mem += s.numel() * s.element_size()
    v_mem = (tracked_mem / 1024**2) - model_mem
    # step tensors
    for p in params:
        if 'step' in optimizer.state[p]:
            tracked_mem += optimizer.state[p]['step'].numel() * optimizer.state[p]['step'].element_size()
    # Our compressed data
    tracked_mem += byte012.numel()
    tracked_mem += packed3.numel()
    tracked_mem += codebook.numel()

    print(f"  Model params: {model_mem:.0f} MB")
    print(f"  v states: {v_mem:.0f} MB")
    print(f"  byte012: {byte012.numel()/1024**2:.0f} MB")
    print(f"  packed3: {packed3.numel()/1024**2:.0f} MB")
    print(f"  Total tracked: {tracked_mem/1024**2:.0f} MB")
    print(f"  Actual allocated: {mem():.0f} MB")
    print(f"  UNTRACKED: {mem() - tracked_mem/1024**2:.0f} MB")

    # Check for any other optimizer state tensors
    other_state_mem = 0
    for p in params:
        for key, val in optimizer.state[p].items():
            if isinstance(val, torch.Tensor) and key not in ('exp_avg', 'exp_avg_sq'):
                other_state_mem += val.numel() * val.element_size()
    print(f"  Other state tensors (step etc): {other_state_mem/1024**2:.2f} MB")

    # Check param_groups
    for group in optimizer.param_groups:
        for key, val in group.items():
            if isinstance(val, torch.Tensor):
                print(f"  param_group tensor {key}: {val.numel() * val.element_size() / 1024**2:.2f} MB")


if __name__ == '__main__':
    main()
