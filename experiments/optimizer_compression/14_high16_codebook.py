"""
Experiment 14: High-16 Codebook Compression with Vectorized GPU Ops

Key insight from entropy analysis:
- FP32 = (high16, low16)
- high16 has ~5000-6000 unique values → needs 13 bits → store as uint16 INDEX
  Wait... uint16 index is same size as raw high16. No savings from indices alone.

REAL insight: We need to compress BELOW 16 bits per high16 value.
Options:
1. Store as 13-bit packed (complex, exp13 showed Python loops are 42x slower)
2. Use torch built-in: store high16 as uint8 pairs with codebook
   - No, same size.

Let's think differently. The ACTUAL savings come from:
- byte3 has 5-6 bits entropy (save 2-3 bits per element)
- byte2 has ~7.5 bits entropy (save 0.5 bits per element)
- bytes 0-1: ~8 bits each (incompressible)

Total theoretical: save ~3 bits per element = 9.4% of FP32.

BUT: bit-packing on GPU with Python is too slow.

NEW APPROACH: Use torch's native uint8 operations.
- For byte3 with N unique values (< 64 typically):
  Pack 2 byte3 indices into 1 byte (if ≤16 unique, use 4-bit; if ≤64, we can't pack 2)
  Actually: if v has 27 unique → 5 bits. We can't pack 2 into 8 bits.
  But if v has ≤16 unique → 4 bits → pack 2 per byte → 50% savings on byte3.

Let's check: from exp13, v typically has 13-27 unique byte3 values.
- Many params have ≤16 unique → 4-bit packing (2 per byte)
- Others have ≤32 unique → 5-bit (can't do simple packing)

SIMPLER APPROACH: Use PyTorch's quantize or just store byte3 as indices with
vectorized 4-bit packing for the common case (≤16 unique).

Even simpler: ELIMINATE byte3 entirely for v states.
- v = β₂*v + (1-β₂)*g² is always non-negative (it's a sum of squares)
- So byte3 (sign+exp MSBs) is always 0x00..0x4F range (positive, small exponent)
- Actually let's just measure and do the simplest thing that works.

ACTUAL SIMPLEST APPROACH:
Store (byte0, byte1, byte2) as 3-byte packed + byte3 as uint8 index into codebook.
This saves nothing because uint8 index = uint8 raw value.

OK, the fundamental issue: to get real savings we MUST do sub-byte packing,
and Python loops make this 42x slower.

SOLUTION: Use CUDA via CuPy or write a simple CUDA kernel, OR use vectorized
torch ops that avoid Python loops.

VECTORIZED 4-BIT PACKING (no Python loops):
For byte3 values with ≤16 unique: indices fit in 4 bits.
Pack pairs: packed[i] = (idx[2i] & 0xF) | (idx[2i+1] << 4)
This is ONE vectorized torch operation, no loops!

For ≤32 unique (5 bits): can't simply pack pairs.
For ≤64 unique (6 bits): can't simply pack pairs.

But we can do nibble packing (4 bits) for params where unique ≤ 16,
and fall back to raw uint8 for others. Let's see how many qualify.

BETTER: Use high-16 approach but with vectorized 12-bit packing.
high16 has ~5800 unique for m, ~4238 for v → 13 bits for m, 12 bits for v.
12-bit packing: pack 2 values in 3 bytes → 1.5 bytes per value vs 2 bytes → 25% savings on high16.
13-bit: more complex but 2 values in ~3.25 bytes.

12-BIT VECTORIZED PACKING (no loops, pure torch):
For N elements with indices in [0, 4095]:
- Reshape to pairs: (N/2, 2)
- packed_bytes = 3 bytes per pair:
  byte_a = idx[0] & 0xFF           (low 8 bits of first)
  byte_b = (idx[0] >> 8) | ((idx[1] & 0xF) << 4)  (high 4 of first + low 4 of second)
  byte_c = idx[1] >> 4             (high 8 bits of second)
- All vectorized!

This gives 25% savings on high16 = 12.5% savings on FP32 for v.
For m (13 bits needed), fall back to uint16 (no savings) or use 13-bit packing.

Let's implement 12-bit packing for high16 of v, and measure if it's fast enough.
"""

import torch
import time
import gc
import math
import numpy as np
from transformers import AutoModelForCausalLM


def pack_12bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint16 indices (max 4095) into 12-bit packed uint8 tensor.

    Vectorized, no Python loops. Packs pairs of 12-bit values into 3 bytes.
    """
    device = indices.device
    n = indices.numel()

    # Pad to even
    if n % 2:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.int32, device=device)])

    idx = indices.to(torch.int32).reshape(-1, 2)  # [N/2, 2]
    a, b = idx[:, 0], idx[:, 1]

    # 3 bytes per pair
    byte0 = (a & 0xFF).to(torch.uint8)
    byte1 = (((a >> 8) & 0xF) | ((b & 0xF) << 4)).to(torch.uint8)
    byte2 = ((b >> 4) & 0xFF).to(torch.uint8)

    # Interleave: [byte0, byte1, byte2] per pair
    packed = torch.stack([byte0, byte1, byte2], dim=1)  # [N/2, 3]
    return packed.flatten()


def unpack_12bit(packed: torch.Tensor, n_elements: int) -> torch.Tensor:
    """Unpack 12-bit packed uint8 tensor back to uint16 indices."""
    device = packed.device
    n_pairs = (n_elements + 1) // 2

    groups = packed[:n_pairs * 3].reshape(n_pairs, 3)
    byte0 = groups[:, 0].to(torch.int32)
    byte1 = groups[:, 1].to(torch.int32)
    byte2 = groups[:, 2].to(torch.int32)

    a = byte0 | ((byte1 & 0xF) << 8)
    b = ((byte1 >> 4) & 0xF) | (byte2 << 4)

    result = torch.stack([a, b], dim=1).flatten()[:n_elements]
    return result.to(torch.int16)


def pack_nibble(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 indices (max 15) into 4-bit nibble-packed tensor.

    2 values per byte, fully vectorized.
    """
    device = indices.device
    n = indices.numel()

    if n % 2:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.uint8, device=device)])

    pairs = indices.reshape(-1, 2)
    packed = (pairs[:, 0] & 0xF) | ((pairs[:, 1] & 0xF) << 4)
    return packed.to(torch.uint8)


def unpack_nibble(packed: torch.Tensor, n_elements: int) -> torch.Tensor:
    """Unpack 4-bit nibble-packed tensor."""
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    result = torch.stack([low, high], dim=1).flatten()[:n_elements]
    return result.to(torch.uint8)


class High16CompressedState:
    """Compress FP32 by codebook-encoding high16, storing low16 raw.

    For states with ≤4096 unique high16 values: 12-bit pack indices (25% savings on high16)
    For states with ≤16 unique byte3 values: nibble-pack byte3 (50% savings on byte3)
    """

    def __init__(self):
        self.low16 = None          # int16, raw
        self.codebook = None       # unique high16 values (int16)
        self.packed_indices = None  # 12-bit or nibble packed
        self.pack_mode = None      # '12bit', 'nibble', 'raw'
        self.n = 0
        self.shape = None
        self.n_unique = 0

    def compress(self, tensor: torch.Tensor):
        assert tensor.dtype == torch.float32
        self.shape = tensor.shape
        self.n = tensor.numel()
        device = tensor.device

        # Split into high16 and low16
        int32_view = tensor.contiguous().flatten().view(torch.int32)
        high16 = ((int32_view >> 16) & 0xFFFF).to(torch.int16)
        self.low16 = (int32_view & 0xFFFF).to(torch.int16)

        # Build codebook
        self.codebook = torch.unique(high16)
        self.n_unique = len(self.codebook)
        bits_needed = max(1, math.ceil(math.log2(max(self.n_unique, 2))))

        # Create LUT (high16 value → index)
        # high16 is int16 (-32768 to 32767), shift to 0-65535 for indexing
        lut = torch.zeros(65536, dtype=torch.int32, device=device)
        # Vectorized LUT build
        shifted_codebook = (self.codebook.to(torch.int32) + 32768) % 65536
        lut[shifted_codebook.long()] = torch.arange(self.n_unique, device=device, dtype=torch.int32)

        # Map to indices
        shifted_high16 = (high16.to(torch.int32) + 32768) % 65536
        indices = lut[shifted_high16.long()]

        # Choose packing strategy
        if self.n_unique <= 4096:
            self.pack_mode = '12bit'
            self.packed_indices = pack_12bit(indices)
        else:
            self.pack_mode = 'raw'
            self.packed_indices = indices.to(torch.int16)

        return self

    def decompress(self) -> torch.Tensor:
        device = self.low16.device

        # Unpack indices
        if self.pack_mode == '12bit':
            indices = unpack_12bit(self.packed_indices, self.n)
        else:
            indices = self.packed_indices[:self.n]

        # Look up high16
        high16 = self.codebook[indices.long()]

        # Reconstruct int32
        result = (high16.to(torch.int32) << 16) | (self.low16.to(torch.int32) & 0xFFFF)
        return result.view(torch.float32).reshape(self.shape)

    def memory_bytes(self):
        total = 0
        for t in [self.low16, self.packed_indices, self.codebook]:
            if t is not None:
                total += t.numel() * t.element_size()
        return total

    def original_bytes(self):
        return self.n * 4

    def ratio(self):
        return self.memory_bytes() / self.original_bytes()


class High16CompressedAdamW:
    """AdamW wrapper with high-16 codebook compression."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._compressed = {}
        self._is_compressed = False

    def _compress_states(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in self.optimizer.state:
                    continue
                state = self.optimizer.state[p]
                if 'exp_avg' not in state:
                    continue

                pid = id(p)
                cm = High16CompressedState().compress(state['exp_avg'])
                cv = High16CompressedState().compress(state['exp_avg_sq'])
                self._compressed[pid] = (cm, cv)

                state['exp_avg'] = torch.empty(0, dtype=torch.float32, device=p.device)
                state['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=p.device)

        self._is_compressed = True

    def _decompress_states(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                pid = id(p)
                if pid not in self._compressed:
                    continue
                cm, cv = self._compressed[pid]
                state = self.optimizer.state[p]
                state['exp_avg'] = cm.decompress()
                state['exp_avg_sq'] = cv.decompress()

        self._compressed.clear()
        self._is_compressed = False

    def step(self):
        if self._is_compressed:
            self._decompress_states()
        self.optimizer.step()
        self._compress_states()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def get_stats(self):
        if not self._compressed:
            return None
        total_c, total_o = 0, 0
        details = []
        for pid, (cm, cv) in self._compressed.items():
            total_c += cm.memory_bytes() + cv.memory_bytes()
            total_o += cm.original_bytes() + cv.original_bytes()
            details.append((cm.n_unique, cm.pack_mode, cm.ratio(),
                           cv.n_unique, cv.pack_mode, cv.ratio()))
        return {
            'compressed_bytes': total_c,
            'original_bytes': total_o,
            'ratio': total_c / total_o,
            'savings_mb': (total_o - total_c) / 1024**2,
            'details': details[:5],
        }


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    print("--- Losslessness Verification ---")

    # Test 12-bit round-trip
    print("  Test 1: 12-bit pack round-trip...")
    for n in [100, 1000, 10001, 100000]:
        data = torch.randint(0, 4096, (n,), dtype=torch.int32, device='cuda')
        packed = pack_12bit(data)
        restored = unpack_12bit(packed, n)
        assert torch.all(data.to(torch.int16) == restored), f"12-bit FAILED for n={n}"
    print("  ✓ 12-bit packing is lossless")

    # Test nibble round-trip
    print("  Test 2: Nibble pack round-trip...")
    for n in [100, 1001, 10000]:
        data = torch.randint(0, 16, (n,), dtype=torch.uint8, device='cuda')
        packed = pack_nibble(data)
        restored = unpack_nibble(packed, n)
        assert torch.all(data == restored), f"Nibble FAILED for n={n}"
    print("  ✓ Nibble packing is lossless")

    # Test FP32 compress/decompress
    print("  Test 3: FP32 compress/decompress...")
    for _ in range(5):
        data = torch.randn(100000, dtype=torch.float32, device='cuda')
        cs = High16CompressedState().compress(data)
        restored = cs.decompress()
        assert torch.all(data == restored), "FP32 compress/decompress FAILED"
    print("  ✓ FP32 compression is lossless")

    # Test with special values
    print("  Test 4: Special values...")
    special = torch.tensor([0.0, -0.0, float('inf'), float('-inf'), 1e-38, 1e38,
                           1.0, -1.0, 3.14159], dtype=torch.float32, device='cuda')
    cs = High16CompressedState().compress(special)
    restored = cs.decompress()
    # NaN != NaN, so check non-nan separately
    mask = ~torch.isnan(special)
    assert torch.all(special[mask] == restored[mask]), "Special values FAILED"
    print("  ✓ Special values preserved")

    # Full optimizer test
    print("  Test 5: Full optimizer comparison...")
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = High16CompressedAdamW(inner)

    for step in range(5):
        torch.manual_seed(step + 100)
        ids = torch.randint(100, 10000, (2, 128), device='cuda')

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            m1(input_ids=ids, labels=ids).loss.backward()
        o1.step(); o1.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            m2(input_ids=ids, labels=ids).loss.backward()
        o2.step(); o2.zero_grad()

    max_diff = max(
        (p1.data - p2.data).abs().max().item()
        for p1, p2 in zip(m1.parameters(), m2.parameters())
    )
    print(f"  Max param diff: {max_diff}" + (" ✓ LOSSLESS" if max_diff == 0 else " ✗ FAILED"))

    del m1, m2, o1, o2, inner
    gc.collect(); torch.cuda.empty_cache()
    return max_diff == 0


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "=" * 80)
    print("High-16 Codebook Compressed FP32 Optimizer Benchmark")
    print("=" * 80)

    results = []
    n_warmup, n_measure = 10, 40

    for name, use_compressed in [("Standard FP32 AdamW", False),
                                  ("High16-Coded FP32 AdamW", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()

        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = High16CompressedAdamW(inner) if use_compressed else inner

        for _ in range(n_warmup):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()

        gc.collect(); torch.cuda.empty_cache()
        gpu_mem = torch.cuda.memory_allocated() / 1024**2

        if use_compressed:
            stats = opt.get_stats()
            if stats:
                print(f"  Ratio: {stats['ratio']*100:.1f}%, Savings: {stats['savings_mb']:.0f} MB")
                for mu, mm, mr, vu, vm, vr in stats['details']:
                    print(f"    m: {mu} unique ({mm}), ratio={mr*100:.1f}% | "
                          f"v: {vu} unique ({vm}), ratio={vr*100:.1f}%")

        torch.cuda.synchronize()
        times = []
        for _ in range(n_measure):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_step = sum(times) / len(times)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        tps = batch_size * seq_len / avg_step

        print(f"  Mem: {gpu_mem:.0f} MB, Peak: {peak:.0f} MB, "
              f"Step: {avg_step*1000:.1f} ms, Tok/s: {tps:.0f}")

        results.append({
            'method': name, 'gpu_mem': gpu_mem, 'peak': peak,
            'step_ms': avg_step * 1000, 'tps': tps,
        })

        del model, inner, opt
        gc.collect(); torch.cuda.empty_cache()

    # Summary
    bl = results[0]
    print(f"\n{'='*80}")
    print(f"{'Method':<30} {'Mem':>7} {'ΔMem':>7} {'Peak':>7} {'Step':>7} {'Slow':>5}")
    print("-" * 65)
    for r in results:
        dm = r['gpu_mem'] - bl['gpu_mem']
        s = r['step_ms'] / bl['step_ms']
        print(f"{r['method']:<30} {r['gpu_mem']:>6.0f}M {dm:>+6.0f}M {r['peak']:>6.0f}M "
              f"{r['step_ms']:>6.1f} {s:>4.2f}x")


if __name__ == '__main__':
    ok = verify_lossless()
    if ok:
        benchmark()
