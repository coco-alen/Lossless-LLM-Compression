"""
Experiment 16: Per-Parameter High-16 Compression with Vectorized Ops

Fix key issues from exp13-15:
- Process per-parameter (no gather → no memory explosion)
- Vectorized 12-bit packing for high16 (no Python bit loops)
- Careful memory management (free originals ASAP)

Compression: FP32 = (high16, low16)
- low16: store raw (2 bytes/element)
- high16: codebook + 12-bit packed indices (1.5 bytes/element if ≤4096 unique)
         OR raw int16 (2 bytes/element if >4096 unique)

Expected:
- For v (≤4096 unique high16 per param): 87.5% ratio → 12.5% savings
- For m (>4096 unique for big params): ~97-100% ratio → 0-3% savings
- Combined: ~6-8% savings, but minimal overhead
"""

import torch
import time
import gc
import math
from transformers import AutoModelForCausalLM


def pack_12bit_vectorized(indices: torch.Tensor) -> torch.Tensor:
    """Pack int32 indices (0-4095) as 12-bit pairs into uint8 tensor.
    2 indices → 3 bytes. Fully vectorized."""
    device = indices.device
    n = indices.numel()
    # Pad to even
    if n % 2:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.int32, device=device)])
    pairs = indices.reshape(-1, 2)
    a, b = pairs[:, 0], pairs[:, 1]
    byte0 = (a & 0xFF).to(torch.uint8)
    byte1 = (((a >> 8) & 0xF) | ((b & 0xF) << 4)).to(torch.uint8)
    byte2 = ((b >> 4) & 0xFF).to(torch.uint8)
    return torch.stack([byte0, byte1, byte2], dim=1).reshape(-1)


def unpack_12bit_vectorized(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack 12-bit pairs from uint8 tensor. Returns int32 indices."""
    n_pairs = (n + 1) // 2
    groups = packed[:n_pairs * 3].reshape(n_pairs, 3)
    b0, b1, b2 = groups[:, 0].to(torch.int32), groups[:, 1].to(torch.int32), groups[:, 2].to(torch.int32)
    a = b0 | ((b1 & 0xF) << 8)
    b = ((b1 >> 4) & 0xF) | (b2 << 4)
    return torch.stack([a, b], dim=1).reshape(-1)[:n]


def pack_bits_int64(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 indices at `bits` bits per value using int64 accumulation.
    Groups of 8 → `bits` bytes. Vectorized."""
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
    """Unpack int64-packed uint8 indices."""
    device = packed.device
    n_groups = ((n + 7) // 8)
    groups = packed[:n_groups * bits].reshape(n_groups, bits).to(torch.int64)
    byte_shifts = torch.arange(bits, device=device, dtype=torch.int64) * 8
    combined = (groups << byte_shifts.unsqueeze(0)).sum(dim=1)
    bit_shifts = torch.arange(8, device=device, dtype=torch.int64) * bits
    mask = (1 << bits) - 1
    result = ((combined.unsqueeze(1) >> bit_shifts.unsqueeze(0)) & mask).to(torch.uint8)
    return result.reshape(-1)[:n]


class CompressedState:
    """Per-parameter compressed FP32 state."""
    __slots__ = ['low16', 'high16_data', 'high16_codebook', 'mode', 'n', 'shape']

    def __init__(self):
        self.low16 = None
        self.high16_data = None      # packed indices or raw int16
        self.high16_codebook = None   # only for codebook modes
        self.mode = None              # '12bit', 'byte3_Xbit', 'raw'
        self.n = 0
        self.shape = None


def compress_fp32(tensor: torch.Tensor) -> CompressedState:
    """Compress one FP32 tensor. Returns CompressedState."""
    cs = CompressedState()
    cs.shape = tensor.shape
    cs.n = tensor.numel()
    device = tensor.device

    int32_view = tensor.contiguous().flatten().view(torch.int32)

    # Extract low16 and high16
    cs.low16 = (int32_view & 0xFFFF).to(torch.int16)
    high16 = ((int32_view >> 16) & 0xFFFF).to(torch.int16)

    # Count unique high16
    codebook = torch.unique(high16)
    n_unique = len(codebook)

    if n_unique <= 4096:
        # 12-bit packing: 1.5 bytes per element for high16
        cs.mode = '12bit'
        cs.high16_codebook = codebook

        # Build LUT
        lut = torch.zeros(65536, dtype=torch.int32, device=device)
        shifted_cb = (codebook.to(torch.int32) + 32768) % 65536
        lut[shifted_cb.long()] = torch.arange(n_unique, device=device, dtype=torch.int32)
        shifted_h16 = (high16.to(torch.int32) + 32768) % 65536
        indices = lut[shifted_h16.long()]

        cs.high16_data = pack_12bit_vectorized(indices)

        del lut, shifted_cb, shifted_h16, indices
    else:
        # Try byte3-only compression
        byte3 = ((int32_view >> 24) & 0xFF).to(torch.uint8)
        byte3_unique = torch.unique(byte3)
        n_byte3_unique = len(byte3_unique)
        bits = max(1, math.ceil(math.log2(max(n_byte3_unique, 2))))

        if bits < 8:
            cs.mode = f'byte3_{bits}bit'
            cs.high16_codebook = byte3_unique  # byte3 codebook

            # Build LUT for byte3
            lut = torch.zeros(256, dtype=torch.uint8, device=device)
            lut[byte3_unique.long()] = torch.arange(n_byte3_unique, device=device, dtype=torch.uint8)
            byte3_indices = lut[byte3.long()]

            # Pack byte3 indices
            packed_byte3 = pack_bits_int64(byte3_indices, bits)

            # Store: byte2 (raw uint8) + packed byte3
            byte2 = ((int32_view >> 16) & 0xFF).to(torch.uint8)
            cs.high16_data = (byte2, packed_byte3, bits)

            del lut, byte3_indices, byte3
        else:
            # Raw: no compression for high16
            cs.mode = 'raw'
            cs.high16_data = high16

    return cs


def decompress_fp32(cs: CompressedState) -> torch.Tensor:
    """Decompress CompressedState back to FP32."""
    device = cs.low16.device

    if cs.mode == '12bit':
        indices = unpack_12bit_vectorized(cs.high16_data, cs.n)
        high16 = cs.high16_codebook[indices.long()]
    elif cs.mode.startswith('byte3_'):
        byte2, packed_byte3, bits = cs.high16_data
        byte3_indices = unpack_bits_int64(packed_byte3, bits, cs.n)
        byte3 = cs.high16_codebook[byte3_indices.long()]
        # Reconstruct high16 from byte2 and byte3
        high16 = (byte2.to(torch.int16) | (byte3.to(torch.int16) << 8))
    else:
        high16 = cs.high16_data[:cs.n]

    result = (cs.low16.to(torch.int32) & 0xFFFF) | (high16.to(torch.int32) << 16)
    return result.view(torch.float32).reshape(cs.shape)


def compressed_bytes(cs: CompressedState) -> int:
    """Count bytes used by compressed state."""
    total = cs.low16.numel() * cs.low16.element_size()

    if cs.mode == '12bit':
        total += cs.high16_data.numel() * cs.high16_data.element_size()
        total += cs.high16_codebook.numel() * cs.high16_codebook.element_size()
    elif cs.mode.startswith('byte3_'):
        byte2, packed_byte3, _ = cs.high16_data
        total += byte2.numel() * byte2.element_size()
        total += packed_byte3.numel() * packed_byte3.element_size()
        total += cs.high16_codebook.numel() * cs.high16_codebook.element_size()
    else:
        total += cs.high16_data.numel() * cs.high16_data.element_size()

    return total


class PerParamCompressedAdamW:
    """AdamW wrapper with per-parameter high-16 compression."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._compressed = {}  # param_id → (cm, cv)
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
                cm = compress_fp32(state['exp_avg'])
                state['exp_avg'] = torch.empty(0, dtype=torch.float32, device=p.device)

                cv = compress_fp32(state['exp_avg_sq'])
                state['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=p.device)

                self._compressed[pid] = (cm, cv)

        self._is_compressed = True

    def _decompress_states(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                pid = id(p)
                if pid not in self._compressed:
                    continue
                cm, cv = self._compressed[pid]
                state = self.optimizer.state[p]
                state['exp_avg'] = decompress_fp32(cm)
                state['exp_avg_sq'] = decompress_fp32(cv)

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
        mode_counts = {}
        for pid, (cm, cv) in self._compressed.items():
            total_c += compressed_bytes(cm) + compressed_bytes(cv)
            total_o += cm.n * 4 + cv.n * 4
            for s in [cm, cv]:
                mode_counts[s.mode] = mode_counts.get(s.mode, 0) + 1
        return {
            'compressed_bytes': total_c,
            'original_bytes': total_o,
            'ratio': total_c / total_o,
            'savings_mb': (total_o - total_c) / 1024**2,
            'mode_counts': mode_counts,
        }


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    print("--- Losslessness Verification ---")

    # Unit tests
    print("  Test 1: 12-bit round-trip...")
    for n in [1, 2, 100, 10001, 1000000]:
        data = torch.randint(0, 4096, (n,), dtype=torch.int32, device='cuda')
        packed = pack_12bit_vectorized(data)
        restored = unpack_12bit_vectorized(packed, n)
        assert torch.all(data == restored), f"12-bit FAILED for n={n}"
    print("  ✓ OK")

    print("  Test 2: Bit-packed round-trip...")
    for bits in range(1, 9):
        for n in [1, 7, 8, 100, 100001]:
            max_val = (1 << bits) - 1
            data = torch.randint(0, max_val + 1, (n,), dtype=torch.uint8, device='cuda')
            packed = pack_bits_int64(data, bits)
            restored = unpack_bits_int64(packed, bits, n)
            assert torch.all(data == restored), f"FAILED bits={bits} n={n}"
    print("  ✓ OK")

    print("  Test 3: FP32 compress/decompress...")
    for _ in range(10):
        data = torch.randn(100000, dtype=torch.float32, device='cuda')
        cs = compress_fp32(data)
        restored = decompress_fp32(cs)
        assert torch.all(data == restored), f"FAILED mode={cs.mode}"
    print("  ✓ OK")

    # Full optimizer test
    print("  Test 4: Full optimizer (5 steps)...")
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = PerParamCompressedAdamW(inner)

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
    print("Per-Param High-16 Compressed FP32 Optimizer Benchmark")
    print("=" * 80)

    results = []
    n_warmup, n_measure = 10, 40

    for name, use_compressed in [("Standard FP32 AdamW", False),
                                  ("PerParam H16 Compressed", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()

        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = PerParamCompressedAdamW(inner) if use_compressed else inner

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
                print(f"  Modes: {stats['mode_counts']}")

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
