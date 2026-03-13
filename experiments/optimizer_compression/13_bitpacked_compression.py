"""
Experiment 13: Bit-Packed GPU Compression of FP32 Optimizer States

From entropy analysis:
- byte3 (sign + exp_MSBs): m has 46 unique (6 bits), v has 28 unique (5 bits)
- bytes 0-2: ~8 bits each (incompressible)

Approach: Extract byte3, encode as indices into codebook, bit-pack at actual
bit-width (5-6 bits instead of 8). Store bytes 0-2 raw.

Savings:
- v: (8-5)/8 of 1 byte = 0.375 bytes/element → 9.4% of FP32
- m: (8-6)/8 of 1 byte = 0.25 bytes/element → 6.25% of FP32
- Combined: ~7.8% of optimizer states → ~355 MB

GPU bit-packing: process elements in groups, pack indices using shifts/ORs
into minimal bytes. Groups of 8 elements pack into exactly B bytes (for B bits/idx).
"""

import torch
import time
import gc
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def bitpack_encode(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 indices at `bits` bits per index into uint8 tensor.

    Process in groups of 8 indices → `bits` bytes.
    Padding with zeros if not a multiple of 8.
    """
    device = indices.device
    n = indices.numel()

    # Pad to multiple of 8
    remainder = n % 8
    if remainder:
        pad = 8 - remainder
        indices = torch.cat([indices, torch.zeros(pad, dtype=torch.uint8, device=device)])

    n_padded = indices.numel()
    n_groups = n_padded // 8

    # Reshape to groups of 8
    groups = indices.reshape(n_groups, 8)  # [n_groups, 8]

    # For each group of 8 indices at B bits, we produce B bytes
    # Bit layout: index 0 occupies bits 0..B-1, index 1 occupies bits B..2B-1, etc.
    # Total: 8*B bits = B bytes

    # Build packed output: B bytes per group
    packed = torch.zeros(n_groups, bits, dtype=torch.uint8, device=device)

    for idx in range(8):
        # idx-th element contributes to multiple output bytes
        val = groups[:, idx].to(torch.int64)  # [n_groups]
        bit_offset = idx * bits

        for bit in range(bits):
            byte_pos = (bit_offset + bit) // 8
            bit_pos = (bit_offset + bit) % 8
            # Extract bit `bit` from val, put into byte_pos at bit_pos
            bit_val = ((val >> bit) & 1).to(torch.uint8)
            packed[:, byte_pos] |= (bit_val << bit_pos)

    return packed.flatten()


def bitpack_decode(packed: torch.Tensor, bits: int, n_elements: int) -> torch.Tensor:
    """Unpack uint8 tensor back to uint8 indices at `bits` bits per index."""
    device = packed.device

    n_padded = ((n_elements + 7) // 8) * 8
    n_groups = n_padded // 8

    # Reshape to groups of `bits` bytes
    groups = packed[:n_groups * bits].reshape(n_groups, bits)

    # Extract 8 indices per group
    result = torch.zeros(n_groups, 8, dtype=torch.uint8, device=device)

    for idx in range(8):
        bit_offset = idx * bits
        val = torch.zeros(n_groups, dtype=torch.int64, device=device)
        for bit in range(bits):
            byte_pos = (bit_offset + bit) // 8
            bit_pos = (bit_offset + bit) % 8
            bit_val = ((groups[:, byte_pos].to(torch.int64) >> bit_pos) & 1)
            val |= (bit_val << bit)
        result[:, idx] = val.to(torch.uint8)

    return result.flatten()[:n_elements]


class BitPackedState:
    """Compressed representation of one FP32 state tensor."""

    def __init__(self):
        self.byte0 = None    # uint8, raw
        self.byte1 = None    # uint8, raw
        self.byte2 = None    # uint8, raw
        self.byte3_packed = None  # bit-packed byte3 indices
        self.byte3_codebook = None  # uint8 codebook
        self.bits_per_idx = 0
        self.n = 0
        self.shape = None

    def compress(self, tensor: torch.Tensor):
        assert tensor.dtype == torch.float32
        self.shape = tensor.shape
        self.n = tensor.numel()
        device = tensor.device

        # View as 4 byte planes
        int32_view = tensor.contiguous().flatten().view(torch.int32)
        self.byte0 = (int32_view & 0xFF).to(torch.uint8)
        self.byte1 = ((int32_view >> 8) & 0xFF).to(torch.uint8)
        self.byte2 = ((int32_view >> 16) & 0xFF).to(torch.uint8)
        byte3 = ((int32_view >> 24) & 0xFF).to(torch.uint8)

        # Build codebook for byte3
        self.byte3_codebook = torch.unique(byte3)
        n_unique = len(self.byte3_codebook)
        self.bits_per_idx = max(1, math.ceil(math.log2(max(n_unique, 2))))

        # Create LUT: byte3 value → index
        lut = torch.zeros(256, dtype=torch.uint8, device=device)
        for i, v in enumerate(self.byte3_codebook):
            lut[v.item()] = i

        # Map to indices
        indices = lut[byte3.long()]

        # Bit-pack
        self.byte3_packed = bitpack_encode(indices, self.bits_per_idx)

        return self

    def decompress(self) -> torch.Tensor:
        device = self.byte0.device

        # Unpack byte3 indices
        indices = bitpack_decode(self.byte3_packed, self.bits_per_idx, self.n)
        byte3 = self.byte3_codebook[indices.long()]

        # Reconstruct int32
        result = (self.byte0.to(torch.int32) |
                  (self.byte1.to(torch.int32) << 8) |
                  (self.byte2.to(torch.int32) << 16) |
                  (byte3.to(torch.int32) << 24))

        return result.view(torch.float32).reshape(self.shape)

    def memory_bytes(self):
        total = 0
        for t in [self.byte0, self.byte1, self.byte2, self.byte3_packed, self.byte3_codebook]:
            if t is not None:
                total += t.numel() * t.element_size()
        return total

    def original_bytes(self):
        return self.n * 4

    def ratio(self):
        return self.memory_bytes() / self.original_bytes()


class BitPackedAdamW:
    """AdamW wrapper with bit-packed byte3 compression of FP32 states."""

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
                cm = BitPackedState().compress(state['exp_avg'])
                cv = BitPackedState().compress(state['exp_avg_sq'])
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
        bits_list = []
        for pid, (cm, cv) in self._compressed.items():
            total_c += cm.memory_bytes() + cv.memory_bytes()
            total_o += cm.original_bytes() + cv.original_bytes()
            bits_list.append((cm.bits_per_idx, cm.n, len(cm.byte3_codebook),
                              cv.bits_per_idx, cv.n, len(cv.byte3_codebook)))
        return {
            'compressed_bytes': total_c,
            'original_bytes': total_o,
            'ratio': total_c / total_o,
            'savings_mb': (total_o - total_c) / 1024**2,
            'bits_sample': bits_list[:5],
        }


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    print("--- Losslessness Verification ---")

    # Test bitpack round-trip
    print("  Test 1: Bitpack round-trip...")
    for bits in range(1, 9):
        n = 10000
        max_val = (1 << bits) - 1
        data = torch.randint(0, max_val + 1, (n,), dtype=torch.uint8, device='cuda')
        packed = bitpack_encode(data, bits)
        restored = bitpack_decode(packed, bits, n)
        assert torch.all(data == restored), f"Bitpack FAILED for {bits} bits"
    print("  ✓ Bitpack is lossless for all bit-widths 1-8")

    # Test full compress/decompress
    print("  Test 2: FP32 compress/decompress...")
    for _ in range(5):
        data = torch.randn(100000, dtype=torch.float32, device='cuda')
        bp = BitPackedState().compress(data)
        restored = bp.decompress()
        assert torch.all(data == restored), "FP32 compress/decompress FAILED"
    print("  ✓ FP32 compression is lossless")

    # Test with optimizer
    print("  Test 3: Full optimizer comparison...")
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = BitPackedAdamW(inner)

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
    print("Bit-Packed FP32 Optimizer Benchmark")
    print("=" * 80)

    results = []
    n_warmup, n_measure = 10, 40

    for name, use_compressed in [("Standard FP32 AdamW", False),
                                  ("BitPacked FP32 AdamW", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
        model.train()

        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = BitPackedAdamW(inner) if use_compressed else inner

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
                for mb, mn, mc, vb, vn, vc in stats['bits_sample']:
                    print(f"    Param ({mn:,} elem): m byte3={mc} unique→{mb}b, v byte3={vc} unique→{vb}b")

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
