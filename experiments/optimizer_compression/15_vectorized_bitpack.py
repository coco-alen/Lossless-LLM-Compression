"""
Experiment 15: Vectorized Bit-Packing via int64 Accumulation

Key insight: Pack 8 values at B bits each into a single int64 (8*B ≤ 64 bits).
Then extract B bytes from the int64. ALL operations are vectorized torch ops.

For B=5: 8 values × 5 bits = 40 bits → 5 bytes per 8 values → 62.5% of raw uint8
For B=6: 8 values × 6 bits = 48 bits → 6 bytes per 8 values → 75% of raw uint8

FP32 byte decomposition:
- byte3 (sign + exp_MSBs): m ~44 unique (6 bits), v ~27 unique (5 bits)
- bytes 0-2: ~8 bits each (incompressible, store raw)

Expected savings:
- v byte3: 5/8 = 37.5% savings on 1 byte per element → 9.4% of FP32
- m byte3: 2/8 = 25% savings on 1 byte per element → 6.25% of FP32
- Combined: ~7.8% of FP32 states → ~355 MB

Approach: Batch ALL parameters into one tensor, single global codebook,
single vectorized operation. No per-parameter Python loops.
"""

import torch
import time
import gc
import math
import numpy as np
from transformers import AutoModelForCausalLM


def pack_bits_vectorized(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 indices at `bits` bits per value into packed uint8 tensor.

    Uses int64 accumulation: groups of 8 values → 1 int64 → `bits` bytes.
    Fully vectorized, no Python loops over data.
    """
    device = indices.device
    n = indices.numel()

    # Pad to multiple of 8
    remainder = n % 8
    if remainder:
        pad = 8 - remainder
        indices = torch.cat([indices, torch.zeros(pad, dtype=torch.uint8, device=device)])

    groups = indices.reshape(-1, 8).to(torch.int64)  # [N/8, 8]

    # Accumulate into int64: combined = v0 | (v1 << B) | (v2 << 2B) | ... | (v7 << 7B)
    shifts = torch.arange(8, device=device, dtype=torch.int64) * bits  # [8]
    combined = (groups << shifts.unsqueeze(0)).sum(dim=1)  # [N/8] int64 values

    # Extract `bits` bytes from each int64
    byte_shifts = torch.arange(bits, device=device, dtype=torch.int64) * 8  # [bits]
    packed = ((combined.unsqueeze(1) >> byte_shifts.unsqueeze(0)) & 0xFF).to(torch.uint8)
    # packed shape: [N/8, bits]

    return packed.flatten()


def unpack_bits_vectorized(packed: torch.Tensor, bits: int, n_elements: int) -> torch.Tensor:
    """Unpack packed uint8 tensor back to uint8 indices at `bits` bits per value."""
    device = packed.device
    n_padded = ((n_elements + 7) // 8) * 8
    n_groups = n_padded // 8

    groups = packed[:n_groups * bits].reshape(n_groups, bits).to(torch.int64)

    # Reconstruct int64 from `bits` bytes
    byte_shifts = torch.arange(bits, device=device, dtype=torch.int64) * 8
    combined = (groups << byte_shifts.unsqueeze(0)).sum(dim=1)  # [N/8]

    # Extract 8 values from each int64
    bit_shifts = torch.arange(8, device=device, dtype=torch.int64) * bits
    mask = (1 << bits) - 1
    result = ((combined.unsqueeze(1) >> bit_shifts.unsqueeze(0)) & mask).to(torch.uint8)
    # result shape: [N/8, 8]

    return result.flatten()[:n_elements]


class BatchedBitPackedOptimizer:
    """AdamW wrapper that compresses FP32 states by bit-packing byte3.

    Key design: Concatenate ALL parameters into single tensors, compress once.
    No per-parameter Python loops during compress/decompress.
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._param_list = []  # ordered list of (param, key) tuples
        self._param_sizes = []  # numel for each param
        self._total_elements = 0

        # Compressed storage
        self._m_compressed = None  # (byte012, packed_byte3, codebook, bits, n)
        self._v_compressed = None
        self._is_compressed = False

        # Build param list on first use
        self._initialized = False

    def _init_param_list(self):
        self._param_list = []
        self._param_sizes = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in self.optimizer.state:
                    continue
                state = self.optimizer.state[p]
                if 'exp_avg' not in state:
                    continue
                self._param_list.append(p)
                self._param_sizes.append(p.numel())
        self._total_elements = sum(self._param_sizes)
        self._initialized = True

    def _gather_state(self, key: str) -> torch.Tensor:
        """Gather one state (exp_avg or exp_avg_sq) from all params into one flat tensor."""
        tensors = []
        for p in self._param_list:
            tensors.append(self.optimizer.state[p][key].flatten())
        return torch.cat(tensors)

    def _scatter_state(self, flat: torch.Tensor, key: str):
        """Scatter flat tensor back into per-param optimizer states."""
        offset = 0
        for p, size in zip(self._param_list, self._param_sizes):
            self.optimizer.state[p][key] = flat[offset:offset+size].reshape(p.shape)
            offset += size

    def _compress_one(self, flat_fp32: torch.Tensor):
        """Compress a flat FP32 tensor by bit-packing byte3."""
        n = flat_fp32.numel()
        device = flat_fp32.device
        int32_view = flat_fp32.view(torch.int32)

        # Extract 4 byte planes
        byte012 = torch.stack([
            (int32_view & 0xFF).to(torch.uint8),
            ((int32_view >> 8) & 0xFF).to(torch.uint8),
            ((int32_view >> 16) & 0xFF).to(torch.uint8),
        ], dim=1)  # [N, 3]

        byte3 = ((int32_view >> 24) & 0xFF).to(torch.uint8)

        # Build global codebook for byte3
        codebook = torch.unique(byte3)
        n_unique = len(codebook)
        bits = max(1, math.ceil(math.log2(max(n_unique, 2))))

        # Create LUT: byte3 value → index (vectorized)
        lut = torch.zeros(256, dtype=torch.uint8, device=device)
        lut[codebook.long()] = torch.arange(n_unique, device=device, dtype=torch.uint8)

        # Map to indices
        indices = lut[byte3.long()]

        # Bit-pack (fully vectorized)
        packed = pack_bits_vectorized(indices, bits)

        return (byte012, packed, codebook, bits, n)

    def _decompress_one(self, compressed) -> torch.Tensor:
        """Decompress back to flat FP32 tensor."""
        byte012, packed, codebook, bits, n = compressed
        device = byte012.device

        # Unpack indices
        indices = unpack_bits_vectorized(packed, bits, n)

        # Look up byte3 from codebook
        byte3 = codebook[indices.long()]

        # Reconstruct int32
        result = (byte012[:, 0].to(torch.int32) |
                  (byte012[:, 1].to(torch.int32) << 8) |
                  (byte012[:, 2].to(torch.int32) << 16) |
                  (byte3.to(torch.int32) << 24))

        return result.view(torch.float32)

    def _compress_states(self):
        if not self._initialized:
            self._init_param_list()

        # Gather all m and v into single tensors
        m_flat = self._gather_state('exp_avg')
        v_flat = self._gather_state('exp_avg_sq')

        # Compress each
        self._m_compressed = self._compress_one(m_flat)
        self._v_compressed = self._compress_one(v_flat)

        # Free original states (replace with empty tensors)
        for p in self._param_list:
            state = self.optimizer.state[p]
            state['exp_avg'] = torch.empty(0, dtype=torch.float32, device=p.device)
            state['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=p.device)

        del m_flat, v_flat
        self._is_compressed = True

    def _decompress_states(self):
        m_flat = self._decompress_one(self._m_compressed)
        v_flat = self._decompress_one(self._v_compressed)

        self._scatter_state(m_flat, 'exp_avg')
        self._scatter_state(v_flat, 'exp_avg_sq')

        self._m_compressed = None
        self._v_compressed = None
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
        stats = {}
        for name, compressed in [('m', self._m_compressed), ('v', self._v_compressed)]:
            if compressed is None:
                continue
            byte012, packed, codebook, bits, n = compressed
            compressed_bytes = byte012.numel() + packed.numel() + codebook.numel()
            original_bytes = n * 4
            stats[name] = {
                'n_unique': len(codebook),
                'bits': bits,
                'compressed_bytes': compressed_bytes,
                'original_bytes': original_bytes,
                'ratio': compressed_bytes / original_bytes,
                'savings_mb': (original_bytes - compressed_bytes) / 1024**2,
            }
        return stats


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    print("--- Losslessness Verification ---")

    # Test vectorized bitpack round-trip
    print("  Test 1: Vectorized bitpack round-trip...")
    for bits in range(1, 9):
        for n in [100, 1000, 10001, 100000, 1000000]:
            max_val = (1 << bits) - 1
            data = torch.randint(0, max_val + 1, (n,), dtype=torch.uint8, device='cuda')
            packed = pack_bits_vectorized(data, bits)
            restored = unpack_bits_vectorized(packed, bits, n)
            assert torch.all(data == restored), f"FAILED for bits={bits}, n={n}"
    print("  ✓ Vectorized bitpack is lossless for all bit-widths 1-8")

    # Speed test
    print("  Test 2: Bitpack speed...")
    n = 600_000_000  # ~596M params
    data = torch.randint(0, 32, (n,), dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    packed = pack_bits_vectorized(data, 5)
    torch.cuda.synchronize()
    t_pack = time.perf_counter() - t0

    t0 = time.perf_counter()
    restored = unpack_bits_vectorized(packed, 5, n)
    torch.cuda.synchronize()
    t_unpack = time.perf_counter() - t0

    assert torch.all(data == restored)
    print(f"  Pack: {t_pack*1000:.1f} ms, Unpack: {t_unpack*1000:.1f} ms "
          f"({n/1e6:.0f}M elements)")
    print(f"  Packed size: {packed.numel()/1024**2:.1f} MB vs raw: {data.numel()/1024**2:.1f} MB "
          f"({packed.numel()/data.numel()*100:.1f}%)")

    del data, packed, restored
    gc.collect(); torch.cuda.empty_cache()

    # Test full compress/decompress
    print("  Test 3: FP32 round-trip...")
    opt = BatchedBitPackedOptimizer(None)
    for _ in range(5):
        data = torch.randn(100000, dtype=torch.float32, device='cuda')
        compressed = opt._compress_one(data)
        restored = opt._decompress_one(compressed)
        assert torch.all(data == restored), "FP32 round-trip FAILED"
    print("  ✓ FP32 compression is lossless")

    # Full optimizer test
    print("  Test 4: Full optimizer comparison...")
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = BatchedBitPackedOptimizer(inner)

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
    print("Vectorized Bit-Packed FP32 Optimizer Benchmark")
    print("=" * 80)

    results = []
    n_warmup, n_measure = 10, 40

    for name, use_compressed in [("Standard FP32 AdamW", False),
                                  ("VecBitPack FP32 AdamW", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()

        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = BatchedBitPackedOptimizer(inner) if use_compressed else inner

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
                for k, s in stats.items():
                    print(f"  {k}: {s['n_unique']} unique → {s['bits']}b, "
                          f"ratio={s['ratio']*100:.1f}%, savings={s['savings_mb']:.0f} MB")

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
