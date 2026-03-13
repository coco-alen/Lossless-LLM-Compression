"""
Experiment 21: Optimized Byte3 Compression

Improvements over exp19:
1. Avoid byte3.long() by using int32 indexing instead of int64
2. Avoid torch.unique by using torch.bincount for faster codebook building
3. Pre-compute codebook from a small sample, then verify coverage
4. Reduce peak memory during pack_6bit by using smaller intermediates

Also explore: 2-byte compression (byte3+byte2 as 16-bit high) for more savings.
"""

import torch
import time
import gc
import math
from transformers import AutoModelForCausalLM


def pack_6bit(indices: torch.Tensor) -> torch.Tensor:
    """Pack uint8 indices (≤63) as 6-bit: 4 values → 3 bytes. Uses int32."""
    device = indices.device
    n = indices.numel()
    remainder = n % 4
    if remainder:
        indices = torch.cat([indices, torch.zeros(4 - remainder, dtype=torch.uint8, device=device)])
    groups = indices.reshape(-1, 4)
    combined = groups[:, 0].to(torch.int32)
    combined |= groups[:, 1].to(torch.int32) << 6
    combined |= groups[:, 2].to(torch.int32) << 12
    combined |= groups[:, 3].to(torch.int32) << 18
    packed = torch.stack([
        (combined & 0xFF).to(torch.uint8),
        ((combined >> 8) & 0xFF).to(torch.uint8),
        ((combined >> 16) & 0xFF).to(torch.uint8),
    ], dim=1).reshape(-1)
    return packed


def unpack_6bit(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack 6-bit: 3 bytes → 4 values."""
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


class OptimizedCompressedAdamW:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._params = None
        self._sizes = None
        self._offsets = None
        self._total_n = 0
        self._initialized = False

        # Compressed state
        self._m_byte012 = None
        self._m_packed3 = None
        self._m_codebook = None
        self._v_byte012 = None
        self._v_packed3 = None
        self._v_codebook = None
        self._is_compressed = False

    def _init(self):
        self._params = []
        self._sizes = []
        self._offsets = []
        offset = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state and 'exp_avg' in self.optimizer.state[p]:
                    self._params.append(p)
                    self._sizes.append(p.numel())
                    self._offsets.append(offset)
                    offset += p.numel()
        self._total_n = offset
        self._initialized = True

    def _gather(self, key):
        device = self._params[0].device
        flat = torch.empty(self._total_n, dtype=torch.float32, device=device)
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            flat[off:off+sz] = self.optimizer.state[p][key].flatten()
        return flat

    def _scatter(self, flat, key):
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            self.optimizer.state[p][key] = flat[off:off+sz].reshape(p.shape).clone()

    def _build_lut_fast(self, byte3: torch.Tensor):
        """Build codebook and LUT using bincount (faster than torch.unique)."""
        device = byte3.device
        # Count occurrences of each byte value
        counts = torch.bincount(byte3.to(torch.int32), minlength=256)
        # Codebook = values with count > 0
        present = (counts > 0).nonzero(as_tuple=True)[0]
        codebook = present.to(torch.uint8)
        n_unique = len(codebook)
        bits = max(1, math.ceil(math.log2(max(n_unique, 2))))

        # LUT: byte3 value → index
        lut = torch.zeros(256, dtype=torch.uint8, device=device)
        lut[present] = torch.arange(n_unique, device=device, dtype=torch.uint8)
        return codebook, lut, bits

    def _compress_one(self, flat_fp32):
        """Compress flat FP32. Returns (byte012, packed3, codebook)."""
        n = flat_fp32.numel()
        device = flat_fp32.device
        int32_view = flat_fp32.view(torch.int32)

        # Extract byte3
        byte3 = ((int32_view >> 24) & 0xFF).to(torch.uint8)

        # Build LUT (using bincount, no torch.unique storage leak)
        codebook, lut, bits = self._build_lut_fast(byte3)

        # Map byte3 → indices using int32 indexing (not int64)
        indices = lut[byte3.to(torch.int32)]
        del byte3

        # Pack indices
        packed3 = pack_6bit(indices) if bits <= 6 else indices
        del indices

        # Extract byte012 efficiently
        byte012 = int32_view.view(torch.uint8).reshape(n, 4)[:, :3].contiguous().reshape(-1)

        return byte012, packed3, codebook

    def _decompress_one(self, byte012, packed3, codebook, n):
        """Decompress back to flat FP32."""
        # Unpack byte3 indices
        bits = max(1, math.ceil(math.log2(max(len(codebook), 2))))
        indices = unpack_6bit(packed3, n) if bits <= 6 else packed3[:n]
        byte3 = codebook[indices.to(torch.int32)]
        del indices

        # Reconstruct
        b = byte012.reshape(n, 3)
        result = (b[:, 0].to(torch.int32) |
                  (b[:, 1].to(torch.int32) << 8) |
                  (b[:, 2].to(torch.int32) << 16) |
                  (byte3.to(torch.int32) << 24))
        return result.view(torch.float32)

    def _compress_states(self):
        if not self._initialized:
            self._init()
        device = self._params[0].device

        # Compress m
        flat_m = self._gather('exp_avg')
        self._m_byte012, self._m_packed3, self._m_codebook = self._compress_one(flat_m)
        del flat_m
        for p in self._params:
            self.optimizer.state[p]['exp_avg'] = torch.empty(0, dtype=torch.float32, device=device)

        # Compress v
        flat_v = self._gather('exp_avg_sq')
        self._v_byte012, self._v_packed3, self._v_codebook = self._compress_one(flat_v)
        del flat_v
        for p in self._params:
            self.optimizer.state[p]['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=device)

        self._is_compressed = True

    def _decompress_states(self):
        flat_m = self._decompress_one(self._m_byte012, self._m_packed3, self._m_codebook, self._total_n)
        self._scatter(flat_m, 'exp_avg')
        self._m_byte012 = self._m_packed3 = self._m_codebook = None
        del flat_m

        flat_v = self._decompress_one(self._v_byte012, self._v_packed3, self._v_codebook, self._total_n)
        self._scatter(flat_v, 'exp_avg_sq')
        self._v_byte012 = self._v_packed3 = self._v_codebook = None
        del flat_v

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
        total_c = 0
        total_o = self._total_n * 4 * 2
        for name, b012, p3, cb in [
            ('m', self._m_byte012, self._m_packed3, self._m_codebook),
            ('v', self._v_byte012, self._v_packed3, self._v_codebook),
        ]:
            if b012 is not None:
                c = b012.numel() + p3.numel() + cb.numel()
                o = self._total_n * 4
                bits = max(1, math.ceil(math.log2(max(len(cb), 2))))
                total_c += c
                print(f"  {name}: {len(cb)} unique → {bits}b, "
                      f"ratio={c/o*100:.1f}%, saves={o-c:.0f} B = {(o-c)/1024**2:.0f} MB")
        return {'ratio': total_c / total_o, 'savings_mb': (total_o - total_c) / 1024**2}


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    print("--- Verify ---")
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = OptimizedCompressedAdamW(inner)

    for s in range(5):
        torch.manual_seed(s + 100)
        ids = torch.randint(100, 10000, (2, 128), device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            m1(input_ids=ids, labels=ids).loss.backward()
        o1.step(); o1.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            m2(input_ids=ids, labels=ids).loss.backward()
        o2.step(); o2.zero_grad()

    max_diff = max((p1.data - p2.data).abs().max().item()
                   for p1, p2 in zip(m1.parameters(), m2.parameters()))
    print(f"  Max diff: {max_diff}" + (" ✓" if max_diff == 0 else " ✗"))
    del m1, m2, o1, o2, inner
    gc.collect(); torch.cuda.empty_cache()
    return max_diff == 0


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "=" * 80)
    print("Optimized Byte3 Compression Benchmark")
    print("=" * 80)

    results = []
    for name, use_comp in [("Standard FP32 AdamW", False),
                            ("Opt Byte3 Compressed", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()
        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = OptimizedCompressedAdamW(inner) if use_comp else inner

        for _ in range(10):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()

        gc.collect(); torch.cuda.empty_cache()
        gpu_mem = torch.cuda.memory_allocated() / 1024**2

        if use_comp:
            stats = opt.get_stats()
            print(f"  Total: ratio={stats['ratio']*100:.1f}%, savings={stats['savings_mb']:.0f} MB")

        torch.cuda.synchronize()
        times = []
        for _ in range(40):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Mem: {gpu_mem:.0f} MB, Peak: {peak:.0f} MB, Step: {avg*1000:.1f} ms")
        results.append({'method': name, 'gpu_mem': gpu_mem, 'peak': peak, 'step_ms': avg*1000})

        del model, inner, opt
        gc.collect(); torch.cuda.empty_cache()

    bl = results[0]
    print(f"\n{'='*65}")
    print(f"{'Method':<25} {'Mem':>7} {'ΔMem':>7} {'Peak':>7} {'Step':>7} {'Slow':>5}")
    print("-" * 60)
    for r in results:
        dm = r['gpu_mem'] - bl['gpu_mem']
        s = r['step_ms'] / bl['step_ms']
        print(f"{r['method']:<25} {r['gpu_mem']:>6.0f}M {dm:>+6.0f}M {r['peak']:>6.0f}M "
              f"{r['step_ms']:>6.1f} {s:>4.2f}x")


if __name__ == '__main__':
    ok = verify_lossless()
    if ok:
        benchmark()
