"""
Experiment 33: Final Optimized Byte3 Compression

Best practical approach: 6-bit fixed-length byte3 coding.
Saves 6.25% = ~285 MB on Qwen3-0.6B (596M params).

Optimizations vs exp19/21:
1. Zero-copy: decompress → flat buffer → views → optimizer.step → compress directly
2. Careful memory management: delete intermediates immediately
3. Avoid torch.compile (causes 6GB peak memory spike)
4. Minimize intermediate tensor creation
"""

import torch
import time
import gc
import math
from transformers import AutoModelForCausalLM


class CompressedAdamW:
    """Lossless FP32 optimizer state compression via byte3 6-bit packing.

    FP32 values have only ~47 unique byte3 (MSB) values.
    Store byte012 (lower 3 bytes) raw + byte3 indices packed at 6 bits.
    Total: 3 + 0.75 = 3.75 bytes per value (93.75%).
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._params = None
        self._sizes = None
        self._offsets = None
        self._total_n = 0
        self._flat_m = None
        self._flat_v = None
        self._m_data = None  # (byte012, packed3, codebook, n)
        self._v_data = None
        self._is_compressed = False
        self._first_step = True

    def _init_params(self):
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

    def _set_views(self, flat, key):
        """Set each param's state as a view into flat buffer."""
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            self.optimizer.state[p][key] = flat[off:off+sz].view(p.shape)

    def _initial_gather(self, key):
        """First-time gather from per-param states to flat buffer."""
        device = self._params[0].device
        flat = torch.empty(self._total_n, dtype=torch.float32, device=device)
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            flat[off:off+sz] = self.optimizer.state[p][key].flatten()
        return flat

    @staticmethod
    @torch.no_grad()
    def _compress(flat_fp32):
        """Compress flat FP32 → (byte012, packed3, codebook, n)."""
        n = flat_fp32.numel()
        device = flat_fp32.device
        int32 = flat_fp32.view(torch.int32)

        # Extract byte3 and build LUT
        byte3 = ((int32 >> 24) & 0xFF).to(torch.uint8)
        counts = torch.bincount(byte3.to(torch.int32), minlength=256)
        present = (counts > 0).nonzero(as_tuple=True)[0]
        codebook = present.to(torch.uint8)
        n_unique = len(codebook)
        lut = torch.zeros(256, dtype=torch.uint8, device=device)
        lut[present] = torch.arange(n_unique, device=device, dtype=torch.uint8)

        # Map byte3 → indices
        indices = lut[byte3.to(torch.int32)]
        del byte3, counts, lut, present

        # Pad to multiple of 4 and pack 6-bit
        pad = (4 - n % 4) % 4
        if pad:
            indices = torch.cat([indices, torch.zeros(pad, dtype=torch.uint8, device=device)])
        groups = indices.reshape(-1, 4)
        combined = (groups[:, 0].to(torch.int32) |
                    (groups[:, 1].to(torch.int32) << 6) |
                    (groups[:, 2].to(torch.int32) << 12) |
                    (groups[:, 3].to(torch.int32) << 18))
        packed3 = torch.stack([
            (combined & 0xFF).to(torch.uint8),
            ((combined >> 8) & 0xFF).to(torch.uint8),
            ((combined >> 16) & 0xFF).to(torch.uint8),
        ], dim=1).reshape(-1)
        del indices, groups, combined

        # Extract byte012
        byte012 = int32.view(torch.uint8).reshape(n, 4)[:, :3].contiguous().reshape(-1)

        return byte012, packed3, codebook, n

    @staticmethod
    @torch.no_grad()
    def _decompress(data):
        """Decompress (byte012, packed3, codebook, n) → flat FP32."""
        byte012, packed3, codebook, n = data

        # Unpack 6-bit
        n_groups = (n + 3) // 4
        groups = packed3[:n_groups * 3].reshape(n_groups, 3)
        combined = (groups[:, 0].to(torch.int32) |
                    (groups[:, 1].to(torch.int32) << 8) |
                    (groups[:, 2].to(torch.int32) << 16))
        indices = torch.stack([
            (combined & 0x3F).to(torch.uint8),
            ((combined >> 6) & 0x3F).to(torch.uint8),
            ((combined >> 12) & 0x3F).to(torch.uint8),
            ((combined >> 18) & 0x3F).to(torch.uint8),
        ], dim=1).reshape(-1)[:n]
        del groups, combined

        byte3 = codebook[indices.to(torch.int32)]
        del indices

        b = byte012.reshape(n, 3)
        result = (b[:, 0].to(torch.int32) |
                  (b[:, 1].to(torch.int32) << 8) |
                  (b[:, 2].to(torch.int32) << 16) |
                  (byte3.to(torch.int32) << 24))
        return result.view(torch.float32)

    def _compress_states(self):
        if self._first_step:
            self._init_params()
            self._flat_m = self._initial_gather('exp_avg')
            self._flat_v = self._initial_gather('exp_avg_sq')
            self._first_step = False

        device = self._params[0].device

        # Compress m, then free flat buffer immediately
        self._m_data = self._compress(self._flat_m)
        del self._flat_m; self._flat_m = None

        # Compress v
        self._v_data = self._compress(self._flat_v)
        del self._flat_v; self._flat_v = None

        # Clear per-param state references
        for p in self._params:
            self.optimizer.state[p]['exp_avg'] = torch.empty(0, dtype=torch.float32, device=device)
            self.optimizer.state[p]['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=device)

        self._is_compressed = True

    def _decompress_states(self):
        self._flat_m = self._decompress(self._m_data)
        self._m_data = None
        self._set_views(self._flat_m, 'exp_avg')

        self._flat_v = self._decompress(self._v_data)
        self._v_data = None
        self._set_views(self._flat_v, 'exp_avg_sq')

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
        total_c, total_o = 0, self._total_n * 4 * 2
        for name, data in [('m', self._m_data), ('v', self._v_data)]:
            if data:
                b012, p3, cb, n = data
                c = b012.numel() + p3.numel() + cb.numel()
                total_c += c
                print(f"  {name}: {len(cb)} unique, ratio={c/(n*4)*100:.1f}%")
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
    o2 = CompressedAdamW(inner)

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
    print("Lossless FP32 Optimizer Compression (Byte3 6-bit)")
    print("=" * 80)

    results = []
    for name, use_comp in [("Standard FP32 AdamW", False),
                            ("Compressed AdamW", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()
        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = CompressedAdamW(inner) if use_comp else inner

        # Warmup
        for _ in range(10):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()

        # Measure steady-state memory AFTER warmup
        gc.collect(); torch.cuda.empty_cache()
        gpu_mem = torch.cuda.memory_allocated() / 1024**2

        if use_comp:
            stats = opt.get_stats()
            print(f"  Total: ratio={stats['ratio']*100:.1f}%, savings={stats['savings_mb']:.0f} MB")

        # Reset peak for timing phase
        torch.cuda.reset_peak_memory_stats()

        # Benchmark
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
    print(f"\n{'='*60}")
    print(f"{'Method':<25} {'Mem':>7} {'ΔMem':>7} {'Peak':>7} {'ΔPeak':>7} {'Step':>7} {'Slow':>5}")
    print("-" * 65)
    for r in results:
        dm = r['gpu_mem'] - bl['gpu_mem']
        dp = r['peak'] - bl['peak']
        s = r['step_ms'] / bl['step_ms']
        print(f"{r['method']:<25} {r['gpu_mem']:>6.0f}M {dm:>+6.0f}M {r['peak']:>6.0f}M "
              f"{dp:>+6.0f}M {r['step_ms']:>6.1f} {s:>4.2f}x")


if __name__ == '__main__':
    ok = verify_lossless()
    if ok:
        benchmark()
