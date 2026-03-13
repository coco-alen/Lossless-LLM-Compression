"""
Experiment 29: Fused Single-Pass Compression

Key insight: instead of separate byte012 extraction and 6-bit packing,
do everything in a SINGLE pass over the data.

For every 4 FP32 values (16 bytes input):
  - Output 12 bytes of byte012 (lower 3 bytes of each value)
  - Output 3 bytes of 6-bit packed byte3 indices
  Total: 15 bytes per 4 values = 93.75%

Single contiguous read + single contiguous write = maximum memory bandwidth.
Eliminate all intermediate tensors.

Also try: keeping byte012 as int32 views (no copy needed!) and only
compressing byte3, which is the approach with minimum memory movement.
"""

import torch
import time
import gc
import math
from transformers import AutoModelForCausalLM


def build_lut(int32_flat):
    """Build codebook + LUT using bincount. Returns (codebook, lut)."""
    byte3 = ((int32_flat >> 24) & 0xFF).to(torch.uint8)
    counts = torch.bincount(byte3.to(torch.int32), minlength=256)
    present = (counts > 0).nonzero(as_tuple=True)[0]
    codebook = present.to(torch.uint8)
    n_unique = len(codebook)
    lut = torch.zeros(256, dtype=torch.uint8, device=int32_flat.device)
    lut[present] = torch.arange(n_unique, device=int32_flat.device, dtype=torch.uint8)
    return codebook, lut


class FusedCompressedAdamW:
    """Approach: store lower 24 bits as-is via masking, compress only byte3.

    Key insight: instead of extracting byte012 as a separate tensor,
    just zero out byte3 in the original int32 and store that.
    For byte3, store the 6-bit packed indices.

    Decompress: restore byte3 from indices, OR it back into the int32.

    This minimizes data movement:
    - Compress: 1 read (full int32) + 1 write (zeroed int32) + 1 write (packed byte3)
    - Decompress: 1 read (zeroed int32) + 1 read (packed byte3) + 1 write (full int32)
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._params = None
        self._sizes = None
        self._offsets = None
        self._total_n = 0
        self._flat_m = None
        self._flat_v = None
        self._m_data = None  # (low24_int32, packed3, codebook, n)
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
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            self.optimizer.state[p][key] = flat[off:off+sz].view(p.shape)

    def _initial_gather(self, key):
        device = self._params[0].device
        flat = torch.empty(self._total_n, dtype=torch.float32, device=device)
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            flat[off:off+sz] = self.optimizer.state[p][key].flatten()
        return flat

    def _compress(self, flat_fp32):
        """Compress: store low24 as int32 + packed byte3 indices."""
        n = flat_fp32.numel()
        int32_view = flat_fp32.view(torch.int32)

        # Extract byte3 and build LUT
        byte3 = ((int32_view >> 24) & 0xFF).to(torch.uint8)
        codebook, lut = build_lut(int32_view)
        indices = lut[byte3.to(torch.int32)]
        del byte3

        # Store low24 by zeroing byte3 (in-place would be ideal but flat may be needed)
        low24 = (int32_view & 0x00FFFFFF).to(torch.int32)

        # Pad indices to multiple of 4 for packing
        pad = (4 - n % 4) % 4
        if pad:
            indices = torch.cat([indices, torch.zeros(pad, dtype=torch.uint8, device=indices.device)])

        # 6-bit pack: 4 values → 3 bytes
        groups = indices.reshape(-1, 4)
        combined = groups[:, 0].to(torch.int32)
        combined = combined | (groups[:, 1].to(torch.int32) << 6)
        combined = combined | (groups[:, 2].to(torch.int32) << 12)
        combined = combined | (groups[:, 3].to(torch.int32) << 18)
        packed3 = torch.stack([
            (combined & 0xFF).to(torch.uint8),
            ((combined >> 8) & 0xFF).to(torch.uint8),
            ((combined >> 16) & 0xFF).to(torch.uint8),
        ], dim=1).reshape(-1)
        del indices, groups, combined

        return low24, packed3, codebook, n

    def _decompress(self, data):
        """Decompress: unpack byte3, OR into low24."""
        low24, packed3, codebook, n = data

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

        result = low24 | (byte3.to(torch.int32) << 24)
        return result.view(torch.float32)

    def _compress_states(self):
        if self._first_step:
            self._init_params()
            self._flat_m = self._initial_gather('exp_avg')
            self._flat_v = self._initial_gather('exp_avg_sq')
            self._first_step = False

        device = self._params[0].device

        self._m_data = self._compress(self._flat_m)
        del self._flat_m; self._flat_m = None

        self._v_data = self._compress(self._flat_v)
        del self._flat_v; self._flat_v = None

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
                low24, p3, cb, n = data
                c = low24.numel() * 4 + p3.numel() + cb.numel()
                # Note: low24 stored as int32 = same 4 bytes. But byte3 is zeroed.
                # Actual savings = n bytes (byte3) - packed3 bytes
                packed_bytes = p3.numel()
                orig_byte3_bytes = n
                savings = orig_byte3_bytes - packed_bytes
                total_c += n * 4 - savings  # effective compressed size
                ratio = (n * 4 - savings) / (n * 4) * 100
                print(f"  {name}: {len(cb)} unique, packed3={packed_bytes/1024**2:.0f} MB, "
                      f"ratio={ratio:.1f}%, saves {savings/1024**2:.0f} MB")
        # Total savings from byte3 compression
        m_low24, m_p3, m_cb, m_n = self._m_data
        v_low24, v_p3, v_cb, v_n = self._v_data
        total_savings = (m_n - m_p3.numel()) + (v_n - v_p3.numel())
        return {'ratio': 1 - total_savings / total_o,
                'savings_mb': total_savings / 1024**2}


class InPlaceCompressedAdamW:
    """Alternative: store data as int32 with byte3 masked + separate packed byte3.

    This avoids creating a separate low24 tensor — just mask in-place in the flat buffer.
    The flat buffer is 4 bytes per value (same as FP32), but byte3 is always 0.
    The packed byte3 is 0.75 bytes per value.
    Total: 4.75 bytes per value vs 4 = WORSE!

    Wait, that doesn't save anything. The trick is:
    The low24 in int32 uses 4 bytes but byte3 is 0. We can store the packed byte3
    in the SPACE FREED by zeroing byte3. But int32 storage is contiguous...

    Better approach: Store low 3 bytes as uint8[N*3] and packed byte3 as uint8[N*3/4].
    That's 3 + 0.75 = 3.75 bytes per value. Saves 6.25%.

    Even better: DON'T create low24 tensor. Instead, store the FULL int32 tensor
    but with byte3 zeroed. Then packed byte3 is extra 0.75N bytes.
    Net: N*4 + 0.75N = 4.75N bytes. WORSE than 4N!

    The savings come from:
    - Original: N*4 bytes (full FP32)
    - Compressed: N*3 (byte012) + N*3/4 (packed byte3) = 3.75N bytes
    - Savings: 0.25N bytes = 6.25%

    So we MUST extract byte012 separately to get savings.
    The low24 as int32 approach stores 4 bytes per value (no savings from byte3).

    Let's go back to the byte012 + packed3 approach but optimize it.
    """
    pass


class OptimizedByte3AdamW:
    """The leanest possible byte3 compression.

    Focus on minimizing kernel launches and intermediate tensors.
    Use contiguous memory and minimal operations.
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
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            self.optimizer.state[p][key] = flat[off:off+sz].view(p.shape)

    def _initial_gather(self, key):
        device = self._params[0].device
        flat = torch.empty(self._total_n, dtype=torch.float32, device=device)
        for p, off, sz in zip(self._params, self._offsets, self._sizes):
            flat[off:off+sz] = self.optimizer.state[p][key].flatten()
        return flat

    @staticmethod
    @torch.no_grad()
    def _compress(flat_fp32):
        n = flat_fp32.numel()
        int32 = flat_fp32.view(torch.int32)
        device = int32.device

        # byte3: extract, build LUT, map indices, pack — all minimal alloc
        byte3 = ((int32 >> 24) & 0xFF).to(torch.uint8)

        # Build LUT via bincount (no torch.unique)
        counts = torch.bincount(byte3.to(torch.int32), minlength=256)
        present = (counts > 0).nonzero(as_tuple=True)[0]
        codebook = present.to(torch.uint8)
        n_unique = len(codebook)
        lut = torch.zeros(256, dtype=torch.uint8, device=device)
        lut[present] = torch.arange(n_unique, device=device, dtype=torch.uint8)

        # Map and pack in one go
        indices = lut[byte3.to(torch.int32)]
        del byte3, counts, lut, present

        # Pad to multiple of 4
        pad = (4 - n % 4) % 4
        if pad:
            indices = torch.cat([indices, torch.zeros(pad, dtype=torch.uint8, device=device)])

        # 6-bit pack
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

        # byte012: view as uint8, take first 3 of every 4
        byte012 = int32.view(torch.uint8).reshape(n, 4)[:, :3].contiguous().reshape(-1)

        return byte012, packed3, codebook, n

    @staticmethod
    @torch.no_grad()
    def _decompress(data):
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

        self._m_data = self._compress(self._flat_m)
        del self._flat_m; self._flat_m = None

        self._v_data = self._compress(self._flat_v)
        del self._flat_v; self._flat_v = None

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


def verify(model_name="Qwen/Qwen3-0.6B"):
    print("--- Verify ---")
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = OptimizedByte3AdamW(inner)

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
    print("Optimized Byte3 FP32 Optimizer")
    print("=" * 80)

    results = []
    for name, use_comp in [("Standard FP32 AdamW", False),
                            ("OptByte3 Compressed", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()
        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = OptimizedByte3AdamW(inner) if use_comp else inner

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
    print(f"\n{'='*60}")
    print(f"{'Method':<25} {'Mem':>7} {'ΔMem':>7} {'Peak':>7} {'Step':>7} {'Slow':>5}")
    print("-" * 58)
    for r in results:
        dm = r['gpu_mem'] - bl['gpu_mem']
        s = r['step_ms'] / bl['step_ms']
        print(f"{r['method']:<25} {r['gpu_mem']:>6.0f}M {dm:>+6.0f}M {r['peak']:>6.0f}M "
              f"{r['step_ms']:>6.1f} {s:>4.2f}x")


if __name__ == '__main__':
    ok = verify()
    if ok:
        benchmark()
