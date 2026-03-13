"""
Experiment 37: Grouped Byte3 Compression

Group parameters into medium-sized batches (~32M elements = 128MB FP32).
Each group gets its own flat buffer for efficient GPU kernels.
During decompress: process one group at a time, free compressed data per group.
During compress: process one group at a time, free flat buffer per group.

Target: ~-285 MB savings, ~1.5x slowdown, minimal peak spike.
"""

import torch
import time
import gc
from transformers import AutoModelForCausalLM

GROUP_SIZE = 64 * 1024 * 1024  # 64M elements per group


@torch.no_grad()
def _pack_byte3(int32_chunk, lut):
    """Extract byte3, map through LUT, pack 6-bit. Returns (byte012, packed3)."""
    n = int32_chunk.numel()
    byte3 = ((int32_chunk >> 24) & 0xFF).to(torch.uint8)
    indices = lut[byte3.to(torch.int32)]
    del byte3
    # Pad to multiple of 4
    pad = (4 - n % 4) % 4
    if pad:
        indices = torch.cat([indices, torch.zeros(pad, dtype=torch.uint8, device=int32_chunk.device)])
    groups = indices.reshape(-1, 4)
    combined = (groups[:, 0].to(torch.int32) |
                (groups[:, 1].to(torch.int32) << 6) |
                (groups[:, 2].to(torch.int32) << 12) |
                (groups[:, 3].to(torch.int32) << 18))
    del indices, groups
    packed3 = torch.stack([
        (combined & 0xFF).to(torch.uint8),
        ((combined >> 8) & 0xFF).to(torch.uint8),
        ((combined >> 16) & 0xFF).to(torch.uint8),
    ], dim=1).reshape(-1)
    del combined
    byte012 = int32_chunk.view(torch.uint8).reshape(n, 4)[:, :3].contiguous().reshape(-1)
    return byte012, packed3


@torch.no_grad()
def _unpack_byte3(byte012, packed3, codebook, n):
    """Unpack 6-bit packed3, reconstruct FP32."""
    n_groups = (n + 3) // 4
    groups = packed3[:n_groups * 3].reshape(n_groups, 3)
    combined = (groups[:, 0].to(torch.int32) |
                (groups[:, 1].to(torch.int32) << 8) |
                (groups[:, 2].to(torch.int32) << 16))
    del groups
    indices = torch.stack([
        (combined & 0x3F).to(torch.uint8),
        ((combined >> 6) & 0x3F).to(torch.uint8),
        ((combined >> 12) & 0x3F).to(torch.uint8),
        ((combined >> 18) & 0x3F).to(torch.uint8),
    ], dim=1).reshape(-1)[:n]
    del combined
    byte3 = codebook[indices.to(torch.int32)]
    del indices
    b = byte012.reshape(n, 3)
    result = (b[:, 0].to(torch.int32) |
              (b[:, 1].to(torch.int32) << 8) |
              (b[:, 2].to(torch.int32) << 16) |
              (byte3.to(torch.int32) << 24))
    return result.view(torch.float32)


# Try to compile the hot functions
try:
    _pack_byte3_compiled = torch.compile(_pack_byte3)
    _unpack_byte3_compiled = torch.compile(_unpack_byte3)
    USE_COMPILE = True
except Exception:
    USE_COMPILE = False

_pack_fn = _pack_byte3_compiled if USE_COMPILE else _pack_byte3
_unpack_fn = _unpack_byte3_compiled if USE_COMPILE else _unpack_byte3


class CompressedAdamW:
    """Grouped lossless FP32 optimizer compression via byte3 6-bit packing."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._params = None
        self._groups = None  # list of (params, sizes, offsets, total_n)
        self._m_data = None  # list of (byte012, packed3, codebook, n) per group
        self._v_data = None
        self._is_compressed = False
        self._first_step = True

    def _init_params(self):
        self._params = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state and 'exp_avg' in self.optimizer.state[p]:
                    self._params.append(p)

        # Build groups of ~GROUP_SIZE elements
        self._groups = []
        cur_params, cur_sizes, cur_offsets = [], [], []
        cur_total = 0
        for p in self._params:
            n = p.numel()
            cur_params.append(p)
            cur_sizes.append(n)
            cur_offsets.append(cur_total)
            cur_total += n
            if cur_total >= GROUP_SIZE:
                self._groups.append((cur_params, cur_sizes, cur_offsets, cur_total))
                cur_params, cur_sizes, cur_offsets = [], [], []
                cur_total = 0
        if cur_params:
            self._groups.append((cur_params, cur_sizes, cur_offsets, cur_total))

        # Convert lists to tuples for immutability
        self._groups = [(tuple(p), tuple(s), tuple(o), t) for p, s, o, t in self._groups]
        self._total_n = sum(g[3] for g in self._groups)

    @staticmethod
    def _compress_chunk(int32_chunk, lut):
        return _pack_fn(int32_chunk, lut)

    @staticmethod
    @torch.no_grad()
    def _compress_flat(flat_fp32):
        """Compress flat FP32 → (byte012, packed3, codebook, n). Chunked for large tensors."""
        n = flat_fp32.numel()
        device = flat_fp32.device
        int32 = flat_fp32.view(torch.int32)

        # Build global LUT from byte3 stats
        byte3_sample = ((int32 >> 24) & 0xFF).to(torch.uint8)
        counts = torch.bincount(byte3_sample.to(torch.int32), minlength=256)
        del byte3_sample
        present = (counts > 0).nonzero(as_tuple=True)[0]
        codebook = present.to(torch.uint8)
        lut = torch.zeros(256, dtype=torch.uint8, device=device)
        lut[present] = torch.arange(len(codebook), device=device, dtype=torch.uint8)
        del counts, present

        # Process in chunks
        all_b012, all_p3 = [], []
        for start in range(0, n, GROUP_SIZE):
            end = min(start + GROUP_SIZE, n)
            b012, p3 = CompressedAdamW._compress_chunk(int32[start:end], lut)
            all_b012.append(b012)
            all_p3.append(p3)
        del lut

        byte012 = torch.cat(all_b012) if len(all_b012) > 1 else all_b012[0]
        packed3 = torch.cat(all_p3) if len(all_p3) > 1 else all_p3[0]
        return byte012, packed3, codebook, n

    @staticmethod
    def _decompress_chunk(byte012, packed3, codebook, n):
        return _unpack_fn(byte012, packed3, codebook, n)

    @staticmethod
    @torch.no_grad()
    def _decompress_flat(data):
        """Decompress (byte012, packed3, codebook, n) → flat FP32. Chunked for large tensors."""
        byte012, packed3, codebook, n = data
        if n <= GROUP_SIZE:
            return CompressedAdamW._decompress_chunk(byte012, packed3, codebook, n)

        device = byte012.device
        result = torch.empty(n, dtype=torch.float32, device=device)
        b012_off, p3_off = 0, 0
        for start in range(0, n, GROUP_SIZE):
            end = min(start + GROUP_SIZE, n)
            cn = end - start
            ng = (cn + 3) // 4
            result[start:end] = CompressedAdamW._decompress_chunk(
                byte012[b012_off:b012_off + cn * 3],
                packed3[p3_off:p3_off + ng * 3],
                codebook, cn)
            b012_off += cn * 3
            p3_off += ng * 3
        return result

    def _gather_group(self, params, sizes, offsets, total_n, key):
        device = params[0].device
        flat = torch.empty(total_n, dtype=torch.float32, device=device)
        for p, sz, off in zip(params, sizes, offsets):
            flat[off:off+sz] = self.optimizer.state[p][key].flatten()
        return flat

    def _set_group_views(self, flat, params, sizes, offsets, key):
        for p, sz, off in zip(params, sizes, offsets):
            self.optimizer.state[p][key] = flat[off:off+sz].view(p.shape)

    def _compress_states(self):
        if self._first_step:
            self._init_params()
            self._first_step = False

        device = self._params[0].device
        self._m_data = []
        self._v_data = []

        for params, sizes, offsets, total_n in self._groups:
            # Gather and compress m for this group
            flat_m = self._gather_group(params, sizes, offsets, total_n, 'exp_avg')
            self._m_data.append(self._compress_flat(flat_m))
            del flat_m

            # Gather and compress v for this group
            flat_v = self._gather_group(params, sizes, offsets, total_n, 'exp_avg_sq')
            self._v_data.append(self._compress_flat(flat_v))
            del flat_v

            # Clear this group's per-param states
            for p in params:
                self.optimizer.state[p]['exp_avg'] = torch.empty(0, dtype=torch.float32, device=device)
                self.optimizer.state[p]['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=device)

        self._is_compressed = True

    def _decompress_states(self):
        for gi, (params, sizes, offsets, total_n) in enumerate(self._groups):
            # Decompress m
            flat_m = self._decompress_flat(self._m_data[gi])
            self._m_data[gi] = None
            self._set_group_views(flat_m, params, sizes, offsets, 'exp_avg')
            # Note: flat_m stays alive as backing for views

            # Decompress v
            flat_v = self._decompress_flat(self._v_data[gi])
            self._v_data[gi] = None
            self._set_group_views(flat_v, params, sizes, offsets, 'exp_avg_sq')

        self._m_data = None
        self._v_data = None
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
        total_c, total_o = 0, 0
        all_cb_m, all_cb_v = set(), set()
        for gi in range(len(self._groups)):
            for name, data_list in [('m', self._m_data), ('v', self._v_data)]:
                b012, p3, cb, n = data_list[gi]
                c = b012.numel() + p3.numel() + cb.numel()
                total_c += c
                total_o += n * 4
                if name == 'm':
                    all_cb_m.update(cb.tolist())
                else:
                    all_cb_v.update(cb.tolist())
        print(f"  m: {len(all_cb_m)} unique, v: {len(all_cb_v)} unique")
        print(f"  ratio={total_c/total_o*100:.1f}%, savings={(total_o-total_c)/1024**2:.0f} MB")
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
    print(f"Lossless FP32 Optimizer Compression (Grouped Byte3 6-bit, {GROUP_SIZE//1024//1024}M/group)")
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

        for _ in range(10):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            opt.step(); opt.zero_grad()

        gc.collect(); torch.cuda.empty_cache()
        gpu_mem = torch.cuda.memory_allocated() / 1024**2

        if use_comp:
            stats = opt.get_stats()

        torch.cuda.reset_peak_memory_stats()

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
