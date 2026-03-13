"""
Experiment 23: Triton-Fused Byte3 Compression

Use Triton kernels to fuse:
1. Compress: read FP32, split into byte012 + byte3 index, pack byte3 6-bit
2. Decompress: read byte012 + packed byte3, reconstruct FP32

This eliminates all intermediate tensor allocations.
Single-pass read of input, single-pass write of output.

Expected: same 6.2% savings but with <<20% overhead.
"""

import torch
import triton
import triton.language as tl
import time
import gc
import math
from transformers import AutoModelForCausalLM


@triton.jit
def _write_byte012(byte012_ptr, offset, v):
    tl.store(byte012_ptr + offset, (v & 0xFF).to(tl.uint8))
    tl.store(byte012_ptr + offset + 1, ((v >> 8) & 0xFF).to(tl.uint8))
    tl.store(byte012_ptr + offset + 2, ((v >> 16) & 0xFF).to(tl.uint8))


@triton.jit
def compress_kernel(
    input_ptr, byte012_ptr, packed3_ptr, lut_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK * 4
    for i in range(BLOCK):
        idx = base + i * 4
        if idx + 3 < N:
            v0 = tl.load(input_ptr + idx)
            v1 = tl.load(input_ptr + idx + 1)
            v2 = tl.load(input_ptr + idx + 2)
            v3 = tl.load(input_ptr + idx + 3)

            i0 = tl.load(lut_ptr + ((v0 >> 24) & 0xFF)).to(tl.int32)
            i1 = tl.load(lut_ptr + ((v1 >> 24) & 0xFF)).to(tl.int32)
            i2 = tl.load(lut_ptr + ((v2 >> 24) & 0xFF)).to(tl.int32)
            i3 = tl.load(lut_ptr + ((v3 >> 24) & 0xFF)).to(tl.int32)

            combined = i0 | (i1 << 6) | (i2 << 12) | (i3 << 18)
            p3_offset = (pid * BLOCK + i) * 3
            tl.store(packed3_ptr + p3_offset, (combined & 0xFF).to(tl.uint8))
            tl.store(packed3_ptr + p3_offset + 1, ((combined >> 8) & 0xFF).to(tl.uint8))
            tl.store(packed3_ptr + p3_offset + 2, ((combined >> 16) & 0xFF).to(tl.uint8))

            _write_byte012(byte012_ptr, (idx + 0) * 3, v0)
            _write_byte012(byte012_ptr, (idx + 1) * 3, v1)
            _write_byte012(byte012_ptr, (idx + 2) * 3, v2)
            _write_byte012(byte012_ptr, (idx + 3) * 3, v3)


@triton.jit
def _read_reconstruct(byte012_ptr, output_ptr, elem_idx, b3):
    b012_offset = elem_idx * 3
    b0 = tl.load(byte012_ptr + b012_offset).to(tl.int32)
    b1 = tl.load(byte012_ptr + b012_offset + 1).to(tl.int32)
    b2 = tl.load(byte012_ptr + b012_offset + 2).to(tl.int32)
    val = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    tl.store(output_ptr + elem_idx, val)


@triton.jit
def decompress_kernel(
    byte012_ptr, packed3_ptr, codebook_ptr, output_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK * 4
    for i in range(BLOCK):
        idx = base + i * 4
        if idx + 3 < N:
            p3_offset = (pid * BLOCK + i) * 3
            pb0 = tl.load(packed3_ptr + p3_offset).to(tl.int32)
            pb1 = tl.load(packed3_ptr + p3_offset + 1).to(tl.int32)
            pb2 = tl.load(packed3_ptr + p3_offset + 2).to(tl.int32)
            combined = pb0 | (pb1 << 8) | (pb2 << 16)

            b3_0 = tl.load(codebook_ptr + (combined & 0x3F)).to(tl.int32)
            b3_1 = tl.load(codebook_ptr + ((combined >> 6) & 0x3F)).to(tl.int32)
            b3_2 = tl.load(codebook_ptr + ((combined >> 12) & 0x3F)).to(tl.int32)
            b3_3 = tl.load(codebook_ptr + ((combined >> 18) & 0x3F)).to(tl.int32)

            _read_reconstruct(byte012_ptr, output_ptr, idx + 0, b3_0)
            _read_reconstruct(byte012_ptr, output_ptr, idx + 1, b3_1)
            _read_reconstruct(byte012_ptr, output_ptr, idx + 2, b3_2)
            _read_reconstruct(byte012_ptr, output_ptr, idx + 3, b3_3)


def triton_compress(flat_fp32, lut):
    """Compress flat FP32 using Triton kernel."""
    n = flat_fp32.numel()
    # Pad to multiple of 4
    n_padded = ((n + 3) // 4) * 4
    if n_padded > n:
        flat_fp32 = torch.cat([flat_fp32, torch.zeros(n_padded - n, dtype=torch.float32, device=flat_fp32.device)])

    byte012 = torch.empty(n_padded * 3, dtype=torch.uint8, device=flat_fp32.device)
    packed3 = torch.empty((n_padded // 4) * 3, dtype=torch.uint8, device=flat_fp32.device)

    BLOCK = 64  # Process 64 groups of 4 per thread block
    n_groups = n_padded // 4
    grid = ((n_groups + BLOCK - 1) // BLOCK,)

    compress_kernel[grid](
        flat_fp32.view(torch.int32), byte012, packed3, lut,
        N=n_padded, BLOCK=BLOCK,
    )

    return byte012[:n * 3], packed3[:(n // 4) * 3 + (3 if n % 4 else 0)], n


def triton_decompress(byte012, packed3, codebook, n):
    """Decompress using Triton kernel."""
    n_padded = ((n + 3) // 4) * 4
    device = byte012.device

    # Pad if needed
    if byte012.numel() < n_padded * 3:
        byte012 = torch.cat([byte012, torch.zeros(n_padded * 3 - byte012.numel(), dtype=torch.uint8, device=device)])
    if packed3.numel() < (n_padded // 4) * 3:
        packed3 = torch.cat([packed3, torch.zeros((n_padded // 4) * 3 - packed3.numel(), dtype=torch.uint8, device=device)])

    output = torch.empty(n_padded, dtype=torch.int32, device=device)

    BLOCK = 64
    n_groups = n_padded // 4
    grid = ((n_groups + BLOCK - 1) // BLOCK,)

    decompress_kernel[grid](
        byte012, packed3, codebook, output,
        N=n_padded, BLOCK=BLOCK,
    )

    return output[:n].view(torch.float32)


def build_lut(flat_fp32):
    """Build LUT from byte3 values using bincount."""
    byte3 = ((flat_fp32.view(torch.int32) >> 24) & 0xFF).to(torch.uint8)
    counts = torch.bincount(byte3.to(torch.int32), minlength=256)
    present = (counts > 0).nonzero(as_tuple=True)[0]
    codebook = present.to(torch.uint8)
    n_unique = len(codebook)

    lut = torch.zeros(256, dtype=torch.uint8, device=flat_fp32.device)
    lut[present] = torch.arange(n_unique, device=flat_fp32.device, dtype=torch.uint8)
    del byte3, counts
    return codebook, lut


class TritonCompressedAdamW:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._params = None
        self._sizes = None
        self._offsets = None
        self._total_n = 0

        self._flat_m = None
        self._flat_v = None
        self._m_data = None  # (byte012, packed3, codebook)
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

    def _compress_after_step(self):
        if self._first_step:
            self._init_params()
            self._flat_m = self._initial_gather('exp_avg')
            self._flat_v = self._initial_gather('exp_avg_sq')
            self._first_step = False

        device = self._params[0].device

        # Compress m
        codebook_m, lut_m = build_lut(self._flat_m)
        byte012_m, packed3_m, _ = triton_compress(self._flat_m, lut_m)
        self._m_data = (byte012_m, packed3_m, codebook_m)
        del self._flat_m, lut_m
        self._flat_m = None

        # Compress v
        codebook_v, lut_v = build_lut(self._flat_v)
        byte012_v, packed3_v, _ = triton_compress(self._flat_v, lut_v)
        self._v_data = (byte012_v, packed3_v, codebook_v)
        del self._flat_v, lut_v
        self._flat_v = None

        for p in self._params:
            self.optimizer.state[p]['exp_avg'] = torch.empty(0, dtype=torch.float32, device=device)
            self.optimizer.state[p]['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=device)

        self._is_compressed = True

    def _decompress_before_step(self):
        byte012_m, packed3_m, codebook_m = self._m_data
        self._flat_m = triton_decompress(byte012_m, packed3_m, codebook_m, self._total_n)
        self._m_data = None
        self._set_views(self._flat_m, 'exp_avg')

        byte012_v, packed3_v, codebook_v = self._v_data
        self._flat_v = triton_decompress(byte012_v, packed3_v, codebook_v, self._total_n)
        self._v_data = None
        self._set_views(self._flat_v, 'exp_avg_sq')

        self._is_compressed = False

    def step(self):
        if self._is_compressed:
            self._decompress_before_step()
        self.optimizer.step()
        self._compress_after_step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def get_stats(self):
        total_c = 0
        total_o = self._total_n * 4 * 2
        for name, data in [('m', self._m_data), ('v', self._v_data)]:
            if data:
                b012, p3, cb = data
                c = b012.numel() + p3.numel() + cb.numel()
                total_c += c
                print(f"  {name}: {len(cb)} unique, ratio={c/(self._total_n*4)*100:.1f}%")
        return {'ratio': total_c / total_o, 'savings_mb': (total_o - total_c) / 1024**2}


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    print("--- Verify ---")

    # Basic round-trip test
    print("  Round-trip test...")
    for n in [100, 10000, 100001, 1000000]:
        data = torch.randn(n, dtype=torch.float32, device='cuda')
        cb, lut = build_lut(data)
        b012, p3, n_out = triton_compress(data, lut)
        restored = triton_decompress(b012, p3, cb, n)
        if not torch.all(data == restored):
            # Show first mismatch
            mask = data != restored
            idx = mask.nonzero()[0].item()
            print(f"  FAILED at index {idx}: {data[idx].item()} != {restored[idx].item()}")
            print(f"  data int32: {data[idx].view(torch.int32).item():#010x}")
            print(f"  restored:   {restored[idx].view(torch.int32).item():#010x}")
            return False
    print("  ✓ Round-trip OK")

    # Full optimizer test
    print("  Full optimizer test...")
    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = TritonCompressedAdamW(inner)

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
    print("Triton-Fused Compressed FP32 Optimizer")
    print("=" * 80)

    results = []
    for name, use_comp in [("Standard FP32 AdamW", False),
                            ("Triton Compressed", True)]:
        print(f"\n--- {name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
        model.train()
        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = TritonCompressedAdamW(inner) if use_comp else inner

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
    ok = verify_lossless()
    if ok:
        benchmark()
