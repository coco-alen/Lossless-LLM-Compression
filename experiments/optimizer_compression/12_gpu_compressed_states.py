"""
Experiment 12: GPU-Resident Lossless Compression of FP32 Optimizer States

Approach: Split FP32 into (high16, low16). The high16 contains sign+exponent+mantissa_MSBs
and has only ~5000-6000 unique values with ~11 bits entropy. Compress it using a
codebook + index approach. Store low16 raw.

Compression methods for high16:
A) 12-bit index packing: If ≤4096 unique values, store as 12-bit indices (pack 2 in 3 bytes)
   Savings: 2 bytes → 1.5 bytes per element for high16 = 12.5% total
B) ANS coding: Entropy-optimal, ~11 bits per high16 = ~16% total savings
C) Huffman coding: Near-optimal, slightly easier to decode on GPU

For simplicity, start with approach A (12-bit packing) since it's GPU-friendly
(no variable-length decoding needed).

Memory layout:
- Original: 4 bytes per element (FP32)
- Compressed: 2 bytes (low16) + 1.5 bytes (high16 as 12-bit index) = 3.5 bytes
- Ratio: 87.5% → saves 12.5% = ~568 MB for this model

Then try 11-bit packing if unique count allows.
"""

import torch
import time
import gc
import math
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class CompressedFP32Buffer:
    """Stores FP32 data in compressed form on GPU.

    Splits FP32 into high16 and low16. High16 is stored as indices into a
    codebook using bit-packing. Low16 is stored raw.

    For N elements:
    - Original: 4N bytes
    - Compressed: 2N (low16) + ceil(N * bits_per_idx / 8) + codebook
    """

    def __init__(self):
        self.low16 = None        # uint16 tensor, N elements
        self.indices = None      # packed indices (uint8)
        self.codebook = None     # unique high16 values (uint16)
        self.bits_per_idx = 0
        self.n_elements = 0
        self.shape = None

    def compress(self, tensor: torch.Tensor) -> 'CompressedFP32Buffer':
        """Compress an FP32 tensor."""
        assert tensor.dtype == torch.float32
        self.shape = tensor.shape
        self.n_elements = tensor.numel()
        device = tensor.device

        # View as uint32 via int32
        flat = tensor.contiguous().flatten()
        int32_view = flat.view(torch.int32)

        # Split into high16 and low16
        high16 = ((int32_view >> 16) & 0xFFFF).to(torch.int16)
        self.low16 = (int32_view & 0xFFFF).to(torch.int16)

        # Build codebook for high16
        self.codebook = torch.unique(high16)
        n_unique = len(self.codebook)
        self.bits_per_idx = max(1, math.ceil(math.log2(max(n_unique, 2))))

        # Create reverse lookup table (value -> index)
        # high16 values can be negative (int16), shift to 0-65535 range for indexing
        lut = torch.zeros(65536, dtype=torch.int32, device=device)
        for i, val in enumerate(self.codebook):
            # Use (val + 32768) as index to handle negative int16
            lut[(val.item() + 32768) % 65536] = i

        # Map high16 to indices
        indices = lut[((high16.to(torch.int32) + 32768) % 65536).long()]

        # Pack indices using bit-packing
        self.indices = self._pack_indices(indices, self.bits_per_idx, device)

        return self

    def decompress(self) -> torch.Tensor:
        """Decompress back to FP32 tensor."""
        device = self.low16.device

        # Unpack indices
        indices = self._unpack_indices(self.indices, self.bits_per_idx,
                                        self.n_elements, device)

        # Look up high16 values
        high16 = self.codebook[indices.long()]

        # Reconstruct int32
        reconstructed = (high16.to(torch.int32) << 16) | (self.low16.to(torch.int32) & 0xFFFF)

        # View as float32
        return reconstructed.view(torch.float32).reshape(self.shape)

    def _pack_indices(self, indices: torch.Tensor, bits: int, device) -> torch.Tensor:
        """Pack integer indices into bit-packed uint8 tensor."""
        if bits <= 8:
            # Simple: just store as uint8 (wastes some bits but fast)
            return indices.to(torch.uint8)
        elif bits <= 16:
            return indices.to(torch.int16)
        else:
            return indices.to(torch.int32)

    def _unpack_indices(self, packed: torch.Tensor, bits: int, n: int, device) -> torch.Tensor:
        """Unpack bit-packed indices."""
        if bits <= 8:
            return packed[:n].to(torch.int32)
        elif bits <= 16:
            return packed[:n].to(torch.int32)
        else:
            return packed[:n]

    def memory_bytes(self) -> int:
        """Total GPU memory used by compressed representation."""
        total = 0
        if self.low16 is not None:
            total += self.low16.numel() * self.low16.element_size()
        if self.indices is not None:
            total += self.indices.numel() * self.indices.element_size()
        if self.codebook is not None:
            total += self.codebook.numel() * self.codebook.element_size()
        return total

    def original_bytes(self) -> int:
        return self.n_elements * 4


class CompressedFP32AdamW:
    """AdamW wrapper that stores m/v compressed in GPU memory.

    After optimizer.step(), m and v are compressed using high16 codebook encoding.
    Before the next step(), they are decompressed in-place.

    This saves GPU memory between optimizer steps (during forward/backward pass).
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._compressed_states = {}  # param_id -> (compressed_m, compressed_v)
        self._is_compressed = False

    def _compress_states(self):
        """Compress all m/v states."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p not in self.optimizer.state:
                    continue
                state = self.optimizer.state[p]
                if 'exp_avg' not in state:
                    continue

                pid = id(p)
                cm = CompressedFP32Buffer().compress(state['exp_avg'])
                cv = CompressedFP32Buffer().compress(state['exp_avg_sq'])
                self._compressed_states[pid] = (cm, cv)

                # Free original FP32 tensors
                state['exp_avg'] = torch.empty(0, dtype=torch.float32, device=p.device)
                state['exp_avg_sq'] = torch.empty(0, dtype=torch.float32, device=p.device)

        self._is_compressed = True

    def _decompress_states(self):
        """Decompress all m/v states back to FP32."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                pid = id(p)
                if pid not in self._compressed_states:
                    continue
                cm, cv = self._compressed_states[pid]
                state = self.optimizer.state[p]
                state['exp_avg'] = cm.decompress()
                state['exp_avg_sq'] = cv.decompress()

        self._compressed_states.clear()
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
        if not self._compressed_states:
            return None
        total_compressed = 0
        total_original = 0
        n_params = 0
        bits_info = {}
        for pid, (cm, cv) in self._compressed_states.items():
            total_compressed += cm.memory_bytes() + cv.memory_bytes()
            total_original += cm.original_bytes() + cv.original_bytes()
            n_params += 1
            bits_info[pid] = (cm.bits_per_idx, cv.bits_per_idx,
                              len(cm.codebook), len(cv.codebook))
        return {
            'compressed_bytes': total_compressed,
            'original_bytes': total_original,
            'ratio': total_compressed / max(total_original, 1),
            'n_params': n_params,
            'savings_mb': (total_original - total_compressed) / 1024**2,
            'bits_info': bits_info,
        }


def verify_lossless(model_name="Qwen/Qwen3-0.6B"):
    """Verify compression is bit-exact lossless."""
    print("--- Losslessness Verification ---")

    torch.manual_seed(42)
    m1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    m1.train()
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    m2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    m2.train()
    inner = torch.optim.AdamW(m2.parameters(), lr=1e-4, weight_decay=0.01)
    o2 = CompressedFP32AdamW(inner)

    for step in range(10):
        torch.manual_seed(step + 100)
        ids = torch.randint(100, 10000, (2, 128), device='cuda')

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            m1(input_ids=ids, labels=ids).loss.backward()
        o1.step(); o1.zero_grad()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            m2(input_ids=ids, labels=ids).loss.backward()
        o2.step(); o2.zero_grad()

    # Compare parameters
    max_diff = 0
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        diff = (p1.data - p2.data).abs().max().item()
        max_diff = max(max_diff, diff)

    print(f"  Max param diff after 10 steps: {max_diff}")

    # Compare optimizer states (decompress first)
    o2._decompress_states()
    max_state_diff = 0
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if p1 in o1.state and p2 in inner.state:
            s1, s2 = o1.state[p1], inner.state[p2]
            if 'exp_avg' in s1 and s2['exp_avg'].numel() > 0:
                d = (s1['exp_avg'] - s2['exp_avg']).abs().max().item()
                max_state_diff = max(max_state_diff, d)
                d = (s1['exp_avg_sq'] - s2['exp_avg_sq']).abs().max().item()
                max_state_diff = max(max_state_diff, d)

    print(f"  Max state diff: {max_state_diff}")
    ok = max_diff == 0 and max_state_diff == 0
    print(f"  {'✓ LOSSLESS' if ok else '✗ FAILED'}")

    del m1, m2, o1, o2, inner
    gc.collect(); torch.cuda.empty_cache()
    return ok


def benchmark(model_name="Qwen/Qwen3-0.6B", batch_size=4, seq_len=256):
    print("\n" + "=" * 80)
    print("GPU-Resident Compressed FP32 Optimizer Benchmark")
    print("=" * 80)

    results = []
    n_warmup, n_measure = 10, 40

    for method_name, use_compressed in [("Standard FP32 AdamW", False),
                                         ("Compressed FP32 AdamW", True)]:
        print(f"\n--- {method_name} ---")
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
        model.train()

        inner = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        opt = CompressedFP32AdamW(inner) if use_compressed else inner

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
                print(f"  Compressed: {stats['compressed_bytes']/1024**2:.1f} MB")
                print(f"  Original: {stats['original_bytes']/1024**2:.1f} MB")
                print(f"  Ratio: {stats['ratio']*100:.1f}%")
                print(f"  Savings: {stats['savings_mb']:.1f} MB")

                # Show some bits_info
                sample = list(stats['bits_info'].values())[:5]
                for i, (mb, vb, mc, vc) in enumerate(sample):
                    print(f"    Param {i}: m={mc} unique ({mb} bits), v={vc} unique ({vb} bits)")

        # Measure speed
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

        # Optimizer-only time
        opt_times = []
        for _ in range(20):
            ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                model(input_ids=ids, labels=ids).loss.backward()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            opt_times.append(time.perf_counter() - t0)
            opt.zero_grad()

        avg_step = sum(times) / len(times)
        avg_opt = sum(opt_times) / len(opt_times)
        peak = torch.cuda.max_memory_allocated() / 1024**2
        tps = batch_size * seq_len / avg_step

        print(f"  GPU mem (steady): {gpu_mem:.0f} MB, Peak: {peak:.0f} MB")
        print(f"  Step: {avg_step*1000:.1f} ms, Opt: {avg_opt*1000:.1f} ms, Tok/s: {tps:.0f}")

        results.append({
            'method': method_name,
            'gpu_mem': gpu_mem,
            'peak': peak,
            'step_ms': avg_step * 1000,
            'opt_ms': avg_opt * 1000,
            'tps': tps,
        })

        del model, inner, opt
        gc.collect(); torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    bl = results[0]
    print(f"{'Method':<30} {'Mem':>7} {'ΔMem':>7} {'Peak':>7} {'Step':>7} {'Slow':>5} {'Tok/s':>7}")
    print("-" * 72)
    for r in results:
        dm = r['gpu_mem'] - bl['gpu_mem']
        s = r['step_ms'] / bl['step_ms']
        print(f"{r['method']:<30} {r['gpu_mem']:>6.0f}M {dm:>+6.0f}M {r['peak']:>6.0f}M "
              f"{r['step_ms']:>6.1f} {s:>4.2f}x {r['tps']:>6.0f}")


if __name__ == '__main__':
    ok = verify_lossless()
    if ok:
        benchmark()
    else:
        print("\n!!! Compression is NOT lossless — fixing before benchmark")
