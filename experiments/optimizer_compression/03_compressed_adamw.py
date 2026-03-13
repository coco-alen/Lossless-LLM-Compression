"""
Experiment 3: Compressed AdamW Optimizer

Approach: After each optimizer step, compress m and v states using byte-level
compression. Before each optimizer step, decompress them. This saves GPU memory
between steps (the majority of training time is forward/backward, not optimizer step).

Method: Split FP32 into 4 byte planes, compress high bytes (which contain the
exponent and have low entropy) using simple LZ4 or zlib, store compressed on CPU.
Decompress to GPU when needed.

Variants tested:
A) CPU offload with compression (compressed on CPU, decompress to GPU for step)
B) GPU byte-plane compression (keep on GPU, compress only high-entropy bytes)
C) Simple truncation baseline (store fp16 — LOSSY, for comparison only)
"""

import torch
import torch.nn as nn
import time
import gc
import struct
import numpy as np
import lz4.frame
import zlib
from typing import Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class CompressedAdamW:
    """AdamW optimizer that stores m/v compressed between steps.

    After each step, m and v are compressed and stored on CPU.
    Before each step, they are decompressed to GPU.
    The master weights (FP32 params) stay on GPU as usual.

    This is lossless — decompressed values are bit-identical to originals.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, compression='lz4', compress_master_weights=False):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.compression = compression  # 'lz4', 'zlib', 'bytegroup_lz4', 'none'
        self.compress_master_weights = compress_master_weights

        self.param_groups = [{'params': list(params), 'lr': lr, 'betas': betas,
                              'eps': eps, 'weight_decay': weight_decay}]

        # State: compressed buffers on CPU
        self.state: Dict[torch.nn.Parameter, dict] = {}
        self.step_count = 0

    def _compress(self, tensor: torch.Tensor) -> bytes:
        """Compress a FP32 tensor to bytes."""
        data = tensor.detach().cpu().numpy().tobytes()

        if self.compression == 'lz4':
            return lz4.frame.compress(data, compression_level=0)  # fastest
        elif self.compression == 'zlib':
            return zlib.compress(data, level=1)  # fast
        elif self.compression == 'bytegroup_lz4':
            # Byte-group: rearrange bytes so all byte3 are together, then byte2, etc.
            arr = np.frombuffer(data, dtype=np.uint8).reshape(-1, 4)
            grouped = np.ascontiguousarray(arr.T).tobytes()  # [all_byte0, all_byte1, all_byte2, all_byte3]
            return lz4.frame.compress(grouped, compression_level=0)
        elif self.compression == 'none':
            return data
        else:
            raise ValueError(f"Unknown compression: {self.compression}")

    def _decompress(self, compressed: bytes, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Decompress bytes back to a FP32 tensor on the given device."""
        n_elements = 1
        for s in shape:
            n_elements *= s

        if self.compression == 'lz4':
            data = lz4.frame.decompress(compressed)
        elif self.compression == 'zlib':
            data = zlib.decompress(compressed)
        elif self.compression == 'bytegroup_lz4':
            raw = lz4.frame.decompress(compressed)
            # Undo byte-grouping
            grouped = np.frombuffer(raw, dtype=np.uint8).reshape(4, n_elements)
            data = np.ascontiguousarray(grouped.T).tobytes()
        elif self.compression == 'none':
            data = compressed
        else:
            raise ValueError(f"Unknown compression: {self.compression}")

        arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
        return torch.from_numpy(arr.copy()).to(device)

    def _compressed_size(self, param) -> int:
        """Get total compressed size for a parameter's state in bytes."""
        if param not in self.state:
            return 0
        state = self.state[param]
        size = 0
        if 'exp_avg_compressed' in state:
            size += len(state['exp_avg_compressed'])
        if 'exp_avg_sq_compressed' in state:
            size += len(state['exp_avg_sq_compressed'])
        return size

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

    def step(self):
        """Perform one optimizer step with compressed state management."""
        self.step_count += 1

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.float()

                # Initialize or decompress state
                if p not in self.state:
                    self.state[p] = {
                        'step': 0,
                        'exp_avg_compressed': None,
                        'exp_avg_sq_compressed': None,
                        'shape': p.data.shape,
                    }
                    exp_avg = torch.zeros_like(p.data, dtype=torch.float32)
                    exp_avg_sq = torch.zeros_like(p.data, dtype=torch.float32)
                else:
                    state = self.state[p]
                    if state['exp_avg_compressed'] is not None:
                        exp_avg = self._decompress(state['exp_avg_compressed'], state['shape'], p.device)
                        exp_avg_sq = self._decompress(state['exp_avg_sq_compressed'], state['shape'], p.device)
                    else:
                        exp_avg = torch.zeros_like(p.data, dtype=torch.float32)
                        exp_avg_sq = torch.zeros_like(p.data, dtype=torch.float32)

                state = self.state[p]
                state['step'] += 1

                # AdamW update
                p_float = p.data.float()

                # Decoupled weight decay
                p_float.mul_(1 - lr * wd)

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p_float.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.copy_(p_float.to(p.data.dtype))

                # Compress and store states
                state['exp_avg_compressed'] = self._compress(exp_avg)
                state['exp_avg_sq_compressed'] = self._compress(exp_avg_sq)

                # Free GPU tensors
                del exp_avg, exp_avg_sq

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        total_compressed = 0
        total_uncompressed = 0
        n_params = 0

        for p, state in self.state.items():
            if state['exp_avg_compressed'] is not None:
                total_compressed += len(state['exp_avg_compressed'])
                total_compressed += len(state['exp_avg_sq_compressed'])
                total_uncompressed += p.numel() * 4 * 2  # m + v, FP32
                n_params += 1

        return {
            'compressed_bytes': total_compressed,
            'uncompressed_bytes': total_uncompressed,
            'ratio': total_compressed / max(total_uncompressed, 1),
            'n_params': n_params,
            'savings_mb': (total_uncompressed - total_compressed) / 1024 / 1024,
        }


def benchmark_compressed_adamw(model, tokenizer, compression, n_warmup=5, n_measure=30,
                                batch_size=4, seq_len=256):
    """Benchmark a compression variant."""
    print(f"\n--- Compression: {compression} ---")

    optimizer = CompressedAdamW(model.parameters(), lr=1e-4, weight_decay=0.01,
                                 compression=compression)

    # Warmup
    for i in range(n_warmup):
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Get compression stats
    stats = optimizer.get_memory_stats()
    print(f"  Compression ratio: {stats['ratio']*100:.2f}%")
    print(f"  Compressed size: {stats['compressed_bytes']/1024/1024:.1f} MB")
    print(f"  Uncompressed size: {stats['uncompressed_bytes']/1024/1024:.1f} MB")
    print(f"  Memory saved: {stats['savings_mb']:.1f} MB")

    # Check GPU memory (states should be on CPU now)
    gc.collect()
    torch.cuda.empty_cache()
    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"  GPU memory (after step, states on CPU): {gpu_mem:.1f} MB")

    # Measure step time
    torch.cuda.synchronize()
    step_times = []
    for i in range(n_measure):
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    avg_time = sum(step_times) / len(step_times)
    tokens_per_sec = batch_size * seq_len / avg_time

    print(f"  Avg step time: {avg_time*1000:.2f} ms")
    print(f"  Tokens/sec: {tokens_per_sec:.0f}")

    # Measure just optimizer step time
    optim_times = []
    for _ in range(20):
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        optimizer.zero_grad()
        optim_times.append(t1 - t0)

    avg_optim = sum(optim_times) / len(optim_times)
    print(f"  Avg optimizer.step() time: {avg_optim*1000:.2f} ms")

    return {
        'compression': compression,
        'ratio': stats['ratio'],
        'compressed_mb': stats['compressed_bytes'] / 1024 / 1024,
        'uncompressed_mb': stats['uncompressed_bytes'] / 1024 / 1024,
        'gpu_mem_mb': gpu_mem,
        'step_time_ms': avg_time * 1000,
        'optim_time_ms': avg_optim * 1000,
        'tokens_per_sec': tokens_per_sec,
    }


def benchmark_standard_adamw(model, batch_size=4, seq_len=256, n_warmup=5, n_measure=30):
    """Benchmark standard PyTorch AdamW for comparison."""
    print(f"\n--- Standard PyTorch AdamW (baseline) ---")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Warmup
    for i in range(n_warmup):
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    gc.collect()
    torch.cuda.empty_cache()
    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"  GPU memory (with optimizer states): {gpu_mem:.1f} MB")

    # Count optimizer state memory
    state_bytes = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state_bytes += v.numel() * v.element_size()
    print(f"  Optimizer state memory: {state_bytes/1024/1024:.1f} MB")

    torch.cuda.synchronize()
    step_times = []
    for i in range(n_measure):
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

    avg_time = sum(step_times) / len(step_times)
    tokens_per_sec = batch_size * seq_len / avg_time

    print(f"  Avg step time: {avg_time*1000:.2f} ms")
    print(f"  Tokens/sec: {tokens_per_sec:.0f}")

    optim_times = []
    for _ in range(20):
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        optimizer.zero_grad()
        optim_times.append(t1 - t0)

    avg_optim = sum(optim_times) / len(optim_times)
    print(f"  Avg optimizer.step() time: {avg_optim*1000:.2f} ms")

    return {
        'compression': 'none (standard)',
        'gpu_mem_mb': gpu_mem,
        'state_mem_mb': state_bytes / 1024 / 1024,
        'step_time_ms': avg_time * 1000,
        'optim_time_ms': avg_optim * 1000,
        'tokens_per_sec': tokens_per_sec,
    }


def main():
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    print("="*80)
    print("Compressed AdamW Optimizer Benchmark")
    print("="*80)

    model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.cuda()
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} ({n_params/1e6:.1f}M params)")

    results = []

    # Standard baseline
    baseline = benchmark_standard_adamw(model)
    results.append(baseline)

    # Need to reload model for each test since optimizer modified weights
    for compression in ['none', 'lz4', 'zlib', 'bytegroup_lz4']:
        # Reload model
        del model
        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        model = model.cuda()
        model.train()

        result = benchmark_compressed_adamw(model, tokenizer, compression)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'GPU Mem':>10} {'Ratio':>8} {'Step(ms)':>10} {'Optim(ms)':>10} {'Tok/s':>10}")
    print("-" * 80)
    for r in results:
        ratio = f"{r.get('ratio', 1.0)*100:.1f}%" if 'ratio' in r else "100.0%"
        print(f"{r['compression']:<25} {r['gpu_mem_mb']:>9.1f} {ratio:>8} {r['step_time_ms']:>9.2f} {r['optim_time_ms']:>9.2f} {r['tokens_per_sec']:>9.0f}")


if __name__ == '__main__':
    main()
