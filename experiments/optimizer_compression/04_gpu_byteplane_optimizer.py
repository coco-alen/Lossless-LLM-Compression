"""
Experiment 4: GPU-Native Byte-Plane Compressed Optimizer

Instead of CPU offload, keep states on GPU but in a compressed format.
FP32 has 4 bytes per element. The high byte (sign + exponent MSBs) has very
low entropy in optimizer states. We can:

Approach A: Byte-plane separation + unique value encoding
  - For each tensor, byte3 typically has <20 unique values
  - Store byte3 as uint8 index into a per-tensor codebook (saves nothing in raw form,
    but enables approach B)

Approach B: Pack 2x uint8 into uint8 (for bytes with <16 unique values)
  - If byte3 has ≤16 unique values, we can pack 2 byte3 values into 1 byte
  - This saves 50% on byte3 = 12.5% of total FP32 storage

Approach C: Bit-packing for low-cardinality byte planes
  - If byte3 has N unique values, we need ceil(log2(N)) bits per element
  - Pack these bits tightly. E.g., 4 unique values → 2 bits → 8x compression on that byte

Approach D: Store as BFloat16 + residual
  - Cast FP32 to BF16 (top 16 bits) + store bottom 16 bits as uint16
  - The BF16 part is compressible (like weight compression); the residual may also compress
  - But this doesn't save raw space — both halves are 2 bytes each

This experiment measures the cardinality of each byte plane to determine which
compression approach is most promising.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import gc
import math
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer


def analyze_byte_planes(tensor: torch.Tensor, name: str):
    """Analyze each byte plane of an FP32 tensor for compressibility."""
    data = tensor.detach().cpu().float().numpy()
    raw = data.tobytes()
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
    n = len(arr)

    print(f"\n  {name} ({n:,} elements, {n*4/1024/1024:.1f} MB)")

    byte_names = ['byte0 (mantissa LSB)', 'byte1 (mantissa mid)',
                  'byte2 (exp_lsb+mant_hi)', 'byte3 (sign+exp_hi)']

    total_compressed_bits = 0
    for i in range(4):
        col = arr[:, i]
        unique_vals = len(np.unique(col))
        counts = Counter(col)
        total = len(col)
        entropy = sum(-c/total * math.log2(c/total) for c in counts.values())

        bits_per_element = entropy
        compressed_bits = bits_per_element * n
        total_compressed_bits += compressed_bits

        # How much we can save with bit-packing
        bits_needed = max(1, math.ceil(math.log2(max(unique_vals, 2))))
        pack_ratio = bits_needed / 8

        print(f"    {byte_names[i]:30s}: {unique_vals:5d} unique, entropy={entropy:.3f} bpe, "
              f"bits_needed={bits_needed}, pack_ratio={pack_ratio:.3f}")

    original_bits = n * 32
    print(f"    Total entropy: {total_compressed_bits/n:.3f} bits/element "
          f"(= {total_compressed_bits/original_bits*100:.2f}% of FP32)")

    # Also analyze 16-bit halves
    uint16_view = np.frombuffer(raw, dtype=np.uint16)
    low16 = uint16_view[0::2]
    high16 = uint16_view[1::2]

    for half_name, half_data in [('low 16-bit', low16), ('high 16-bit', high16)]:
        unique = len(np.unique(half_data))
        vals, counts = np.unique(half_data, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        print(f"    {half_name:30s}: {unique:7d} unique, entropy={entropy:.3f} bpe")


class BytePlaneCompressor:
    """GPU-native byte-plane compression for FP32 tensors.

    Compresses byte3 (sign+exponent MSBs) using bit-packing since it has
    very few unique values. Stores byte0-2 raw.
    """

    def __init__(self):
        self.codebooks = {}

    def compress(self, tensor: torch.Tensor, key: str) -> dict:
        """Compress an FP32 tensor into byte planes with bit-packing."""
        assert tensor.dtype == torch.float32
        device = tensor.device
        n = tensor.numel()

        # View as uint8 bytes
        raw = tensor.contiguous().view(torch.uint8).reshape(-1, 4)

        byte3 = raw[:, 3]  # sign + exp high bits
        byte2 = raw[:, 2]  # exp low bit + mantissa high
        byte1 = raw[:, 1]
        byte0 = raw[:, 0]

        # Analyze byte3 for bit-packing
        unique_vals = torch.unique(byte3)
        n_unique = len(unique_vals)
        bits_needed = max(1, math.ceil(math.log2(max(n_unique, 2))))

        compressed = {
            'shape': tensor.shape,
            'n': n,
            'byte0': byte0.clone(),  # raw
            'byte1': byte1.clone(),  # raw
            'byte2': byte2.clone(),  # raw
        }

        if bits_needed <= 4:
            # Pack byte3 using indices into codebook
            codebook = unique_vals.to(device)
            # Create reverse mapping
            index_map = torch.zeros(256, dtype=torch.uint8, device=device)
            for idx, val in enumerate(unique_vals):
                index_map[val] = idx

            indices = index_map[byte3]

            if bits_needed <= 4:
                # Pack 2 indices per byte (4 bits each)
                n_padded = (n + 1) // 2 * 2
                if n % 2 != 0:
                    indices = torch.cat([indices, torch.zeros(1, dtype=torch.uint8, device=device)])
                packed = (indices[0::2] << 4) | indices[1::2]
                compressed['byte3_packed'] = packed
                compressed['byte3_codebook'] = codebook
                compressed['byte3_bits'] = bits_needed
                compressed['byte3_method'] = '4bit_pack'
            else:
                compressed['byte3'] = byte3.clone()
                compressed['byte3_method'] = 'raw'
        else:
            compressed['byte3'] = byte3.clone()
            compressed['byte3_method'] = 'raw'

        return compressed

    def decompress(self, compressed: dict, device: torch.device) -> torch.Tensor:
        """Decompress byte planes back to FP32 tensor."""
        n = compressed['n']
        shape = compressed['shape']

        byte0 = compressed['byte0'].to(device)
        byte1 = compressed['byte1'].to(device)
        byte2 = compressed['byte2'].to(device)

        if compressed['byte3_method'] == '4bit_pack':
            packed = compressed['byte3_packed'].to(device)
            codebook = compressed['byte3_codebook'].to(device)
            # Unpack
            high = (packed >> 4) & 0x0F
            low = packed & 0x0F
            indices = torch.zeros(len(packed) * 2, dtype=torch.uint8, device=device)
            indices[0::2] = high
            indices[1::2] = low
            indices = indices[:n]
            byte3 = codebook[indices.long()]
        else:
            byte3 = compressed['byte3'].to(device)

        # Reconstruct FP32
        raw = torch.stack([byte0, byte1, byte2, byte3], dim=1).contiguous()
        return raw.view(torch.float32).reshape(shape)

    def compressed_size(self, compressed: dict) -> int:
        """Total bytes stored."""
        size = 0
        for k, v in compressed.items():
            if isinstance(v, torch.Tensor):
                size += v.numel() * v.element_size()
        return size


class BytePlaneAdamW:
    """AdamW with byte-plane compressed optimizer states on GPU."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.compressor = BytePlaneCompressor()

        self.param_groups = [{'params': list(params), 'lr': lr, 'betas': betas,
                              'eps': eps, 'weight_decay': weight_decay}]
        self.state = {}
        self.step_count = 0

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.zero_()

    def step(self):
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
                device = p.device

                if p not in self.state:
                    self.state[p] = {
                        'step': 0,
                        'exp_avg_c': None,
                        'exp_avg_sq_c': None,
                    }
                    exp_avg = torch.zeros_like(p.data, dtype=torch.float32)
                    exp_avg_sq = torch.zeros_like(p.data, dtype=torch.float32)
                else:
                    state = self.state[p]
                    if state['exp_avg_c'] is not None:
                        exp_avg = self.compressor.decompress(state['exp_avg_c'], device)
                        exp_avg_sq = self.compressor.decompress(state['exp_avg_sq_c'], device)
                    else:
                        exp_avg = torch.zeros_like(p.data, dtype=torch.float32)
                        exp_avg_sq = torch.zeros_like(p.data, dtype=torch.float32)

                state = self.state[p]
                state['step'] += 1

                p_float = p.data.float()
                p_float.mul_(1 - lr * wd)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                p_float.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.copy_(p_float.to(p.data.dtype))

                # Compress and store
                state['exp_avg_c'] = self.compressor.compress(exp_avg, f'{id(p)}_m')
                state['exp_avg_sq_c'] = self.compressor.compress(exp_avg_sq, f'{id(p)}_v')

                del exp_avg, exp_avg_sq

    def get_memory_stats(self):
        total_compressed = 0
        total_uncompressed = 0
        for p, state in self.state.items():
            if state['exp_avg_c'] is not None:
                total_compressed += self.compressor.compressed_size(state['exp_avg_c'])
                total_compressed += self.compressor.compressed_size(state['exp_avg_sq_c'])
                total_uncompressed += p.numel() * 4 * 2
        return {
            'compressed_bytes': total_compressed,
            'uncompressed_bytes': total_uncompressed,
            'ratio': total_compressed / max(total_uncompressed, 1),
            'savings_mb': (total_uncompressed - total_compressed) / 1024 / 1024,
        }


def main():
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    print("="*80)
    print("GPU Byte-Plane Compressed Optimizer")
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

    # First: analyze byte planes of optimizer states after some training
    print("\n--- Phase 1: Byte-Plane Analysis ---")
    temp_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    for i in range(50):
        input_ids = torch.randint(100, 10000, (2, 128), device='cuda')
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        temp_optimizer.step()
        temp_optimizer.zero_grad()

    # Analyze a few representative tensors
    analyzed = 0
    for group in temp_optimizer.param_groups:
        for p in group['params']:
            if p not in temp_optimizer.state:
                continue
            state = temp_optimizer.state[p]
            if 'exp_avg' in state and p.numel() > 100000:
                analyze_byte_planes(state['exp_avg'], f'm[{p.shape}]')
                analyze_byte_planes(state['exp_avg_sq'], f'v[{p.shape}]')
                analyzed += 1
                if analyzed >= 3:
                    break
        if analyzed >= 3:
            break

    # Now benchmark the byte-plane optimizer
    print("\n\n--- Phase 2: BytePlane AdamW Benchmark ---")
    del temp_optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.cuda()
    model.train()

    optimizer = BytePlaneAdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    batch_size = 4
    seq_len = 256

    # Warmup
    for i in range(5):
        input_ids = torch.randint(100, 10000, (batch_size, seq_len), device='cuda')
        outputs = model(input_ids=input_ids, labels=input_ids)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    stats = optimizer.get_memory_stats()
    print(f"\nCompression ratio: {stats['ratio']*100:.2f}%")
    print(f"Compressed: {stats['compressed_bytes']/1024/1024:.1f} MB")
    print(f"Uncompressed: {stats['uncompressed_bytes']/1024/1024:.1f} MB")
    print(f"Memory saved: {stats['savings_mb']:.1f} MB")

    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"GPU memory: {gpu_mem:.1f} MB")

    # Measure speed
    torch.cuda.synchronize()
    step_times = []
    for i in range(30):
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
    print(f"\nAvg step time: {avg_time*1000:.2f} ms")
    print(f"Tokens/sec: {tokens_per_sec:.0f}")

    # Verify losslessness: compare one step with standard AdamW
    print("\n--- Phase 3: Losslessness Verification ---")
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(42)
    model1 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    model1.train()
    opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-4, weight_decay=0.01)

    torch.manual_seed(42)
    model2 = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
    model2.train()
    opt2 = BytePlaneAdamW(model2.parameters(), lr=1e-4, weight_decay=0.01)

    # Run 5 steps with identical inputs
    for step in range(5):
        torch.manual_seed(step)
        input_ids = torch.randint(100, 10000, (2, 128), device='cuda')

        out1 = model1(input_ids=input_ids, labels=input_ids)
        out1.loss.backward()
        opt1.step()
        opt1.zero_grad()

        out2 = model2(input_ids=input_ids, labels=input_ids)
        out2.loss.backward()
        opt2.step()
        opt2.zero_grad()

    # Compare weights
    max_diff = 0
    n_diff = 0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        diff = (p1.data - p2.data).abs().max().item()
        if diff > 0:
            n_diff += 1
        max_diff = max(max_diff, diff)

    print(f"Max weight difference after 5 steps: {max_diff}")
    print(f"Parameters with differences: {n_diff}")
    if max_diff == 0:
        print("VERIFIED: Byte-plane compression is bit-exact lossless!")
    else:
        print("WARNING: Differences detected — investigating...")


if __name__ == '__main__':
    main()
