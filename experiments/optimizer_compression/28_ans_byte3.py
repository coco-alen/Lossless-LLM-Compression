"""
Experiment 28: ANS coding for byte3 via constriction (CPU)

byte3 has 3.7 bits entropy but fixed-length coding uses 6 bits.
ANS can approach entropy (~3.7 bits), saving ~13.5% vs 6.25%.

Strategy:
- Transfer byte3 (568 MB) to CPU
- ANS encode with constriction
- Transfer compressed back to GPU
- Store byte012 raw on GPU

Timing budget: PCIe transfer ~23ms each way, ANS encode/decode unknown.
If total overhead < 100ms, this gives ~620 MB savings with ~1.5x slowdown.
"""

import torch
import time
import gc
import math
import numpy as np
from transformers import AutoModelForCausalLM

try:
    import constriction
    HAS_CONSTRICTION = True
except ImportError:
    HAS_CONSTRICTION = False
    print("WARNING: constriction not installed. pip install constriction")


def benchmark_ans_speed():
    """Benchmark ANS encode/decode speed on byte3-like data."""
    if not HAS_CONSTRICTION:
        return

    print("=== ANS Speed Benchmark ===\n")

    # Simulate byte3 distribution
    n = 596_049_920
    # Top byte3 values for m: 0xB7(14.3%), 0x37(14.3%), 0xB8(11.1%), etc.
    probs = torch.zeros(256)
    probs[0xB7] = 14.3
    probs[0x37] = 14.3
    probs[0xB8] = 11.1
    probs[0x38] = 11.1
    probs[0x36] = 8.1
    probs[0xB6] = 8.1
    probs[0x39] = 7.0
    probs[0xB9] = 7.0
    probs[0x35] = 4.5
    probs[0xB5] = 4.5
    probs[0x3A] = 2.5
    probs[0xBA] = 2.5
    remaining = 100 - probs.sum().item()
    n_remaining = 256 - (probs > 0).sum().item()
    if n_remaining > 0:
        probs[probs == 0] = remaining / n_remaining * 0.01  # tiny prob for unseen
    # Only keep nonzero
    probs = probs / probs.sum()

    # Generate test data on GPU
    print(f"Generating {n:,} byte3 values on GPU...")
    byte3_gpu = torch.multinomial(probs, n, replacement=True).to(torch.uint8).cuda()

    # 1. GPU → CPU transfer time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    byte3_cpu = byte3_gpu.cpu()
    t1 = time.perf_counter()
    print(f"GPU→CPU transfer: {(t1-t0)*1000:.1f} ms ({byte3_cpu.numel()/1024**2:.0f} MB)")

    # Convert to numpy
    byte3_np = byte3_cpu.numpy().astype(np.int32)

    # 2. Build ANS model
    # Use empirical frequencies
    freqs = np.bincount(byte3_np, minlength=256).astype(np.float64)
    freqs[freqs == 0] = 1e-10  # avoid zero prob
    freqs = freqs / freqs.sum()

    # Try block-based ANS for parallelism
    block_size = 1024 * 1024  # 1M values per block
    n_blocks = (n + block_size - 1) // block_size

    print(f"\nBlock ANS: {n_blocks} blocks of {block_size:,} values")

    # Encode
    t0 = time.perf_counter()
    encoded_blocks = []
    total_compressed_bytes = 0
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block = byte3_np[start:end]

        # Create ANS encoder for this block
        model = constriction.stream.model.Categorical(freqs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(block, model)
        compressed = encoder.get_compressed()
        encoded_blocks.append(compressed)
        total_compressed_bytes += compressed.nbytes

    t1 = time.perf_counter()
    encode_time = (t1 - t0) * 1000
    ratio = total_compressed_bytes / n * 100
    print(f"ANS encode: {encode_time:.0f} ms, ratio={ratio:.1f}%, "
          f"compressed={total_compressed_bytes/1024**2:.0f} MB")

    # Decode
    t0 = time.perf_counter()
    decoded_blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, n)
        block_len = end - start

        decoder = constriction.stream.stack.AnsCoder(encoded_blocks[i])
        model = constriction.stream.model.Categorical(freqs, perfect=False)
        decoded = decoder.decode(model, block_len)
        decoded_blocks.append(decoded)

    t1 = time.perf_counter()
    decode_time = (t1 - t0) * 1000
    print(f"ANS decode: {decode_time:.0f} ms")

    # Verify
    decoded_full = np.concatenate(decoded_blocks)
    assert np.array_equal(decoded_full, byte3_np), "DECODE MISMATCH"
    print("✓ Round-trip verified")

    # 3. CPU → GPU transfer (compressed)
    compressed_tensor = torch.from_numpy(
        np.concatenate([b.view(np.uint8) for b in encoded_blocks])
    )
    t0 = time.perf_counter()
    compressed_gpu = compressed_tensor.cuda()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"CPU→GPU transfer (compressed): {(t1-t0)*1000:.1f} ms ({compressed_tensor.numel()/1024**2:.0f} MB)")

    # Summary
    total_compress = 23 + encode_time + 1  # GPU→CPU + encode + CPU→GPU(compressed)
    total_decompress = 1 + decode_time + 23  # GPU→CPU(compressed) + decode + CPU→GPU
    print(f"\nTotal compress overhead: ~{total_compress:.0f} ms")
    print(f"Total decompress overhead: ~{total_decompress:.0f} ms")
    print(f"Total per-step overhead: ~{total_compress + total_decompress:.0f} ms")
    print(f"Savings: {(n - total_compressed_bytes)/1024**2:.0f} MB per state, "
          f"{2*(n - total_compressed_bytes)/1024**2:.0f} MB total")

    del byte3_gpu, byte3_cpu, byte3_np
    gc.collect(); torch.cuda.empty_cache()


def full_optimizer_test(model_name="Qwen/Qwen3-0.6B"):
    """Test actual optimizer state compression with ANS byte3."""
    if not HAS_CONSTRICTION:
        return

    print("\n\n=== Full Optimizer Test ===\n")

    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Do a few steps
    for s in range(3):
        torch.manual_seed(s + 100)
        ids = torch.randint(100, 10000, (2, 128), device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(input_ids=ids, labels=ids).loss.backward()
        opt.step(); opt.zero_grad()

    # Gather m into flat
    params = []
    for group in opt.param_groups:
        for p in group['params']:
            if p in opt.state and 'exp_avg' in opt.state[p]:
                params.append(p)

    total_n = sum(p.numel() for p in params)
    flat = torch.empty(total_n, dtype=torch.float32, device='cuda')
    offset = 0
    for p in params:
        n = p.numel()
        flat[offset:offset+n] = opt.state[p]['exp_avg'].flatten()
        offset += n

    int32 = flat.view(torch.int32)

    # Extract byte3
    byte3_gpu = ((int32 >> 24) & 0xFF).to(torch.uint8)

    # Time actual compression pipeline
    print(f"Data size: {total_n:,} values")

    # GPU→CPU
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    byte3_cpu = byte3_gpu.cpu().numpy().astype(np.int32)
    t1 = time.perf_counter()
    print(f"GPU→CPU: {(t1-t0)*1000:.1f} ms")

    # Build model from actual frequencies
    freqs = np.bincount(byte3_cpu, minlength=256).astype(np.float64)
    freqs[freqs == 0] = 1e-10
    freqs = freqs / freqs.sum()

    # Encode
    block_size = 1024 * 1024
    n_blocks = (total_n + block_size - 1) // block_size

    t0 = time.perf_counter()
    encoded_blocks = []
    total_compressed = 0
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, total_n)
        block = byte3_cpu[start:end]
        model_ans = constriction.stream.model.Categorical(freqs, perfect=False)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(block, model_ans)
        compressed = encoder.get_compressed()
        encoded_blocks.append(compressed)
        total_compressed += compressed.nbytes
    t1 = time.perf_counter()
    encode_time = (t1 - t0) * 1000

    ratio = total_compressed / total_n * 100
    savings_mb = (total_n - total_compressed) / 1024**2
    print(f"ANS encode: {encode_time:.0f} ms, {ratio:.1f}%, saves {savings_mb:.0f} MB")

    # Decode
    t0 = time.perf_counter()
    decoded_blocks = []
    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, total_n)
        block_len = end - start
        decoder = constriction.stream.stack.AnsCoder(encoded_blocks[i])
        model_ans = constriction.stream.model.Categorical(freqs, perfect=False)
        decoded = decoder.decode(model_ans, block_len)
        decoded_blocks.append(decoded)
    t1 = time.perf_counter()
    decode_time = (t1 - t0) * 1000

    decoded_full = np.concatenate(decoded_blocks).astype(np.uint8)
    original = byte3_cpu.astype(np.uint8)
    assert np.array_equal(decoded_full, original), "MISMATCH"
    print(f"ANS decode: {decode_time:.0f} ms ✓")

    # CPU→GPU
    t0 = time.perf_counter()
    decoded_gpu = torch.from_numpy(decoded_full).cuda()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"CPU→GPU: {(t1-t0)*1000:.1f} ms")

    print(f"\nTotal compress time: ~{23 + encode_time:.0f} ms")
    print(f"Total decompress time: ~{decode_time + 23:.0f} ms")
    print(f"Per-step overhead: ~{23 + encode_time + decode_time + 23:.0f} ms × 2 states")
    print(f"Savings per state: {savings_mb:.0f} MB (of {total_n*4/1024**2:.0f} MB)")

    del model, opt, flat, int32, byte3_gpu
    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    full_optimizer_test()
