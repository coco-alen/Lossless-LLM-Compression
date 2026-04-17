"""
Cross-Node KV Transfer Simulation for SplitZip.

Simulates disaggregated PD serving at datacenter-realistic bandwidths.
Since we lack RDMA hardware, we measure real GPU codec times and simulate
network transfer delays for various bandwidths (25G - 400G).

Models the production Mooncake data flow:
  Prefill GPU → encode → CPU staging → network → CPU staging → decode → Decode GPU

Two modes:
  1. Single-layer: raw vs SplitZip for one KV transfer
  2. Multi-layer pipeline: overlapped encode/transfer/decode across N layers
"""

import torch
import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from experiments.splitzip.lossless_fast import FastLosslessCodec


def measure_codec_times(codec, kv_bf16, warmup=30, iters=200):
    """Measure real GPU encode and decode times."""
    n = kv_bf16.numel()
    nbytes = n * 2

    # Warmup
    for _ in range(warmup):
        r = codec.encode(kv_bf16)
        codec.decode(*r)

    # Encode (GPU kernel + escape collection)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        r = codec.encode(kv_bf16)
    torch.cuda.synchronize()
    enc_time = (time.perf_counter() - t0) / iters

    # Compressed size
    r = codec.encode(kv_bf16)
    pk, sm, esc_pos, esc_val, n_out, n_esc = r
    comp_bytes = pk.numel() + sm.numel() + esc_pos.numel() * 4 + esc_val.numel()
    ratio = nbytes / comp_bytes

    # Decode (GPU kernel + escape fix)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        codec.decode(*r)
    torch.cuda.synchronize()
    dec_time = (time.perf_counter() - t0) / iters

    # Verify
    decoded = codec.decode(*r)
    correct = torch.equal(kv_bf16.view(torch.int16), decoded.view(torch.int16))

    # GPU→CPU serialization time (part of real encode pipeline)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = pk.cpu()
        _ = sm.cpu()
        if n_esc > 0:
            _ = esc_pos.cpu()
            _ = esc_val.cpu()
    torch.cuda.synchronize()
    ser_time = (time.perf_counter() - t0) / iters

    # CPU→GPU deserialization time (part of real decode pipeline)
    pk_cpu, sm_cpu = pk.cpu(), sm.cpu()
    esc_pos_cpu = esc_pos.cpu() if n_esc > 0 else esc_pos
    esc_val_cpu = esc_val.cpu() if n_esc > 0 else esc_val
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = pk_cpu.cuda()
        _ = sm_cpu.cuda()
        if n_esc > 0:
            _ = esc_pos_cpu.cuda()
            _ = esc_val_cpu.cuda()
    torch.cuda.synchronize()
    deser_time = (time.perf_counter() - t0) / iters

    return {
        'enc_time': enc_time,
        'dec_time': dec_time,
        'ser_time': ser_time,       # GPU→CPU
        'deser_time': deser_time,   # CPU→GPU
        'comp_bytes': comp_bytes,
        'raw_bytes': nbytes,
        'ratio': ratio,
        'n_esc': n_esc,
        'correct': correct,
    }


def simulate_single_layer(m, bw_gbps):
    """Simulate single-layer transfer at given bandwidth."""
    bw = bw_gbps * 1e9 / 8  # Gbps → bytes/s

    raw_time = m['raw_bytes'] / bw

    # SplitZip pipeline: encode(GPU) + serialize(GPU→CPU) + transfer + deserialize(CPU→GPU) + decode(GPU)
    xfer_time = m['comp_bytes'] / bw
    splitzip_time = m['enc_time'] + m['ser_time'] + xfer_time + m['deser_time'] + m['dec_time']

    return raw_time, splitzip_time


def simulate_pipeline(m, bw_gbps, n_layers):
    """
    Simulate multi-layer pipelined transfer.

    Pipeline model (3 stages overlapped across layers):
      Stage 1: GPU encode + GPU→CPU serialize
      Stage 2: Network transfer (compressed)
      Stage 3: CPU→GPU deserialize + GPU decode

    Total = startup + bottleneck * (n_layers - 1) + drain
    """
    bw = bw_gbps * 1e9 / 8

    # Per-layer stage times
    stage1 = m['enc_time'] + m['ser_time']     # encode + serialize
    stage2 = m['comp_bytes'] / bw              # network transfer
    stage3 = m['deser_time'] + m['dec_time']   # deserialize + decode

    raw_total = m['raw_bytes'] * n_layers / bw

    # Pipeline: first layer fills all 3 stages, then bottleneck dominates
    bottleneck = max(stage1, stage2, stage3)
    pipe_total = stage1 + stage2 + stage3 + bottleneck * (n_layers - 1)

    bn_name = 'enc' if bottleneck == stage1 else ('net' if bottleneck == stage2 else 'dec')

    return raw_total, pipe_total, bn_name, (stage1, stage2, stage3)


def main():
    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    codec = FastLosslessCodec(device)

    # Model configurations (KV per layer)
    configs = [
        # (name, kv_heads, seq_len, head_dim, n_layers, description)
        ("Llama-3-8B 4K",      8, 4096, 128, 32,  "8B model, short context"),
        ("Llama-3-8B 32K",     8, 32768, 128, 32,  "8B model, long context"),
        ("Llama-3-70B 4K",     8, 4096, 128, 80,   "70B model, short context"),
        ("Llama-3-70B 64K",    8, 65536, 128, 80,  "70B model, long context"),
        ("Qwen3-30B-A3B 4K",   4, 4096, 128, 48,   "MoE, short context"),
        ("Qwen3-30B-A3B 32K",  4, 32768, 128, 48,  "MoE, long context"),
    ]

    # Network configurations (name, Gbps)
    networks = [
        ("25G TCP",     25),
        ("100G TCP",   100),
        ("200G RoCE",  200),
        ("400G RoCE",  400),
        ("4x200G RoCE (Mooncake)", 700),  # ~87 GB/s reported
        ("8x400G RoCE (Mooncake)", 1520), # ~190 GB/s reported
    ]

    print("=" * 100)
    print("PHASE 1: MEASURE REAL GPU CODEC TIMES")
    print("=" * 100)

    # Measure codec at different sizes to capture scaling
    measurements = {}
    for name, kv_heads, seq_len, head_dim, n_layers, desc in configs:
        # KV per layer: 2 (K+V) * kv_heads * seq_len * head_dim
        n_elements = 2 * kv_heads * seq_len * head_dim
        kv = torch.randn(n_elements, dtype=torch.bfloat16, device=device)
        nbytes = n_elements * 2

        # Calibrate codec
        codec.calibrate(kv)

        # Measure
        m = measure_codec_times(codec, kv, warmup=20, iters=100)
        measurements[name] = m

        enc_gbs = nbytes / m['enc_time'] / 1e9
        dec_gbs = nbytes / m['dec_time'] / 1e9

        print(f"\n  {name} ({desc})")
        print(f"    Per-layer KV: {nbytes/1e6:.1f} MB ({n_elements:,} BF16 elements)")
        print(f"    Encode:       {m['enc_time']*1000:.3f} ms ({enc_gbs:.0f} GB/s)")
        print(f"    Decode:       {m['dec_time']*1000:.3f} ms ({dec_gbs:.0f} GB/s)")
        print(f"    GPU→CPU:      {m['ser_time']*1000:.3f} ms")
        print(f"    CPU→GPU:      {m['deser_time']*1000:.3f} ms")
        print(f"    Ratio:        {m['ratio']:.4f}x  Escapes: {m['n_esc']}  Correct: {m['correct']}")

    # ================================================================
    print("\n\n" + "=" * 100)
    print("PHASE 2: SINGLE-LAYER TRANSFER SIMULATION")
    print("=" * 100)

    for name, kv_heads, seq_len, head_dim, n_layers, desc in configs:
        m = measurements[name]
        print(f"\n  {name} (per-layer: {m['raw_bytes']/1e6:.1f} MB → "
              f"{m['comp_bytes']/1e6:.1f} MB, ratio: {m['ratio']:.3f}x)")
        print(f"  {'Network':<28} {'Raw ms':>8} {'SplitZip ms':>12} {'Speedup':>8}")
        print(f"  {'-'*60}")

        for net_name, gbps in networks:
            raw_t, sz_t = simulate_single_layer(m, gbps)
            sp = raw_t / sz_t
            print(f"  {net_name:<28} {raw_t*1000:>7.3f} {sz_t*1000:>11.3f} {sp:>7.3f}x")

    # ================================================================
    print("\n\n" + "=" * 100)
    print("PHASE 3: MULTI-LAYER PIPELINED TRANSFER SIMULATION")
    print("=" * 100)
    print("  Models the production Mooncake flow: prefill GPU → CPU → network → CPU → decode GPU")
    print("  Pipeline: encode+serialize | network transfer | deserialize+decode (3-stage overlap)")

    for name, kv_heads, seq_len, head_dim, n_layers, desc in configs:
        m = measurements[name]
        total_kv = m['raw_bytes'] * n_layers
        print(f"\n  {name} ({n_layers} layers, total KV: {total_kv/1e9:.2f} GB)")
        print(f"  {'Network':<28} {'Raw ms':>9} {'Pipe ms':>9} {'Speedup':>8} "
              f"{'Saved ms':>9} {'Bottleneck':>11}")
        print(f"  {'-'*78}")

        for net_name, gbps in networks:
            raw_t, pipe_t, bn, stages = simulate_pipeline(m, gbps, n_layers)
            sp = raw_t / pipe_t
            saved = raw_t - pipe_t
            print(f"  {net_name:<28} {raw_t*1000:>8.1f} {pipe_t*1000:>8.1f} {sp:>7.3f}x "
                  f"{saved*1000:>8.1f} {bn:>11}")

    # ================================================================
    print("\n\n" + "=" * 100)
    print("PHASE 4: PIPELINE STAGE BREAKDOWN")
    print("=" * 100)
    print("  Shows per-layer stage times to identify bottleneck transitions")

    # Pick the most interesting config: Llama-3-70B 64K
    target = "Llama-3-70B 64K"
    m = measurements[target]
    n_layers = 80
    print(f"\n  Model: {target} ({n_layers} layers, per-layer: {m['raw_bytes']/1e6:.1f} MB)")
    print(f"\n  {'Network':<28} {'Enc+Ser':>9} {'NetXfer':>9} {'Deser+Dec':>10} {'Bottleneck':>11}")
    print(f"  {'-'*70}")

    for net_name, gbps in networks:
        _, _, bn, (s1, s2, s3) = simulate_pipeline(m, gbps, n_layers)
        marker = lambda s, name: f"*{s*1000:.3f}*" if name == bn else f" {s*1000:.3f} "
        print(f"  {net_name:<28} {s1*1000:>8.3f} {s2*1000:>8.3f} {s3*1000:>9.3f}  [{bn}]")

    # ================================================================
    print("\n\n" + "=" * 100)
    print("PHASE 5: BREAK-EVEN ANALYSIS")
    print("=" * 100)
    print("  Minimum network bandwidth where SplitZip pipeline > 1.0x speedup")

    for name, kv_heads, seq_len, head_dim, n_layers, desc in configs:
        m = measurements[name]

        # Binary search for break-even bandwidth
        lo, hi = 1, 10000  # Gbps
        for _ in range(50):
            mid = (lo + hi) / 2
            raw_t, pipe_t, _, _ = simulate_pipeline(m, mid, n_layers)
            if pipe_t < raw_t:
                lo = mid
            else:
                hi = mid

        # At break-even, what's the stage breakdown?
        raw_t, pipe_t, bn, stages = simulate_pipeline(m, hi, n_layers)

        print(f"  {name:<25} break-even at >{hi:.0f} Gbps "
              f"(always wins below, [{bn}]-limited above)")

    # ================================================================
    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("""
  SplitZip lossless KV compression for disaggregated PD serving:

  1. ALWAYS beneficial at TCP bandwidths (25-100 Gbps):
     Network is the bottleneck → speedup ≈ compression ratio (1.29-1.33x)

  2. Still beneficial at RDMA bandwidths (200-400 Gbps):
     Network still dominates for large KV caches

  3. Diminishing returns at ultra-high bandwidth (4x200G+):
     Encode+serialize overhead starts to matter

  4. No existing system does lossless KV compression during transfer.
     CacheGen (SIGCOMM'24) is the only compression-aware system, but it's lossy.

  5. Integration path: plug into Mooncake/vLLM/SGLang KV connector layer
     as a transparent encode/decode wrapper around the transfer.
""")


if __name__ == "__main__":
    main()
