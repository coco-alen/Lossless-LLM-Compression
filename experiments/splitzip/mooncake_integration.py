"""
Real Mooncake Transfer Engine Integration for SplitZip.

This test uses Mooncake's actual TransferEngine with TCP transport
and etcd metadata to transfer KV cache data between two endpoints
(simulating prefill→decode on the same machine).

We measure: raw BF16 transfer vs SplitZip-compressed transfer,
including real serialization, compression, network transfer, and decompression.
"""

import torch
import time
import sys, os
import threading
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Set CUDA library path before importing mooncake
os.environ['LD_LIBRARY_PATH'] = (
    '/data02/home/yipin/miniconda3/envs/quant/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:'
    + os.environ.get('LD_LIBRARY_PATH', '')
)

import mooncake.engine as mte


def init_transfer_engine(name, port, metadata_server="localhost:2379"):
    """Initialize a Mooncake transfer engine endpoint."""
    te = mte.TransferEngine()
    ret = te.initialize(f'localhost:{port}', metadata_server, 'tcp', 'cpu')
    if ret != 0:
        raise RuntimeError(f"Failed to initialize TransferEngine '{name}': ret={ret}")
    print(f"  [{name}] TransferEngine initialized on port {port}")
    return te


def run_mooncake_kv_transfer():
    """Run real Mooncake KV transfer benchmark."""
    print("=" * 90)
    print("REAL MOONCAKE TRANSFER ENGINE: SplitZip KV Cache Transfer")
    print("=" * 90)

    # Initialize two transfer engine endpoints (prefill and decode)
    print("\nInitializing transfer engines...")
    try:
        te_prefill = init_transfer_engine("prefill", 23456)
        te_decode = init_transfer_engine("decode", 23457)
    except RuntimeError as e:
        print(f"  Failed: {e}")
        print("  Falling back to direct memory benchmark (no Mooncake transport)")
        run_direct_benchmark()
        return

    # Test data: KV cache for one layer
    # Qwen2.5-7B: 4 KV heads, head_dim=128
    for n_tokens in [1024, 4096]:
        kv_shape = (2, 4, n_tokens, 128)  # K+V, heads, seq, dim
        kv_data = torch.randn(kv_shape, dtype=torch.bfloat16)
        kv_bytes = kv_data.numel() * 2
        kv_flat = kv_data.numpy().tobytes() if hasattr(kv_data, 'numpy') else kv_data.view(torch.uint8).numpy().tobytes()

        print(f"\n--- KV: {n_tokens} tokens, {kv_bytes/1024:.0f} KB ---")

        # Allocate managed buffers
        buf_prefill = te_prefill.allocate_managed_buffer(kv_bytes)
        buf_decode = te_decode.allocate_managed_buffer(kv_bytes)

        # Write KV data to prefill buffer
        te_prefill.write_bytes_to_buffer(buf_prefill, kv_flat)

        # Raw transfer (no compression)
        t0 = time.perf_counter()
        ret = te_prefill.transfer_sync_write(
            buf_prefill, "localhost:23457", buf_decode, kv_bytes)
        raw_time = time.perf_counter() - t0

        # Verify
        received = te_decode.read_bytes_from_buffer(buf_decode, kv_bytes)
        match = (received == kv_flat)

        raw_bw = kv_bytes / raw_time / 1e9 if raw_time > 0 else float('inf')
        print(f"  Raw transfer: {raw_time*1000:.2f} ms, {raw_bw:.1f} GB/s, match={match}")

        # SplitZip compressed transfer
        # Step 1: Compress on prefill side
        from experiments.splitzip.lossless_fast import FastLosslessCodec
        codec = FastLosslessCodec('cpu')  # CPU for Mooncake managed buffers
        codec.calibrate(kv_data.view(-1))

        t0 = time.perf_counter()
        kv_gpu = kv_data.cuda()
        r = codec.encode(kv_gpu.view(-1))
        pk, sm, esc_pos, esc_val, n, n_esc = r
        # Serialize compressed data
        pk_bytes = pk.cpu().numpy().tobytes()
        sm_bytes = sm.cpu().numpy().tobytes()
        esc_bytes = b''
        if n_esc > 0:
            esc_bytes = (esc_pos.cpu().numpy().tobytes() +
                        esc_val.cpu().numpy().tobytes())
        header = struct.pack('III', n, n_esc, len(esc_bytes))
        compressed = header + pk_bytes + sm_bytes + esc_bytes
        compress_time = time.perf_counter() - t0

        comp_bytes = len(compressed)
        ratio = kv_bytes / comp_bytes

        # Step 2: Transfer compressed
        comp_buf_p = te_prefill.allocate_managed_buffer(comp_bytes)
        comp_buf_d = te_decode.allocate_managed_buffer(comp_bytes)
        te_prefill.write_bytes_to_buffer(comp_buf_p, compressed)

        t0 = time.perf_counter()
        ret = te_prefill.transfer_sync_write(
            comp_buf_p, "localhost:23457", comp_buf_d, comp_bytes)
        transfer_time = time.perf_counter() - t0

        # Step 3: Decompress on decode side
        t0 = time.perf_counter()
        recv_comp = te_decode.read_bytes_from_buffer(comp_buf_d, comp_bytes)
        # Deserialize
        hdr = struct.unpack('III', recv_comp[:12])
        n_recv, n_esc_recv, esc_len = hdr
        offset = 12
        pk_recv = torch.frombuffer(bytearray(recv_comp[offset:offset+n_recv//2]), dtype=torch.uint8).cuda()
        offset += n_recv // 2
        sm_recv = torch.frombuffer(bytearray(recv_comp[offset:offset+n_recv]), dtype=torch.uint8).cuda()
        offset += n_recv
        if n_esc_recv > 0:
            ep_recv = torch.frombuffer(bytearray(recv_comp[offset:offset+n_esc_recv*4]), dtype=torch.int32).cuda()
            offset += n_esc_recv * 4
            ev_recv = torch.frombuffer(bytearray(recv_comp[offset:offset+n_esc_recv]), dtype=torch.uint8).cuda()
        else:
            ep_recv = torch.empty(0, dtype=torch.int32, device='cuda')
            ev_recv = torch.empty(0, dtype=torch.uint8, device='cuda')
        decoded = codec.decode(pk_recv, sm_recv, ep_recv, ev_recv, n_recv, n_esc_recv)
        decompress_time = time.perf_counter() - t0

        # Verify lossless
        lossless = torch.equal(kv_gpu.view(torch.int16).view(-1),
                               decoded.view(torch.int16).view(-1))

        total_comp = compress_time + transfer_time + decompress_time
        speedup = raw_time / total_comp if total_comp > 0 else float('inf')

        print(f"  Compressed: ratio={ratio:.3f}x")
        print(f"    Compress: {compress_time*1000:.2f} ms")
        print(f"    Transfer: {transfer_time*1000:.2f} ms ({comp_bytes/transfer_time/1e9:.1f} GB/s)")
        print(f"    Decompress: {decompress_time*1000:.2f} ms")
        print(f"    Total: {total_comp*1000:.2f} ms vs raw {raw_time*1000:.2f} ms")
        print(f"    Speedup: {speedup:.3f}x, Lossless: {lossless}")

        # Cleanup
        te_prefill.free_managed_buffer(buf_prefill, kv_bytes)
        te_decode.free_managed_buffer(buf_decode, kv_bytes)
        te_prefill.free_managed_buffer(comp_buf_p, comp_bytes)
        te_decode.free_managed_buffer(comp_buf_d, comp_bytes)


def run_direct_benchmark():
    """Fallback: measure compress→memcpy→decompress without Mooncake."""
    print("\n" + "=" * 90)
    print("DIRECT BENCHMARK (without Mooncake transport)")
    print("=" * 90)

    device = 'cuda'
    from experiments.splitzip.lossless_fast import FastLosslessCodec

    codec = FastLosslessCodec(device)
    calib = torch.randn(1024*1024, dtype=torch.bfloat16, device=device)
    codec.calibrate(calib)

    for n_tokens in [256, 1024, 4096]:
        n = 2 * 4 * n_tokens * 128  # K+V, 4 heads, seq, 128 dim
        kv = torch.randn(n, dtype=torch.bfloat16, device=device)
        nbytes = n * 2

        # Warmup
        for _ in range(10):
            r = codec.encode(kv)
            codec.decode(*r)

        # Measure full pipeline
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 100
        for _ in range(iters):
            # Encode
            r = codec.encode(kv)
            # Simulate transfer: GPU→CPU→GPU (like Mooncake TCP path)
            pk_cpu = r[0].cpu()
            sm_cpu = r[1].cpu()
            pk_back = pk_cpu.cuda()
            sm_back = sm_cpu.cuda()
            # Decode
            decoded = codec.decode(pk_back, sm_back, r[2], r[3], r[4], r[5])
        torch.cuda.synchronize()
        total_t = (time.perf_counter() - t0) / iters

        # Raw GPU→CPU→GPU
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            cpu_buf = kv.cpu()
            gpu_buf = cpu_buf.cuda()
        torch.cuda.synchronize()
        raw_t = (time.perf_counter() - t0) / iters

        lossless = torch.equal(kv.view(torch.int16), decoded.view(torch.int16))
        comp_bytes = r[0].numel() + r[1].numel() + r[2].numel() * 4 + r[3].numel()
        ratio = nbytes / comp_bytes

        print(f"\n  {n_tokens} tokens ({nbytes/1024:.0f} KB):")
        print(f"    Raw GPU→CPU→GPU:        {raw_t*1000:.3f} ms")
        print(f"    SplitZip full pipeline:  {total_t*1000:.3f} ms")
        print(f"    Speedup: {raw_t/total_t:.3f}x, Ratio: {ratio:.3f}x, Lossless: {lossless}")


if __name__ == "__main__":
    try:
        run_mooncake_kv_transfer()
    except Exception as e:
        print(f"\nMooncake transfer failed: {e}")
        print("Running direct benchmark instead...")
        run_direct_benchmark()
