from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from experiments.splitzip_v2.benchmark_utils import write_json
from experiments.splitzip_v2.codec_gpu import (
    ChunkLocalSplitZipGPU,
    _count_escapes_chunk,
    _dec_4bit,
    _dec_4bit_pair32,
    _dec_4bit_quad64,
    _enc_4bit,
    _fix_escapes_local_linear,
    _write_escapes_chunk,
)


def bench_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def make_input(numel: int, device: str, seed: int):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.randn(numel, dtype=torch.bfloat16, device=device, generator=gen)


def run(
    numel: int,
    chunk_size: int,
    device: str,
    warmup: int,
    iters: int,
    decode_block: int = 256,
    decode_num_warps: int = 4,
    decode_vector: str = "pair32",
    fix_block: int = 256,
    fix_num_warps: int = 4,
):
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    x = make_input(numel, device, seed=7)
    codec = ChunkLocalSplitZipGPU(device=device, chunk_size=chunk_size)
    coverage = codec.calibrate(x)
    encoded = codec.encode(x)
    decoded = codec.decode(encoded)
    if not torch.equal(x.view(torch.int16), decoded.view(torch.int16)):
        raise RuntimeError("GPU chunk-local codec failed lossless round trip")

    flat = x.contiguous().view(torch.int16)
    n = int(flat.numel())
    n_pairs = (n + 1) // 2
    enc_block = 256
    n_chunks = (n + chunk_size - 1) // chunk_size
    pk = torch.empty(n_pairs, dtype=torch.uint8, device=device)
    sm = torch.empty(n, dtype=torch.uint8, device=device)
    counts = torch.empty(n_chunks, dtype=torch.int32, device=device)

    def dense_encode():
        _enc_4bit[((n_pairs + enc_block * 4 - 1) // (enc_block * 4),)](
            flat, codec.enc_lut, pk, sm, n, BLOCK=enc_block
        )

    def count_escapes():
        _count_escapes_chunk[(n_chunks,)](flat, codec.common_lut, counts, n, CHUNK=chunk_size)

    dense_encode()
    count_escapes()
    starts = torch.cumsum(counts, dim=0) - counts
    n_esc = int((starts[-1] + counts[-1]).item()) if n_chunks else 0
    chunk_id = torch.empty(n_esc, dtype=torch.int32, device=device)
    local_pos = torch.empty(n_esc, dtype=torch.uint16, device=device)
    esc_val = torch.empty(n_esc, dtype=torch.uint8, device=device)
    out = torch.empty(n, dtype=torch.int16, device=device)

    def prefix_sum():
        torch.cumsum(counts, dim=0)

    def scatter_escapes():
        if n_esc:
            _write_escapes_chunk[(n_chunks,)](
                flat, codec.common_lut, starts, chunk_id, local_pos, esc_val, n, CHUNK=chunk_size
            )

    def dense_decode():
        if decode_vector == "quad64" and n % 4 == 0:
            n_quads = n // 4
            _dec_4bit_quad64[((n_quads + decode_block - 1) // decode_block,)](
                pk.view(torch.int16),
                sm.view(torch.int32),
                codec.dec_lut,
                out.view(torch.int64),
                n_quads,
                BLOCK=decode_block,
                num_warps=decode_num_warps,
            )
        elif n % 2 == 0:
            _dec_4bit_pair32[((n_pairs + decode_block - 1) // decode_block,)](
                pk,
                sm,
                codec.dec_lut,
                out.view(torch.int32),
                n_pairs,
                BLOCK=decode_block,
                num_warps=decode_num_warps,
            )
        else:
            _dec_4bit[((n_pairs + decode_block * 4 - 1) // (decode_block * 4),)](
                pk, sm, codec.dec_lut, out, n, BLOCK=decode_block, num_warps=decode_num_warps
            )

    def fix_escapes():
        if n_esc:
            _fix_escapes_local_linear[((n_esc + fix_block - 1) // fix_block,)](
                chunk_id,
                local_pos,
                esc_val,
                sm,
                out,
                n_esc,
                CHUNK=chunk_size,
                BLOCK_ESC=fix_block,
                num_warps=fix_num_warps,
            )

    scatter_escapes()
    dense_decode()
    fix_escapes()

    stage_s = {
        "dense_encode": bench_cuda(dense_encode, warmup, iters),
        "count_escapes": bench_cuda(count_escapes, warmup, iters),
        "prefix_sum": bench_cuda(prefix_sum, warmup, iters),
        "scatter_escapes": bench_cuda(scatter_escapes, warmup, iters),
        "dense_decode": bench_cuda(dense_decode, warmup, iters),
        "fix_escapes": bench_cuda(fix_escapes, warmup, iters),
    }
    raw_bytes = n * 2
    return {
        "numel": n,
        "raw_bytes": raw_bytes,
        "chunk_size": chunk_size,
        "decode_block": decode_block,
        "decode_num_warps": decode_num_warps,
        "decode_vector": decode_vector,
        "fix_block": fix_block,
        "fix_num_warps": fix_num_warps,
        "n_chunks": n_chunks,
        "n_escapes": n_esc,
        "coverage": coverage,
        "compressed_bytes": encoded.compressed_bytes,
        "ratio": raw_bytes / encoded.compressed_bytes,
        "stage_s": stage_s,
        "stage_gbs": {k: raw_bytes / v / 1e9 for k, v in stage_s.items()},
        "encode_total_s": stage_s["dense_encode"] + stage_s["count_escapes"] + stage_s["prefix_sum"] + stage_s["scatter_escapes"],
        "decode_total_s": stage_s["dense_decode"] + stage_s["fix_escapes"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=65536 * 4096)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--decode-block", type=int, default=256)
    parser.add_argument("--decode-num-warps", type=int, default=4)
    parser.add_argument("--decode-vector", choices=["pair32", "quad64"], default="pair32")
    parser.add_argument("--fix-block", type=int, default=256)
    parser.add_argument("--fix-num-warps", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/chunklocal_gpu_breakdown.json"))
    args = parser.parse_args()
    write_json(
        args.output,
        run(
            args.numel,
            args.chunk_size,
            args.device,
            args.warmup,
            args.iters,
            decode_block=args.decode_block,
            decode_num_warps=args.decode_num_warps,
            decode_vector=args.decode_vector,
            fix_block=args.fix_block,
            fix_num_warps=args.fix_num_warps,
        ),
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
