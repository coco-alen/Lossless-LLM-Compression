from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import triton
import triton.language as tl

from experiments.splitzip_v2.benchmark_utils import write_json


@triton.jit
def _decode_traffic_bound(pk16, sm32, out64, n_quads, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_quads
    pk = tl.load(pk16 + offs, mask=mask, other=0).to(tl.uint64)
    sm = tl.load(sm32 + offs, mask=mask, other=0).to(tl.uint64)
    # Same mandatory read/write footprint as quad64 decode, minimal ALU.
    tl.store(out64 + offs, pk | (sm << 16), mask=mask)


def bench_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def run(numel: int, device: str, warmup: int, iters: int, blocks: list[int], warps: list[int]):
    n_pairs = (numel + 1) // 2
    n_quads = numel // 4
    pk = torch.empty(n_pairs, dtype=torch.uint8, device=device)
    sm = torch.empty(numel, dtype=torch.uint8, device=device)
    out = torch.empty(numel, dtype=torch.int16, device=device)
    pk16 = pk.view(torch.int16)
    sm32 = sm.view(torch.int32)
    out64 = out.view(torch.int64)

    rows = []
    best = None
    raw_bytes = numel * 2
    mandatory_bytes = n_quads * (2 + 4 + 8)
    for block in blocks:
        for num_warps in warps:
            t = bench_cuda(
                lambda: _decode_traffic_bound[((n_quads + block - 1) // block,)](
                    pk16, sm32, out64, n_quads, BLOCK=block, num_warps=num_warps
                ),
                warmup,
                iters,
            )
            row = {
                "block": block,
                "num_warps": num_warps,
                "seconds": t,
                "raw_gbs": raw_bytes / t / 1e9,
                "mandatory_traffic_gbs": mandatory_bytes / t / 1e9,
            }
            rows.append(row)
            if best is None or row["raw_gbs"] > best["raw_gbs"]:
                best = row
            print(
                f"block={block} warps={num_warps} raw={row['raw_gbs']:.2f} GB/s "
                f"traffic={row['mandatory_traffic_gbs']:.2f} GB/s",
                flush=True,
            )
    return {
        "numel": numel,
        "raw_bytes": raw_bytes,
        "mandatory_bytes": mandatory_bytes,
        "rows": rows,
        "best": best,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=65536 * 4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--blocks", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/decode_roofline_bench.json"))
    args = parser.parse_args()
    result = run(args.numel, args.device, args.warmup, args.iters, args.blocks, args.warps)
    write_json(args.output, result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
