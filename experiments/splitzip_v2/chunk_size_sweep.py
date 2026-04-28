from __future__ import annotations

import argparse
from pathlib import Path

from experiments.splitzip_v2.benchmark_utils import write_json
from experiments.splitzip_v2.gpu_breakdown_bench import run


def summarize(result: dict) -> dict:
    raw = result["raw_bytes"]
    return {
        "chunk_size": result["chunk_size"],
        "decode_vector": result.get("decode_vector", "pair32"),
        "decode_block": result.get("decode_block"),
        "decode_num_warps": result.get("decode_num_warps"),
        "fix_block": result.get("fix_block"),
        "fix_num_warps": result.get("fix_num_warps"),
        "n_chunks": result["n_chunks"],
        "n_escapes": result["n_escapes"],
        "ratio": result["ratio"],
        "encode_gbs": raw / result["encode_total_s"] / 1e9,
        "decode_gbs": raw / result["decode_total_s"] / 1e9,
        "stage_gbs": result["stage_gbs"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=65536 * 4096)
    parser.add_argument("--chunk-sizes", nargs="+", type=int,
                        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--decode-vector", choices=["pair32", "quad64"], default="quad64")
    parser.add_argument("--decode-block", type=int, default=512)
    parser.add_argument("--decode-num-warps", type=int, default=4)
    parser.add_argument("--fix-block", type=int, default=128)
    parser.add_argument("--fix-num-warps", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/chunk_size_sweep.json"))
    args = parser.parse_args()

    rows = []
    full_results = []
    for chunk_size in args.chunk_sizes:
        print(f"chunk_size={chunk_size}", flush=True)
        result = run(
            args.numel,
            chunk_size,
            args.device,
            args.warmup,
            args.iters,
            decode_block=args.decode_block,
            decode_num_warps=args.decode_num_warps,
            decode_vector=args.decode_vector,
            fix_block=args.fix_block,
            fix_num_warps=args.fix_num_warps,
        )
        full_results.append(result)
        row = summarize(result)
        rows.append(row)
        print(
            f"  ratio={row['ratio']:.4f} encode={row['encode_gbs']:.2f} GB/s "
            f"decode={row['decode_gbs']:.2f} GB/s",
            flush=True,
        )

    best_encode = max(rows, key=lambda x: x["encode_gbs"])
    best_decode = max(rows, key=lambda x: x["decode_gbs"])
    best_hmean = max(rows, key=lambda x: 2.0 / (1.0 / x["encode_gbs"] + 1.0 / x["decode_gbs"]))
    write_json(args.output, {
        "numel": args.numel,
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "decode_vector": args.decode_vector,
        "decode_block": args.decode_block,
        "decode_num_warps": args.decode_num_warps,
        "fix_block": args.fix_block,
        "fix_num_warps": args.fix_num_warps,
        "rows": rows,
        "best_encode": best_encode,
        "best_decode": best_decode,
        "best_harmonic_mean": best_hmean,
        "full_results": full_results,
    })
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
