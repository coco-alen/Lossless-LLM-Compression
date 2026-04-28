from __future__ import annotations

import argparse
from pathlib import Path

from experiments.splitzip_v2.benchmark_utils import write_json
from experiments.splitzip_v2.gpu_breakdown_bench import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=65536 * 4096)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--decode-blocks", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--decode-warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--decode-vectors", choices=["pair32", "quad64"], nargs="+", default=["pair32"])
    parser.add_argument("--fix-blocks", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--fix-warps", type=int, nargs="+", default=[4])
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/decode_tuning_sweep.json"))
    args = parser.parse_args()

    rows = []
    best = None
    for decode_vector in args.decode_vectors:
        for decode_block in args.decode_blocks:
            for decode_warps in args.decode_warps:
                for fix_block in args.fix_blocks:
                    for fix_warps in args.fix_warps:
                        print(
                            f"decode_vector={decode_vector} decode_block={decode_block} decode_warps={decode_warps} "
                            f"fix_block={fix_block} fix_warps={fix_warps}",
                            flush=True,
                        )
                        result = run(
                            args.numel,
                            args.chunk_size,
                            args.device,
                            args.warmup,
                            args.iters,
                            decode_block=decode_block,
                            decode_num_warps=decode_warps,
                            decode_vector=decode_vector,
                            fix_block=fix_block,
                            fix_num_warps=fix_warps,
                        )
                        decode_gbs = result["raw_bytes"] / result["decode_total_s"] / 1e9
                        row = {
                            "decode_vector": decode_vector,
                            "decode_block": decode_block,
                            "decode_num_warps": decode_warps,
                            "fix_block": fix_block,
                            "fix_num_warps": fix_warps,
                            "decode_gbs": decode_gbs,
                            "dense_decode_gbs": result["stage_gbs"]["dense_decode"],
                            "fix_escapes_gbs": result["stage_gbs"]["fix_escapes"],
                            "decode_total_s": result["decode_total_s"],
                            "stage_s": result["stage_s"],
                        }
                        rows.append(row)
                        if best is None or row["decode_gbs"] > best["decode_gbs"]:
                            best = row
                        print(
                            f"  total={decode_gbs:.2f} GB/s dense={row['dense_decode_gbs']:.2f} "
                            f"fix={row['fix_escapes_gbs']:.2f}",
                            flush=True,
                        )
                        write_json(args.output, {"rows": rows, "best": best})

    write_json(args.output, {"rows": rows, "best": best})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
