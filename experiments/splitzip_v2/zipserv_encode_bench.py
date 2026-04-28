from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import sys
import time

import torch

from experiments.splitzip.thesis_experiment_dump import assemble_row_prefix, load_model_activation_blocks
from experiments.splitzip_v2.benchmark_utils import mean_std, write_json


@contextlib.contextmanager
def silence_fds():
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def assemble_matrix_width(blocks, target_rows: int, target_width: int):
    pieces = []
    width = 0
    start_idx = 0
    while width < target_width:
        block = assemble_row_prefix(blocks, target_rows, start_idx=start_idx)
        take = min(int(block.shape[1]), target_width - width)
        pieces.append(block[:, :take])
        width += take
        start_idx += 1
    return torch.cat(pieces, dim=1).contiguous()


def compressed_bytes(meta) -> int:
    return sum(
        int(value.numel() * value.element_size())
        for value in meta.values()
        if torch.is_tensor(value)
    )


def bench_zipserv_encode(matrix: torch.Tensor, repeats: int, warmup: int, zipserv_root: Path):
    sys.path.insert(0, str(zipserv_root / "bindings"))
    sys.path.insert(0, str(zipserv_root / "python"))
    from bf16_zip import compress_tensor_bf16

    if matrix.dtype != torch.bfloat16 or matrix.device.type != "cpu":
        raise ValueError("ZipServ encode expects CPU BF16 matrix")
    if matrix.dim() != 2 or matrix.shape[0] % 64 or matrix.shape[1] % 64:
        raise ValueError(f"ZipServ encode expects 2D 64-aligned shape, got {tuple(matrix.shape)}")

    with silence_fds():
        for _ in range(warmup):
            meta = compress_tensor_bf16(matrix)

    times = []
    last_meta = None
    with silence_fds():
        for _ in range(repeats):
            t0 = time.perf_counter()
            last_meta = compress_tensor_bf16(matrix)
            times.append(time.perf_counter() - t0)

    raw_bytes = int(matrix.numel() * matrix.element_size())
    comp_bytes = compressed_bytes(last_meta)
    return {
        "shape": list(matrix.shape),
        "raw_bytes": raw_bytes,
        "compressed_bytes": comp_bytes,
        "ratio": raw_bytes / comp_bytes,
        "encode_s": mean_std(times),
        "encode_gbs": mean_std([raw_bytes / t / 1e9 for t in times]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3-8B")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1024, 4096, 16384, 65536])
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--zipserv-root", type=Path, default=Path("/data02/home/yilian2/project/ZipServ_BF16"))
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/zipserv_encode_bench.json"))
    args = parser.parse_args()

    blocks, meta = load_model_activation_blocks(args.model, torch.device(args.device))
    rows = []
    for seq_len in args.seq_lens:
        print(f"ZipServ encode shape=({seq_len}, {args.hidden_dim})", flush=True)
        matrix = assemble_matrix_width(blocks, seq_len, args.hidden_dim).cpu().contiguous()
        row = bench_zipserv_encode(matrix, args.repeats, args.warmup, args.zipserv_root)
        rows.append(row)
        print(
            f"  ratio={row['ratio']:.4f} encode={row['encode_gbs']['mean']:.4f} GB/s "
            f"+/- {row['encode_gbs']['std']:.4f}",
            flush=True,
        )
    write_json(args.output, {"model": args.model, "activation_meta": meta, "rows": rows})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
