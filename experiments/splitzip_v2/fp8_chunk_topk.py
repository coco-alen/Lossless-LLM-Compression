from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.splitzip_v2.benchmark_utils import write_json


def fp8_exponent(raw: torch.Tensor, fmt: str) -> torch.Tensor:
    raw = raw.to(torch.uint8).contiguous().view(-1)
    if fmt == "e4m3":
        return ((raw >> 3) & 0x0F).to(torch.uint8)
    if fmt == "e5m2":
        return ((raw >> 2) & 0x1F).to(torch.uint8)
    raise ValueError(fmt)


def topk_coverage(exp: torch.Tensor, k: int):
    vals, counts = torch.unique(exp.cpu(), return_counts=True)
    order = torch.argsort(counts, descending=True)
    top = vals[order[: min(k, vals.numel())]]
    lut = torch.zeros(256, dtype=torch.bool)
    lut[top.long()] = True
    return float(lut[exp.cpu().long()].float().mean().item()), [int(v) for v in top.tolist()]


def chunk_local_topk_coverage(exp: torch.Tensor, k: int, chunk_size: int):
    exp = exp.cpu().contiguous().view(-1)
    total_common = 0
    codebooks = []
    for start in range(0, exp.numel(), chunk_size):
        chunk = exp[start:start + chunk_size]
        cov, top = topk_coverage(chunk, k)
        total_common += int(round(cov * chunk.numel()))
        codebooks.append(top)
    return total_common / exp.numel(), codebooks


def estimate_fp8_ratio(n: int, fmt: str, coverage: float, chunk_size: int, codebook_per_chunk: bool):
    sm_bits = 4 if fmt == "e4m3" else 3
    code_bits = 3
    dense_bytes = (n * (sm_bits + code_bits) + 7) // 8
    n_chunks = (n + chunk_size - 1) // chunk_size
    esc = int(round(n * (1.0 - coverage)))
    pos_bytes = 1 if chunk_size <= 256 else 2
    exp_bits = 4 if fmt == "e4m3" else 5
    esc_bytes = esc * pos_bytes + (esc * exp_bits + 7) // 8
    codebook_bytes = n_chunks * 8 if codebook_per_chunk else 8
    count_bytes = n_chunks * 2
    comp = dense_bytes + esc_bytes + codebook_bytes + count_bytes
    return n / comp


def analyze_raw(raw: torch.Tensor, fmt: str, chunk_size: int = 256):
    exp = fp8_exponent(raw, fmt)
    global_cov, global_top = topk_coverage(exp, 8)
    chunk_cov, _ = chunk_local_topk_coverage(exp, 8, chunk_size)
    return {
        "fmt": fmt,
        "numel": int(exp.numel()),
        "chunk_size": chunk_size,
        "global_top8_coverage": global_cov,
        "chunk_local_top8_coverage": chunk_cov,
        "global_top8_values": global_top,
        "global_top8_ratio_est": estimate_fp8_ratio(exp.numel(), fmt, global_cov, chunk_size, False),
        "chunk_local_top8_ratio_est": estimate_fp8_ratio(exp.numel(), fmt, chunk_cov, chunk_size, True),
    }


def synthetic_raw(numel: int, fmt: str, seed: int):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    bf16 = torch.randn(numel, dtype=torch.bfloat16, generator=gen)
    if fmt == "e4m3":
        return bf16.to(torch.float8_e4m3fn).view(torch.uint8)
    return bf16.to(torch.float8_e5m2).view(torch.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt", choices=["e4m3", "e5m2"], default="e5m2")
    parser.add_argument("--numel", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/fp8_chunk_topk.json"))
    args = parser.parse_args()
    raw = synthetic_raw(args.numel, args.fmt, seed=31)
    write_json(args.output, analyze_raw(raw, args.fmt, args.chunk_size))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

