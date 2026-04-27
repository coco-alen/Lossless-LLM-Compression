from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from experiments.splitzip_v2.benchmark_utils import StageBreakdown, write_json
from experiments.splitzip_v2.codec_cpu import ChunkLocalSplitZipCPU, reviewer_compaction_paragraph


def make_input(numel: int, seed: int):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(numel, dtype=torch.bfloat16, generator=gen)


def run_cpu_breakdown(numel: int, chunk_size: int, repeats: int):
    x = make_input(numel, 17)
    codec = ChunkLocalSplitZipCPU(chunk_size=chunk_size)
    coverage = codec.calibrate(x)
    breakdown = StageBreakdown()
    result = None
    for _ in range(repeats):
        enc = codec.encode(x, profile=True)
        dec = codec.decode(enc, profile=True)
        if not torch.equal(x.view(torch.int16), dec.view(torch.int16)):
            raise RuntimeError("chunk-local CPU codec failed lossless round trip")
        breakdown.add(enc.timings_s)
        result = enc
    return {
        "numel": numel,
        "chunk_size": chunk_size,
        "coverage": coverage,
        "ratio": result.ratio,
        "raw_bytes": result.raw_bytes,
        "compressed_bytes": result.compressed_bytes,
        "n_chunks": result.n_chunks,
        "n_escapes": result.n_escapes,
        "summary_s": breakdown.summary(),
        "paper_paragraph": reviewer_compaction_paragraph(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/chunklocal_cpu_breakdown.json"))
    args = parser.parse_args()
    write_json(args.output, run_cpu_breakdown(args.numel, args.chunk_size, args.repeats))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

