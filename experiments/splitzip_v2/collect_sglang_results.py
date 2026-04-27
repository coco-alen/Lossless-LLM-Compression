from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.splitzip_v2.benchmark_utils import write_json


def parse_name(path: Path):
    stem = path.stem
    parts = stem.split("_")
    out = {"file": str(path)}
    for part in parts:
        if part.startswith("bs") and part[2:].isdigit():
            out["batch_size"] = int(part[2:])
        if part.startswith("seq") and part[3:].isdigit():
            out["seq_len"] = int(part[3:])
    if "bs1_seq" in stem:
        out["sweep"] = "bs1_seq"
    elif "bs16_seq" in stem:
        out["sweep"] = "bs16_seq"
    elif "seq1024_bs" in stem:
        out["sweep"] = "seq1024_bs"
    elif "seq32768_bs" in stem:
        out["sweep"] = "seq32768_bs"
    return out


def read_last_jsonl(path: Path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty SGLang result file: {path}")
    return rows[-1]


def collect(root: Path):
    rows = []
    for mode_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        mode = mode_dir.name
        for path in sorted(mode_dir.glob("*.jsonl")):
            result = read_last_jsonl(path)
            rows.append({
                "mode": mode,
                **parse_name(path),
                "completed": result.get("completed"),
                "mean_ttft_ms": result.get("mean_ttft_ms"),
                "median_ttft_ms": result.get("median_ttft_ms"),
                "p99_ttft_ms": result.get("p99_ttft_ms"),
                "request_throughput": result.get("request_throughput"),
                "output_throughput": result.get("output_throughput"),
                "total_throughput": result.get("total_throughput"),
                "mean_tpot_ms": result.get("mean_tpot_ms"),
                "median_tpot_ms": result.get("median_tpot_ms"),
            })
    return rows


def add_speedups(rows):
    by_key = {}
    for row in rows:
        key = (row.get("sweep"), row.get("batch_size"), row.get("seq_len"))
        by_key.setdefault(key, {})[row["mode"]] = row
    out = []
    for key, modes in sorted(by_key.items(), key=lambda item: (str(item[0][0]), item[0][1] or 0, item[0][2] or 0)):
        native = modes.get("native")
        splitzip = modes.get("splitzip")
        if native and splitzip:
            out.append({
                "sweep": key[0],
                "batch_size": key[1],
                "seq_len": key[2],
                "native_median_ttft_ms": native["median_ttft_ms"],
                "splitzip_median_ttft_ms": splitzip["median_ttft_ms"],
                "ttft_speedup": native["median_ttft_ms"] / splitzip["median_ttft_ms"]
                if native["median_ttft_ms"] and splitzip["median_ttft_ms"] else None,
                "native_output_throughput": native["output_throughput"],
                "splitzip_output_throughput": splitzip["output_throughput"],
                "throughput_speedup": splitzip["output_throughput"] / native["output_throughput"]
                if native["output_throughput"] and splitzip["output_throughput"] else None,
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/splitzip_v2/results/sglang"))
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/sglang_summary.json"))
    args = parser.parse_args()
    rows = collect(args.root)
    write_json(args.output, {"rows": rows, "paired_speedups": add_speedups(rows)})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
