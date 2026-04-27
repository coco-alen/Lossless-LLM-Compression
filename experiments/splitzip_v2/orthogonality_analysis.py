from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.splitzip_v2.benchmark_utils import write_json


DEFAULT_TECHNIQUES = [
    {
        "name": "FlowKV",
        "mechanism": "transfer consolidation / fewer KV movements",
        "optimized_transfer_fraction": 0.60,
    },
    {
        "name": "HybridServe",
        "mechanism": "hybrid placement and scheduling",
        "optimized_transfer_fraction": 0.70,
    },
    {
        "name": "KVPR",
        "mechanism": "reuse / placement-aware KV routing",
        "optimized_transfer_fraction": 0.65,
    },
]


def codec_ms(raw_bytes: int, ratio: float, encode_gbs: float, decode_gbs: float) -> dict[str, float]:
    raw_gb = raw_bytes / 1e9
    return {
        "encode_ms": raw_gb / encode_gbs * 1000.0,
        "decode_ms": raw_gb / decode_gbs * 1000.0,
        "compressed_bytes": raw_bytes / ratio,
    }


def compose(
    raw_bytes: int,
    native_transfer_ms: float,
    optimized_transfer_fraction: float,
    ratio: float,
    encode_gbs: float,
    decode_gbs: float,
    overlap: bool,
) -> dict[str, float]:
    codec = codec_ms(raw_bytes, ratio, encode_gbs, decode_gbs)
    software_ms = native_transfer_ms * optimized_transfer_fraction
    splitzip_transfer_ms = native_transfer_ms / ratio
    combined_transfer_ms = software_ms / ratio
    if overlap:
        splitzip_ms = max(codec["encode_ms"], splitzip_transfer_ms, codec["decode_ms"])
        combined_ms = max(codec["encode_ms"], combined_transfer_ms, codec["decode_ms"])
    else:
        splitzip_ms = codec["encode_ms"] + splitzip_transfer_ms + codec["decode_ms"]
        combined_ms = codec["encode_ms"] + combined_transfer_ms + codec["decode_ms"]
    return {
        "native_ms": native_transfer_ms,
        "software_only_ms": software_ms,
        "splitzip_only_ms": splitzip_ms,
        "combined_ms": combined_ms,
        "software_speedup": native_transfer_ms / software_ms,
        "splitzip_speedup": native_transfer_ms / splitzip_ms,
        "combined_speedup_vs_native": native_transfer_ms / combined_ms,
        "combined_speedup_vs_software": software_ms / combined_ms,
        **codec,
    }


def load_techniques(path: Path | None):
    if path is None:
        return DEFAULT_TECHNIQUES, True
    return json.loads(path.read_text()), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-bytes", type=int, default=2 * 32768 * 4096)
    parser.add_argument("--native-transfer-ms", type=float, default=3.0)
    parser.add_argument("--ratio", type=float, default=1.316)
    parser.add_argument("--encode-gbs", type=float, default=300.0)
    parser.add_argument("--decode-gbs", type=float, default=500.0)
    parser.add_argument("--techniques", type=Path, default=None)
    parser.add_argument("--no-overlap", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/orthogonality_analysis.json"))
    args = parser.parse_args()

    techniques, assumptions = load_techniques(args.techniques)
    rows = []
    for technique in techniques:
        rows.append({
            **technique,
            **compose(
                raw_bytes=args.raw_bytes,
                native_transfer_ms=args.native_transfer_ms,
                optimized_transfer_fraction=float(technique["optimized_transfer_fraction"]),
                ratio=args.ratio,
                encode_gbs=args.encode_gbs,
                decode_gbs=args.decode_gbs,
                overlap=not args.no_overlap,
            ),
        })
    write_json(args.output, {
        "assumption_only": assumptions,
        "overlap_model": not args.no_overlap,
        "rows": rows,
        "interpretation": (
            "FlowKV/HybridServe/KVPR reduce when or how often KV is moved; SplitZip reduces bytes per move. "
            "Under this model their gains multiply until codec time or another non-transfer stage dominates."
        ),
    })
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
