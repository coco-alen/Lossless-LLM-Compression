from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Dict, List

from experiments.splitzip_v2.benchmark_utils import write_json


BASELINES = [
    {
        "name": "nvcomp_lz4",
        "family": "nvCOMP",
        "status": "implemented_v1",
        "runner": "experiments.splitzip.codec_ablation_bench:nvcomp_bench",
        "notes": "Existing v1 byte-stream LZ4 baseline.",
    },
    {
        "name": "nvcomp_cascaded",
        "family": "nvCOMP",
        "status": "adapter",
        "algorithm": "Cascaded",
        "notes": "Use nvidia.nvcomp.Codec(algorithm='Cascaded') on raw BF16 bytes when available.",
    },
    {
        "name": "nvcomp_bitcomp",
        "family": "nvCOMP",
        "status": "adapter",
        "algorithm": "Bitcomp",
        "notes": "Use nvidia.nvcomp.Codec(algorithm='Bitcomp') when the installed nvCOMP wheel exposes it.",
    },
    {
        "name": "zipserv_tca_tbe",
        "family": "ZipServ",
        "status": "implemented_v2_boundary",
        "runner": "experiments.splitzip.check_zipserv_baseline_boundary",
        "notes": "GPU-resident pybind path keeps wrapper allocation overhead but excludes CPU/GPU transfer.",
    },
    {
        "name": "tca_tbe",
        "family": "TCA-TBE",
        "status": "alias_of_zipserv_encoding",
        "notes": "ZipServ's TCA-TBE is the triple-bitmap encoding in the local ZipServ_BF16 source tree.",
    },
    {
        "name": "falcon",
        "family": "Falcon",
        "status": "external_required",
        "notes": "Provide --falcon-root when source is available; adapter records exact command and result JSON.",
    },
]


def check_nvcomp_algorithms() -> Dict[str, str]:
    try:
        from nvidia import nvcomp
    except Exception as exc:
        return {"available": "false", "error": repr(exc)}
    out = {"available": "true"}
    for alg in ("LZ4", "Cascaded", "Bitcomp"):
        try:
            nvcomp.Codec(algorithm=alg)
            out[alg] = "ok"
        except Exception as exc:
            out[alg] = f"unavailable: {exc!r}"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/baseline_matrix.json"))
    args = parser.parse_args()
    write_json(args.output, {"baselines": BASELINES, "nvcomp_probe": check_nvcomp_algorithms()})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

