from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import torch

from experiments.splitzip.thesis_experiment_dump import assemble_row_prefix, load_model_activation_blocks
from experiments.splitzip_v2.benchmark_utils import mean_std, write_json


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "splitzip_v2" / "falcon_float_bench.cu"
RESULT_RE = re.compile(r"(\w+)=([^\s]+)")


def build_binary(falcon_root: Path, output: Path, arch: str, rebuild: bool) -> Path:
    if output.exists() and not rebuild:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        f"-arch={arch}",
        "-I",
        str(falcon_root / "include"),
        "-I",
        str(falcon_root / "src" / "utils"),
        str(SOURCE),
        str(falcon_root / "src" / "gpu" / "Falcon_float_compressor.cu"),
        str(falcon_root / "src" / "gpu" / "Falcon_float_decompressor.cu"),
        str(falcon_root / "src" / "gpu" / "Falcon_float_pipeline.cu"),
        "-o",
        str(output),
    ]
    subprocess.run(cmd, check=True)
    return output


def assemble_matrix_width(blocks, target_rows: int, target_width: int = 4096):
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


def export_fp32_activation(path: Path, model: str, rows: int, width: int, device: str, force: bool):
    numel = rows * width
    if path.exists() and path.stat().st_size == numel * 4 and not force:
        return {"path": str(path), "num_elements": numel, "reused": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks, meta = load_model_activation_blocks(model, torch.device(device))
    matrix = assemble_matrix_width(blocks, rows, width)
    fp32 = matrix.to(torch.float32).cpu().contiguous()
    fp32.numpy().tofile(path)
    return {
        "path": str(path),
        "num_elements": numel,
        "reused": False,
        "activation_meta": meta,
    }


def parse_results(stdout: str):
    rows = []
    config = {}
    for line in stdout.splitlines():
        if line.startswith("FALCON_CONFIG"):
            config = {k: v for k, v in RESULT_RE.findall(line)}
        elif line.startswith("FALCON_RESULT"):
            row = {k: v for k, v in RESULT_RE.findall(line)}
            rows.append(
                {
                    "repeat": int(row["repeat"]),
                    "ok": row["ok"] == "1",
                    "compressed_bytes": int(row["compressed_bytes"]),
                    "ratio_vs_fp32": float(row["ratio_fp32"]),
                    "comp_ms": float(row["comp_ms"]),
                    "decomp_ms": float(row["decomp_ms"]),
                    "comp_gbs_fp32": float(row["comp_gbs"]),
                    "decomp_gbs_fp32": float(row["decomp_gbs"]),
                }
            )
    if not rows:
        raise RuntimeError(f"Falcon benchmark produced no parseable results:\n{stdout}")
    return config, rows


def summarize(rows, bf16_raw_bytes: int):
    compressed = [r["compressed_bytes"] for r in rows]
    comp_gbs_fp32 = [r["comp_gbs_fp32"] for r in rows]
    decomp_gbs_fp32 = [r["decomp_gbs_fp32"] for r in rows]
    comp_gbs_bf16_equiv = [x * 0.5 for x in comp_gbs_fp32]
    decomp_gbs_bf16_equiv = [x * 0.5 for x in decomp_gbs_fp32]
    ratios_bf16 = [bf16_raw_bytes / c for c in compressed]
    ratios_fp32 = [(bf16_raw_bytes * 2) / c for c in compressed]
    return {
        "ok": all(r["ok"] for r in rows),
        "compressed_bytes": mean_std(compressed),
        "ratio_vs_fp32_input": mean_std(ratios_fp32),
        "ratio_vs_bf16_payload": mean_std(ratios_bf16),
        "encode_gbs_fp32_input": mean_std(comp_gbs_fp32),
        "decode_gbs_fp32_input": mean_std(decomp_gbs_fp32),
        "encode_gbs_bf16_equiv": mean_std(comp_gbs_bf16_equiv),
        "decode_gbs_bf16_equiv": mean_std(decomp_gbs_bf16_equiv),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--falcon-root", type=Path, default=Path("/data02/home/yilian2/project/Falcon"))
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--falcon-device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--arch", default="sm_90")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--work-dir", type=Path, default=Path("experiments/splitzip_v2/results/falcon"))
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/falcon_baseline_qwen32.json"))
    args = parser.parse_args()

    binary = build_binary(args.falcon_root, args.work_dir / "falcon_float_bench", args.arch, args.rebuild)
    activation_path = args.work_dir / f"{args.model.replace('/', '_')}_{args.rows}x{args.width}_fp32.bin"
    export_meta = export_fp32_activation(
        activation_path,
        args.model,
        args.rows,
        args.width,
        args.device,
        args.force_export,
    )

    cmd = [
        str(binary),
        "--input",
        str(activation_path),
        "--num-elements",
        str(args.rows * args.width),
        "--repeats",
        str(args.repeats),
        "--device",
        str(args.falcon_device),
        "--streams",
        str(args.streams),
    ]
    if args.chunk_size:
        cmd += ["--chunk-size", str(args.chunk_size)]

    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    config, rows = parse_results(proc.stdout)
    bf16_raw_bytes = args.rows * args.width * 2
    result = {
        "method": "falcon_fp32_cast",
        "model": args.model,
        "shape": [args.rows, args.width],
        "falcon_root": str(args.falcon_root),
        "binary": str(binary),
        "command": cmd,
        "config": config,
        "export": export_meta,
        "summary": summarize(rows, bf16_raw_bytes),
        "repeats": rows,
        "note": (
            "Falcon exposes FP32/FP64 floating-point codecs in this checkout, not BF16. "
            "This benchmark casts the same BF16 activation values to FP32 before running Falcon. "
            "BF16-equivalent ratio and throughput are reported against the original BF16 payload size."
        ),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
    }
    write_json(args.output, result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
