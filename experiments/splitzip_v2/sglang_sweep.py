from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from experiments.splitzip_v2.benchmark_utils import write_json
from experiments.splitzip_v2.config import SGLANG_MODEL, requested_transfer_grid


def _quote_cmd(cmd: list[str]) -> str:
    return shlex.join(cmd)


def server_commands(args, mode: str) -> dict[str, object]:
    router_python = args.sglang_root / "sgl-model-gateway" / "bindings" / "python" / "src"
    env = {
        "PYTHONPATH": f"{args.repo_root}:{args.sglang_root / 'python'}:{router_python}",
        "SGLANG_TEST_PD_DISAGG_BACKEND": args.transfer_backend,
        "SGLANG_FORCE_STREAM_INTERVAL": "1",
    }
    if args.ib_device:
        env["SGLANG_TEST_PD_DISAGG_DEVICES"] = args.ib_device
    if mode == "splitzip":
        env["SPLITZIP_SGLANG_ENABLE"] = "1"
        env["SPLITZIP_SGLANG_CODEC"] = "chunklocal_top16"
        env["SPLITZIP_SGLANG_CHUNK_SIZE"] = str(args.chunk_size)

    common = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--trust-remote-code",
        "--tp",
        str(args.tp),
        "--kv-cache-dtype",
        "bfloat16",
    ]
    transfer = ["--disaggregation-transfer-backend", args.transfer_backend]
    if args.ib_device:
        transfer += ["--disaggregation-ib-device", args.ib_device]

    prefill = common + [
        "--disaggregation-mode",
        "prefill",
        "--base-gpu-id",
        str(args.prefill_base_gpu),
        "--port",
        str(args.prefill_port),
        "--disaggregation-bootstrap-port",
        str(args.prefill_bootstrap_port),
    ] + transfer
    decode = common + [
        "--disaggregation-mode",
        "decode",
        "--base-gpu-id",
        str(args.decode_base_gpu),
        "--port",
        str(args.decode_port),
    ] + transfer
    router = [
        "python",
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--mini-lb",
        "--prefill",
        f"http://{args.host}:{args.prefill_port}",
        str(args.prefill_bootstrap_port),
        "--decode",
        f"http://{args.host}:{args.decode_port}",
        "--host",
        args.host,
        "--port",
        str(args.router_port),
    ]
    return {
        "mode": mode,
        "env": env,
        "commands": {
            "prefill": _quote_cmd(prefill),
            "decode": _quote_cmd(decode),
            "router": _quote_cmd(router),
        },
    }


def benchmark_commands(args, mode: str) -> list[dict[str, object]]:
    rows = []
    for point in requested_transfer_grid():
        output = args.output_dir / mode / f"{point['sweep']}_bs{point['batch_size']}_seq{point['seq_len']}.jsonl"
        cmd = [
            "python",
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--base-url",
            f"http://{args.host}:{args.router_port}",
            "--dataset-name",
            "random",
            "--model",
            args.model,
            "--tokenize-prompt",
            "--num-prompts",
            str(max(args.min_prompts, point["batch_size"])),
            "--max-concurrency",
            str(point["batch_size"]),
            "--random-input-len",
            str(point["seq_len"]),
            "--random-output-len",
            str(args.output_len),
            "--random-range-ratio",
            "0.0",
            "--disable-ignore-eos",
            "--warmup-requests",
            str(args.warmup_requests),
            "--output-file",
            str(output),
        ]
        rows.append({**point, "mode": mode, "output_file": str(output), "command": _quote_cmd(cmd)})
    return rows


def build_plan(args) -> dict[str, object]:
    modes = args.modes
    return {
        "model": {"display_name": SGLANG_MODEL.display_name, "hf_name": args.model},
        "note": (
            "This plan launches native and SplitZip PD-disaggregated SGLang runs. "
            "The SplitZip mode expects the SGLang-side hook to read SPLITZIP_SGLANG_ENABLE=1."
        ),
        "servers": [server_commands(args, mode) for mode in modes],
        "benchmarks": [row for mode in modes for row in benchmark_commands(args, mode)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=SGLANG_MODEL.hf_name)
    parser.add_argument("--sglang-root", type=Path, default=Path("/data02/home/yilian2/project/sglang"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=30000)
    parser.add_argument("--prefill-port", type=int, default=30100)
    parser.add_argument("--decode-port", type=int, default=30200)
    parser.add_argument("--prefill-bootstrap-port", type=int, default=8998)
    parser.add_argument("--prefill-base-gpu", type=int, default=1)
    parser.add_argument("--decode-base-gpu", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--transfer-backend", default="mooncake")
    parser.add_argument("--ib-device", default="")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--min-prompts", type=int, default=16)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--modes", nargs="+", choices=["native", "splitzip"], default=["native", "splitzip"])
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/splitzip_v2/results/sglang"))
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/sglang_sweep_plan.json"))
    args = parser.parse_args()
    plan = build_plan(args)
    write_json(args.output, plan)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
