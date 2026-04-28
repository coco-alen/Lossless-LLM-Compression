from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import statistics
import subprocess
import sys
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from experiments.splitzip_v2.benchmark_utils import write_json


DEFAULT_PYTHON = "/data02/home/yilian2/miniconda3/envs/quant/bin/python"
DEFAULT_PLAN = Path("experiments/splitzip_v2/results/sglang_sweep_plan.json")
DEFAULT_RESULT_DIR = Path("experiments/splitzip_v2/results/sglang_realworld")
DEFAULT_SGLANG_ROOT = Path("/data02/home/yilian2/project/sglang")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_jsonl_from_offset(path: Path, offset: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as fin:
        fin.seek(offset)
        data = fin.read().decode("utf-8", errors="replace")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def replace_arg(tokens: list[str], flag: str, value: str) -> None:
    if flag in tokens:
        idx = tokens.index(flag)
        if idx + 1 < len(tokens):
            tokens[idx + 1] = value
            return
    tokens.extend([flag, value])


def add_flag(tokens: list[str], flag: str) -> None:
    if flag not in tokens:
        tokens.append(flag)


def rewrite_python_command(command: str, python: str) -> list[str]:
    tokens = shlex.split(command)
    if tokens and tokens[0] in {"python", "python3", sys.executable}:
        tokens[0] = python
    return tokens


def prepend_pythonpath(env: dict[str, str], entries: list[Path]) -> None:
    existing = env.get("PYTHONPATH", "")
    prefix = ":".join(str(p) for p in entries)
    env["PYTHONPATH"] = prefix + (":" + existing if existing else "")


def plan_env(base: dict[str, str], args: argparse.Namespace, mode: str, metrics_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in base.items()})
    prepend_pythonpath(
        env,
        [
            args.repo_root,
            args.sglang_root / "python",
            args.sglang_root / "sgl-model-gateway" / "bindings" / "python" / "src",
        ],
    )
    env["SGLANG_FORCE_STREAM_INTERVAL"] = "1"
    env["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    # Long-context KV payloads can exceed Mooncake's default per-transfer timeout
    # when the local testbed falls back to TCP instead of RDMA.
    env.setdefault("MC_TRANSFER_TIMEOUT", str(args.mooncake_transfer_timeout_s))
    env.setdefault("MC_SLICE_TIMEOUT", str(args.mooncake_slice_timeout_s))
    # High-concurrency long-context PD runs can spend several minutes before the
    # decode side sends bootstrap KV indices back to prefill.
    env.setdefault("SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", str(args.sglang_bootstrap_timeout_s))
    env["SPLITZIP_SGLANG_METRICS_PATH"] = str(metrics_path.resolve())
    env["SPLITZIP_SGLANG_RATIO"] = str(args.splitzip_ratio)
    env["SPLITZIP_SGLANG_ENCODE_GBS"] = str(args.splitzip_encode_gbs)
    env["SPLITZIP_SGLANG_DECODE_GBS"] = str(args.splitzip_decode_gbs)
    python_bin = str(Path(args.python).resolve().parent)
    env["PATH"] = python_bin + (":" + env["PATH"] if env.get("PATH") else "")
    if mode == "splitzip":
        env["SPLITZIP_SGLANG_ENABLE"] = "1"
        env.setdefault("SPLITZIP_SGLANG_CODEC", "chunklocal_top16")
        env.setdefault("SPLITZIP_SGLANG_CHUNK_SIZE", str(args.chunk_size))
    return env


def maybe_load_splitzip_baseline(args: argparse.Namespace) -> None:
    if not args.baseline_table.exists():
        return
    data = read_json(args.baseline_table)
    for row in data.get("rows", []):
        if row.get("method") == "splitzip_v2":
            args.splitzip_ratio = args.splitzip_ratio or float(row["ratio"])
            args.splitzip_encode_gbs = args.splitzip_encode_gbs or float(row["encode_gbs"])
            args.splitzip_decode_gbs = args.splitzip_decode_gbs or float(row["decode_gbs"])
            return


def qwen32_kv_bytes_per_token(tp: int) -> int:
    # Qwen3-32B: 64 layers, K+V, 8 KV heads, 128 head dim, BF16.
    return (64 * 2 * 8 * 128 * 2) // max(tp, 1)


def feasible(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str, float]:
    total_tokens = int(row["batch_size"]) * int(row["seq_len"])
    kv_gib = total_tokens * qwen32_kv_bytes_per_token(args.tp) / float(1024**3)
    if args.ignore_memory_estimator:
        return True, "", kv_gib
    estimated_gib = kv_gib + args.model_reserve_gib
    if estimated_gib > args.decode_memory_budget_gib:
        return (
            False,
            f"estimated decode memory {estimated_gib:.1f} GiB exceeds budget "
            f"{args.decode_memory_budget_gib:.1f} GiB",
            kv_gib,
        )
    if int(row["seq_len"]) > args.max_seq_len:
        return False, f"seq_len {row['seq_len']} exceeds --max-seq-len {args.max_seq_len}", kv_gib
    return True, "", kv_gib


def select_benchmarks(plan: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requested_modes = set(args.modes)
    requested_sweeps = set(args.sweeps) if args.sweeps else None
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in plan["benchmarks"]:
        if row["mode"] not in requested_modes:
            continue
        if requested_sweeps is not None and row["sweep"] not in requested_sweeps:
            continue
        if args.smoke and not (
            row["sweep"] == "bs1_seq" and row["batch_size"] == 1 and row["seq_len"] == 512
        ):
            continue
        if args.min_batch_size and int(row["batch_size"]) < args.min_batch_size:
            continue
        if args.max_batch_size and int(row["batch_size"]) > args.max_batch_size:
            continue
        if args.min_seq_len_filter and int(row["seq_len"]) < args.min_seq_len_filter:
            continue
        if args.max_seq_len_filter and int(row["seq_len"]) > args.max_seq_len_filter:
            continue
        ok, reason, kv_gib = feasible(row, args)
        enriched = {**row, "estimated_kv_gib": kv_gib}
        if ok:
            rows.append(enriched)
        else:
            skipped.append({**enriched, "skip_reason": reason})
    if args.max_points:
        kept = rows[: args.max_points]
        skipped.extend({**row, "skip_reason": "--max-points limit"} for row in rows[args.max_points :])
        rows = kept
    return rows, skipped


def wait_url(url: str, timeout_s: float, process: subprocess.Popen | None = None) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"process exited before {url} became ready: code={process.returncode}")
        try:
            with urlopen(url, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for {url}; last_error={last_error}")


def fetch_json(url: str) -> dict[str, Any] | None:
    try:
        with urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)


def launch_process(
    name: str,
    tokens: list[str],
    env: dict[str, str],
    log_path: Path,
    cwd: Path,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fout = log_path.open("ab", buffering=0)
    fout.write((" ".join(shlex.quote(tok) for tok in tokens) + "\n").encode("utf-8"))
    return subprocess.Popen(
        tokens,
        cwd=str(cwd),
        env=env,
        stdout=fout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def rewrite_server_command(command: str, args: argparse.Namespace, role: str, max_total_tokens: int) -> list[str]:
    tokens = rewrite_python_command(command, args.python)
    if role in {"prefill", "decode"}:
        replace_arg(tokens, "--context-length", str(args.context_length))
        replace_arg(tokens, "--max-total-tokens", str(max_total_tokens))
        replace_arg(tokens, "--mem-fraction-static", str(args.mem_fraction_static))
        if args.disable_cuda_graph:
            add_flag(tokens, "--disable-cuda-graph")
        if args.enable_memory_saver:
            add_flag(tokens, "--enable-memory-saver")
    return tokens


def rewrite_bench_command(row: dict[str, Any], args: argparse.Namespace, out_path: Path) -> list[str]:
    tokens = rewrite_python_command(row["command"], args.python)
    replace_arg(tokens, "--output-file", str(out_path))
    if args.num_prompts:
        replace_arg(tokens, "--num-prompts", str(args.num_prompts))
    if args.warmup_requests is not None:
        replace_arg(tokens, "--warmup-requests", str(args.warmup_requests))
    return tokens


def aggregate_transfer_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    kv = [ev for ev in events if ev.get("status") == 0 and ev.get("tag") in {"kv", "kv_staged"}]
    all_ok = [ev for ev in events if ev.get("status") == 0]
    failed = [ev for ev in events if int(ev.get("status", 0)) != 0]
    source = kv or all_ok
    if not source:
        return {
            "num_events": len(events),
            "kv_events": 0,
            "failed_events": len(failed),
            "raw_bytes": 0,
            "actual_transfer_ms": None,
            "projected_splitzip_total_ms": None,
        }
    actual_ms = sum(float(ev.get("elapsed_ms", 0.0)) for ev in source)
    projected_ms = sum(float(ev.get("projected_splitzip_total_ms", 0.0)) for ev in source)
    actual_values = [float(ev.get("elapsed_ms", 0.0)) for ev in source]
    projected_values = [float(ev.get("projected_splitzip_total_ms", 0.0)) for ev in source]
    raw_bytes = sum(int(ev.get("total_bytes", 0)) for ev in source)
    encoded_bytes = sum(int(ev.get("projected_encoded_bytes", 0)) for ev in source)
    return {
        "num_events": len(events),
        "kv_events": len(kv),
        "failed_events": len(failed),
        "raw_bytes": raw_bytes,
        "encoded_bytes_projected": encoded_bytes,
        "actual_transfer_ms_sum": actual_ms,
        "actual_transfer_ms_mean": statistics.fmean(actual_values),
        "actual_transfer_ms_median": statistics.median(actual_values),
        "projected_splitzip_total_ms_sum": projected_ms,
        "projected_splitzip_total_ms_mean": statistics.fmean(projected_values),
        "projected_splitzip_total_ms_median": statistics.median(projected_values),
        "projected_transfer_ms_sum": sum(float(ev.get("projected_transfer_ms", 0.0)) for ev in source),
        "projected_encode_ms_sum": sum(float(ev.get("projected_encode_ms", 0.0)) for ev in source),
        "projected_decode_ms_sum": sum(float(ev.get("projected_decode_ms", 0.0)) for ev in source),
        "actual_effective_gbs": (raw_bytes / float(1024**3)) / (actual_ms / 1000.0)
        if actual_ms > 0
        else None,
    }


def last_bench_row(path: Path) -> dict[str, Any] | None:
    rows = read_jsonl(path)
    return rows[-1] if rows else None


def run_mode(
    plan: dict[str, Any],
    mode: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    group_name: str = "",
) -> list[dict[str, Any]]:
    server_plan = next(item for item in plan["servers"] if item["mode"] == mode)
    mode_dir = args.output_dir / mode
    log_dir = args.output_dir / "logs" / mode / group_name if group_name else args.output_dir / "logs" / mode
    metrics_name = f"{mode}_{group_name}.jsonl" if group_name else f"{mode}.jsonl"
    metrics_path = args.output_dir / "metrics" / metrics_name
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("")
    env = plan_env(server_plan.get("env", {}), args, mode, metrics_path)
    requested_total_tokens = max(
        int(row["batch_size"]) * (int(row["seq_len"]) + args.output_len) for row in rows
    )
    max_total_tokens = args.max_total_tokens or (
        max(requested_total_tokens, args.min_total_tokens) + args.token_pool_margin
    )

    commands = server_plan["commands"]
    procs: dict[str, subprocess.Popen] = {}
    results: list[dict[str, Any]] = []
    try:
        for role in ["prefill", "decode"]:
            tokens = rewrite_server_command(commands[role], args, role, max_total_tokens)
            procs[role] = launch_process(role, tokens, env, log_dir / f"{role}.log", args.sglang_root)
        for role in ["prefill", "decode"]:
            port = args.prefill_port if role == "prefill" else args.decode_port
            wait_url(
                f"http://{args.host}:{port}/model_info",
                args.server_timeout_s,
                procs[role],
            )

        router_tokens = rewrite_server_command(commands["router"], args, "router", max_total_tokens)
        procs["router"] = launch_process("router", router_tokens, env, log_dir / "router.log", args.sglang_root)
        wait_url(f"http://{args.host}:{args.router_port}/health", args.server_timeout_s, procs["router"])

        for row in rows:
            out_path = mode_dir / Path(row["output_file"]).name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            offset = metrics_path.stat().st_size if metrics_path.exists() else 0
            bench_cmd = rewrite_bench_command(row, args, out_path)
            started = time.perf_counter()
            proc = subprocess.run(
                bench_cmd,
                cwd=str(args.repo_root),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.bench_timeout_s,
            )
            elapsed_s = time.perf_counter() - started
            bench_log = log_dir / f"bench_{row['sweep']}_bs{row['batch_size']}_seq{row['seq_len']}.log"
            bench_log.write_text(proc.stdout)
            events = read_jsonl_from_offset(metrics_path, offset)
            bench = last_bench_row(out_path)
            result = {
                "mode": mode,
                "sweep": row["sweep"],
                "batch_size": row["batch_size"],
                "seq_len": row["seq_len"],
                "estimated_kv_gib": row["estimated_kv_gib"],
                "returncode": proc.returncode,
                "elapsed_s": elapsed_s,
                "output_file": str(out_path),
                "bench_log": str(bench_log),
                "bench": bench,
                "transfer": aggregate_transfer_events(events),
                "server_info": {
                    "router": fetch_json(f"http://{args.host}:{args.router_port}/server_info"),
                    "prefill": fetch_json(f"http://{args.host}:{args.prefill_port}/server_info"),
                    "decode": fetch_json(f"http://{args.host}:{args.decode_port}/server_info"),
                },
            }
            results.append(result)
            write_json(args.output_dir / "partial_summary.json", {"rows": results})
            if proc.returncode != 0 and args.stop_on_error:
                break
    finally:
        for proc in reversed(list(procs.values())):
            terminate_process(proc)
    return results


def add_paired_speedups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["sweep"], int(row["batch_size"]), int(row["seq_len"]))
        by_key.setdefault(key, {})[row["mode"]] = row
    paired = []
    for (sweep, bs, seq), modes in sorted(by_key.items()):
        native = modes.get("native")
        splitzip = modes.get("splitzip")
        if not native or not splitzip:
            continue
        native_bench = native.get("bench") or {}
        splitzip_bench = splitzip.get("bench") or {}
        native_ttft = native_bench.get("median_ttft_ms")
        splitzip_ttft = splitzip_bench.get("median_ttft_ms")
        native_transfer = native["transfer"].get("actual_transfer_ms_median")
        projected = native["transfer"].get("projected_splitzip_total_ms_median")
        paired.append(
            {
                "sweep": sweep,
                "batch_size": bs,
                "seq_len": seq,
                "native_median_ttft_ms": native_ttft,
                "splitzip_observed_median_ttft_ms": splitzip_ttft,
                "observed_ttft_speedup": native_ttft / splitzip_ttft
                if native_ttft and splitzip_ttft
                else None,
                "native_kv_transfer_ms_median": native_transfer,
                "splitzip_projected_kv_stage_ms_median": projected,
                "projected_kv_stage_speedup": native_transfer / projected
                if native_transfer and projected
                else None,
                "projection_note": (
                    "The SGLang hook records native Mooncake bytes/time and applies measured "
                    "SplitZip ratio plus GPU encode/decode throughput. It is not a compressed "
                    "scratch-buffer data path."
                ),
            }
        )
    return paired


def failed_rows_for_group(
    mode: str, rows: list[dict[str, Any]], exc: Exception, max_total_tokens: int
) -> list[dict[str, Any]]:
    message = str(exc)
    lower = message.lower()
    is_oom = "outofmemory" in lower or "out of memory" in lower or "oom" in lower
    return [
        {
            "mode": mode,
            "sweep": row["sweep"],
            "batch_size": row["batch_size"],
            "seq_len": row["seq_len"],
            "estimated_kv_gib": row["estimated_kv_gib"],
            "returncode": None,
            "max_total_tokens": max_total_tokens,
            "startup_error": message,
            "startup_error_type": type(exc).__name__,
            "startup_oom": is_oom,
            "bench": None,
            "transfer": None,
        }
        for row in rows
    ]


def run_mode_grouped(
    plan: dict[str, Any], mode: str, rows: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        total_tokens = int(row["batch_size"]) * (int(row["seq_len"]) + args.output_len)
        grouped.setdefault(max(total_tokens, args.min_total_tokens) + args.token_pool_margin, []).append(row)

    out: list[dict[str, Any]] = []
    original_max_total_tokens = args.max_total_tokens
    try:
        for total_tokens in sorted(grouped):
            group_rows = grouped[total_tokens]
            args.max_total_tokens = total_tokens
            group_name = f"tokens{total_tokens}"
            try:
                out.extend(run_mode(plan, mode, group_rows, args, group_name=group_name))
            except Exception as exc:
                out.extend(failed_rows_for_group(mode, group_rows, exc, args.max_total_tokens))
                if args.stop_on_error:
                    raise
    finally:
        args.max_total_tokens = original_max_total_tokens
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SGLang Real-World SplitZip Sweep",
        "",
        "This file is generated by `experiments/splitzip_v2/sglang_realworld_runner.py`.",
        "The current SGLang hook records real Mooncake KV transfer events and projects the SplitZip KV stage using measured codec throughput.",
        "",
        "## Rows",
        "",
        "| Mode | Sweep | BS | Seq Len | TTFT median (ms) | Output tok/s | KV transfer median (ms) | SplitZip projected KV stage median (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        bench = row.get("bench") or {}
        transfer = row.get("transfer") or {}
        lines.append(
            "| {mode} | {sweep} | {bs} | {seq} | {ttft} | {tps} | {kv} | {proj} |".format(
                mode=row["mode"],
                sweep=row["sweep"],
                bs=row["batch_size"],
                seq=row["seq_len"],
                ttft=f"{bench.get('median_ttft_ms'):.3f}" if bench.get("median_ttft_ms") is not None else "NA",
                tps=f"{bench.get('output_throughput'):.3f}" if bench.get("output_throughput") is not None else "NA",
                kv=f"{transfer.get('actual_transfer_ms_median'):.3f}"
                if transfer.get("actual_transfer_ms_median") is not None
                else "NA",
                proj=f"{transfer.get('projected_splitzip_total_ms_median'):.3f}"
                if transfer.get("projected_splitzip_total_ms_median") is not None
                else "NA",
            )
        )
    if payload.get("selected") and not payload["rows"]:
        lines.extend(
            [
                "",
                "## Selected Dry-Run Points",
                "",
                "| Mode | Sweep | BS | Seq Len | Estimated KV (GiB) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in payload["selected"]:
            lines.append(
                "| {mode} | {sweep} | {bs} | {seq} | {kv:.3f} |".format(
                    mode=row["mode"],
                    sweep=row["sweep"],
                    bs=row["batch_size"],
                    seq=row["seq_len"],
                    kv=row["estimated_kv_gib"],
                )
            )
    if payload.get("skipped"):
        lines.extend(["", "## Skipped", "", "| Mode | Sweep | BS | Seq Len | Reason |", "|---|---:|---:|---:|---|"])
        for row in payload["skipped"]:
            lines.append(
                f"| {row['mode']} | {row['sweep']} | {row['batch_size']} | {row['seq_len']} | {row['skip_reason']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=30000)
    parser.add_argument("--prefill-port", type=int, default=30100)
    parser.add_argument("--decode-port", type=int, default=30200)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=131072)
    parser.add_argument("--max-seq-len", type=int, default=131072)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--min-total-tokens", type=int, default=2048)
    parser.add_argument("--token-pool-margin", type=int, default=0)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.82)
    parser.add_argument("--decode-memory-budget-gib", type=float, default=125.0)
    parser.add_argument("--model-reserve-gib", type=float, default=70.0)
    parser.add_argument("--ignore-memory-estimator", action="store_true")
    parser.add_argument("--group-by-total-tokens", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--baseline-table", type=Path, default=Path("experiments/splitzip_v2/results/baseline_table_current.json"))
    parser.add_argument("--splitzip-ratio", type=float, default=0.0)
    parser.add_argument("--splitzip-encode-gbs", type=float, default=0.0)
    parser.add_argument("--splitzip-decode-gbs", type=float, default=0.0)
    parser.add_argument("--modes", nargs="+", choices=["native", "splitzip"], default=["native", "splitzip"])
    parser.add_argument("--sweeps", nargs="*", default=[])
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--num-prompts", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=None)
    parser.add_argument("--server-timeout-s", type=float, default=1800.0)
    parser.add_argument("--bench-timeout-s", type=float, default=3600.0)
    parser.add_argument("--mooncake-transfer-timeout-s", type=int, default=900)
    parser.add_argument("--mooncake-slice-timeout-s", type=int, default=900)
    parser.add_argument("--sglang-bootstrap-timeout-s", type=int, default=1200)
    parser.add_argument("--min-batch-size", type=int, default=0)
    parser.add_argument("--max-batch-size", type=int, default=0)
    parser.add_argument("--min-seq-len-filter", type=int, default=0)
    parser.add_argument("--max-seq-len-filter", type=int, default=0)
    parser.add_argument("--disable-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-memory-saver", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.output_dir = args.output_dir.resolve()
    args.repo_root = args.repo_root.resolve()
    args.sglang_root = args.sglang_root.resolve()
    args.baseline_table = args.baseline_table.resolve()

    maybe_load_splitzip_baseline(args)
    if args.splitzip_ratio <= 0 or args.splitzip_encode_gbs <= 0 or args.splitzip_decode_gbs <= 0:
        raise ValueError("SplitZip ratio/throughput values must be positive")

    plan = read_json(args.plan)
    rows, skipped = select_benchmarks(plan, args)
    json_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload: dict[str, Any] = {
        "plan": str(args.plan),
        "model": plan.get("model"),
        "args": json_args,
        "rows": [],
        "skipped": skipped,
        "note": (
            "Current integration is instrumentation/projection. Native SGLang still transfers BF16 KV; "
            "SplitZip projected fields use measured compression ratio and GPU codec throughput."
        ),
    }

    if args.dry_run:
        payload["selected"] = rows
        write_json(args.output_dir / "summary.json", payload)
        write_markdown(args.output_dir / "summary.md", payload)
        return

    for mode in args.modes:
        mode_rows = [row for row in rows if row["mode"] == mode]
        if not mode_rows:
            continue
        if args.group_by_total_tokens:
            payload["rows"].extend(run_mode_grouped(plan, mode, mode_rows, args))
        else:
            payload["rows"].extend(run_mode(plan, mode, mode_rows, args))
        payload["paired_speedups"] = add_paired_speedups(payload["rows"])
        write_json(args.output_dir / "summary.json", payload)
        write_markdown(args.output_dir / "summary.md", payload)

    payload["paired_speedups"] = add_paired_speedups(payload["rows"])
    write_json(args.output_dir / "summary.json", payload)
    write_markdown(args.output_dir / "summary.md", payload)


if __name__ == "__main__":
    main()
