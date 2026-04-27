from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time

import torch

from experiments.splitzip.thesis_experiment_dump import assemble_row_prefix, load_model_activation_blocks
from experiments.splitzip_v2.benchmark_utils import write_json
from experiments.splitzip_v2.codec_cpu import ChunkLocalSplitZipCPU
from experiments.splitzip_v2.config import TRANSFER_MODELS, requested_transfer_grid
from experiments.splitzip_v2.serialization import deserialize_chunklocal, payload_size_breakdown, serialize_chunklocal


def init_engine(name: str, port: int, protocol: str, device: str, metadata: str):
    import mooncake.engine as mte

    engine = mte.TransferEngine()
    ret = engine.initialize(f"localhost:{port}", metadata, protocol, device)
    if ret != 0:
        raise RuntimeError(f"Mooncake TransferEngine init failed for {name}: ret={ret}")
    return engine


def transfer_median_ms(src, dst, target: str, payload: bytes, warmup: int, runs: int):
    size = len(payload)
    src_buf = src.allocate_managed_buffer(size)
    dst_buf = dst.allocate_managed_buffer(size)
    src.write_bytes_to_buffer(src_buf, payload, size)
    for _ in range(warmup):
        src.transfer_sync_write(target, src_buf, dst_buf, size)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        src.transfer_sync_write(target, src_buf, dst_buf, size)
        times.append((time.perf_counter() - t0) * 1000)
    received = dst.read_bytes_from_buffer(dst_buf, size)
    src.free_managed_buffer(src_buf, size)
    dst.free_managed_buffer(dst_buf, size)
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.fmean(times),
        "times_ms": times,
        "ok": received == payload,
    }


def matrix_payload(blocks, rows: int):
    mat = assemble_row_prefix(blocks, rows).contiguous()
    return mat.view(torch.uint8).numpy().tobytes(), tuple(mat.shape)


def matrix_for_rows(blocks, rows: int):
    return assemble_row_prefix(blocks, rows).contiguous()


def splitzip_payload(codec: ChunkLocalSplitZipCPU, matrix: torch.Tensor):
    t0 = time.perf_counter()
    encoded = codec.encode(matrix, profile=True)
    encode_ms = (time.perf_counter() - t0) * 1000
    payload = serialize_chunklocal(encoded)
    return payload, encoded, encode_ms


def decode_splitzip_payload(payload: bytes, expected: torch.Tensor):
    t0 = time.perf_counter()
    encoded = deserialize_chunklocal(payload)
    codec = ChunkLocalSplitZipCPU(chunk_size=encoded.chunk_size)
    decoded = codec.decode(encoded, profile=True)
    decode_ms = (time.perf_counter() - t0) * 1000
    ok = torch.equal(expected.view(torch.int16), decoded.view(torch.int16))
    return ok, decode_ms


def run_model(model_spec, args):
    blocks, meta = load_model_activation_blocks(model_spec.hf_name, torch.device(args.device))
    codec = ChunkLocalSplitZipCPU(chunk_size=args.chunk_size)
    calib_rows = min(args.calibration_rows, sum(int(block.shape[0]) for block in blocks))
    coverage = codec.calibrate(matrix_for_rows(blocks, calib_rows))
    prefill = init_engine("prefill", args.prefill_port, args.protocol, args.mooncake_device, args.metadata)
    decode = init_engine("decode", args.decode_port, args.protocol, args.mooncake_device, args.metadata)
    target = f"localhost:{args.decode_port}"

    rows = []
    for point in requested_transfer_grid():
        total_rows = point["batch_size"] * point["seq_len"]
        matrix = matrix_for_rows(blocks, total_rows)
        payload = matrix.view(torch.uint8).numpy().tobytes()
        shape = tuple(matrix.shape)
        raw = transfer_median_ms(prefill, decode, target, payload, args.warmup, args.runs)
        splitzip = None
        if not args.native_only:
            comp_payload, encoded, encode_ms = splitzip_payload(codec, matrix)
            comp_transfer = transfer_median_ms(prefill, decode, target, comp_payload, args.warmup, args.runs)
            ok, decode_ms = decode_splitzip_payload(comp_payload, matrix)
            splitzip = {
                "codec": "chunklocal_cpu_reference_transport_payload",
                "coverage": coverage,
                "encoded_bytes": len(comp_payload),
                "compressed_bytes_without_wire_header": encoded.compressed_bytes,
                "payload_breakdown": payload_size_breakdown(encoded),
                "ratio_vs_wire_payload": len(payload) / len(comp_payload),
                "ratio_without_wire_header": len(payload) / encoded.compressed_bytes,
                "n_chunks": encoded.n_chunks,
                "n_escapes": encoded.n_escapes,
                "encode_ms_cpu_reference": encode_ms,
                "decode_ms_cpu_reference": decode_ms,
                "transfer": comp_transfer,
                "ok": ok and comp_transfer["ok"],
            }
        rows.append({
            **point,
            "matrix_shape": list(shape),
            "raw_bytes": len(payload),
            "native_mooncake": raw,
            "splitzip_mooncake": splitzip,
        })
        print(
            f"{model_spec.display_name} {point['sweep']} bs={point['batch_size']} "
            f"seq={point['seq_len']} bytes={len(payload)} median={raw['median_ms']:.3f} ms ok={raw['ok']}",
            flush=True,
        )
    return {
        "model": model_spec.__dict__,
        "meta": meta,
        "codec_note": (
            "Mooncake measures real payload transfer. The included compressed payload uses the CPU "
            "reference serializer for portability; GPU codec stage timings should come from gpu_breakdown_bench.py."
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[m.display_name for m in TRANSFER_MODELS])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protocol", default="tcp")
    parser.add_argument("--mooncake-device", default="cpu")
    parser.add_argument("--metadata", default="localhost:2379")
    parser.add_argument("--prefill-port", type=int, default=23456)
    parser.add_argument("--decode-port", type=int, default=23457)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=9)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--calibration-rows", type=int, default=65536)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/mooncake_kv_sweep.json"))
    args = parser.parse_args()

    selected = [m for m in TRANSFER_MODELS if m.display_name in set(args.models) or m.hf_name in set(args.models)]
    if args.dry_run:
        payload = {
            "models": [m.__dict__ for m in selected],
            "grid": requested_transfer_grid(),
            "note": "Dry run only; no model loading or Mooncake transfer executed.",
        }
    else:
        payload = {"results": [run_model(model, args) for model in selected]}
    write_json(args.output, payload)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
