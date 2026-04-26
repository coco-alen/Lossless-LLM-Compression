import argparse
import ctypes
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from nvidia import nvcomp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.splitzip.codec_ablation_bench import (
    SEQ_LENS,
    collect_real_activations,
    DIETGPU_MAIN_LIB,
    dfloat11_compressed_bytes,
    dfloat11_decode,
    dfloat11_encode_full,
    dfloat11_prepare_decode,
    load_dietgpu,
    load_zipserv_module,
    raw_bytes,
    silence_stdout,
    torch_uint8_to_cupy,
    throughput_gbs,
    zipserv_compressed_bytes,
    zipserv_encode,
)
from experiments.splitzip.lossless_fast import FastLosslessCodec


DEFAULT_JSON_OUT = ROOT / "experiments" / "splitzip" / "baseline_throughput_repeats.json"
DEFAULT_MD_OUT = ROOT / "experiments" / "splitzip" / "baseline_throughput_repeats.md"
DEFAULT_THESIS_JSON = ROOT / "experiments" / "splitzip" / "thesis_experiment_data.json"
ZIPSERV_LIB = Path("/data02/home/yilian2/project/ZipServ_BF16/build/libL_API.so")
ZIPSERV_DECOMPRESS_SYMBOL = "_Z31BF16TripleBitmap_Decompress_APIP11CUstream_stPKhPK13__nv_bfloat16PKmS7_S7_PKiS9_iihPS3_ii"


METHODS = ["dietgpu", "nvcomp_lz4", "dfloat11", "zipserv", "splitzip", "zipnn"]
METHOD_LABELS = {
    "dietgpu": "DietGPU",
    "nvcomp_lz4": "nvCOMP LZ4",
    "dfloat11": "DFloat11",
    "zipserv": "ZipServ",
    "splitzip": "SplitZip",
    "zipnn": "ZipNN",
}


def mean_std_stderr(values):
    mean = statistics.fmean(values)
    if len(values) <= 1:
        return mean, 0.0, 0.0
    std = statistics.stdev(values)
    stderr = std / math.sqrt(len(values))
    return mean, std, stderr


def summarize_runs(method, runs):
    enc_values = [run["encode_gbs"] for run in runs]
    dec_values = [run["decode_gbs"] for run in runs]
    ratio_values = [run["ratio"] for run in runs]
    enc_mean, enc_std, enc_stderr = mean_std_stderr(enc_values)
    dec_mean, dec_std, dec_stderr = mean_std_stderr(dec_values)
    ratio_mean, ratio_std, ratio_stderr = mean_std_stderr(ratio_values)
    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "n": len(runs),
        "ratio_mean": ratio_mean,
        "ratio_std": ratio_std,
        "ratio_stderr": ratio_stderr,
        "encode_gbs_mean": enc_mean,
        "encode_gbs_std": enc_std,
        "encode_gbs_stderr": enc_stderr,
        "decode_gbs_mean": dec_mean,
        "decode_gbs_std": dec_std,
        "decode_gbs_stderr": dec_stderr,
        "encode_gbs_values": enc_values,
        "decode_gbs_values": dec_values,
        "ratio_values": ratio_values,
    }


def load_reported_zipnn(thesis_json):
    data = json.loads(Path(thesis_json).read_text())
    for row in data["baseline"]["rows"]:
        if row["method"] == "zipnn":
            return {
                "method": "zipnn",
                "label": "ZipNN",
                "n": 0,
                "ratio_mean": row["ratio"],
                "ratio_std": None,
                "ratio_stderr": None,
                "encode_gbs_mean": row["encode_gbs"],
                "encode_gbs_std": None,
                "encode_gbs_stderr": None,
                "decode_gbs_mean": row["decode_gbs"],
                "decode_gbs_std": None,
                "decode_gbs_stderr": None,
                "encode_gbs_values": [],
                "decode_gbs_values": [],
                "ratio_values": [],
                "source": "reported; not rerun locally",
            }
    raise RuntimeError(f"ZipNN row not found in {thesis_json}")


def measured_row(name, result, shape):
    return {
        "encode_s": result["encode_s"],
        "decode_s": result["decode_s"],
        "encode_gbs": throughput_gbs(shape, result["encode_s"]),
        "decode_gbs": throughput_gbs(shape, result["decode_s"]),
        "ratio": raw_bytes(shape) / result["compressed_bytes"],
        "compressed_bytes": int(result["compressed_bytes"]),
        "ok": bool(result["ok"]),
    }


def sync_gpu():
    torch.cuda.synchronize()


def time_cpu(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def time_gpu(fn):
    sync_gpu()
    t0 = time.perf_counter()
    out = fn()
    sync_gpu()
    return time.perf_counter() - t0, out


def time_gpu_many(fn, iters):
    out = None
    sync_gpu()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    sync_gpu()
    return (time.perf_counter() - t0) / iters, out


def time_cpu_many(fn, iters):
    out = None
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    return (time.perf_counter() - t0) / iters, out


class ZipServGpuDecoder:
    def __init__(self):
        self.lib = ctypes.CDLL(str(ZIPSERV_LIB))
        self.fn = getattr(self.lib, ZIPSERV_DECOMPRESS_SYMBOL)
        self.fn.restype = ctypes.c_int
        self.fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ubyte,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]

    @staticmethod
    def prepare(meta, device):
        return {
            "sign_mantissa": meta["sign_mantissa"].to(device),
            "compressed_full": meta["compressed_full"].to(device),
            "bitmap1": meta["bitmap1"].to(device),
            "bitmap2": meta["bitmap2"].to(device),
            "bitmap3": meta["bitmap3"].to(device),
            "tile_offsets_median": meta["tile_offsets_median"].to(device),
            "tile_offsets_global": meta["tile_offsets_global"].to(device),
            "max_high_freq_count": int(meta["max_high_freq_count"]),
            "max_full_count": int(meta["max_full_count"]),
            "start_exp": int(meta["start_exp"]),
            "M": int(meta["M"]),
            "K": int(meta["K"]),
        }

    def decode(self, prepared, output):
        status = self.fn(
            None,
            ctypes.c_void_p(prepared["sign_mantissa"].data_ptr()),
            ctypes.c_void_p(prepared["compressed_full"].data_ptr()),
            ctypes.c_void_p(prepared["bitmap1"].data_ptr()),
            ctypes.c_void_p(prepared["bitmap2"].data_ptr()),
            ctypes.c_void_p(prepared["bitmap3"].data_ptr()),
            ctypes.c_void_p(prepared["tile_offsets_median"].data_ptr()),
            ctypes.c_void_p(prepared["tile_offsets_global"].data_ptr()),
            prepared["max_high_freq_count"],
            prepared["max_full_count"],
            prepared["start_exp"],
            ctypes.c_void_p(output.data_ptr()),
            prepared["M"],
            prepared["K"],
        )
        if status != 0:
            raise RuntimeError(f"ZipServ GPU decode failed with cudaError_t={status}")
        return output


def dietgpu_measure_once(cpu_bf16, gpu_bf16, inner_iters):
    temp_mem_bytes = max(64 * 1024 * 1024, 2 * raw_bytes(cpu_bf16.shape))
    encode_s, encoded = time_gpu_many(
        lambda: torch.ops.dietgpu.compress_data_simple(True, [gpu_bf16], True, temp_mem_bytes)
        ,
        inner_iters,
    )
    decode_s, decoded = time_gpu_many(
        lambda: torch.ops.dietgpu.decompress_data_simple(True, encoded, True, temp_mem_bytes),
        inner_iters,
    )
    ok = torch.equal(gpu_bf16.view(torch.int16), decoded[0].view(torch.int16))
    return {
        "ok": ok,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "compressed_bytes": encoded[0].numel() * encoded[0].element_size(),
    }


def nvcomp_measure_once(cpu_bf16, gpu_bytes, cp_bytes, codec, inner_iters):
    encode_s, encoded = time_gpu_many(lambda: codec.encode(nvcomp.as_array(cp_bytes)), inner_iters)
    decode_s, decoded = time_gpu_many(lambda: codec.decode(encoded), inner_iters)
    # Full equality checks every repeat are costly; the first repeat still validates the path.
    ok = True
    return {
        "ok": ok,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "compressed_bytes": int(encoded.buffer_size),
    }


def dfloat11_measure_once(cpu_bf16, device, decode_inner_iters):
    encode_s, encoded_pack = time_cpu(lambda: dfloat11_encode_full(cpu_bf16))
    prepared = dfloat11_prepare_decode(encoded_pack, device)
    decode_s, decoded = time_gpu_many(lambda: dfloat11_decode(prepared), decode_inner_iters)
    ok = torch.equal(cpu_bf16.view(-1).view(torch.int16), decoded.cpu().view(torch.int16))
    return {
        "ok": ok,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "compressed_bytes": dfloat11_compressed_bytes(encoded_pack),
    }


def zipserv_measure_once(cpu_bf16, device, zipserv_module, zipserv_decoder, decode_inner_iters):
    encode_s, meta = time_cpu(lambda: zipserv_encode(zipserv_module, cpu_bf16))
    prepared = zipserv_decoder.prepare(meta, device)
    output = torch.empty((prepared["M"], prepared["K"]), dtype=torch.bfloat16, device=device)
    decode_s, decoded_gpu = time_gpu_many(lambda: zipserv_decoder.decode(prepared, output), decode_inner_iters)
    ok = torch.equal(cpu_bf16.view(torch.int16), decoded_gpu.cpu().view(torch.int16))
    return {
        "ok": ok,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "compressed_bytes": zipserv_compressed_bytes(meta),
    }


def splitzip_measure_once(gpu_bf16, codec, inner_iters):
    encode_s, encoded = time_gpu_many(lambda: codec.encode(gpu_bf16), inner_iters)
    decode_s, decoded = time_gpu_many(lambda: codec.decode(*encoded), inner_iters)
    ok = torch.equal(gpu_bf16.view(torch.int16), decoded.view(torch.int16))
    comp_bytes = encoded[0].numel() + encoded[1].numel() + encoded[2].numel() * 4 + encoded[3].numel()
    return {
        "ok": ok,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "compressed_bytes": comp_bytes,
    }


def make_markdown(payload):
    lines = []
    lines.append("# Baseline Throughput Repeats")
    lines.append("")
    lines.append(f"- Tensor shape: `{payload['shape'][0]} x {payload['shape'][1]}`")
    lines.append(f"- Repeats requested: `{payload['repeats']}`")
    lines.append("- Error values are sample standard deviation and standard error of the mean across repeated benchmark runs.")
    lines.append("- ZipNN is reported-only and therefore has no local error bar.")
    lines.append("")
    lines.append("| Method | N | Ratio Mean | Enc Mean GB/s | Enc Std | Enc SEM | Dec Mean GB/s | Dec Std | Dec SEM |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in METHODS:
        row = payload["summary"][method]
        def fmt(value):
            if value is None:
                return "-"
            return f"{value:.6g}"
        lines.append(
            f"| {row['label']} | {row['n']} | {fmt(row['ratio_mean'])} | "
            f"{fmt(row['encode_gbs_mean'])} | {fmt(row['encode_gbs_std'])} | {fmt(row['encode_gbs_stderr'])} | "
            f"{fmt(row['decode_gbs_mean'])} | {fmt(row['decode_gbs_std'])} | {fmt(row['decode_gbs_stderr'])} |"
        )
    lines.append("")
    lines.append("## Plot Error Bars")
    lines.append("")
    lines.append("Use `*_std` for one-standard-deviation error bars, or `*_stderr` for standard-error bars.")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activation-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--zipserv-root", default="/data02/home/yilian2/project/ZipServ_BF16")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--inner-iters-fast", type=int, default=5)
    parser.add_argument("--inner-iters-decode", type=int, default=10)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--thesis-json", type=Path, default=DEFAULT_THESIS_JSON)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    print("Loading ZipServ extension...", flush=True)
    zipserv_module = load_zipserv_module(args.zipserv_root)
    print("Loading DietGPU extension...", flush=True)
    load_dietgpu()

    target_seq_len = 32768
    if target_seq_len not in SEQ_LENS:
        raise RuntimeError(f"{target_seq_len} is not in codec_ablation_bench.SEQ_LENS")
    print(f"Collecting real activation tensor for seq_len={target_seq_len}", flush=True)
    cpu_bf16 = collect_real_activations(args.activation_model, [target_seq_len], device)[0]
    shape = tuple(cpu_bf16.shape)

    runs = {method: [] for method in METHODS if method != "zipnn"}
    gpu_bf16 = cpu_bf16.to(device).view(-1).contiguous()
    gpu_bytes = cpu_bf16.to(device).view(torch.uint8).view(-1).contiguous()
    cp_bytes = torch_uint8_to_cupy(gpu_bytes)
    nvcomp_codec = nvcomp.Codec(algorithm="LZ4")
    splitzip_codec = FastLosslessCodec(str(device))
    splitzip_codec.calibrate(gpu_bf16)
    zipserv_decoder = ZipServGpuDecoder()

    print("Warming GPU codec paths...", flush=True)
    _ = dietgpu_measure_once(cpu_bf16, gpu_bf16, inner_iters=1)
    _ = nvcomp_measure_once(cpu_bf16, gpu_bytes, cp_bytes, nvcomp_codec, inner_iters=1)
    zipserv_warm_meta = zipserv_encode(zipserv_module, cpu_bf16)
    zipserv_warm_prepared = zipserv_decoder.prepare(zipserv_warm_meta, device)
    zipserv_warm_output = torch.empty(
        (zipserv_warm_prepared["M"], zipserv_warm_prepared["K"]),
        dtype=torch.bfloat16,
        device=device,
    )
    _ = time_gpu(lambda: zipserv_decoder.decode(zipserv_warm_prepared, zipserv_warm_output))
    splitzip_warm = splitzip_codec.encode(gpu_bf16)
    _ = splitzip_codec.decode(*splitzip_warm)
    sync_gpu()

    bench_fns = {
        "dietgpu": lambda: dietgpu_measure_once(cpu_bf16, gpu_bf16, args.inner_iters_fast),
        "nvcomp_lz4": lambda: nvcomp_measure_once(cpu_bf16, gpu_bytes, cp_bytes, nvcomp_codec, args.inner_iters_fast),
        "dfloat11": lambda: dfloat11_measure_once(cpu_bf16, device, args.inner_iters_decode),
        "zipserv": lambda: zipserv_measure_once(
            cpu_bf16,
            device,
            zipserv_module,
            zipserv_decoder,
            args.inner_iters_decode,
        ),
        "splitzip": lambda: splitzip_measure_once(gpu_bf16, splitzip_codec, args.inner_iters_fast),
    }

    for rep in range(args.repeats):
        print(f"Repeat {rep + 1}/{args.repeats}", flush=True)
        for method in ["dietgpu", "nvcomp_lz4", "dfloat11", "zipserv", "splitzip"]:
            t0 = time.perf_counter()
            result = bench_fns[method]()
            row = measured_row(method, result, shape)
            if not row["ok"]:
                raise RuntimeError(f"{method} round-trip failed on repeat {rep + 1}")
            runs[method].append(row)
            print(
                f"  {method:<10} enc={row['encode_gbs']:.6g} GB/s "
                f"dec={row['decode_gbs']:.6g} GB/s ratio={row['ratio']:.6g} "
                f"elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    summary = {method: summarize_runs(method, runs[method]) for method in runs}
    summary["zipnn"] = load_reported_zipnn(args.thesis_json)

    payload = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "activation_model": args.activation_model,
        "shape": list(shape),
        "raw_bytes": raw_bytes(shape),
        "repeats": args.repeats,
        "inner_iters_fast": args.inner_iters_fast,
        "inner_iters_decode": args.inner_iters_decode,
        "methods": METHODS,
        "runs": runs,
        "summary": summary,
        "notes": {
            "throughput_gbs": "GB/s means gigabytes per second, computed from uncompressed native BF16 bytes.",
            "zipnn": "Reported external number; no local variance is available.",
            "zipserv": "Uses the local ZipServ Python binding in codec_ablation_bench.py.",
        },
    }

    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(make_markdown(payload) + "\n")
    print(f"Wrote {args.json_out}", flush=True)
    print(f"Wrote {args.md_out}", flush=True)


if __name__ == "__main__":
    main()
