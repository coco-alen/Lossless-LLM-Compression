from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import torch

from experiments.splitzip.thesis_experiment_dump import assemble_row_prefix, load_model_activation_blocks
from experiments.splitzip_v2.benchmark_utils import mean_std, write_json


def bench_cuda(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def raw_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def torch_uint8_to_cupy(tensor):
    import cupy as cp

    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def nvcomp_algorithm_bench(cpu_bf16: torch.Tensor, algorithm: str, device: str, repeats: int):
    from nvidia import nvcomp
    import cupy as cp

    gpu_bytes = cpu_bf16.to(device=device, dtype=torch.bfloat16).view(torch.uint8).view(-1).contiguous()
    cp_bytes = torch_uint8_to_cupy(gpu_bytes)
    codec = nvcomp.Codec(algorithm=algorithm)

    encoded = codec.encode(nvcomp.as_array(cp_bytes))
    decoded = codec.decode(encoded)
    ok = bool(cp.array_equal(cp.asarray(decoded).view(cp.uint8), cp_bytes.view(cp.uint8)))

    encode_s = []
    decode_s = []
    for _ in range(repeats):
        encode_s.append(bench_cuda(lambda: codec.encode(nvcomp.as_array(cp_bytes)), 2, 5))
        encoded = codec.encode(nvcomp.as_array(cp_bytes))
        decode_s.append(bench_cuda(lambda: codec.decode(encoded), 2, 8))
    rb = raw_bytes(cpu_bf16)
    return {
        "method": f"nvcomp_{algorithm.lower()}",
        "algorithm": algorithm,
        "ok": ok,
        "raw_bytes": rb,
        "compressed_bytes": int(encoded.buffer_size),
        "ratio": rb / int(encoded.buffer_size),
        "encode_s": mean_std(encode_s),
        "decode_s": mean_std(decode_s),
        "encode_gbs": mean_std([rb / t / 1e9 for t in encode_s]),
        "decode_gbs": mean_std([rb / t / 1e9 for t in decode_s]),
    }


def zipserv_bench(cpu_bf16: torch.Tensor, zipserv_root: Path, device: str, repeats: int):
    sys.path.insert(0, str(zipserv_root / "python"))
    import bf16_zip

    if cpu_bf16.ndim != 2 or cpu_bf16.shape[0] % 64 or cpu_bf16.shape[1] % 64:
        raise ValueError(f"ZipServ expects a 2D 64-aligned matrix, got {tuple(cpu_bf16.shape)}")
    meta = bf16_zip.compress_tensor_bf16(cpu_bf16.contiguous())
    if hasattr(importlib.import_module("zipserv_bf16"), "decompress_bf16_gpu_resident"):
        import zipserv_bf16

        gpu_meta = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in meta.items()
        }

        def dec():
            return zipserv_bf16.decompress_bf16_gpu_resident(
                gpu_meta["sign_mantissa"],
                gpu_meta["compressed_full"],
                gpu_meta["bitmap1"],
                gpu_meta["bitmap2"],
                gpu_meta["bitmap3"],
                gpu_meta["tile_offsets_median"],
                gpu_meta["tile_offsets_global"],
                gpu_meta["max_high_freq_count"],
                gpu_meta["max_full_count"],
                gpu_meta["start_exp"],
                gpu_meta["M"],
                gpu_meta["K"],
            )

        decoded = dec()
        ok = torch.equal(cpu_bf16.to(device).view(torch.int16), decoded.view(torch.int16))
    else:
        gpu_meta = meta

        def dec():
            return bf16_zip.decompress_tensor_bf16(gpu_meta, device=device)

        decoded = dec()
        ok = torch.equal(cpu_bf16.view(torch.int16), decoded.view(torch.int16))

    decode_s = [bench_cuda(dec, 2, 8) for _ in range(repeats)]
    compressed_bytes = sum(
        int(value.numel() * value.element_size())
        for value in meta.values()
        if torch.is_tensor(value)
    )
    rb = raw_bytes(cpu_bf16)
    return {
        "method": "zipserv_tca_tbe",
        "ok": bool(ok),
        "raw_bytes": rb,
        "compressed_bytes": compressed_bytes,
        "ratio": rb / compressed_bytes,
        "encode_s": None,
        "decode_s": mean_std(decode_s),
        "encode_gbs": None,
        "decode_gbs": mean_std([rb / t / 1e9 for t in decode_s]),
        "note": "ZipServ compression is CPU-side in the public wrapper; decode uses GPU-resident inputs if the local binding exposes decompress_bf16_gpu_resident.",
    }


def falcon_command(shape: tuple[int, int], falcon_root: Path | None):
    if falcon_root is None:
        return {
            "method": "falcon",
            "status": "external_required",
            "note": "Pass --falcon-root when a local Falcon checkout is available.",
        }
    return {
        "method": "falcon",
        "status": "command_template",
        "command": f"cd {falcon_root} && python bench.py --dtype bf16 --shape {shape[0]} {shape[1]} --lossless",
    }


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3-8B")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[1024, 4096, 16384, 65536])
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=["nvcomp_cascaded", "nvcomp_bitcomp", "zipserv", "falcon"])
    parser.add_argument("--zipserv-root", type=Path, default=Path("/data02/home/yilian2/project/ZipServ_BF16"))
    parser.add_argument("--falcon-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/additional_baselines.json"))
    args = parser.parse_args()

    blocks, meta = load_model_activation_blocks(args.model, torch.device(args.device))
    matrices = [assemble_matrix_width(blocks, seq_len, args.hidden_dim) for seq_len in args.seq_lens]
    rows = []
    for matrix in matrices:
        matrix = matrix.contiguous()
        shape = tuple(int(x) for x in matrix.shape)
        methods = []
        if "nvcomp_cascaded" in args.methods:
            methods.append(nvcomp_algorithm_bench(matrix, "Cascaded", args.device, args.repeats))
        if "nvcomp_bitcomp" in args.methods:
            methods.append(nvcomp_algorithm_bench(matrix, "Bitcomp", args.device, args.repeats))
        if "zipserv" in args.methods:
            methods.append(zipserv_bench(matrix, args.zipserv_root, args.device, args.repeats))
        if "falcon" in args.methods:
            methods.append(falcon_command(shape, args.falcon_root))
        rows.append({"shape": shape, "methods": methods})
    write_json(args.output, {"model": args.model, "activation_meta": meta, "rows": rows})
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
