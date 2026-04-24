import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
import ctypes
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load as load_cpp_extension
from transformers import AutoModelForCausalLM, AutoTokenizer
from nvidia import nvcomp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dfloat11.dfloat11 import _decode as dfloat11_decode_kernel
from dfloat11.dfloat11 import bytes_per_thread as dfloat11_bytes_per_thread
from dfloat11.dfloat11 import threads_per_block as dfloat11_threads_per_block
from dfloat11.dfloat11_utils import encode_weights, get_32bit_codec, get_codec, get_luts
from experiments.splitzip.lossless_fast import FastLosslessCodec

import cupy as cp


SEQ_LENS = [1024, 4096, 8192, 16384, 32768]
TARGET_HIDDEN_DIM = 4096
BASE_BLOCK_DIM = 512
NUM_FEATURE_CHUNKS = TARGET_HIDDEN_DIM // BASE_BLOCK_DIM
DIETGPU_BUILD_ROOT = Path("/data02/home/yilian2/project/dietgpu/build")
DIETGPU_LIB_DIR = DIETGPU_BUILD_ROOT / "lib"
DIETGPU_MAIN_LIB = DIETGPU_LIB_DIR / "libdietgpu.so"
TORCH_LIB_DIR = Path(torch.__file__).resolve().parent / "lib"
TORCH_PRELOAD_LIBS = [
    TORCH_LIB_DIR / "libtorch.so",
    TORCH_LIB_DIR / "libtorch_cpu.so",
    TORCH_LIB_DIR / "libtorch_python.so",
    TORCH_LIB_DIR / "libc10.so",
    TORCH_LIB_DIR / "libc10_cuda.so",
]


def raw_bytes(shape):
    return math.prod(shape) * 2


def pick_iters(method, shape):
    size_mb = raw_bytes(shape) / (1024 * 1024)
    if method in ("dfloat11_encode", "zipserv_encode"):
        if size_mb <= 1:
            return 2, 12
        if size_mb <= 8:
            return 2, 8
        if size_mb <= 32:
            return 1, 5
        return 1, 3
    if method in ("dietgpu_encode", "nvcomp_encode"):
        if size_mb <= 8:
            return 3, 12
        if size_mb <= 32:
            return 2, 8
        return 1, 5
    if method in ("dietgpu_decode", "nvcomp_decode"):
        if size_mb <= 8:
            return 3, 16
        if size_mb <= 32:
            return 2, 10
        return 1, 6
    if method == "zipserv_decode":
        if size_mb <= 1:
            return 3, 20
        if size_mb <= 8:
            return 2, 12
        if size_mb <= 32:
            return 1, 8
        return 1, 5
    if size_mb <= 1:
        return 5, 40
    if size_mb <= 8:
        return 3, 20
    if size_mb <= 32:
        return 2, 10
    return 1, 5


def bench(fn, warmup, iters, cuda=False):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def build_token_batch(tokenizer, target_len, salt=""):
    base = (
        "Disaggregated prefill decode serving moves KV cache tensors between workers. "
        "Lossless compression can reduce transfer time if the codec preserves exact BF16 values "
        "while keeping encode and decode throughput high. "
        + salt
    )
    ids = tokenizer(base, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError("Tokenizer returned no ids for the base prompt")
    repeated = (ids * ((target_len + len(ids) - 1) // len(ids)))[:target_len]
    return torch.tensor([repeated], dtype=torch.long)


def kv_to_rows(key, value):
    key_rows = key.detach().squeeze(0).permute(1, 0, 2).contiguous().flatten(1)
    value_rows = value.detach().squeeze(0).permute(1, 0, 2).contiguous().flatten(1)
    rows = torch.cat([key_rows, value_rows], dim=1).to(torch.bfloat16)
    if rows.shape[0] % 64 != 0 or rows.shape[1] % 64 != 0:
        raise RuntimeError(f"KV row matrix shape {tuple(rows.shape)} is not 64-aligned for ZipServ")
    if rows.shape[1] != BASE_BLOCK_DIM:
        raise RuntimeError(f"Expected {BASE_BLOCK_DIM}-wide KV rows, got {rows.shape[1]}")
    return rows


def assemble_row_prefix(blocks, target_rows, start_idx=0):
    if not blocks:
        raise RuntimeError("No activation blocks were collected")
    pieces = []
    remaining = target_rows
    idx = start_idx
    n_blocks = len(blocks)
    while remaining > 0:
        block = blocks[idx % n_blocks]
        take = min(remaining, block.shape[0])
        pieces.append(block[:take])
        remaining -= take
        idx += 1
    return torch.cat(pieces, dim=0).contiguous()


def assemble_matrix_4096(blocks, target_rows):
    feature_chunks = []
    for chunk_idx in range(NUM_FEATURE_CHUNKS):
        feature_chunks.append(assemble_row_prefix(blocks, target_rows, start_idx=chunk_idx))
    matrix = torch.cat(feature_chunks, dim=1).contiguous()
    if matrix.shape != (target_rows, TARGET_HIDDEN_DIM):
        raise RuntimeError(f"Expected {(target_rows, TARGET_HIDDEN_DIM)}, got {tuple(matrix.shape)}")
    return matrix


def collect_real_activations(model_name, seq_lens, device):
    print(f"Loading activation model: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    target_max_rows = max(seq_lens)
    blocks = []
    prompt_idx = 0
    with torch.no_grad():
        while len(blocks) < NUM_FEATURE_CHUNKS or sum(block.shape[0] for block in blocks) < target_max_rows:
            print(f"Collecting KV blocks from 1024-token prompt {prompt_idx}", flush=True)
            input_ids = build_token_batch(tokenizer, 1024, salt=f" prompt-{prompt_idx}").to(device)
            attention_mask = torch.ones_like(input_ids)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
            for key, value in outputs.past_key_values:
                blocks.append(kv_to_rows(key, value).cpu())
                if len(blocks) >= NUM_FEATURE_CHUNKS and sum(block.shape[0] for block in blocks) >= target_max_rows:
                    break
            prompt_idx += 1

    del model
    torch.cuda.empty_cache()
    return [assemble_matrix_4096(blocks, seq_len) for seq_len in seq_lens]


def preload_torch_global_symbols():
    for lib in TORCH_PRELOAD_LIBS:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def load_dietgpu():
    if not DIETGPU_MAIN_LIB.exists():
        raise RuntimeError(f"DietGPU library not found at {DIETGPU_MAIN_LIB}")
    os.environ["LD_LIBRARY_PATH"] = f"{DIETGPU_LIB_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    preload_torch_global_symbols()
    ctypes.CDLL(str(DIETGPU_MAIN_LIB), mode=ctypes.RTLD_GLOBAL)


def torch_uint8_to_cupy(tensor):
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


@contextlib.contextmanager
def silence_stdout():
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def dfloat11_encode_full(cpu_bf16):
    weights = [cpu_bf16.flatten()]
    sink_out = io.StringIO()
    sink_err = io.StringIO()
    with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
        _, counter = get_codec(cpu_bf16)
        codec, _, table = get_32bit_codec(counter)
        luts = get_luts(table)
        encoded, other_8bits, output_positions, gaps, _ = encode_weights(
            weights, codec, dfloat11_bytes_per_thread, dfloat11_threads_per_block[0]
        )
    return {
        "luts": luts,
        "encoded": encoded,
        "sign_mantissa": other_8bits,
        "output_positions": output_positions,
        "gaps": gaps,
    }


def dfloat11_prepare_decode(encoded_pack, device):
    luts = encoded_pack["luts"].to(device)
    encoded = encoded_pack["encoded"].to(device)
    sign_mantissa = encoded_pack["sign_mantissa"].to(device)
    output_positions = encoded_pack["output_positions"].to(device)
    gaps = encoded_pack["gaps"].to(device)
    n_luts = luts.shape[0]
    n_bytes = encoded.numel()
    n_elements = sign_mantissa.numel()
    blocks_per_grid = (
        int(math.ceil(n_bytes / (dfloat11_threads_per_block[0] * dfloat11_bytes_per_thread))),
    )
    output_positions_np = encoded_pack["output_positions"].view(torch.uint32).numpy()
    shared_mem_size = (
        dfloat11_threads_per_block[0] * 4
        + 4
        + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
    )
    return {
        "luts": luts,
        "encoded": encoded,
        "sign_mantissa": sign_mantissa,
        "output_positions": output_positions,
        "gaps": gaps,
        "n_luts": n_luts,
        "n_bytes": n_bytes,
        "n_elements": n_elements,
        "blocks_per_grid": blocks_per_grid,
        "shared_mem_size": shared_mem_size,
        "device": device,
    }


def dfloat11_decode(prepared):
    out = torch.empty(prepared["n_elements"], dtype=torch.bfloat16, device=prepared["device"])
    with cp.cuda.Device(prepared["device"].index):
        dfloat11_decode_kernel(
            grid=prepared["blocks_per_grid"],
            block=dfloat11_threads_per_block,
            shared_mem=prepared["shared_mem_size"],
            args=[
                prepared["luts"].data_ptr(),
                prepared["encoded"].data_ptr(),
                prepared["sign_mantissa"].data_ptr(),
                prepared["output_positions"].data_ptr(),
                prepared["gaps"].data_ptr(),
                out.data_ptr(),
                prepared["n_luts"],
                prepared["n_bytes"],
                prepared["n_elements"],
            ],
        )
    return out


def dfloat11_compressed_bytes(encoded_pack):
    return (
        encoded_pack["encoded"].numel()
        + encoded_pack["sign_mantissa"].numel()
        + encoded_pack["output_positions"].numel() * 4
        + encoded_pack["gaps"].numel()
    )


def load_zipserv_module(zipserv_root):
    zipserv_root = Path(zipserv_root).resolve()
    build_dir = zipserv_root / "build"
    source = zipserv_root / "bindings" / "zipserv_bindings.cpp"
    os.environ["LD_LIBRARY_PATH"] = f"{build_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    return load_cpp_extension(
        name="zipserv_bf16",
        sources=[str(source)],
        extra_include_paths=[
            str(build_dir),
            str(zipserv_root / "kernel_benchmark"),
            "/usr/local/cuda/include",
        ],
        extra_cflags=["-O3"],
        extra_ldflags=[
            f"-L{build_dir}",
            "-lL_API",
            f"-Wl,-rpath,{build_dir}",
        ],
        verbose=False,
    )


def zipserv_encode(module, cpu_bf16_2d):
    with silence_stdout():
        return module.compress_bf16_cpu(cpu_bf16_2d)


def zipserv_decode(module, meta):
    with silence_stdout():
        return module.decompress_bf16_gpu(
            meta["sign_mantissa"],
            meta["compressed_full"],
            meta["bitmap1"],
            meta["bitmap2"],
            meta["bitmap3"],
            meta["tile_offsets_median"],
            meta["tile_offsets_global"],
            int(meta["max_high_freq_count"]),
            int(meta["max_full_count"]),
            int(meta["start_exp"]),
            int(meta["M"]),
            int(meta["K"]),
            "cuda",
        )


def zipserv_compressed_bytes(meta):
    return (
        meta["sign_mantissa"].numel()
        + meta["compressed_full"].numel() * 2
        + meta["bitmap1"].numel() * 8
        + meta["bitmap2"].numel() * 8
        + meta["bitmap3"].numel() * 8
        + meta["tile_offsets_median"].numel() * 4
        + meta["tile_offsets_global"].numel() * 4
    )


def splitzip_bench(cpu_bf16, device):
    gpu_bf16 = cpu_bf16.to(device).view(-1).contiguous()
    codec = FastLosslessCodec(device)
    calibrate_s = bench(lambda: codec.calibrate(gpu_bf16), warmup=0, iters=1, cuda=True)
    codec.calibrate(gpu_bf16)

    encode_warmup, encode_iters = pick_iters("splitzip_encode", cpu_bf16.shape)
    decode_warmup, decode_iters = pick_iters("splitzip_decode", cpu_bf16.shape)

    encoded = codec.encode(gpu_bf16)
    decoded = codec.decode(*encoded)
    ok = torch.equal(gpu_bf16.view(torch.int16), decoded.view(torch.int16))
    encode_s = bench(lambda: codec.encode(gpu_bf16), encode_warmup, encode_iters, cuda=True)
    encoded = codec.encode(gpu_bf16)
    decode_s = bench(lambda: codec.decode(*encoded), decode_warmup, decode_iters, cuda=True)
    comp_bytes = encoded[0].numel() + encoded[1].numel() + encoded[2].numel() * 4 + encoded[3].numel()
    return {
        "ok": ok,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "compressed_bytes": comp_bytes,
        "escapes": int(encoded[5]),
        "calibrate_s": calibrate_s,
    }


def dietgpu_bench(cpu_bf16, device):
    gpu_bf16 = cpu_bf16.to(device).view(-1).contiguous()
    encode_warmup, encode_iters = pick_iters("dietgpu_encode", cpu_bf16.shape)
    decode_warmup, decode_iters = pick_iters("dietgpu_decode", cpu_bf16.shape)
    temp_mem_bytes = max(64 * 1024 * 1024, 2 * raw_bytes(cpu_bf16.shape))

    encoded = torch.ops.dietgpu.compress_data_simple(True, [gpu_bf16], True, temp_mem_bytes)
    decoded = torch.ops.dietgpu.decompress_data_simple(True, encoded, True, temp_mem_bytes)
    ok = torch.equal(gpu_bf16.view(torch.int16), decoded[0].view(torch.int16))

    encode_s = bench(
        lambda: torch.ops.dietgpu.compress_data_simple(True, [gpu_bf16], True, temp_mem_bytes),
        encode_warmup,
        encode_iters,
        cuda=True,
    )
    encoded = torch.ops.dietgpu.compress_data_simple(True, [gpu_bf16], True, temp_mem_bytes)
    decode_s = bench(
        lambda: torch.ops.dietgpu.decompress_data_simple(True, encoded, True, temp_mem_bytes),
        decode_warmup,
        decode_iters,
        cuda=True,
    )
    return {
        "ok": ok,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "compressed_bytes": encoded[0].numel() * encoded[0].element_size(),
    }


def nvcomp_bench(cpu_bf16, device):
    gpu_bytes = cpu_bf16.to(device).view(torch.uint8).view(-1).contiguous()
    cp_bytes = torch_uint8_to_cupy(gpu_bytes)
    codec = nvcomp.Codec(algorithm="LZ4")
    encode_warmup, encode_iters = pick_iters("nvcomp_encode", cpu_bf16.shape)
    decode_warmup, decode_iters = pick_iters("nvcomp_decode", cpu_bf16.shape)

    encoded = codec.encode(nvcomp.as_array(cp_bytes))
    decoded = codec.decode(encoded)
    ok = bool(cp.array_equal(cp.asarray(decoded).view(cp.uint8), cp_bytes.view(cp.uint8)))

    encode_s = bench(
        lambda: codec.encode(nvcomp.as_array(cp_bytes)),
        encode_warmup,
        encode_iters,
        cuda=True,
    )
    encoded = codec.encode(nvcomp.as_array(cp_bytes))
    decode_s = bench(
        lambda: codec.decode(encoded),
        decode_warmup,
        decode_iters,
        cuda=True,
    )
    return {
        "ok": ok,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "compressed_bytes": int(encoded.buffer_size),
    }


def dfloat11_bench(cpu_bf16, device):
    encode_warmup, encode_iters = pick_iters("dfloat11_encode", cpu_bf16.shape)
    decode_warmup, decode_iters = pick_iters("dfloat11_decode", cpu_bf16.shape)

    encoded_pack = dfloat11_encode_full(cpu_bf16)
    prepared = dfloat11_prepare_decode(encoded_pack, device)
    decoded = dfloat11_decode(prepared)
    ok = torch.equal(cpu_bf16.view(-1).view(torch.int16), decoded.cpu().view(torch.int16))

    encode_s = bench(
        lambda: dfloat11_encode_full(cpu_bf16),
        encode_warmup,
        encode_iters,
        cuda=False,
    )
    encoded_pack = dfloat11_encode_full(cpu_bf16)
    prepared = dfloat11_prepare_decode(encoded_pack, device)
    decode_s = bench(
        lambda: dfloat11_decode(prepared),
        decode_warmup,
        decode_iters,
        cuda=True,
    )
    return {
        "ok": ok,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "compressed_bytes": dfloat11_compressed_bytes(encoded_pack),
    }


def zipserv_bench(cpu_bf16, device, zipserv_module):
    encode_warmup, encode_iters = pick_iters("zipserv_encode", cpu_bf16.shape)
    decode_warmup, decode_iters = pick_iters("zipserv_decode", cpu_bf16.shape)

    meta = zipserv_encode(zipserv_module, cpu_bf16)
    decoded = zipserv_decode(zipserv_module, meta)
    ok = torch.equal(cpu_bf16.view(torch.int16), decoded.view(torch.int16))

    encode_s = bench(
        lambda: zipserv_encode(zipserv_module, cpu_bf16),
        encode_warmup,
        encode_iters,
        cuda=False,
    )
    meta = zipserv_encode(zipserv_module, cpu_bf16)
    decode_s = bench(
        lambda: zipserv_decode(zipserv_module, meta),
        decode_warmup,
        decode_iters,
        cuda=True,
    )
    return {
        "ok": ok,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "compressed_bytes": zipserv_compressed_bytes(meta),
    }


def throughput_gbs(shape, seconds):
    return raw_bytes(shape) / seconds / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--zipserv-root", default="/data02/home/yilian2/project/ZipServ_BF16")
    parser.add_argument("--max-shapes", type=int, default=None)
    parser.add_argument("--activation-model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--output",
        default="experiments/splitzip/codec_ablation_results.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print("Loading ZipServ extension...", flush=True)
    zipserv_module = load_zipserv_module(args.zipserv_root)
    print("ZipServ extension ready.", flush=True)
    print("Loading DietGPU extension...", flush=True)
    load_dietgpu()
    print("DietGPU extension ready.", flush=True)

    seq_lens = SEQ_LENS if args.max_shapes is None else SEQ_LENS[:args.max_shapes]
    activation_tensors = collect_real_activations(args.activation_model, seq_lens, device)

    results = {
        "device": str(device),
        "activation_model": args.activation_model,
        "shapes": [],
        "methods": ["dfloat11", "zipserv", "dietgpu", "nvcomp_lz4", "splitzip"],
        "notes": {
            "splitzip": "Encode/decode throughput excludes offline calibration. calibration_s is reported separately.",
            "dfloat11": "Encode throughput includes codec construction and Huffman emission for the tensor.",
            "zipserv": "Decode path uses the provided Python binding, which returns a CPU tensor after GPU decompression.",
            "dietgpu": "DietGPU is loaded from the local build tree and benchmarks its BF16 float codec via torch.ops.",
            "nvcomp_lz4": "nvCOMP benchmarks the generic GPU LZ4 codec on raw BF16 byte streams resident on GPU.",
            "activations": "All tensors are assembled as SEQ_LEN x 4096 BF16 matrices by concatenating eight real 512-wide KV activation row streams collected from 1024-token forwards of the activation_model.",
        },
    }

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"{'Shape':<12} {'Method':<10} {'Enc GB/s':>10} {'Dec GB/s':>10} {'Ratio':>8} {'OK':>5}")
    print("-" * 64)

    for seq_len, cpu_bf16 in zip(seq_lens, activation_tensors):
        shape = tuple(cpu_bf16.shape)
        shape_result = {
            "shape": list(shape),
            "seq_len": seq_len,
            "elements": math.prod(shape),
            "raw_bytes": raw_bytes(shape),
            "results": {},
        }

        dfloat11_result = dfloat11_bench(cpu_bf16, device)
        zipserv_result = zipserv_bench(cpu_bf16, device, zipserv_module)
        dietgpu_result = dietgpu_bench(cpu_bf16, device)
        nvcomp_result = nvcomp_bench(cpu_bf16, device)
        splitzip_result = splitzip_bench(cpu_bf16, device)

        for name, result in (
            ("dfloat11", dfloat11_result),
            ("zipserv", zipserv_result),
            ("dietgpu", dietgpu_result),
            ("nvcomp_lz4", nvcomp_result),
            ("splitzip", splitzip_result),
        ):
            result["encode_gbs"] = throughput_gbs(shape, result["encode_s"])
            result["decode_gbs"] = throughput_gbs(shape, result["decode_s"])
            result["ratio"] = raw_bytes(shape) / result["compressed_bytes"]
            shape_result["results"][name] = result
            print(
                f"{shape[0]}x{shape[1]:<6} {name:<10} "
                f"{result['encode_gbs']:>10.1f} {result['decode_gbs']:>10.1f} "
                f"{result['ratio']:>8.3f} {str(result['ok']):>5}"
            )

        results["shapes"].append(shape_result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
