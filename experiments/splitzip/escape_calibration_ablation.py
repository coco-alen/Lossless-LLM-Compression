import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.splitzip.lossless_fast import (
    FastLosslessCodec,
    _enc_4bit,
    _dec_4bit,
)


ACTIVATION_MODEL = "Qwen/Qwen2.5-1.5B"
SEQ_LEN = 32768
TARGET_HIDDEN_DIM = 4096
BASE_BLOCK_DIM = 512
NUM_FEATURE_CHUNKS = TARGET_HIDDEN_DIM // BASE_BLOCK_DIM
DEFAULT_JSON_OUT = ROOT / "experiments" / "splitzip" / "escape_calibration_ablation.json"
DEFAULT_MD_OUT = ROOT / "experiments" / "splitzip" / "escape_calibration_ablation.md"


@triton.jit
def _count_sentinel_escapes(pk, counts, n_pairs, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    packed = tl.load(pk + offs, mask=mask, other=0)
    hi_esc = mask & (((packed >> 4) & 0x0F) == 15)
    lo_pos = offs * 2 + 1
    lo_mask = mask & (lo_pos < n)
    lo_esc = lo_mask & ((packed & 0x0F) == 15)
    total = tl.sum(hi_esc.to(tl.int32), axis=0) + tl.sum(lo_esc.to(tl.int32), axis=0)
    tl.store(counts + pid, total)


@triton.jit
def _write_sentinel_values(inp, pk, starts, esc_val, n_pairs, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    packed = tl.load(pk + offs, mask=mask, other=0)

    hi_pos = offs * 2
    lo_pos = hi_pos + 1

    hi_raw = tl.load(inp + hi_pos, mask=mask, other=0).to(tl.int32)
    lo_mask = mask & (lo_pos < n)
    lo_raw = tl.load(inp + lo_pos, mask=lo_mask, other=0).to(tl.int32)
    hi_exp = ((hi_raw >> 7) & 0xFF).to(tl.uint8)
    lo_exp = ((lo_raw >> 7) & 0xFF).to(tl.uint8)

    hi_esc = mask & (((packed >> 4) & 0x0F) == 15)
    lo_esc = lo_mask & ((packed & 0x0F) == 15)

    hi_i = hi_esc.to(tl.int32)
    lo_i = lo_esc.to(tl.int32)
    n_hi = tl.sum(hi_i, axis=0)
    hi_rank = tl.cumsum(hi_i, axis=0) - 1
    lo_rank = n_hi + tl.cumsum(lo_i, axis=0) - 1
    base = tl.load(starts + pid)

    tl.store(esc_val + base + hi_rank, hi_exp, mask=hi_esc)
    tl.store(esc_val + base + lo_rank, lo_exp, mask=lo_esc)


@triton.jit
def _fix_sentinel_escapes(pk, starts, esc_val, sm, out, n_pairs, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    packed = tl.load(pk + offs, mask=mask, other=0)

    hi_pos = offs * 2
    lo_pos = hi_pos + 1
    lo_mask = mask & (lo_pos < n)

    hi_esc = mask & (((packed >> 4) & 0x0F) == 15)
    lo_esc = lo_mask & ((packed & 0x0F) == 15)

    hi_i = hi_esc.to(tl.int32)
    lo_i = lo_esc.to(tl.int32)
    n_hi = tl.sum(hi_i, axis=0)
    hi_rank = tl.cumsum(hi_i, axis=0) - 1
    lo_rank = n_hi + tl.cumsum(lo_i, axis=0) - 1
    base = tl.load(starts + pid)

    hi_exp = tl.load(esc_val + base + hi_rank, mask=hi_esc, other=0).to(tl.int16)
    lo_exp = tl.load(esc_val + base + lo_rank, mask=lo_esc, other=0).to(tl.int16)
    hi_sm = tl.load(sm + hi_pos, mask=hi_esc, other=0).to(tl.int16)
    lo_sm = tl.load(sm + lo_pos, mask=lo_esc, other=0).to(tl.int16)

    tl.store(out + hi_pos, ((hi_sm & 0x80) << 8) | (hi_exp << 7) | (hi_sm & 0x7F), mask=hi_esc)
    tl.store(out + lo_pos, ((lo_sm & 0x80) << 8) | (lo_exp << 7) | (lo_sm & 0x7F), mask=lo_esc)


def bench_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
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
    if rows.shape[1] != BASE_BLOCK_DIM:
        raise RuntimeError(f"Expected {BASE_BLOCK_DIM}-wide KV rows, got {rows.shape[1]}")
    return rows


def assemble_row_prefix(blocks, target_rows, start_idx=0):
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
    chunks = []
    for chunk_idx in range(NUM_FEATURE_CHUNKS):
        chunks.append(assemble_row_prefix(blocks, target_rows, start_idx=chunk_idx))
    matrix = torch.cat(chunks, dim=1).contiguous()
    if matrix.shape != (target_rows, TARGET_HIDDEN_DIM):
        raise RuntimeError(f"Expected {(target_rows, TARGET_HIDDEN_DIM)}, got {tuple(matrix.shape)}")
    return matrix


def collect_real_activation(device):
    tokenizer = AutoTokenizer.from_pretrained(ACTIVATION_MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        ACTIVATION_MODEL,
        dtype=torch.bfloat16,
        device_map={"": device.index if device.index is not None else 0},
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    blocks = []
    prompt_idx = 0
    with torch.no_grad():
        while len(blocks) < NUM_FEATURE_CHUNKS or sum(block.shape[0] for block in blocks) < SEQ_LEN:
            input_ids = build_token_batch(tokenizer, 1024, salt=f" ablation-{prompt_idx}").to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True,
                return_dict=True,
            )
            pkv = outputs.past_key_values
            if hasattr(pkv, "to_legacy_cache"):
                pkv = pkv.to_legacy_cache()
            for key, value in pkv:
                blocks.append(kv_to_rows(key, value).cpu())
                if len(blocks) >= NUM_FEATURE_CHUNKS and sum(block.shape[0] for block in blocks) >= SEQ_LEN:
                    break
            prompt_idx += 1

    del model
    torch.cuda.empty_cache()
    return assemble_matrix_4096(blocks, SEQ_LEN)


class Top15SentinelCodec:
    def __init__(self, device="cuda"):
        self.device = device
        self.enc_lut = None
        self.dec_lut = None

    def calibrate(self, sample):
        int16 = sample.contiguous().view(torch.int16)
        exponents = ((int16 >> 7) & 0xFF).to(torch.uint8)
        vals, counts = torch.unique(exponents, return_counts=True)
        order = torch.argsort(counts, descending=True)
        self.enc_lut = torch.full((256,), 15, dtype=torch.uint8, device=self.device)
        self.dec_lut = torch.zeros((16,), dtype=torch.uint8, device=self.device)
        top = min(15, vals.numel())
        for i in range(top):
            v = int(vals[order[i]].item())
            self.enc_lut[v] = i
            self.dec_lut[i] = v
        coverage = counts[order[:top]].sum().item() / counts.sum().item()
        return coverage

    def encode(self, tensor):
        n = tensor.numel()
        int16 = tensor.contiguous().view(torch.int16)
        n_pairs = n // 2

        pk = torch.empty(n_pairs, dtype=torch.uint8, device=self.device)
        sm = torch.empty(n, dtype=torch.uint8, device=self.device)
        block = 256
        grid = ((n_pairs + block * 4 - 1) // (block * 4),)
        _enc_4bit[grid](int16, self.enc_lut, pk, sm, n, BLOCK=block)

        scan_block = 256
        n_blocks = (n_pairs + scan_block - 1) // scan_block
        counts = torch.empty(n_blocks, dtype=torch.int32, device=self.device)
        _count_sentinel_escapes[(n_blocks,)](pk, counts, n_pairs, n, BLOCK=scan_block)
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts
        n_esc = int(offsets[-1].item()) if offsets.numel() else 0
        max_esc = max(n // 10, 1024)
        if n_esc > max_esc:
            raise RuntimeError(f"escape buffer too small: {n_esc} > {max_esc}")
        esc_val = torch.empty(max_esc, dtype=torch.uint8, device=self.device)
        _write_sentinel_values[(n_blocks,)](int16, pk, starts, esc_val, n_pairs, n, BLOCK=scan_block)
        return pk, sm, torch.empty(0, dtype=torch.int32, device=self.device), esc_val[:n_esc], n, n_esc

    def decode(self, pk, sm, esc_pos, esc_val, n, n_esc):
        output = torch.empty(n, dtype=torch.int16, device=self.device)
        n_pairs = n // 2
        block = 256
        grid = ((n_pairs + block * 4 - 1) // (block * 4),)
        _dec_4bit[grid](pk, sm, self.dec_lut, output, n, BLOCK=block)
        if n_esc > 0:
            scan_block = 256
            n_blocks = (n_pairs + scan_block - 1) // scan_block
            counts = torch.empty(n_blocks, dtype=torch.int32, device=self.device)
            _count_sentinel_escapes[(n_blocks,)](pk, counts, n_pairs, n, BLOCK=scan_block)
            offsets = torch.cumsum(counts, dim=0)
            starts = offsets - counts
            _fix_sentinel_escapes[(n_blocks,)](pk, starts, esc_val, sm, output, n_pairs, n, BLOCK=scan_block)
        return output.view(torch.bfloat16)


def compressed_bytes(encoded):
    pk, sm, esc_pos, esc_val, _, _ = encoded
    return pk.numel() + sm.numel() + esc_pos.numel() * 4 + esc_val.numel()


def benchmark_codec(name, codec, flat, encode_iters, decode_iters):
    encoded = codec.encode(flat)
    decoded = codec.decode(*encoded)
    if not torch.equal(flat.view(torch.int16), decoded.view(torch.int16)):
        raise RuntimeError(f"{name} round-trip failed")
    raw_bytes = flat.numel() * 2
    enc_s = bench_cuda(lambda: codec.encode(flat), warmup=5, iters=encode_iters)
    encoded = codec.encode(flat)
    dec_s = bench_cuda(lambda: codec.decode(*encoded), warmup=5, iters=decode_iters)
    exp = ((flat.view(torch.int16) >> 7) & 0xFF).to(torch.uint8)
    vals, counts = torch.unique(exp, return_counts=True)
    order = torch.argsort(counts, descending=True)
    topk = 16 if name == "top16_escape_positions" else 15
    coverage = counts[order[: min(topk, vals.numel())]].sum().item() / counts.sum().item()
    return {
        "name": name,
        "coverage": coverage,
        "escapes": int(encoded[5]),
        "escape_rate": float(encoded[5] / flat.numel()),
        "compressed_bytes": int(compressed_bytes(encoded)),
        "ratio": raw_bytes / compressed_bytes(encoded),
        "encode_gbs": raw_bytes / enc_s / 1e9,
        "decode_gbs": raw_bytes / dec_s / 1e9,
    }


def benchmark_dynamic_top16(flat, device):
    codec = FastLosslessCodec(str(device))

    def only_calibrate():
        codec.calibrate(flat)

    def dynamic_encode():
        codec.calibrate(flat)
        return codec.encode(flat)

    calib_s = bench_cuda(only_calibrate, warmup=2, iters=8)
    encoded = dynamic_encode()
    decoded = codec.decode(*encoded)
    if not torch.equal(flat.view(torch.int16), decoded.view(torch.int16)):
        raise RuntimeError("dynamic_top16 round-trip failed")
    dyn_enc_s = bench_cuda(dynamic_encode, warmup=1, iters=4)
    encoded = dynamic_encode()
    dyn_dec_s = bench_cuda(lambda: codec.decode(*encoded), warmup=5, iters=40)

    pre = FastLosslessCodec(str(device))
    pre.calibrate(flat)
    pre_encoded = pre.encode(flat)
    pre_dec_s = bench_cuda(lambda: pre.decode(*pre_encoded), warmup=5, iters=40)
    pre_enc_s = bench_cuda(lambda: pre.encode(flat), warmup=5, iters=20)

    raw_bytes = flat.numel() * 2
    return {
        "precalibrated_top16": {
            "encode_gbs": raw_bytes / pre_enc_s / 1e9,
            "decode_gbs": raw_bytes / pre_dec_s / 1e9,
            "escapes": int(pre_encoded[5]),
            "escape_rate": float(pre_encoded[5] / flat.numel()),
            "ratio": raw_bytes / compressed_bytes(pre_encoded),
        },
        "dynamic_top16": {
            "calibration_gbs": raw_bytes / calib_s / 1e9,
            "encode_gbs": raw_bytes / dyn_enc_s / 1e9,
            "decode_gbs": raw_bytes / dyn_dec_s / 1e9,
            "escapes": int(encoded[5]),
            "escape_rate": float(encoded[5] / flat.numel()),
            "ratio": raw_bytes / compressed_bytes(encoded),
        },
    }


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def to_pct(x):
    return f"{x * 100:.2f}%"


def make_markdown(data):
    lines = []
    lines.append("# Escape / Calibration Ablation")
    lines.append("")
    lines.append(f"- Tensor: real BF16 activation matrix from `{ACTIVATION_MODEL}`, shape `{SEQ_LEN} x {TARGET_HIDDEN_DIM}`")
    lines.append("- Throughput is measured on the same tensor for all variants.")
    lines.append("")
    lines.append("## 1. Escape Position vs. Mask/Sentinel")
    lines.append("")
    lines.append("| Variant | Coverage | Escape Rate | Ratio | Encode GB/s | Decode GB/s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in data["escape_position_vs_mask"]:
        lines.append(
            f"| {row['name']} | {to_pct(row['coverage'])} | {to_pct(row['escape_rate'])} | {fmt(row['ratio'])} | "
            f"{fmt(row['encode_gbs'])} | {fmt(row['decode_gbs'])} |"
        )
    better = data["escape_position_vs_mask_conclusion"]
    lines.append("")
    lines.append(f"- Faster encode: `{better['faster_encode']}`")
    lines.append(f"- Faster decode: `{better['faster_decode']}`")
    lines.append("")
    lines.append("## 2. Pre-Calibration vs. Dynamic Top-16")
    lines.append("")
    lines.append("| Variant | Ratio | Escape Rate | Encode GB/s | Decode GB/s | Calibration GB/s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    pre = data["calibration_ablation"]["precalibrated_top16"]
    dyn = data["calibration_ablation"]["dynamic_top16"]
    lines.append(
        f"| precalibrated_top16 | {fmt(pre['ratio'])} | {to_pct(pre['escape_rate'])} | {fmt(pre['encode_gbs'])} | {fmt(pre['decode_gbs'])} | - |"
    )
    lines.append(
        f"| dynamic_top16 | {fmt(dyn['ratio'])} | {to_pct(dyn['escape_rate'])} | {fmt(dyn['encode_gbs'])} | {fmt(dyn['decode_gbs'])} | {fmt(dyn['calibration_gbs'])} |"
    )
    lines.append("")
    lines.append(f"- Encode slowdown from dynamic top16: `{fmt(pre['encode_gbs'] / dyn['encode_gbs'])}x`")
    lines.append(f"- Decode slowdown from dynamic top16: `{fmt(pre['decode_gbs'] / dyn['decode_gbs'])}x`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args()

    device = torch.device(args.device)
    cpu_matrix = collect_real_activation(device)
    flat = cpu_matrix.to(device=device, dtype=torch.bfloat16).contiguous().view(-1)

    codec_pos = FastLosslessCodec(str(device))
    codec_pos.calibrate(flat)
    pos = benchmark_codec("top16_escape_positions", codec_pos, flat, encode_iters=20, decode_iters=30)

    codec_mask = Top15SentinelCodec(str(device))
    codec_mask.calibrate(flat)
    mask = benchmark_codec("top15_sentinel_mask", codec_mask, flat, encode_iters=20, decode_iters=30)

    calibration = benchmark_dynamic_top16(flat, device)

    payload = {
        "tensor_shape": [SEQ_LEN, TARGET_HIDDEN_DIM],
        "activation_model": ACTIVATION_MODEL,
        "escape_position_vs_mask": [pos, mask],
        "escape_position_vs_mask_conclusion": {
            "faster_encode": pos["name"] if pos["encode_gbs"] > mask["encode_gbs"] else mask["name"],
            "faster_decode": pos["name"] if pos["decode_gbs"] > mask["decode_gbs"] else mask["name"],
        },
        "calibration_ablation": calibration,
        "artifacts": {
            "json": str(args.json_out),
            "markdown": str(args.md_out),
        },
    }
    args.json_out.write_text(json.dumps(payload, indent=2))
    args.md_out.write_text(make_markdown(payload))
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")


if __name__ == "__main__":
    main()
