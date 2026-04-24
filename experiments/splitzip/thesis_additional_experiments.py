import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.splitzip.codec_ablation_bench import collect_real_activations
from experiments.splitzip.fp8_e5m2_top8_compact_bench import FP8Top8CompactCodec
from experiments.splitzip.lossless_fast import FastLosslessCodec
from experiments.splitzip.opt_rounds3 import _dec_3bit_vec, _enc_3bit_vec
from experiments.splitzip.thesis_experiment_dump import assemble_row_prefix, load_model_activation_blocks


DEFAULT_JSON = ROOT / "experiments" / "splitzip" / "thesis_additional_experiments.json"
DEFAULT_MD = ROOT / "experiments" / "splitzip" / "thesis_additional_experiments.md"

REP_ACTIVATION_MODEL = "Qwen/Qwen2.5-1.5B"
REP_SHAPE = (32768, 4096)
CAL_MODEL = "Qwen/Qwen2.5-1.5B"
SERVING_MODEL = "NousResearch/Meta-Llama-3-8B"
SERVING_SEQ_LEN = 32768
SERVING_LAYER_IDX = 0


def bench_cuda(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def load_representative_bf16_matrix(device):
    mats = collect_real_activations(REP_ACTIVATION_MODEL, [REP_SHAPE[0]], str(device))
    return mats[0]


def bf16_to_raw_fp8(flat_bf16, fmt):
    if fmt == "e4m3":
        raw = flat_bf16.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    else:
        raw = flat_bf16.to(torch.float8_e5m2).view(torch.uint8).contiguous()
    usable = raw.numel() - (raw.numel() % 8)
    return raw[:usable].contiguous()


def run_fp8_exact_results(cpu_bf16, device):
    flat = cpu_bf16.to(device=device, dtype=torch.bfloat16).contiguous().view(-1)
    results = []
    for fmt in ("e4m3", "e5m2"):
        raw = bf16_to_raw_fp8(flat, fmt)
        codec = FP8Top8CompactCodec(fmt, raw)
        n_esc = codec.encode_full()
        max_count = codec.counts.max().item()
        codec.decode_full(max_count)
        correct = torch.equal(raw, codec.core.out)
        if not correct:
            raise RuntimeError(f"FP8 round-trip failed for {fmt}")
        comp_bytes = codec.compressed_bytes(n_esc)
        enc_full_s = bench_cuda(codec.encode_full, warmup=10, iters=60)
        n_esc = codec.encode_full()
        max_count = codec.counts.max().item()
        dec_full_s = bench_cuda(lambda: codec.decode_full(max_count), warmup=10, iters=80)
        raw_bytes = raw.numel()
        results.append(
            {
                "format": fmt,
                "raw_fp8_bytes": int(raw_bytes),
                "compressed_bytes": int(comp_bytes),
                "ratio_vs_fp8": raw_bytes / comp_bytes,
                "ratio_vs_bf16": 2.0 * raw_bytes / comp_bytes,
                "coverage_top8": float(codec.core.coverage),
                "escapes": int(n_esc),
                "escape_rate": float(n_esc / raw.numel()),
                "encode_gbs": raw_bytes / enc_full_s / 1e9,
                "decode_gbs": raw_bytes / dec_full_s / 1e9,
            }
        )
    return results


@triton.jit
def _count_external_bf16(inp, common_lut, counts, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    raw = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
    exp = (raw >> 7) & 0xFF
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    count = tl.sum(esc.to(tl.int32), axis=0)
    tl.store(counts + pid, count)


@triton.jit
def _write_external_bf16(inp, common_lut, starts, local_pos, esc_val, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base_elem = pid * BLOCK
    offs = base_elem + tl.arange(0, BLOCK)
    mask = offs < n
    raw = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
    exp = (raw >> 7) & 0xFF
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    rank = tl.cumsum(esc.to(tl.int32), 0) - 1
    start = tl.load(starts + pid)
    out_idx = start + rank
    local = offs - base_elem
    tl.store(local_pos + out_idx, local.to(tl.uint8), mask=esc)
    tl.store(esc_val + out_idx, exp.to(tl.uint8), mask=esc)


@triton.jit
def _fix_external_bf16(counts, starts, local_pos, esc_val, sm, out, BLOCK_ELEMS: tl.constexpr, BLOCK_ESC: tl.constexpr):
    pid = tl.program_id(0)
    count = tl.load(counts + pid)
    start = tl.load(starts + pid)
    offs = tl.arange(0, BLOCK_ESC)
    mask = offs < count
    idx = start + offs
    local = tl.load(local_pos + idx, mask=mask, other=0).to(tl.int32)
    exp = tl.load(esc_val + idx, mask=mask, other=0).to(tl.int32)
    pos = pid * BLOCK_ELEMS + local
    raw_sm = tl.load(sm + pos, mask=mask, other=0).to(tl.int32)
    fixed = ((raw_sm & 0x80) << 8) | (exp << 7) | (raw_sm & 0x7F)
    tl.store(out + pos, fixed.to(tl.int16), mask=mask)


@triton.jit
def _enc_4bit_compact(inp, lut, pk, sm, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    po = tl.arange(0, BLOCK)
    b = pid * BLOCK * 4
    for step in range(4):
        pi = b + step * BLOCK + po
        ei = pi * 2
        oi = ei + 1
        em = ei < n
        om = oi < n
        v0 = tl.load(inp + ei, mask=em, other=0).to(tl.int16)
        v1 = tl.load(inp + oi, mask=om, other=0).to(tl.int16)
        i0 = tl.load(lut + ((v0 >> 7) & 0xFF).to(tl.int32), mask=em, other=0).to(tl.uint8)
        i1 = tl.load(lut + ((v1 >> 7) & 0xFF).to(tl.int32), mask=om, other=0).to(tl.uint8)
        tl.store(pk + pi, (i0 << 4) | i1, mask=em)
        tl.store(sm + ei, (((v0 >> 8) & 0x80) | (v0 & 0x7F)).to(tl.uint8), mask=em)
        tl.store(sm + oi, (((v1 >> 8) & 0x80) | (v1 & 0x7F)).to(tl.uint8), mask=om)


@triton.jit
def _dec_4bit_compact(pk, sm, dlut, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    po = tl.arange(0, BLOCK)
    b = pid * BLOCK * 4
    for step in range(4):
        pi = b + step * BLOCK + po
        ei = pi * 2
        oi = ei + 1
        em = ei < n
        om = oi < n
        packed = tl.load(pk + pi, mask=em, other=0)
        e0 = tl.load(dlut + ((packed >> 4) & 0x0F).to(tl.int32), mask=em, other=0).to(tl.int16)
        e1 = tl.load(dlut + (packed & 0x0F).to(tl.int32), mask=om, other=0).to(tl.int16)
        s0 = tl.load(sm + ei, mask=em, other=0).to(tl.int16)
        s1 = tl.load(sm + oi, mask=om, other=0).to(tl.int16)
        tl.store(out + ei, ((s0 & 0x80) << 8) | (e0 << 7) | (s0 & 0x7F), mask=em)
        tl.store(out + oi, ((s1 & 0x80) << 8) | (e1 << 7) | (s1 & 0x7F), mask=om)


class BF16Top8ExactCodec:
    def __init__(self, flat_bf16, device, escape_block=256):
        self.device = device
        self.raw = flat_bf16.contiguous().view(torch.int16)
        self.n = self.raw.numel()
        if self.n % 8 != 0:
            raise ValueError("top8 exact benchmark expects numel divisible by 8")
        self.escape_block = escape_block
        self.n_blocks = (self.n + escape_block - 1) // escape_block
        self.block = 128
        self.code_bytes = self.n * 3 // 8
        self.sm_bytes = self.n
        self.enc_lut, self.dec_lut, self.common_lut, self.coverage = self.build_codebook()
        self.code_pk = torch.empty(self.code_bytes, dtype=torch.uint8, device=self.device)
        self.sm = torch.empty(self.n, dtype=torch.uint8, device=self.device)
        self.out = torch.empty(self.n, dtype=torch.int16, device=self.device)
        self.counts = torch.empty(self.n_blocks, dtype=torch.int32, device=self.device)
        self.starts = torch.empty(self.n_blocks, dtype=torch.int32, device=self.device)
        self.max_esc = max(self.n // 4, 1024)
        self.local_pos = torch.empty(self.max_esc, dtype=torch.uint8, device=self.device)
        self.esc_val = torch.empty(self.max_esc, dtype=torch.uint8, device=self.device)

    def build_codebook(self):
        exponents = ((self.raw >> 7) & 0xFF).to(torch.uint8)
        vals, counts = torch.unique(exponents, return_counts=True)
        order = torch.argsort(counts, descending=True)
        enc = torch.zeros((256,), dtype=torch.uint8, device=self.device)
        dec = torch.zeros((8,), dtype=torch.uint8, device=self.device)
        common = torch.zeros((256,), dtype=torch.uint8, device=self.device)
        top = min(8, vals.numel())
        for i in range(top):
            v = vals[order[i]].item()
            enc[v] = i
            dec[i] = v
            common[v] = 1
        coverage = counts[order[:top]].sum().item() / counts.sum().item()
        return enc, dec, common, coverage

    def encode_core(self):
        grid = ((self.n // 8 + self.block - 1) // self.block,)
        _enc_3bit_vec[grid](self.raw, self.enc_lut, self.code_pk, self.sm, self.n, BLOCK=self.block)

    def count_escapes(self):
        _count_external_bf16[(self.n_blocks,)](self.raw, self.common_lut, self.counts, self.n, BLOCK=self.escape_block)

    def write_escapes(self):
        _write_external_bf16[(self.n_blocks,)](
            self.raw, self.common_lut, self.starts, self.local_pos, self.esc_val, self.n, BLOCK=self.escape_block
        )

    def encode_full(self):
        self.encode_core()
        self.count_escapes()
        offsets = torch.cumsum(self.counts, dim=0)
        self.starts = offsets - self.counts
        n_esc = int(offsets[-1].item()) if offsets.numel() else 0
        if n_esc > self.max_esc:
            raise RuntimeError(f"escape buffer too small: {n_esc} > {self.max_esc}")
        self.write_escapes()
        return n_esc

    def decode_core(self):
        grid = ((self.n // 8 + self.block - 1) // self.block,)
        _dec_3bit_vec[grid](self.code_pk, self.sm, self.dec_lut, self.out, self.n, BLOCK=self.block)

    def decode_full(self, max_count):
        self.decode_core()
        block_esc = 1 << (max(1, int(max_count)) - 1).bit_length() if max_count > 0 else 1
        _fix_external_bf16[(self.n_blocks,)](
            self.counts,
            self.starts,
            self.local_pos,
            self.esc_val,
            self.sm,
            self.out,
            BLOCK_ELEMS=self.escape_block,
            BLOCK_ESC=block_esc,
        )
        return self.out

    def compressed_bytes(self, n_esc):
        return self.code_bytes + self.sm_bytes + self.n_blocks + n_esc * 2


def run_bf16_topk_ablation(cpu_bf16, device):
    flat = cpu_bf16.to(device=device, dtype=torch.bfloat16).contiguous().view(-1)

    codec16 = FastLosslessCodec(str(device))
    codec16.calibrate(flat)
    encoded16 = codec16.encode(flat)
    decoded16 = codec16.decode(*encoded16)
    if not torch.equal(flat.view(torch.int16), decoded16.view(torch.int16)):
        raise RuntimeError("top16 lossless round-trip failed")
    comp16 = encoded16[0].numel() + encoded16[1].numel() + encoded16[2].numel() * 4 + encoded16[3].numel()
    enc16_s = bench_cuda(lambda: codec16.encode(flat), warmup=10, iters=60)
    encoded16 = codec16.encode(flat)
    dec16_s = bench_cuda(lambda: codec16.decode(*encoded16), warmup=10, iters=80)
    exp = ((flat.view(torch.int16) >> 7) & 0xFF).to(torch.uint8)
    vals, counts = torch.unique(exp, return_counts=True)
    order = torch.argsort(counts, descending=True)
    cov16 = counts[order[: min(16, vals.numel())]].sum().item() / counts.sum().item()

    codec8 = BF16Top8ExactCodec(flat, device)
    n_esc8 = codec8.encode_full()
    max_count8 = codec8.counts.max().item() if codec8.counts.numel() else 0
    decoded8 = codec8.decode_full(max_count8)
    if not torch.equal(flat.view(torch.int16), decoded8.view(torch.int16)):
        raise RuntimeError("top8 lossless round-trip failed")
    comp8 = codec8.compressed_bytes(n_esc8)
    enc8_s = bench_cuda(codec8.encode_full, warmup=10, iters=40)
    n_esc8 = codec8.encode_full()
    max_count8 = codec8.counts.max().item() if codec8.counts.numel() else 0
    dec8_s = bench_cuda(lambda: codec8.decode_full(max_count8), warmup=10, iters=60)

    raw_bytes = flat.numel() * 2
    return {
        "top8_3bit_exact": {
            "coverage": float(codec8.coverage),
            "escapes": int(n_esc8),
            "escape_rate": float(n_esc8 / flat.numel()),
            "compressed_bytes": int(comp8),
            "ratio": raw_bytes / comp8,
            "encode_gbs": raw_bytes / enc8_s / 1e9,
            "decode_gbs": raw_bytes / dec8_s / 1e9,
        },
        "top16_4bit_exact": {
            "coverage": float(cov16),
            "escapes": int(encoded16[5]),
            "escape_rate": float(encoded16[5] / flat.numel()),
            "compressed_bytes": int(comp16),
            "ratio": raw_bytes / comp16,
            "encode_gbs": raw_bytes / enc16_s / 1e9,
            "decode_gbs": raw_bytes / dec16_s / 1e9,
        },
    }


def load_prompt_texts(dataset_name, count):
    if dataset_name == "A":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [x["text"].strip() for x in ds if x["text"].strip()]
    elif dataset_name == "B":
        ds = load_dataset("openai_humaneval", split="test")
        texts = [x["prompt"].strip() for x in ds if x["prompt"].strip()]
    else:
        raise ValueError(dataset_name)
    return texts[:count]


def collect_exponents_for_prompts(model_name, prompts, device, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map={"": device.index if isinstance(device, torch.device) else 0},
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    exponents = []
    with torch.inference_mode():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(str(device)) for k, v in inputs.items()}
            outputs = model(**inputs, use_cache=True, return_dict=True)
            pkv = outputs.past_key_values
            if hasattr(pkv, "to_legacy_cache"):
                pkv = pkv.to_legacy_cache()
            for key, value in pkv:
                exponents.append(((key.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).view(-1).cpu())
                exponents.append(((value.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).view(-1).cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(exponents, dim=0)


def topk_coverage_from_codebook(sample_exp, eval_exp, k=16):
    vals, counts = torch.unique(sample_exp, return_counts=True)
    order = torch.argsort(counts, descending=True)
    top_vals = vals[order[: min(k, vals.numel())]]
    lut = torch.zeros((256,), dtype=torch.uint8)
    lut[top_vals.long()] = 1
    return float(lut[eval_exp.long()].float().mean().item())


def run_cross_dataset_calibration(device):
    prompts_a = load_prompt_texts("A", 32)
    prompts_b = load_prompt_texts("B", 32)
    exp_a = collect_exponents_for_prompts(CAL_MODEL, prompts_a, device)
    exp_b = collect_exponents_for_prompts(CAL_MODEL, prompts_b, device)
    cov_aa = topk_coverage_from_codebook(exp_a, exp_a, 16)
    cov_ab = topk_coverage_from_codebook(exp_a, exp_b, 16)
    cov_bb = topk_coverage_from_codebook(exp_b, exp_b, 16)
    return {
        "model": CAL_MODEL,
        "dataset_a": "wikitext-2-raw-v1",
        "dataset_b": "openai_humaneval",
        "calibrate_A_eval_A_top16": cov_aa,
        "calibrate_A_eval_B_top16": cov_ab,
        "calibrate_B_eval_B_top16": cov_bb,
    }


def build_top16_ratio_with_metadata(exp_matrix, grouping):
    rows, cols = exp_matrix.shape
    n = rows * cols
    if grouping == "per_tensor":
        groups = [exp_matrix.reshape(-1)]
    elif grouping == "per_token":
        groups = [exp_matrix[i, :] for i in range(rows)]
    elif grouping == "per_channel":
        groups = [exp_matrix[:, j] for j in range(cols)]
    else:
        raise ValueError(grouping)

    total_esc = 0
    total_common = 0
    for g in groups:
        vals, counts = torch.unique(g, return_counts=True)
        order = torch.argsort(counts, descending=True)
        top_vals = vals[order[: min(16, vals.numel())]]
        lut = torch.zeros((256,), dtype=torch.uint8)
        lut[top_vals.long()] = 1
        common = int(lut[g.long()].sum().item())
        total_common += common
        total_esc += int(g.numel() - common)

    codebook_bytes = len(groups) * 16
    n_blocks = math.ceil(n / 256)
    compressed_bytes = (n // 2) + n + n_blocks + total_esc * 2 + codebook_bytes
    return {
        "coverage": total_common / n,
        "escapes": total_esc,
        "escape_rate": total_esc / n,
        "codebook_bytes": codebook_bytes,
        "projected_ratio": (n * 2) / compressed_bytes,
    }


def run_calibration_granularity(cpu_bf16):
    exp = ((cpu_bf16.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8).view(cpu_bf16.shape[0], cpu_bf16.shape[1])
    return {
        "per_tensor": build_top16_ratio_with_metadata(exp, "per_tensor"),
        "per_token": build_top16_ratio_with_metadata(exp, "per_token"),
        "per_channel": build_top16_ratio_with_metadata(exp, "per_channel"),
    }


def build_serving_kv(rows_cpu, seq_len, device):
    rows = assemble_row_prefix([rows_cpu], seq_len).to(device=device, dtype=torch.bfloat16)
    half = rows.shape[1] // 2
    k = rows[:, :half].contiguous().view(seq_len, 8, 128).permute(1, 0, 2).contiguous()
    v = rows[:, half:].contiguous().view(seq_len, 8, 128).permute(1, 0, 2).contiguous()
    return k, v


def attention_proxy(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(1, 2)) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
    out = torch.matmul(probs, v)
    return out


def run_serving_compute_proxy(device):
    blocks, meta = load_model_activation_blocks(SERVING_MODEL, device)
    rows_cpu = blocks[SERVING_LAYER_IDX]
    k, v = build_serving_kv(rows_cpu, SERVING_SEQ_LEN, device)
    q = torch.randn((k.shape[0], 1, k.shape[2]), dtype=torch.bfloat16, device=device)

    codec = FastLosslessCodec(str(device))
    codec.calibrate(torch.cat([k.view(-1), v.view(-1)]))
    k_enc = codec.encode(k.view(-1))
    v_enc = codec.encode(v.view(-1))

    _ = attention_proxy(q, k, v)
    original_s = bench_cuda(lambda: attention_proxy(q, k, v), warmup=20, iters=80)

    def decompress_then_compute():
        k_dec = codec.decode(*k_enc).view_as(k)
        v_dec = codec.decode(*v_enc).view_as(v)
        return attention_proxy(q, k_dec, v_dec)

    _ = decompress_then_compute()
    dec_compute_s = bench_cuda(decompress_then_compute, warmup=10, iters=40)

    materialized_k = torch.empty_like(k)
    materialized_v = torch.empty_like(v)

    def materialize_copy():
        materialized_k.copy_(k)
        materialized_v.copy_(v)

    copy_s = bench_cuda(materialize_copy, warmup=20, iters=80)
    merged_proj_s = max(original_s, dec_compute_s - copy_s)

    return {
        "model": SERVING_MODEL,
        "seq_len": SERVING_SEQ_LEN,
        "layer_index": SERVING_LAYER_IDX,
        "heads": int(k.shape[0]),
        "head_dim": int(k.shape[2]),
        "original_kv_compute_ms": original_s * 1000,
        "decompress_then_compute_ms": dec_compute_s * 1000,
        "projected_merged_kernel_ms": merged_proj_s * 1000,
        "materialization_copy_ms": copy_s * 1000,
        "note": "Merged-kernel number is a projection: explicit decompress+compute minus one measured BF16 materialization copy, floored by original compute time.",
    }


def to_pct(x):
    return f"{x * 100:.2f}%"


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def make_markdown(data):
    lines = []
    lines.append("# Additional Thesis Experiments")
    lines.append("")
    lines.append("## 1. FP8 Exact Results")
    lines.append("")
    lines.append("| Format | Top-8 Coverage | Ratio vs FP8 | Total Ratio vs BF16 | Encode GB/s | Decode GB/s | Escape Rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in data["fp8_results"]:
        lines.append(
            f"| {row['format'].upper()} | {to_pct(row['coverage_top8'])} | {fmt(row['ratio_vs_fp8'])} | "
            f"{fmt(row['ratio_vs_bf16'])} | {fmt(row['encode_gbs'])} | {fmt(row['decode_gbs'])} | {to_pct(row['escape_rate'])} |"
        )
    lines.append("")
    lines.append("## 2. Serving Compute Feasibility")
    lines.append("")
    s = data["serving_compute"]
    lines.append(f"- Model proxy: `{s['model']}`, seq-len `{s['seq_len']}`, layer `{s['layer_index']}`")
    lines.append(f"- Original KV compute time: `{fmt(s['original_kv_compute_ms'])} ms`")
    lines.append(f"- KV compute time with explicit decompression: `{fmt(s['decompress_then_compute_ms'])} ms`")
    lines.append(f"- KV compute time with projected merged kernel: `{fmt(s['projected_merged_kernel_ms'])} ms`")
    lines.append(f"- Materialization copy removed in projection: `{fmt(s['materialization_copy_ms'])} ms`")
    lines.append(f"- Note: {s['note']}")
    lines.append("")
    lines.append("## 3. BF16 Top-8 vs Top-16 (Exact, With Escapes)")
    lines.append("")
    lines.append("| Variant | Coverage | Ratio | Encode GB/s | Decode GB/s | Escape Rate |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    top8 = data["bf16_topk_ablation"]["top8_3bit_exact"]
    top16 = data["bf16_topk_ablation"]["top16_4bit_exact"]
    lines.append(
        f"| Top-8 (3-bit) | {to_pct(top8['coverage'])} | {fmt(top8['ratio'])} | {fmt(top8['encode_gbs'])} | {fmt(top8['decode_gbs'])} | {to_pct(top8['escape_rate'])} |"
    )
    lines.append(
        f"| Top-16 (4-bit) | {to_pct(top16['coverage'])} | {fmt(top16['ratio'])} | {fmt(top16['encode_gbs'])} | {fmt(top16['decode_gbs'])} | {to_pct(top16['escape_rate'])} |"
    )
    better = data["bf16_topk_ablation"]["better_format"]
    lines.append("")
    lines.append(f"- Better format on this real BF16 tensor: `{better}`")
    lines.append("")
    lines.append("## 4. Cross-Dataset Calibration")
    lines.append("")
    c = data["cross_dataset_calibration"]
    lines.append(f"- Dataset A: `{c['dataset_a']}`")
    lines.append(f"- Dataset B: `{c['dataset_b']}`")
    lines.append(f"- Calibrate on A, Top-16 coverage on A: `{to_pct(c['calibrate_A_eval_A_top16'])}`")
    lines.append(f"- Calibrate on A, Top-16 coverage on B: `{to_pct(c['calibrate_A_eval_B_top16'])}`")
    lines.append(f"- Calibrate on B, Top-16 coverage on B: `{to_pct(c['calibrate_B_eval_B_top16'])}`")
    lines.append("")
    lines.append("## 5. Calibration Granularity")
    lines.append("")
    lines.append("| Scope | Coverage | Projected Ratio | Escape Rate | Codebook Bytes |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, row in data["calibration_granularity"].items():
        lines.append(
            f"| {name} | {to_pct(row['coverage'])} | {fmt(row['projected_ratio'])} | {to_pct(row['escape_rate'])} | {row['codebook_bytes']} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- JSON: `{data['artifacts']['json']}`")
    lines.append(f"- Markdown: `{data['artifacts']['markdown']}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_MD))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading representative BF16 activations...", flush=True)
    cpu_bf16 = load_representative_bf16_matrix(device)

    print("Running FP8 exact benchmarks...", flush=True)
    fp8_results = run_fp8_exact_results(cpu_bf16, device)

    print("Running BF16 top-8 vs top-16 ablation...", flush=True)
    bf16_topk = run_bf16_topk_ablation(cpu_bf16, device)
    better_format = "top16_4bit_exact"
    if bf16_topk["top8_3bit_exact"]["ratio"] > bf16_topk["top16_4bit_exact"]["ratio"] and \
       bf16_topk["top8_3bit_exact"]["decode_gbs"] >= bf16_topk["top16_4bit_exact"]["decode_gbs"]:
        better_format = "top8_3bit_exact"
    bf16_topk["better_format"] = better_format

    print("Running cross-dataset calibration study...", flush=True)
    cross_dataset = run_cross_dataset_calibration(device)

    print("Running calibration granularity study...", flush=True)
    granularity = run_calibration_granularity(cpu_bf16)

    print("Running serving compute proxy...", flush=True)
    serving = run_serving_compute_proxy(device)

    out = {
        "representative_tensor": {
            "activation_model": REP_ACTIVATION_MODEL,
            "shape": list(REP_SHAPE),
        },
        "fp8_results": fp8_results,
        "bf16_topk_ablation": bf16_topk,
        "cross_dataset_calibration": cross_dataset,
        "calibration_granularity": granularity,
        "serving_compute": serving,
        "artifacts": {
            "json": str(Path(args.json_out).resolve()),
            "markdown": str(Path(args.md_out).resolve()),
        },
    }

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.write_text(json.dumps(out, indent=2))
    md_path.write_text(make_markdown(out))
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
