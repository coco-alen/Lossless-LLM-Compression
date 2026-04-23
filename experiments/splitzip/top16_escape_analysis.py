"""
Analyze top-16 escape coding and escape-position width.

This script is intentionally byte-accounting focused. It does not depend on the
current BF16 escape sentinel implementation, because a top-16 codebook must
collect escapes from the original exponent stream rather than from nibble == 15.
E4M3 is reported with the existing top-8, 3-bit exponent-code format because
using top-16 with 4-bit codes consumes the whole native FP8 byte.
"""

import argparse
import json
from pathlib import Path

import torch


def _entropy(counts):
    probs = counts.float() / counts.sum()
    return -(probs * torch.log2(probs)).sum().item()


def _coverage(exp, k):
    vals, counts = torch.unique(exp, return_counts=True)
    order = torch.argsort(counts, descending=True)
    used = min(k, vals.numel())
    return {
        "unique": vals.numel(),
        "entropy": _entropy(counts),
        "coverage": counts[order[:used]].sum().item() / counts.sum().item(),
        "top_values": [int(vals[order[i]].item()) for i in range(used)],
    }


def _ratio(native_bits, base_bits, esc_rate, pos_bits, val_bits):
    return native_bits / (base_bits + esc_rate * (pos_bits + val_bits))


def summarize_exponents(name, exp, native_bits, base_bits, val_bits):
    top15 = _coverage(exp, 15)
    top16 = _coverage(exp, 16)
    esc15 = 1.0 - top15["coverage"]
    esc16 = 1.0 - top16["coverage"]
    return {
        "name": name,
        "elements": int(exp.numel()),
        "unique_exponents": int(top16["unique"]),
        "entropy_bits": top16["entropy"],
        "top15_coverage": top15["coverage"],
        "top16_coverage": top16["coverage"],
        "top16_values": top16["top_values"],
        "escape_rate_top15": esc15,
        "escape_rate_top16": esc16,
        "ratio_top15_pos32": _ratio(native_bits, base_bits, esc15, 32, val_bits),
        "ratio_top16_pos32": _ratio(native_bits, base_bits, esc16, 32, val_bits),
        "ratio_top16_pos16": _ratio(native_bits, base_bits, esc16, 16, val_bits),
    }


def summarize_top8_compact(name, exp, native_bits, base_bits, val_bits,
                           escape_block=256, offset_bits=8):
    top8 = _coverage(exp, 8)
    esc8 = 1.0 - top8["coverage"]
    count_bits = 8.0 / escape_block
    bits = base_bits + count_bits + esc8 * (offset_bits + val_bits)
    return {
        "name": name,
        "elements": int(exp.numel()),
        "unique_exponents": int(top8["unique"]),
        "entropy_bits": top8["entropy"],
        "top8_coverage": top8["coverage"],
        "top8_values": top8["top_values"],
        "escape_rate_top8": esc8,
        "ratio_top8_compact": native_bits / bits,
    }


def _topk_values(exp, k):
    vals, counts = torch.unique(exp, return_counts=True)
    order = torch.argsort(counts, descending=True)
    return vals[order[:min(k, vals.numel())]].to(torch.long)


def verify_bf16_dummy_escape(tensor):
    raw = tensor.contiguous().view(torch.int16)
    exp = ((raw >> 7) & 0xFF).to(torch.uint8)
    sm = (((raw >> 8) & 0x80) | (raw & 0x7F)).to(torch.uint8)

    top = _topk_values(exp, 16)
    lut = torch.zeros(256, dtype=torch.uint8, device=exp.device)
    common = torch.zeros(256, dtype=torch.bool, device=exp.device)
    dlut = torch.zeros(16, dtype=torch.uint8, device=exp.device)
    for code, value in enumerate(top.tolist()):
        lut[value] = code
        common[value] = True
        dlut[code] = value

    is_common = common[exp.long()]
    codes = torch.where(is_common, lut[exp.long()], torch.zeros_like(exp))
    decoded_exp = dlut[codes.long()]
    esc_pos = (~is_common).nonzero(as_tuple=True)[0]
    if esc_pos.numel() > 0:
        decoded_exp[esc_pos] = exp[esc_pos]

    out = (((sm.to(torch.int16) & 0x80) << 8)
           | (decoded_exp.to(torch.int16) << 7)
           | (sm.to(torch.int16) & 0x7F))
    return bool(torch.equal(raw, out))


def verify_fp8_dummy_escape(tensor, fmt):
    if fmt == "e4m3":
        raw = tensor.to(torch.float8_e4m3fn).view(torch.uint8)
        exp = ((raw >> 3) & 0x0F).to(torch.uint8)
        sm = (((raw >> 4) & 0x08) | (raw & 0x07)).to(torch.uint8)
        lut_size = 16
    else:
        raw = tensor.to(torch.float8_e5m2).view(torch.uint8)
        exp = ((raw >> 2) & 0x1F).to(torch.uint8)
        sm = (((raw >> 5) & 0x04) | (raw & 0x03)).to(torch.uint8)
        lut_size = 32

    top = _topk_values(exp, 16)
    lut = torch.zeros(lut_size, dtype=torch.uint8, device=exp.device)
    common = torch.zeros(lut_size, dtype=torch.bool, device=exp.device)
    dlut = torch.zeros(16, dtype=torch.uint8, device=exp.device)
    for code, value in enumerate(top.tolist()):
        lut[value] = code
        common[value] = True
        dlut[code] = value

    is_common = common[exp.long()]
    codes = torch.where(is_common, lut[exp.long()], torch.zeros_like(exp))
    decoded_exp = dlut[codes.long()]
    esc_pos = (~is_common).nonzero(as_tuple=True)[0]
    if esc_pos.numel() > 0:
        decoded_exp[esc_pos] = exp[esc_pos]

    if fmt == "e4m3":
        out = (((sm & 0x08) << 4) | (decoded_exp << 3) | (sm & 0x07)).to(torch.uint8)
    else:
        out = (((sm & 0x04) << 5) | (decoded_exp << 2) | (sm & 0x03)).to(torch.uint8)
    return bool(torch.equal(raw, out))


def verify_fp8_top8_dummy_escape(tensor, fmt):
    if fmt == "e4m3":
        raw = tensor.to(torch.float8_e4m3fn).view(torch.uint8)
        exp = ((raw >> 3) & 0x0F).to(torch.uint8)
        sm = (((raw >> 4) & 0x08) | (raw & 0x07)).to(torch.uint8)
        lut_size = 16
    else:
        raw = tensor.to(torch.float8_e5m2).view(torch.uint8)
        exp = ((raw >> 2) & 0x1F).to(torch.uint8)
        sm = (((raw >> 5) & 0x04) | (raw & 0x03)).to(torch.uint8)
        lut_size = 32

    top = _topk_values(exp, 8)
    lut = torch.zeros(lut_size, dtype=torch.uint8, device=exp.device)
    common = torch.zeros(lut_size, dtype=torch.bool, device=exp.device)
    dlut = torch.zeros(8, dtype=torch.uint8, device=exp.device)
    for code, value in enumerate(top.tolist()):
        lut[value] = code
        common[value] = True
        dlut[code] = value

    is_common = common[exp.long()]
    codes = torch.where(is_common, lut[exp.long()], torch.zeros_like(exp))
    decoded_exp = dlut[codes.long()]
    esc_pos = (~is_common).nonzero(as_tuple=True)[0]
    if esc_pos.numel() > 0:
        decoded_exp[esc_pos] = exp[esc_pos]

    if fmt == "e4m3":
        out = (((sm & 0x08) << 4) | (decoded_exp << 3) | (sm & 0x07)).to(torch.uint8)
    else:
        out = (((sm & 0x04) << 5) | (decoded_exp << 2) | (sm & 0x03)).to(torch.uint8)
    return bool(torch.equal(raw, out))


def summarize_bf16(tensor):
    exp = ((tensor.contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.uint8)
    out = summarize_exponents("bf16", exp, native_bits=16, base_bits=12, val_bits=8)
    out["top16_dummy_escape_roundtrip"] = verify_bf16_dummy_escape(tensor)
    return out


def summarize_fp8(tensor, fmt):
    if fmt == "e4m3":
        raw = tensor.to(torch.float8_e4m3fn).view(torch.uint8)
        exp = ((raw >> 3) & 0x0F).to(torch.uint8)
        out = summarize_top8_compact("e4m3", exp, native_bits=8, base_bits=7, val_bits=4)
        out["top8_dummy_escape_roundtrip"] = verify_fp8_top8_dummy_escape(tensor, fmt)
        return out
    raw = tensor.to(torch.float8_e5m2).view(torch.uint8)
    exp = ((raw >> 2) & 0x1F).to(torch.uint8)
    out = summarize_exponents("e5m2", exp, native_bits=8, base_bits=7, val_bits=5)
    out["top16_dummy_escape_roundtrip"] = verify_fp8_dummy_escape(tensor, fmt)
    return out


def synthetic_bf16(elements, device, seed):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randn(elements, dtype=torch.bfloat16, device=device, generator=g)


def model_kv_bf16(model_name, device, max_new_tokens, local_files_only):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [
        "The theory of general relativity describes gravity as curved spacetime.",
        "Explain KV cache transfer in disaggregated prefill decode serving.",
        "Summarize lossless compression for floating point tensors.",
        "A researcher benchmarks GPU kernels for transformer cache compression.",
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model.eval()

    pieces = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                use_cache=True,
            )
            for key, value in outputs.past_key_values:
                pieces.append(key.detach().contiguous().view(-1))
                pieces.append(value.detach().contiguous().view(-1))
    return torch.cat(pieces).contiguous()


def max_uint16_tokens(kv_width_elems):
    return 65536 // kv_width_elems


def architecture_limits():
    rows = []
    configs = [
        ("Qwen3-30B-A3B", 4 * 128, 40960),
        ("DeepSeek-V3/R1 MLA latent", 512 + 64, 163840),
        ("DeepSeek-V3/R1 materialized K+V", 128 * (128 + 64 + 128), 163840),
    ]
    for name, kv_width, max_ctx in configs:
        rows.append({
            "name": name,
            "kv_width_elements_per_token_per_layer": kv_width,
            "max_tokens_with_global_uint16_pos": max_uint16_tokens(kv_width),
            "elements_at_4k_tokens": kv_width * 4096,
            "elements_at_max_context": kv_width * max_ctx,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--size-mb", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.model:
        tensor = model_kv_bf16(
            args.model, args.device, args.max_new_tokens, args.local_files_only)
    else:
        elements = args.size_mb * 1024 * 1024 // 2
        tensor = synthetic_bf16(elements, args.device, args.seed)

    results = {
        "source": args.model or f"synthetic_randn_{args.size_mb}MiB_bf16",
        "bf16_bytes": int(tensor.numel() * 2),
        "formats": [
            summarize_bf16(tensor),
            summarize_fp8(tensor, "e4m3"),
            summarize_fp8(tensor, "e5m2"),
        ],
        "architecture_limits": architecture_limits(),
    }

    text = json.dumps(results, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")


if __name__ == "__main__":
    main()
