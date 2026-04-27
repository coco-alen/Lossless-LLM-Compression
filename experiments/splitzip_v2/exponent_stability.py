from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.splitzip.profile_requested_exponents import iter_past_key_values
from experiments.splitzip_v2.benchmark_utils import write_json
from experiments.splitzip_v2.config import SGLANG_MODEL, TABLE1_EXTRA_MODELS


def build_token_batch(tokenizer, target_len: int, salt: str = ""):
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


def summarize_counts(counts: torch.Tensor):
    total = int(counts.sum().item())
    probs = counts.float() / max(total, 1)
    nonzero = counts > 0
    order = torch.argsort(counts, descending=True)
    return {
        "total": total,
        "unique": int(nonzero.sum().item()),
        "top8": float(counts[order[:8]].sum().item() / total) if total else 0.0,
        "top16": float(counts[order[:16]].sum().item() / total) if total else 0.0,
        "entropy": float(-(probs[nonzero] * torch.log2(probs[nonzero])).sum().item()) if total else 0.0,
        "top16_values": [int(x.item()) for x in order[:16]],
    }


def add_counts(counts, tensor):
    exp = ((tensor.detach().contiguous().view(torch.int16) >> 7) & 0xFF).to(torch.long)
    counts += torch.bincount(exp.view(-1), minlength=256).cpu()


def profile(model_id: str, device: str, prompt_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    input_ids = build_token_batch(tokenizer, prompt_tokens, salt=model_id).to(device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True)
    per_layer = []
    global_k = torch.zeros(256, dtype=torch.long)
    global_v = torch.zeros(256, dtype=torch.long)
    for layer_idx, (key, value) in enumerate(iter_past_key_values(outputs.past_key_values)):
        k_counts = torch.zeros(256, dtype=torch.long)
        v_counts = torch.zeros(256, dtype=torch.long)
        add_counts(k_counts, key)
        add_counts(v_counts, value)
        global_k += k_counts
        global_v += v_counts
        per_layer.append({
            "layer": layer_idx,
            "k": summarize_counts(k_counts),
            "v": summarize_counts(v_counts),
        })
    return {
        "model_id": model_id,
        "prompt_tokens": prompt_tokens,
        "global_k": summarize_counts(global_k),
        "global_v": summarize_counts(global_v),
        "per_layer": per_layer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[SGLANG_MODEL.hf_name] + [m.hf_name for m in TABLE1_EXTRA_MODELS])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("experiments/splitzip_v2/results/exponent_stability.json"))
    args = parser.parse_args()
    payload = {"models": args.models}
    if not args.dry_run:
        payload["results"] = [profile(model, args.device, args.prompt_tokens) for model in args.models]
    write_json(args.output, payload)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
