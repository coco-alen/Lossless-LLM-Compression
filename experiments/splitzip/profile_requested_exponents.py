"""
Profile BF16 KV-cache exponent coverage for a requested model list.

The output is the aggregate distribution over K and V tensors from several
short prompts. It is intended to feed paper/tables/exponent_statistics.tex.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REQUESTED_MODELS = [
    ("Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B", "Qwen-MoE"),
    ("Qwen3-32B", "Qwen/Qwen3-32B", "Qwen"),
    ("Qwen3.6-27B", "Qwen/Qwen3.6-27B", "Qwen"),
    ("Qwen3.6-35B-A3B", "Qwen/Qwen3.6-35B-A3B", "Qwen-MoE"),
    ("Llama-3.1-70B-Instruct", "NousResearch/Meta-Llama-3.1-70B-Instruct", "Llama"),
    ("Llama-3-8B", "NousResearch/Meta-Llama-3-8B", "Llama"),
    ("Phi-2", "microsoft/phi-2", "Phi"),
]

PROMPTS = [
    "The theory of general relativity describes gravity as curved spacetime caused by mass and energy.",
    "Explain how KV cache transfer affects time to first token in disaggregated LLM serving.",
    "Summarize why exponent fields in BF16 tensors can be more compressible than sign and mantissa bits.",
    "A researcher benchmarks GPU kernels for lossless compression of transformer key value caches.",
]


def cache_dir_name(model_id):
    return "models--" + model_id.replace("/", "--")


def model_cached(model_id, cache_root):
    return (cache_root / cache_dir_name(model_id)).exists()


def iter_past_key_values(past_key_values):
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None:
                key = getattr(layer, "key_cache", None)
            if value is None:
                value = getattr(layer, "value_cache", None)
            if key is not None and value is not None:
                yield key, value
        return

    for layer in past_key_values:
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            yield layer[0], layer[1]
        elif hasattr(layer, "keys") and hasattr(layer, "values"):
            yield layer.keys, layer.values


def update_counts(counts, tensor):
    flat = tensor.detach().contiguous().view(-1)
    exp = ((flat.view(torch.int16) >> 7) & 0xFF).to(torch.long)
    counts += torch.bincount(exp, minlength=256).cpu()


def summarize_counts(counts):
    total = int(counts.sum().item())
    if total == 0:
        raise RuntimeError("No KV elements were collected")
    probs = counts.float() / total
    nonzero = counts > 0
    entropy = float(-(probs[nonzero] * torch.log2(probs[nonzero])).sum().item())
    order = torch.argsort(counts, descending=True)
    top8 = float(counts[order[:8]].sum().item() / total)
    top15 = float(counts[order[:15]].sum().item() / total)
    top16 = float(counts[order[:16]].sum().item() / total)
    return {
        "total_elements": total,
        "unique_exponents": int(nonzero.sum().item()),
        "top8_coverage": top8,
        "top15_coverage": top15,
        "top16_coverage": top16,
        "entropy_bits": entropy,
        "top16_values": [int(x.item()) for x in order[:16]],
    }


def profile_model(display_name, model_id, family, args):
    cache_root = Path(args.cache_root).expanduser()
    if args.local_files_only and not model_cached(model_id, cache_root):
        return {
            "model": display_name,
            "model_id": model_id,
            "family": family,
            "status": "skipped_not_cached",
        }

    try:
        AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=args.local_files_only)
    except Exception as exc:
        return {
            "model": display_name,
            "model_id": model_id,
            "family": family,
            "status": "skipped_config_error",
            "error": repr(exc),
        }

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=args.local_files_only)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=args.device_map,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        model.eval()

        input_device = args.input_device
        counts = torch.zeros(256, dtype=torch.long)
        n_layers = None
        with torch.no_grad():
            for prompt in PROMPTS[:args.num_prompts]:
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_length,
                ).to(input_device)
                outputs = model(**inputs, use_cache=True)
                layers = list(iter_past_key_values(outputs.past_key_values))
                if n_layers is None:
                    n_layers = len(layers)
                for key, value in layers:
                    update_counts(counts, key)
                    update_counts(counts, value)

        result = summarize_counts(counts)
        result.update({
            "model": display_name,
            "model_id": model_id,
            "family": family,
            "status": "ok",
            "num_prompts": min(args.num_prompts, len(PROMPTS)),
            "layers": n_layers,
        })
        del model
        torch.cuda.empty_cache()
        return result
    except Exception as exc:
        try:
            del model
        except UnboundLocalError:
            pass
        torch.cuda.empty_cache()
        return {
            "model": display_name,
            "model_id": model_id,
            "family": family,
            "status": "failed",
            "error": repr(exc),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/splitzip/requested_exponent_stats.json")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--input-device", default="cuda")
    parser.add_argument("--cache-root", default="~/.cache/huggingface/hub")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--models", nargs="*", default=None,
                        help="Optional display-name filter, e.g. Qwen3-32B Phi-2")
    args = parser.parse_args()

    selected = REQUESTED_MODELS
    if args.models:
        wanted = set(args.models)
        selected = [m for m in REQUESTED_MODELS if m[0] in wanted or m[1] in wanted]

    results = []
    for display_name, model_id, family in selected:
        print(f"==> {display_name} ({model_id})", flush=True)
        result = profile_model(display_name, model_id, family, args)
        print(json.dumps(result, indent=2), flush=True)
        results.append(result)

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
