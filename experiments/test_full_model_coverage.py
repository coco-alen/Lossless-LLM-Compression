"""
Analyze what fraction of model parameters are covered by current compression,
and how much more could be gained by compressing ALL parameters.
"""

import torch
import numpy as np
import constriction
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoConfig

WEIGHT_TYPES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def ans16_ratio(W):
    """Compute ANS-16bit compression ratio for int16 array."""
    n = len(W)
    vals, counts = np.unique(W, return_counts=True)
    probs = (counts / n).astype(np.float32)
    mapping = np.zeros(65536, dtype=np.int32)
    for i, v in enumerate(vals):
        mapping[int(v) + 32768] = i
    data_idx = mapping[(W.astype(np.int32) + 32768)].astype(np.int32)
    model = constriction.stream.model.Categorical(probs, perfect=False)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(data_idx, model)
    compressed = encoder.get_compressed()
    comp_bytes = len(compressed) * 4 + len(vals) * 6
    return comp_bytes / (n * 2) * 100


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    num_layers = config.num_hidden_layers
    print(f"Model: {args.model_name_or_path}  ({num_layers} layers)")
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Catalog ALL parameters
    covered_params = 0
    uncovered_params = {}
    total_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n

        is_covered = False
        for wt in WEIGHT_TYPES:
            if wt in name and "model.layers" in name:
                is_covered = True
                break

        if is_covered:
            covered_params += n
        else:
            # Group by parameter type
            # Strip layer numbers for grouping
            import re
            clean_name = re.sub(r'model\.layers\.\d+\.', 'model.layers.X.', name)
            if clean_name not in uncovered_params:
                uncovered_params[clean_name] = {"count": 0, "numel": 0, "dtype": str(param.dtype)}
            uncovered_params[clean_name]["count"] += 1
            uncovered_params[clean_name]["numel"] += n

    print(f"\n{'='*80}")
    print("MODEL PARAMETER COVERAGE")
    print(f"{'='*80}")
    print(f"Total parameters:    {total_params:>12,}")
    print(f"Covered (7 types):   {covered_params:>12,}  ({covered_params/total_params*100:.1f}%)")
    print(f"Uncovered:           {total_params-covered_params:>12,}  ({(total_params-covered_params)/total_params*100:.1f}%)")

    print(f"\nUncovered parameters:")
    for name, info in sorted(uncovered_params.items(), key=lambda x: -x[1]["numel"]):
        pct = info["numel"] / total_params * 100
        print(f"  {name:<50} {info['numel']:>12,}  ({pct:.2f}%)  ×{info['count']}  {info['dtype']}")

    # Compress uncovered parameters
    print(f"\n{'='*80}")
    print("COMPRESSION OF UNCOVERED PARAMETERS")
    print(f"{'='*80}")

    uncovered_total_original = 0
    uncovered_total_compressed = 0

    for name, param in model.named_parameters():
        is_covered = False
        for wt in WEIGHT_TYPES:
            if wt in name and "model.layers" in name:
                is_covered = True
                break
        if is_covered:
            continue

        if param.dtype == torch.bfloat16 and param.numel() > 100:
            W = param.data.detach().cpu().contiguous().view(torch.int16).flatten().numpy()
            n = len(W)
            original = n * 2
            uncovered_total_original += original

            # Quick entropy estimate
            vals, counts = np.unique(W, return_counts=True)
            p = counts / n
            h = -np.sum(p * np.log2(p))

            # ANS compression
            ratio = ans16_ratio(W)
            compressed = int(original * ratio / 100)
            uncovered_total_compressed += compressed

            import re
            clean_name = re.sub(r'model\.layers\.\d+\.', 'model.layers.X.', name)
            if n > 10000:  # Only print significant ones
                print(f"  {name:<50} {n:>10,} params  H={h:.4f}  ANS={ratio:.2f}%")

    if uncovered_total_original > 0:
        uc_ratio = uncovered_total_compressed / uncovered_total_original * 100
        print(f"\n  Uncovered total: {uncovered_total_original/1e6:.1f}MB → {uncovered_total_compressed/1e6:.1f}MB ({uc_ratio:.2f}%)")
        savings = (uncovered_total_original - uncovered_total_compressed) / 1e6
        print(f"  Additional savings from compressing uncovered: {savings:.1f}MB")
    else:
        print(f"\n  No compressible uncovered bf16 parameters found.")

    del model


if __name__ == "__main__":
    main()
