"""
Test different weight grouping strategies for ANS-16bit.

Current approach: group same weight type across all layers (e.g., all q_proj)
Alternative: group all types within each layer, or all types across all layers.

The optimal grouping maximizes the data-to-overhead ratio while minimizing
per-group entropy.
"""

import time
from argparse import ArgumentParser

import torch
import numpy as np
import constriction
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


def extract_weights(model, num_layers):
    groups = {wt: [] for wt in WEIGHT_TYPES}
    for idx in range(num_layers):
        layer = model.model.layers[idx]
        for wt in WEIGHT_TYPES:
            parts = wt.split(".")
            mod = layer
            for p in parts:
                mod = getattr(mod, p)
            groups[wt].append(mod.weight.data.detach().cpu().to(torch.bfloat16))
    return groups


def ans16_size(W):
    """ANS-16bit compressed size in bytes for int16 array W."""
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
    return len(compressed) * 4 + len(vals) * 6


def strategy_per_type(groups, num_layers):
    """Current strategy: one group per weight type (across all layers)."""
    total = 0
    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
        total += ans16_size(all_w.numpy())
    return total


def strategy_per_layer(groups, num_layers):
    """One group per transformer layer (all 7 types combined)."""
    total = 0
    for idx in range(num_layers):
        parts = []
        for wt in WEIGHT_TYPES:
            parts.append(groups[wt][idx].contiguous().view(torch.int16).flatten())
        all_w = torch.cat(parts)
        total += ans16_size(all_w.numpy())
    return total


def strategy_global(groups, num_layers):
    """Single global group for all weights."""
    parts = []
    for wt in WEIGHT_TYPES:
        for t in groups[wt]:
            parts.append(t.contiguous().view(torch.int16).flatten())
    all_w = torch.cat(parts)
    return ans16_size(all_w.numpy())


def strategy_attn_mlp_split(groups, num_layers):
    """Two groups: all attention weights + all MLP weights."""
    attn_parts = []
    mlp_parts = []
    for wt in WEIGHT_TYPES:
        for t in groups[wt]:
            w = t.contiguous().view(torch.int16).flatten()
            if "self_attn" in wt:
                attn_parts.append(w)
            else:
                mlp_parts.append(w)

    total = 0
    total += ans16_size(torch.cat(attn_parts).numpy())
    total += ans16_size(torch.cat(mlp_parts).numpy())
    return total


def strategy_per_type_per_layer_group(groups, num_layers, group_size):
    """Group layers into chunks of group_size, separate tables per weight type."""
    total = 0
    for wt in WEIGHT_TYPES:
        for start in range(0, num_layers, group_size):
            end = min(start + group_size, num_layers)
            chunk = groups[wt][start:end]
            all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in chunk])
            total += ans16_size(all_w.numpy())
    return total


def strategy_hybrid_optimal(groups, num_layers):
    """
    Hybrid: for each weight type, choose the best grouping
    (global or per-N-layer) based on data size.
    Small weight types → global table. Large weight types → per-7-layer table.
    """
    total = 0
    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        n = sum(t.numel() for t in tensors)

        # Try global
        all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in tensors])
        global_size = ans16_size(all_w.numpy())

        # Try per-7-layer
        per7_size = 0
        for start in range(0, num_layers, 7):
            end = min(start + 7, num_layers)
            chunk = tensors[start:end]
            all_w = torch.cat([t.contiguous().view(torch.int16).flatten() for t in chunk])
            per7_size += ans16_size(all_w.numpy())

        # Pick better one
        best = min(global_size, per7_size)
        total += best

    return total


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
    print("Extracting weights...", flush=True)
    groups = extract_weights(model, num_layers)

    # Also get embedding
    embed = model.model.embed_tokens.weight.data.detach().cpu().to(torch.bfloat16)
    embed_w = embed.contiguous().view(torch.int16).flatten().numpy()
    embed_original = len(embed_w) * 2

    del model

    # Compute total original size (decoder weights only)
    total_n = sum(sum(t.numel() for t in groups[wt]) for wt in WEIGHT_TYPES)
    total_original = total_n * 2

    print(f"\nDecoder weights: {total_n:,} params ({total_original/1e6:.1f}MB)")
    print(f"Embedding: {len(embed_w):,} params ({embed_original/1e6:.1f}MB)")

    strategies = [
        ("Per-type (current)", lambda: strategy_per_type(groups, num_layers)),
        ("Per-layer (7 types/layer)", lambda: strategy_per_layer(groups, num_layers)),
        ("Global (single table)", lambda: strategy_global(groups, num_layers)),
        ("Attn/MLP split (2 groups)", lambda: strategy_attn_mlp_split(groups, num_layers)),
        ("Per-type, 4-layer groups", lambda: strategy_per_type_per_layer_group(groups, num_layers, 4)),
        ("Per-type, 7-layer groups", lambda: strategy_per_type_per_layer_group(groups, num_layers, 7)),
        ("Per-type, 14-layer groups", lambda: strategy_per_type_per_layer_group(groups, num_layers, 14)),
        ("Hybrid optimal", lambda: strategy_hybrid_optimal(groups, num_layers)),
    ]

    print(f"\n{'='*80}")
    print("GROUPING STRATEGY COMPARISON (decoder weights)")
    print(f"{'='*80}")

    baseline = None
    for name, fn in strategies:
        t0 = time.time()
        size = fn()
        t1 = time.time()
        ratio = size / total_original * 100
        if baseline is None:
            baseline = size
            print(f"  {name:<32} {ratio:.3f}%  ({size/1e6:.1f}MB)  [{t1-t0:.1f}s]")
        else:
            delta = ratio - baseline / total_original * 100
            marker = " ***" if delta < -0.01 else ""
            print(f"  {name:<32} {ratio:.3f}%  delta={delta:+.3f}%{marker}  [{t1-t0:.1f}s]")

    # Embedding compression
    t0 = time.time()
    embed_size = ans16_size(embed_w)
    t1 = time.time()
    embed_ratio = embed_size / embed_original * 100
    print(f"\n  Embedding ANS-16bit:           {embed_ratio:.3f}%  ({embed_size/1e6:.1f}MB)  [{t1-t0:.1f}s]")

    # Total model compression (decoder + embedding)
    total_model_original = total_original + embed_original
    total_model_compressed = baseline + embed_size
    total_ratio = total_model_compressed / total_model_original * 100
    print(f"\n  Full model (decoder+embed):    {total_ratio:.3f}%")
    print(f"  Original: {total_model_original/1e6:.1f}MB  Compressed: {total_model_compressed/1e6:.1f}MB")
    print(f"  Savings:  {(total_model_original - total_model_compressed)/1e6:.1f}MB")


if __name__ == "__main__":
    main()
