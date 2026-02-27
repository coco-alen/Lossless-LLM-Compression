"""
Analyze cross-layer weight distributions for Qwen/Qwen3-1.7B.

For each weight type (q_proj, k_proj, ...):
  1. Compute element-wise mean weight across ALL layers: mean = avg(W[0], W[1], ..., W[L-1])
  2. For each layer l, compute delta[l] = W[l] - mean
  3. Plot 3D figure:
     - Left:  each layer's original weight value distribution (waterfall)
     - Right: each layer's delta-from-mean distribution (waterfall)

Usage:
    python analyze_cross_layer.py
    python analyze_cross_layer.py --model_name_or_path Qwen/Qwen3-1.7B --output_dir ./figures
"""

import os
from argparse import ArgumentParser

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from transformers import AutoModelForCausalLM, AutoConfig
from tqdm import tqdm

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
    """Extract weight tensors grouped by type, each as a flat float32 tensor."""
    groups = {wt: [] for wt in WEIGHT_TYPES}
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for wt in WEIGHT_TYPES:
            parts = wt.split('.')
            module = layer
            for p in parts:
                module = getattr(module, p)
            groups[wt].append(module.weight.data.detach().cpu().float().flatten())
    return groups


def plot_weight_type(wt_name, weight_list, output_dir):
    """
    For one weight type, create a figure with two 3D waterfall plots.

    Left:  original weight distribution per layer
    Right: delta-from-mean distribution per layer

    The element-wise mean across all layers is computed first, then
    delta[l] = W[l] - mean for each layer l.
    """
    num_layers = len(weight_list)

    # --- Step 1: compute element-wise mean across all layers ---
    stacked = torch.stack(weight_list)       # (num_layers, num_elements)
    elem_mean = stacked.mean(dim=0)          # (num_elements,)

    # --- Step 2: compute per-layer delta ---
    deltas = [weight_list[l] - elem_mean for l in range(num_layers)]

    # --- Step 3: build per-layer histograms ---
    n_bins = 300

    # Bin edges (shared across layers for a consistent X axis)
    all_orig = torch.cat(weight_list).numpy()
    all_delta = torch.cat(deltas).numpy()

    orig_lo, orig_hi = np.percentile(all_orig, [0.1, 99.9])
    delta_lo, delta_hi = np.percentile(all_delta, [0.1, 99.9])

    orig_bins = np.linspace(orig_lo, orig_hi, n_bins + 1)
    delta_bins = np.linspace(delta_lo, delta_hi, n_bins + 1)

    orig_centers = (orig_bins[:-1] + orig_bins[1:]) / 2
    delta_centers = (delta_bins[:-1] + delta_bins[1:]) / 2

    orig_hists = np.zeros((num_layers, n_bins))
    delta_hists = np.zeros((num_layers, n_bins))

    for l in range(num_layers):
        h_o, _ = np.histogram(weight_list[l].numpy(), bins=orig_bins, density=True)
        h_d, _ = np.histogram(deltas[l].numpy(), bins=delta_bins, density=True)
        orig_hists[l] = h_o
        delta_hists[l] = h_d

    # --- Step 4: plot ---
    colors = cm.tab20(np.linspace(0, 1, num_layers))

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle(f'{wt_name}   (element-wise mean across {num_layers} layers)',
                 fontsize=16, fontweight='bold')

    # ---- Left: original weight distributions ----
    ax1 = fig.add_subplot(121, projection='3d')
    for l in range(num_layers):
        ax1.plot(orig_centers, orig_hists[l],
                 zs=l, zdir='y', color=colors[l % len(colors)],
                 alpha=0.8, linewidth=0.7)
    ax1.set_xlabel('Weight Value', fontsize=11, labelpad=10)
    ax1.set_ylabel('Layer', fontsize=11, labelpad=10)
    ax1.set_zlabel('Density', fontsize=11, labelpad=10)
    ax1.set_title('Original Weight Distribution', fontsize=14, pad=15)
    ax1.set_yticks(np.arange(0, num_layers, max(1, num_layers // 7)))
    ax1.view_init(elev=25, azim=-60)

    # ---- Right: delta-from-mean distributions ----
    ax2 = fig.add_subplot(122, projection='3d')
    for l in range(num_layers):
        ax2.plot(delta_centers, delta_hists[l],
                 zs=l, zdir='y', color=colors[l % len(colors)],
                 alpha=0.8, linewidth=0.7)
    ax2.set_xlabel('Delta (W[l] − mean)', fontsize=11, labelpad=10)
    ax2.set_ylabel('Layer', fontsize=11, labelpad=10)
    ax2.set_zlabel('Density', fontsize=11, labelpad=10)
    ax2.set_title('Delta from Layer-Mean Distribution', fontsize=14, pad=15)
    ax2.set_yticks(np.arange(0, num_layers, max(1, num_layers // 7)))
    ax2.view_init(elev=25, azim=-60)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    save_name = wt_name.replace('.', '_')
    path = os.path.join(output_dir, f'{save_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def entropy_estimate(int16_data):
    """Estimate Shannon entropy (bits) of an int16 tensor."""
    vals, counts = torch.unique(int16_data, return_counts=True)
    probs = counts.float() / counts.sum()
    return -(probs * probs.log2()).sum().item()


def main():
    parser = ArgumentParser("Analyze cross-layer weight distributions")
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--output_dir', type=str, default='./figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    num_layers = config.num_hidden_layers
    print(f"Model: {args.model_name_or_path}  ({num_layers} layers)")
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Extract weights
    print("Extracting weights...")
    groups = extract_weights(model, num_layers)

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # ---- Generate 3D plots ----
    print(f"\nGenerating 3D plots → {args.output_dir}/")
    for wt in tqdm(WEIGHT_TYPES, desc="Plotting"):
        plot_weight_type(wt, groups[wt], args.output_dir)

    # ---- Print entropy statistics ----
    print("\n" + "=" * 72)
    print("Entropy Statistics  (lower = more compressible)")
    print("=" * 72)
    print(f"  {'Weight Type':<23} {'Original H':>12} {'Delta H':>12} {'Reduction':>10}")
    print(f"  {'-'*23} {'-'*12} {'-'*12} {'-'*10}")

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        stacked = torch.stack(tensors)
        elem_mean = stacked.mean(dim=0)

        orig_int16 = torch.cat(tensors).to(torch.bfloat16).view(torch.int16)
        delta_int16 = torch.cat(
            [t - elem_mean for t in tensors]
        ).to(torch.bfloat16).view(torch.int16)

        h_orig = entropy_estimate(orig_int16)
        h_delta = entropy_estimate(delta_int16)
        reduction = (1 - h_delta / h_orig) * 100

        print(f"  {wt:<23} {h_orig:>10.2f} b {h_delta:>10.2f} b {reduction:>8.1f}%")

    # ---- Print std statistics for quick overview ----
    print("\n" + "=" * 72)
    print("Standard Deviation  (delta std << original std ⇒ strong cross-layer correlation)")
    print("=" * 72)
    print(f"  {'Weight Type':<23} {'Orig std':>12} {'Delta std':>12} {'Ratio':>10}")
    print(f"  {'-'*23} {'-'*12} {'-'*12} {'-'*10}")

    for wt in WEIGHT_TYPES:
        tensors = groups[wt]
        stacked = torch.stack(tensors)
        elem_mean = stacked.mean(dim=0)

        orig_std = torch.cat(tensors).std().item()
        delta_std = torch.cat([t - elem_mean for t in tensors]).std().item()

        print(f"  {wt:<23} {orig_std:>12.6f} {delta_std:>12.6f} {delta_std/orig_std:>9.2%}")

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
