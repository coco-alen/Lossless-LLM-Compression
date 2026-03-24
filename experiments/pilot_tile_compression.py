"""
Pilot experiment: Tiled Block Compression for LLM Weights
=========================================================
Compress weight matrices in tiles matching GPU Tensor Core fragment sizes.
Within each tile, share an exponent base and store per-element offsets.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from collections import defaultdict
import math
import time

TILE_SIZES = [(8, 8), (16, 8), (16, 16), (32, 16)]

def extract_exponents_bf16(tensor_bf16):
    """Extract 8-bit exponent from BF16 tensor."""
    raw = tensor_bf16.view(torch.int16)
    exponents = ((raw >> 7) & 0xFF).to(torch.int32)
    return exponents

def analyze_tiles(weight_2d, tile_h, tile_w):
    """Analyze exponent statistics for tiles of given size on a 2D weight matrix."""
    H, W = weight_2d.shape
    # Pad if needed
    pad_h = (tile_h - H % tile_h) % tile_h
    pad_w = (tile_w - W % tile_w) % tile_w
    if pad_h > 0 or pad_w > 0:
        weight_padded = torch.nn.functional.pad(weight_2d, (0, pad_w, 0, pad_h), value=0.0)
    else:
        weight_padded = weight_2d

    Hp, Wp = weight_padded.shape
    exponents = extract_exponents_bf16(weight_padded)

    # Reshape into tiles: (num_tiles_h, tile_h, num_tiles_w, tile_w)
    tiles = exponents.reshape(Hp // tile_h, tile_h, Wp // tile_w, tile_w)
    tiles = tiles.permute(0, 2, 1, 3).reshape(-1, tile_h, tile_w)  # (num_tiles, tile_h, tile_w)

    num_tiles = tiles.shape[0]
    N = tile_h * tile_w  # elements per tile

    flat_tiles = tiles.reshape(num_tiles, N)

    # Exponent range per tile
    tile_max = flat_tiles.max(dim=1).values
    tile_min = flat_tiles.min(dim=1).values
    ranges = tile_max - tile_min  # (num_tiles,)

    mean_range = ranges.float().mean().item()
    pct_le3 = (ranges <= 3).float().mean().item() * 100
    pct_le7 = (ranges <= 7).float().mean().item() * 100
    pct_le15 = (ranges <= 15).float().mean().item() * 100
    pct_gt15 = (ranges > 15).float().mean().item() * 100

    # Theoretical compression ratio for block floating point
    # For each tile: 1 base_exponent (8 bits) + N * (offset_bits + 7 mantissa + 1 sign)
    # vs uncompressed: N * 16 bits
    # offset_bits depends on actual range of that tile
    ranges_np = ranges.numpy().astype(np.float64)

    # Bits needed per offset = ceil(log2(range + 1)), minimum 0 (if range=0)
    offset_bits = np.where(ranges_np == 0, 0, np.ceil(np.log2(ranges_np + 1)))

    # Per tile compressed bits = 8 (base) + N * (offset_bits + 7 + 1)
    compressed_bits_per_tile = 8.0 + N * (offset_bits + 8.0)
    original_bits_per_tile = N * 16.0

    total_compressed = compressed_bits_per_tile.sum()
    total_original = num_tiles * original_bits_per_tile

    compression_ratio = total_compressed / total_original

    # Average bits per weight
    avg_bpw = total_compressed / (num_tiles * N)

    # Also compute with capped offset bits (fallback for bad tiles)
    # If range > 15, use 4-bit offset and accept that some values need more
    # Actually for "fallback", store tile uncompressed
    offset_bits_capped = offset_bits.copy()
    fallback_mask = ranges_np > 15
    compressed_bits_fallback = compressed_bits_per_tile.copy()
    compressed_bits_fallback[fallback_mask] = original_bits_per_tile  # fallback to raw

    total_compressed_fallback = compressed_bits_fallback.sum()
    compression_ratio_fallback = total_compressed_fallback / total_original

    return {
        'mean_range': mean_range,
        'pct_le3': pct_le3,
        'pct_le7': pct_le7,
        'pct_le15': pct_le15,
        'pct_gt15': pct_gt15,
        'compression_ratio': compression_ratio,
        'compression_ratio_fallback': compression_ratio_fallback,
        'avg_bpw': avg_bpw,
        'num_tiles': num_tiles,
        'original_elements': H * W,
    }


def analyze_tiles_bimodal(weight_2d, tile_h, tile_w):
    """
    Bimodal variant: allow 2 shared exponents per tile.
    Split tile into 2 groups by k-means on exponents (1D, so just split at median).
    Cost: 2 * 8 bits (bases) + 1 bit selector per value + N * (offset_bits + 8)
    """
    H, W = weight_2d.shape
    pad_h = (tile_h - H % tile_h) % tile_h
    pad_w = (tile_w - W % tile_w) % tile_w
    if pad_h > 0 or pad_w > 0:
        weight_padded = torch.nn.functional.pad(weight_2d, (0, pad_w, 0, pad_h), value=0.0)
    else:
        weight_padded = weight_2d

    Hp, Wp = weight_padded.shape
    exponents = extract_exponents_bf16(weight_padded)

    tiles = exponents.reshape(Hp // tile_h, tile_h, Wp // tile_w, tile_w)
    tiles = tiles.permute(0, 2, 1, 3).reshape(-1, tile_h, tile_w)
    num_tiles = tiles.shape[0]
    N = tile_h * tile_w

    flat_tiles = tiles.reshape(num_tiles, N)  # (num_tiles, N)

    # Split each tile into 2 groups by median exponent
    medians = flat_tiles.float().median(dim=1).values.unsqueeze(1)  # (num_tiles, 1)
    group_high = flat_tiles.float() >= medians  # (num_tiles, N)

    # Compute range for each group
    INF = 999
    high_vals = flat_tiles.clone().float()
    low_vals = flat_tiles.clone().float()
    high_vals[~group_high] = -INF
    low_vals[group_high] = INF

    # Group 1 (high): range
    g1_max = high_vals.max(dim=1).values
    g1_min_vals = flat_tiles.clone().float()
    g1_min_vals[~group_high] = INF
    g1_min = g1_min_vals.min(dim=1).values
    g1_range = (g1_max - g1_min).clamp(min=0)

    # Group 0 (low): range
    g0_max_vals = flat_tiles.clone().float()
    g0_max_vals[group_high] = -INF
    g0_max = g0_max_vals.max(dim=1).values
    g0_min = low_vals.min(dim=1).values
    g0_range = (g0_max - g0_min).clamp(min=0)

    # Max of the two group ranges determines offset bits
    max_group_range = torch.maximum(g1_range, g0_range).numpy().astype(np.float64)

    offset_bits = np.where(max_group_range == 0, 0, np.ceil(np.log2(max_group_range + 1)))

    # Cost: 2 bases (16 bits) + N * (1 selector + offset_bits + 7 mantissa + 1 sign)
    compressed_bits_per_tile = 16.0 + N * (1.0 + offset_bits + 8.0)
    original_bits_per_tile = N * 16.0

    total_compressed = compressed_bits_per_tile.sum()
    total_original = num_tiles * original_bits_per_tile
    compression_ratio = total_compressed / total_original

    mean_max_group_range = max_group_range.mean()

    return {
        'compression_ratio_bimodal': compression_ratio,
        'mean_max_group_range': mean_max_group_range,
    }


def main():
    print("=" * 80)
    print("Pilot Experiment: Tiled Block Compression for LLM Weights")
    print("=" * 80)

    print("\nLoading Qwen3-0.6B in BF16...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Collect all 2D weight matrices
    weight_info = []
    total_params = 0
    skipped_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.ndim == 2 and param.dtype == torch.bfloat16:
            weight_info.append((name, param.data))
        else:
            skipped_params += param.numel()

    print(f"\nTotal parameters: {total_params:,}")
    print(f"2D BF16 weights: {total_params - skipped_params:,} ({(total_params - skipped_params) / total_params * 100:.1f}%)")
    print(f"Skipped (1D bias, etc.): {skipped_params:,} ({skipped_params / total_params * 100:.1f}%)")
    print(f"Number of 2D weight matrices: {len(weight_info)}")

    # Aggregate results per tile size
    results = {ts: defaultdict(float) for ts in TILE_SIZES}
    results_bimodal = {ts: defaultdict(float) for ts in TILE_SIZES}
    total_elements = sum(w.numel() for _, w in weight_info)

    for tile_size in TILE_SIZES:
        tile_h, tile_w = tile_size
        print(f"\n--- Analyzing tile size {tile_h}x{tile_w} ---")

        weighted_stats = defaultdict(float)
        total_tiles = 0

        for name, weight in weight_info:
            r = analyze_tiles(weight, tile_h, tile_w)
            rb = analyze_tiles_bimodal(weight, tile_h, tile_w)
            n = weight.numel()
            frac = n / total_elements

            weighted_stats['mean_range'] += r['mean_range'] * frac
            weighted_stats['pct_le3'] += r['pct_le3'] * frac
            weighted_stats['pct_le7'] += r['pct_le7'] * frac
            weighted_stats['pct_le15'] += r['pct_le15'] * frac
            weighted_stats['pct_gt15'] += r['pct_gt15'] * frac
            weighted_stats['compression_ratio'] += r['compression_ratio'] * frac
            weighted_stats['compression_ratio_fallback'] += r['compression_ratio_fallback'] * frac
            weighted_stats['avg_bpw'] += r['avg_bpw'] * frac
            weighted_stats['bimodal_ratio'] += rb['compression_ratio_bimodal'] * frac
            weighted_stats['bimodal_range'] += rb['mean_max_group_range'] * frac
            total_tiles += r['num_tiles']

        results[tile_size] = weighted_stats
        results[tile_size]['total_tiles'] = total_tiles

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Tiled Block Compression Results (Qwen3-0.6B)")
    print("=" * 80)

    print(f"\nTotal 2D weight elements: {total_elements:,}")

    print("\n{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Tile Size", "Mean Range", "<=3 (%)", "<=7 (%)", "<=15 (%)", ">15 (%)"))
    print("-" * 62)
    for ts in TILE_SIZES:
        s = results[ts]
        print("{:<12} {:>10.2f} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}".format(
            f"{ts[0]}x{ts[1]}",
            s['mean_range'], s['pct_le3'], s['pct_le7'], s['pct_le15'], s['pct_gt15']))

    print("\n\nCompression Ratios:")
    print("{:<12} {:>14} {:>14} {:>14} {:>10}".format(
        "Tile Size", "BlockFP (%)", "Fallback (%)", "Bimodal (%)", "Avg BPW"))
    print("-" * 68)
    for ts in TILE_SIZES:
        s = results[ts]
        print("{:<12} {:>14.2f} {:>14.2f} {:>14.2f} {:>10.2f}".format(
            f"{ts[0]}x{ts[1]}",
            s['compression_ratio'] * 100,
            s['compression_ratio_fallback'] * 100,
            s['bimodal_ratio'] * 100,
            s['avg_bpw']))

    print("\n\nComparison with existing methods:")
    print(f"  ANS-16bit:  65.96% (10.55 bpw)")
    print(f"  DFloat11:   ~66.6% (10.66 bpw)")
    for ts in TILE_SIZES:
        s = results[ts]
        print(f"  BlockFP {ts[0]}x{ts[1]:>2}: {s['compression_ratio']*100:.2f}% ({s['avg_bpw']:.2f} bpw)"
              f"  | fallback: {s['compression_ratio_fallback']*100:.2f}%"
              f"  | bimodal: {s['bimodal_ratio']*100:.2f}%")

    print("\n\nBimodal (2-base) analysis:")
    print("{:<12} {:>18} {:>18}".format(
        "Tile Size", "Mean Group Range", "Bimodal Ratio (%)"))
    print("-" * 50)
    for ts in TILE_SIZES:
        s = results[ts]
        print("{:<12} {:>18.2f} {:>18.2f}".format(
            f"{ts[0]}x{ts[1]}",
            s['bimodal_range'],
            s['bimodal_ratio'] * 100))

    # Detailed per-layer analysis for the best tile size
    print("\n\n--- Per-layer detail for 16x8 tiles ---")
    tile_h, tile_w = 16, 8
    print(f"{'Layer':<50} {'Range':>6} {'<=7%':>6} {'Ratio%':>8}")
    print("-" * 74)
    for name, weight in weight_info:
        r = analyze_tiles(weight, tile_h, tile_w)
        if weight.numel() > 100000:  # only show large layers
            print(f"{name:<50} {r['mean_range']:>6.1f} {r['pct_le7']:>6.1f} {r['compression_ratio']*100:>8.2f}")

    print("\n\nConclusion Notes:")
    best_ts = min(TILE_SIZES, key=lambda ts: results[ts]['compression_ratio'])
    best_r = results[best_ts]['compression_ratio'] * 100
    best_bi = min(results[ts]['bimodal_ratio'] for ts in TILE_SIZES) * 100
    print(f"  Best single-base block FP: {best_ts[0]}x{best_ts[1]} at {best_r:.2f}%")
    print(f"  Best bimodal block FP: {best_bi:.2f}%")
    print(f"  ANS-16bit baseline: 65.96%")
    if best_r > 65.96:
        print(f"  -> Block FP is WORSE than ANS-16bit by {best_r - 65.96:.2f}pp")
    else:
        print(f"  -> Block FP is BETTER than ANS-16bit by {65.96 - best_r:.2f}pp")
    if best_bi > 65.96:
        print(f"  -> Bimodal block FP is WORSE than ANS-16bit by {best_bi - 65.96:.2f}pp")
    else:
        print(f"  -> Bimodal block FP is BETTER than ANS-16bit by {65.96 - best_bi:.2f}pp")


if __name__ == "__main__":
    main()
