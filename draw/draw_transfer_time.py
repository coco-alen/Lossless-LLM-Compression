import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [ROOT / "experiments" / "splitzip_v2" / "results" / "mooncake_kv_sweep.json"]
DEFAULT_OUTPUT = ROOT / "draw" / "transfer_time_vs_seq_len"

SWEEP_ORDER = [
    ("bs1_seq", "BS=1, Seq Sweep", "Seq Len", "seq_len"),
    ("bs16_seq", "BS=16, Seq Sweep", "Seq Len", "seq_len"),
    ("seq1024_bs", "Seq=1024, BS Sweep", "Batch Size", "batch_size"),
    ("seq32768_bs", "Seq=32768, BS Sweep", "Batch Size", "batch_size"),
]

MODEL_ORDER = ["Llama-3-8B", "Qwen3-30B-A3B"]

AXIS_LABEL_FONT_SIZE = 40
TICK_LABEL_FONT_SIZE = 40
TITLE_FONT_SIZE = 35
ROW_LABEL_FONT_SIZE = 45
LEGEND_FONT_SIZE = 45
LINE_WIDTH = 10
MARKER_SIZE = 28


def format_power2(value):
    value = int(value)
    if value >= 1024:
        return f"{value // 1024}K"
    return str(value)


def load_mooncake_results(paths):
    results = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        for item in data["results"]:
            name = item["model"]["display_name"]
            results[name] = item
    return results


def row_value(row, stat):
    native = row["native_mooncake"][f"{stat}_ms"]
    splitzip = row["splitzip_mooncake"]["transfer"][f"{stat}_ms"]
    return native, splitzip


def collect_panel(model_result, sweep_name, x_key, stat):
    rows = [r for r in model_result["rows"] if r["sweep"] == sweep_name]
    rows = sorted(rows, key=lambda r: int(r[x_key]))
    x = np.array([int(r[x_key]) for r in rows])
    native = np.array([row_value(r, stat)[0] for r in rows], dtype=float)
    splitzip = np.array([row_value(r, stat)[1] for r in rows], dtype=float)
    speedup = native / splitzip
    return rows, x, native, splitzip, speedup


def draw_transfer_time(results, output_prefix, stat):
    palette = sns.color_palette()
    native_color = palette[0]
    splitzip_color = palette[2]

    model_names = [name for name in MODEL_ORDER if name in results]
    missing = [name for name in MODEL_ORDER if name not in results]
    if missing:
        raise ValueError(f"Missing Mooncake results for: {missing}")

    fig, axes = plt.subplots(
        len(model_names),
        len(SWEEP_ORDER),
        figsize=(42, 20),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.86, bottom=0.16, hspace=0.55, wspace=0.35)

    for row_idx, model_name in enumerate(model_names):
        model_result = results[model_name]
        for col_idx, (sweep_name, title, xlabel, x_key) in enumerate(SWEEP_ORDER):
            ax = axes[row_idx][col_idx]
            _, x, native, splitzip, speedup = collect_panel(model_result, sweep_name, x_key, stat)

            ax.plot(
                x,
                native,
                color=native_color,
                linestyle="-",
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
                label="Native",
            )
            ax.plot(
                x,
                splitzip,
                color=splitzip_color,
                linestyle="-",
                linewidth=LINE_WIDTH,
                marker="v",
                markersize=MARKER_SIZE,
                label="SplitZip",
            )

            ax.set_xscale("log", base=2)
            ax.set_xticks(x)
            ax.set_xticklabels([format_power2(v) for v in x], rotation=30, ha="right", fontsize=TICK_LABEL_FONT_SIZE)
            ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE, width=0)
            ax.tick_params(axis="x", width=0)
            ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Mooncake Transfer (ms)", fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")
            ax.set_title(title, fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=10)

            ymax = max(float(native.max()), float(splitzip.max()))
            ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1)
            ax.grid(True, linewidth=0.55, alpha=0.7)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_linewidth(1.4)

            last_speedup = speedup[-1]
            ax.text(
                0.98,
                0.92,
                f"{last_speedup:.2f}x",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=TICK_LABEL_FONT_SIZE,
                fontweight="bold",
            )

        y_center = 0.69 if row_idx == 0 else 0.31
        fig.text(
            0.018,
            y_center,
            model_name,
            rotation=90,
            ha="center",
            va="center",
            fontsize=ROW_LABEL_FONT_SIZE,
            fontweight="bold",
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.54, 0.965),
        prop={"size": LEGEND_FONT_SIZE},
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    for text in legend.get_texts():
        if text.get_text() == "SplitZip":
            text.set_fontweight("bold")

    output_prefix = Path(output_prefix)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stat", choices=["median", "mean"], default="median")
    args = parser.parse_args()

    results = load_mooncake_results(args.input)
    draw_transfer_time(results, args.output, args.stat)
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
