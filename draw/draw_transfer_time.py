import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments" / "splitzip" / "thesis_experiment_data.json"
DEFAULT_OUTPUT = ROOT / "draw" / "transfer_time_vs_seq_len"
THEORETICAL_RATIO = 1.316

PANEL_ORDER = [
    ("Llama-3-8B", "CPU-RDMA"),
    ("Llama-3-8B", "RoCE 4x200G"),
    ("Qwen3-30B-A3B", "CPU-RDMA"),
    ("Qwen3-30B-A3B", "RoCE 4x200G"),
]


def load_transfer_rows(path):
    data = json.loads(Path(path).read_text())
    return data["transfer"]


def format_seq_len(value):
    if value >= 1024:
        return f"{value // 1024}K"
    return str(value)


def draw_transfer_time(transfer, output_prefix):
    palette = sns.color_palette()
    native_color = palette[0]
    splitzip_color = palette[2]
    theory_color = palette[3]

    fig, axes = plt.subplots(1, 4, figsize=(21, 4), sharex=True)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.76, bottom=0.25, wspace=0.22)

    panel_series = []
    common_ymax = 0.0
    for model_name, mode_name in PANEL_ORDER:
        rows = transfer[model_name]["modes"][mode_name]
        native = np.array([row["native_ms"] for row in rows])
        splitzip = np.array([row["splitzip_ms"] for row in rows])
        theoretical = native / THEORETICAL_RATIO
        panel_series.append((model_name, mode_name, rows, native, splitzip, theoretical))
        common_ymax = max(common_ymax, native.max(), splitzip.max(), theoretical.max())
    common_ymax *= 1.08

    for ax, (model_name, mode_name, rows, native, splitzip, theoretical) in zip(axes, panel_series):
        seq_len = np.array([row["seq_len"] for row in rows])

        ax.plot(
            seq_len,
            native,
            color=native_color,
            alpha=1,
            linestyle="-",
            linewidth=3,
            marker="o",
            markersize=9,
            label="Native (ms)",
        )
        ax.plot(
            seq_len,
            splitzip,
            color=splitzip_color,
            alpha=1,
            linestyle="-",
            linewidth=3,
            marker="v",
            markersize=9,
            label="SplitZip (ms)",
        )
        ax.plot(
            seq_len,
            theoretical,
            color=theory_color,
            alpha=1,
            linestyle="--",
            linewidth=3,
            label="Theoretical Optimum",
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_len)
        ax.set_xticklabels([format_seq_len(x) for x in seq_len], rotation=30, fontsize=12)
        ax.set_xlabel("Seq Len", fontsize=15, fontweight="bold")
        ax.set_ylabel("Time (ms)", fontsize=15, fontweight="bold")
        ax.set_title(f"{model_name} / {mode_name}", fontsize=13, fontweight="bold", pad=8)
        ax.set_ylim(0, common_ymax)
        ax.grid(True, linewidth=0.5)
        ax.tick_params(axis="y", labelsize=12, width=0)
        ax.tick_params(axis="x", width=0)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(1.6)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.0),
        prop={"size": 14},
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
        markerscale=1.25,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold" if text.get_text() == "SplitZip (ms)" else "normal")

    output_prefix = Path(output_prefix)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    transfer = load_transfer_rows(args.input)
    draw_transfer_time(transfer, args.output)
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
