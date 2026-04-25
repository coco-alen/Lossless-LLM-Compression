import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments" / "splitzip" / "thesis_experiment_data.json"
DEFAULT_OUTPUT = ROOT / "draw" / "qwen32_transmission_breakdown"
SEQ_LENS = [2048, 16384, 65536]
BAR_HEIGHT = 0.42
Y_NATIVE = 0.62
Y_OURS = 0.16


def load_breakdown_rows(path):
    data = json.loads(Path(path).read_text())
    rows = {row["seq_len"]: row for row in data["breakdown"]["rows"]}
    return [rows[seq_len] for seq_len in SEQ_LENS]


def format_seq_len(value):
    if value >= 1024:
        return f"{value // 1024}K"
    return str(value)


def annotate_total(ax, value, y_pos, x_pad):
    ax.text(
        value + x_pad,
        y_pos,
        f"{value:.1f}",
        va="center",
        ha="left",
        fontsize=12,
        fontweight="bold",
    )


def draw_qwen32_breakdown(rows, output_prefix):
    palette = sns.color_palette()
    plt.rcParams["hatch.linewidth"] = 1.1
    colors = {
        "native": palette[0],
        "encode": palette[1],
        "transfer": palette[2],
        "decode": palette[3],
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 3))
    fig.subplots_adjust(left=0.045, right=0.995, top=0.78, bottom=0.24, wspace=0.24)

    legend_handles = None
    legend_labels = None

    for ax, row in zip(axes, rows):
        native = row["native_transfer_ms"]
        encode = row["splitzip_encode_ms"]
        transfer = row["splitzip_transfer_ms"]
        decode = row["splitzip_decode_ms"]
        ours_total = row["splitzip_total_ms"]
        x_max = max(native, ours_total) * 1.18
        x_pad = x_max * 0.015

        native_bar = ax.barh(
            Y_NATIVE,
            native,
            height=BAR_HEIGHT,
            color=colors["native"],
            edgecolor="white",
            hatch="/",
            label="Native transfer",
        )
        encode_bar = ax.barh(
            Y_OURS,
            encode,
            height=BAR_HEIGHT,
            color=colors["encode"],
            edgecolor="white",
            hatch="\\",
            label="Encode",
        )
        transfer_bar = ax.barh(
            Y_OURS,
            transfer,
            left=encode,
            height=BAR_HEIGHT,
            color=colors["transfer"],
            edgecolor="white",
            hatch="x",
            label="Transfer",
        )
        decode_bar = ax.barh(
            Y_OURS,
            decode,
            left=encode + transfer,
            height=BAR_HEIGHT,
            color=colors["decode"],
            edgecolor="white",
            hatch="-",
            label="Decode",
        )

        annotate_total(ax, native, Y_NATIVE, x_pad)
        annotate_total(ax, ours_total, Y_OURS, x_pad)

        ax.set_yticks([Y_NATIVE, Y_OURS], ["Native", "Ours"], fontsize=13, fontweight="bold")
        ax.set_xlabel("Time (ms)", fontsize=14, fontweight="bold")
        ax.set_title(f"Seq Len = {format_seq_len(row['seq_len'])}", fontsize=15, fontweight="bold", pad=8)
        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.12, 0.9)
        ax.grid(axis="x", linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=12, width=0)
        ax.tick_params(axis="y", width=0)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_linewidth(1.6)

        if legend_handles is None:
            legend_handles = [native_bar[0], encode_bar[0], transfer_bar[0], decode_bar[0]]
            legend_labels = ["Native transfer", "Encode", "Transfer", "Decode"]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        prop={"size": 12},
        frameon=False,
        columnspacing=1.6,
    )

    output_prefix = Path(output_prefix)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_breakdown_rows(args.input)
    draw_qwen32_breakdown(rows, args.output)
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
