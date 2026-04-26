import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments" / "splitzip" / "thesis_additional_experiments.json"
DEFAULT_OUTPUT = ROOT / "draw" / "fp8_e2e_speedup_vs_seq_len"

AXIS_LABEL_FONT_SIZE = 45
TICK_LABEL_FONT_SIZE = 45
TITLE_FONT_SIZE = 35
LEGEND_FONT_SIZE = 50
LINE_WIDTH = 18
MARKER_SIZE = 40

SCHEME_ORDER = ["e4m3_top8_exact", "e5m2_top8_exact", "e5m2_top16_exact"]
SCHEME_LABELS = {
    "e4m3_top8_exact": "E4M3 Top-8",
    "e5m2_top8_exact": "E5M2 Top-8",
    "e5m2_top16_exact": "E5M2 Top-16",
}
SCHEME_MARKERS = {
    "e4m3_top8_exact": "o",
    "e5m2_top8_exact": "v",
    "e5m2_top16_exact": "s",
}


def load_fp8_transfer(path):
    data = json.loads(Path(path).read_text())
    return data["fp8_transfer"]


def format_seq_len(value):
    if value >= 1024:
        return f"{value // 1024}K"
    return str(value)


def draw_fp8_speedup(fp8_transfer, output_prefix):
    palette = sns.color_palette()
    colors = {
        "e4m3_top8_exact": palette[0],
        "e5m2_top8_exact": palette[2],
        "e5m2_top16_exact": palette[4],
    }

    fig, ax = plt.subplots(1, 1, figsize=(30, 24))
    fig.subplots_adjust(left=0.12, right=0.985, top=0.72, bottom=0.28)

    all_speedups = []
    seq_len = None
    for scheme in SCHEME_ORDER:
        rows = fp8_transfer["rows"][scheme]
        seq_len = np.array([row["seq_len"] for row in rows])
        speedup = np.array([row["speedup"] for row in rows])
        all_speedups.extend(speedup.tolist())

        ax.plot(
            seq_len,
            speedup,
            color=colors[scheme],
            alpha=1,
            linestyle="-",
            linewidth=LINE_WIDTH,
            marker=SCHEME_MARKERS[scheme],
            markersize=MARKER_SIZE,
            label=SCHEME_LABELS[scheme],
        )

    ax.axhline(
        1.0,
        color=palette[3],
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="Native FP8",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(seq_len)
    ax.set_xticklabels(
        [format_seq_len(x) for x in seq_len],
        rotation=30,
        fontsize=TICK_LABEL_FONT_SIZE,
    )
    ax.set_xlabel("Seq Len", fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")
    ax.set_ylabel("E2E Speedup (x)", fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold")

    y_min = min(all_speedups + [1.0])
    y_max = max(all_speedups + [1.0])
    y_pad = max(0.06, (y_max - y_min) * 0.18)
    ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)
    ax.grid(True, linewidth=0.5)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT_SIZE, width=0)
    ax.tick_params(axis="x", width=0)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(1.6)

    legend = fig.legend(
        *ax.get_legend_handles_labels(),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.70),
        prop={"size": LEGEND_FONT_SIZE},
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
        markerscale=1.0,
    )
    for text in legend.get_texts():
        if text.get_text() == "E5M2 Top-8":
            text.set_fontweight("bold")

    output_prefix = Path(output_prefix)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    fp8_transfer = load_fp8_transfer(args.input)
    draw_fp8_speedup(fp8_transfer, args.output)
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
