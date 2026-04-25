import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns


sns.set_palette("colorblind")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "experiments" / "splitzip" / "thesis_experiment_data.json"
DEFAULT_OUTPUT = ROOT / "draw" / "baseline_comparison"

METHOD_ORDER = ["nvcomp_lz4", "dfloat11", "zipserv", "zipnn", "splitzip"]
METHOD_LABELS = {
    "nvcomp_lz4": "nvCOMP\nLZ4",
    "dfloat11": "DFloat11",
    "zipserv": "ZipServ",
    "splitzip": "SplitZip",
    "zipnn": "ZipNN",
}
PANEL_TITLES = [
    "(a) Compression Ratio",
    "(b) Compression Throughput",
    "(c) Decompression Throughput",
]


def load_rows(path):
    data = json.loads(Path(path).read_text())
    rows = {row["method"]: row for row in data["baseline"]["rows"]}
    return [rows[name] for name in METHOD_ORDER]


def add_panel_title_below(fig, ax, title, y=0.08):
    bbox = ax.get_position()
    x_center = 0.5 * (bbox.x0 + bbox.x1)
    fig.text(x_center, y, title, ha="center", va="top", fontsize=16, fontweight="bold")


def annotate_bars(ax, bars, values, fmt, offset_ratio=0.015):
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height() + span * offset_ratio, ymin + span * 0.03),
            fmt(value),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )


def draw_baseline_comparison(rows, output_prefix):
    labels = [METHOD_LABELS[row["method"]] for row in rows]
    ratio = [row["ratio"] for row in rows]
    encode = [row["encode_gbs"] for row in rows]
    decode = [row["decode_gbs"] for row in rows]

    x = np.arange(len(rows))
    palette = sns.color_palette()
    style_map = {
        "nvcomp_lz4": {"color": palette[1], "hatch": "/"},
        "dfloat11": {"color": palette[0], "hatch": "\\"},
        "zipserv": {"color": palette[3], "hatch": "-"},
        "zipnn": {"color": palette[4], "hatch": "+"},
        "splitzip": {"color": palette[2], "hatch": "x"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(21, 4))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.92, bottom=0.18, wspace=0.25)

    panels = [
        {
            "ax": axes[0],
            "values": ratio,
            "ylabel": "Compression Ratio (x)",
            "formatter": lambda v: f"{v:.3f}",
            "ylim_pad": 0.10,
        },
        {
            "ax": axes[1],
            "values": encode,
            "ylabel": "Throughput (GB/s)",
            "formatter": lambda v: f"{v:.3g}" if v < 1 else f"{v:.1f}",
            "ylim_pad": 0.12,
        },
        {
            "ax": axes[2],
            "values": decode,
            "ylabel": "Throughput (GB/s)",
            "formatter": lambda v: f"{v:.3g}" if v < 1 else f"{v:.1f}",
            "ylim_pad": 0.12,
        },
    ]

    for idx, panel in enumerate(panels):
        ax = panel["ax"]
        values = panel["values"]
        bars = []
        for i, (row, value) in enumerate(zip(rows, values)):
            style = style_map[row["method"]]
            bars.append(
                ax.bar(
                    x[i],
                    value,
                    width=0.72,
                    color=style["color"],
                    edgecolor="white",
                    hatch=style["hatch"],
                    linewidth=1.0,
                )[0]
            )

        ax.set_xticks(x, labels, fontsize=14)
        ax.tick_params(axis="y", labelsize=13, width=0)
        ax.set_ylabel(panel["ylabel"], fontsize=15, fontweight="bold")
        ax.grid(axis="y", linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        ymax = max(values)
        ax.set_ylim(0, ymax * (1 + panel["ylim_pad"]))
        if idx == 0:
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
        else:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y:g}"))

        for spine in ["top", "right", "bottom", "left"]:
            ax.spines[spine].set_linewidth(2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for tick_label, row in zip(ax.get_xticklabels(), rows):
            if row["method"] == "splitzip":
                tick_label.set_fontweight("bold")

        annotate_bars(
            ax,
            bars,
            values,
            panel["formatter"],
            offset_ratio=0.02 if idx == 0 else 0.015,
        )
        add_panel_title_below(fig, ax, PANEL_TITLES[idx], y=0.042)

    output_prefix = Path(output_prefix)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_rows(args.input)
    draw_baseline_comparison(rows, args.output)
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
