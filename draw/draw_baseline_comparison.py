import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRENT = ROOT / "experiments" / "splitzip_v2" / "results" / "current_baseline_summary_chunk1024.json"
DEFAULT_REPEATS = ROOT / "experiments" / "splitzip" / "baseline_throughput_repeats.json"
DEFAULT_ADDITIONAL = ROOT / "experiments" / "splitzip_v2" / "results" / "additional_baselines_llama3_8b.json"
DEFAULT_ZIPSERV_ENC = ROOT / "experiments" / "splitzip_v2" / "results" / "zipserv_encode_bench.json"
DEFAULT_SPLITZIP = ROOT / "experiments" / "splitzip_v2" / "results" / "paper_rerun_v2_qwen32_chunk1024.json"
DEFAULT_SPLITZIP_KERNEL = (
    ROOT
    / "experiments"
    / "splitzip_v2"
    / "results"
    / "splitzip_kernel_stage_repeats_chunk1024_idle_cuda1.json"
)
DEFAULT_FALCON = ROOT / "experiments" / "splitzip_v2" / "results" / "falcon_baseline_qwen32.json"
DEFAULT_JSON = ROOT / "experiments" / "splitzip_v2" / "results" / "baseline_table_current.json"
DEFAULT_MD = ROOT / "experiments" / "splitzip_v2" / "results" / "baseline_table_current.md"
DEFAULT_TEX = ROOT / "lossless-paper" / "tables" / "baseline_comparison.tex"

METHOD_ORDER = [
    "dietgpu",
    "nvcomp_lz4",
    "nvcomp_cascaded",
    "nvcomp_bitcomp",
    "dfloat11",
    "zipnn",
    "zipserv",
    "zipserv_tca_tbe",
    "falcon_fp32_cast_bf16_equiv",
    "splitzip_v2",
]


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def stat_from_values(values, center=None):
    values = [float(v) for v in values if v is not None]
    if not values:
        return None, None, None
    if center is None:
        center = sum(values) / len(values)
    return center, center - min(values), max(values) - center


def stat_from_summary(summary, key):
    if summary is None:
        return None, None, None
    if isinstance(summary.get(key), dict):
        item = summary[key]
        mean = item.get("mean")
        if "neg" in item or "pos" in item:
            return mean, item.get("neg"), item.get("pos")
        if "min" in item and "max" in item and mean is not None:
            return mean, mean - item["min"], item["max"] - mean
        stderr = item.get("stderr")
        return mean, stderr, stderr
    mean = summary.get(f"{key}_mean")
    values = summary.get(f"{key}_values")
    if values:
        return stat_from_values(values, mean)
    stderr = summary.get(f"{key}_stderr")
    return mean, stderr, stderr


def fmt_num(value, digits=3):
    if value is None:
        return "N/A"
    value = float(value)
    if not math.isfinite(value):
        return "N/A"
    if abs(value) < 0.01 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def fmt_pm(mean, neg, pos, digits=3):
    if mean is None:
        return "N/A"
    if neg is None or pos is None:
        return fmt_num(mean, digits)
    return f"{fmt_num(mean, digits)} (-{fmt_num(neg, digits)}/+{fmt_num(pos, digits)})"


def fmt_latex_pm(mean, neg, pos, digits=3):
    if mean is None:
        return "N/A"
    if neg is None or pos is None:
        return fmt_num(mean, digits)
    return f"${fmt_num(mean, digits)}_{{-{fmt_num(neg, digits)}}}^{{+{fmt_num(pos, digits)}}}$"


def row(
    method,
    label,
    ratio,
    encode,
    encode_neg,
    encode_pos,
    decode,
    decode_neg,
    decode_pos,
    source,
    note="",
):
    return {
        "method": method,
        "label": label,
        "ratio": ratio,
        "encode_gbs": encode,
        "encode_gbs_neg": encode_neg,
        "encode_gbs_pos": encode_pos,
        "decode_gbs": decode,
        "decode_gbs_neg": decode_neg,
        "decode_gbs_pos": decode_pos,
        "source": source,
        "note": note,
    }


def current_row_map(current):
    if not current:
        return {}
    return {r["method"]: r for r in current.get("rows", [])}


def largest_shape_methods(additional):
    out = {}
    if not additional:
        return out
    rows = additional.get("rows", [])
    if not rows:
        return out
    largest = max(rows, key=lambda r: int(r["shape"][0]) * int(r["shape"][1]))
    for method in largest.get("methods", []):
        out[method.get("method")] = method
    out["_shape"] = largest.get("shape")
    return out


def build_rows(args):
    current = load_json(args.current)
    repeats = load_json(args.repeats)
    additional = load_json(args.additional)
    zipserv_enc = load_json(args.zipserv_encode)
    splitzip = load_json(args.splitzip)
    splitzip_kernel = load_json(args.splitzip_kernel)
    falcon = load_json(args.falcon)

    current_rows = current_row_map(current)
    repeat_summary = (repeats or {}).get("summary", {})
    additional_methods = largest_shape_methods(additional)

    rows = []

    if "dietgpu" in repeat_summary:
        s = repeat_summary["dietgpu"]
        enc, enc_neg, enc_pos = stat_from_summary(s, "encode_gbs")
        dec, dec_neg, dec_pos = stat_from_summary(s, "decode_gbs")
        rows.append(
            row(
                "dietgpu",
                "DietGPU",
                s.get("ratio_mean"),
                enc,
                enc_neg,
                enc_pos,
                dec,
                dec_neg,
                dec_pos,
                "legacy 10-repeat baseline",
                "Included for completeness; older 32768x4096 Qwen2.5 activation workload.",
            )
        )

    for method, label in [("nvcomp_lz4", "nvCOMP LZ4"), ("dfloat11", "DFloat11")]:
        c = current_rows.get(method, {})
        rows.append(
            row(
                method,
                label,
                c.get("ratio"),
                c.get("encode_gbs"),
                c.get("encode_stderr"),
                c.get("encode_stderr"),
                c.get("decode_gbs"),
                c.get("decode_stderr"),
                c.get("decode_stderr"),
                c.get("source", "existing measured baseline"),
            )
        )

    for method, label in [("nvcomp_cascaded", "nvCOMP Cascaded"), ("nvcomp_bitcomp", "nvCOMP Bitcomp")]:
        a = additional_methods.get(method)
        if not a:
            continue
        enc, enc_neg, enc_pos = stat_from_summary(a, "encode_gbs")
        dec, dec_neg, dec_pos = stat_from_summary(a, "decode_gbs")
        rows.append(
            row(
                method,
                label,
                a.get("ratio"),
                enc,
                enc_neg,
                enc_pos,
                dec,
                dec_neg,
                dec_pos,
                "nvCOMP adapter, largest available shape",
                f"Shape {additional_methods.get('_shape')}.",
            )
        )

    c = current_rows.get("zipnn", {})
    rows.append(
        row(
            "zipnn",
            "ZipNN",
            c.get("ratio"),
            c.get("encode_gbs"),
            c.get("encode_stderr"),
            c.get("encode_stderr"),
            c.get("decode_gbs"),
            c.get("decode_stderr"),
            c.get("decode_stderr"),
            c.get("source", "reported"),
            "User-provided reported result.",
        )
    )

    zipserv_largest = None
    if zipserv_enc and zipserv_enc.get("rows"):
        zipserv_largest = max(zipserv_enc["rows"], key=lambda r: int(r["shape"][0]) * int(r["shape"][1]))
    c = current_rows.get("zipserv", {})
    zenc, zenc_neg, zenc_pos = stat_from_summary(zipserv_largest or {}, "encode_gbs")
    rows.append(
        row(
            "zipserv",
            "ZipServ",
            (zipserv_largest or c).get("ratio"),
            zenc if zenc is not None else c.get("encode_gbs"),
            zenc_neg if zenc is not None else c.get("encode_stderr"),
            zenc_pos if zenc is not None else c.get("encode_stderr"),
            c.get("decode_gbs"),
            c.get("decode_stderr"),
            c.get("decode_stderr"),
            "ZipServ CPU encode bench + provided decode",
            "Public wrapper encode; decode value kept at 499.5 GB/s per current baseline.",
        )
    )

    a = additional_methods.get("zipserv_tca_tbe")
    if a:
        dec, dec_neg, dec_pos = stat_from_summary(a, "decode_gbs")
        rows.append(
            row(
                "zipserv_tca_tbe",
                "ZipServ/TCA-TBE",
                a.get("ratio"),
                None,
                None,
                None,
                dec,
                dec_neg,
                dec_pos,
                "GPU-resident ZipServ/TCA-TBE boundary",
                "Decode-only GPU-resident path; public encode is CPU-side.",
            )
        )

    if falcon:
        s = falcon.get("summary", {})
        enc_values = [r["comp_gbs_fp32"] * 0.5 for r in falcon.get("repeats", []) if r.get("comp_gbs_fp32") is not None]
        dec_values = [r["decomp_gbs_fp32"] * 0.5 for r in falcon.get("repeats", []) if r.get("decomp_gbs_fp32") is not None]
        enc_mean = s.get("encode_gbs_bf16_equiv", {}).get("mean")
        dec_mean = s.get("decode_gbs_bf16_equiv", {}).get("mean")
        _, enc_neg, enc_pos = stat_from_values(enc_values, enc_mean)
        _, dec_neg, dec_pos = stat_from_values(dec_values, dec_mean)
        rows.append(
            row(
                "falcon_fp32_cast_bf16_equiv",
                "Falcon",
                s.get("ratio_vs_bf16_payload", {}).get("mean"),
                enc_mean,
                enc_neg,
                enc_pos,
                dec_mean,
                dec_neg,
                dec_pos,
                "current Falcon FP32-cast run",
                "Falcon exposes FP32/FP64 codecs here; BF16 values are cast to FP32 and reported as BF16-equivalent payload metrics.",
            )
        )

    if splitzip:
        top16 = splitzip["bf16_topk"]["top16"]
        encode = top16.get("encode_gbs", {}).get("mean")
        encode_neg = top16.get("encode_gbs", {}).get("stderr")
        encode_pos = top16.get("encode_gbs", {}).get("stderr")
        decode = top16.get("decode_gbs", {}).get("mean")
        decode_neg = top16.get("decode_gbs", {}).get("stderr")
        decode_pos = top16.get("decode_gbs", {}).get("stderr")
        note = f"Chunk size {splitzip.get('chunk_size')}."

        if splitzip_kernel and splitzip_kernel.get("summary"):
            encode, encode_neg, encode_pos = stat_from_summary(splitzip_kernel["summary"], "encode_gbs")
            decode, decode_neg, decode_pos = stat_from_summary(splitzip_kernel["summary"], "decode_gbs")
            note = (
                f"Chunk size {splitzip.get('chunk_size')}; ratio from real Qwen3-32B activations; "
                "throughput from idle preallocated kernel-stage repeats."
            )

        rows.append(
            row(
                "splitzip_v2",
                "SplitZip",
                top16.get("ratio"),
                encode,
                encode_neg,
                encode_pos,
                decode,
                decode_neg,
                decode_pos,
                "current chunk-local Top-16",
                note,
            )
        )

    order = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    rows = sorted(rows, key=lambda r: order.get(r["method"], 999))
    if not args.include_dietgpu:
        rows = [r for r in rows if r["method"] != "dietgpu"]
    return rows


def write_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "columns": [
            "method",
            "ratio",
            "encode_gbs",
            "encode_gbs_neg",
            "encode_gbs_pos",
            "decode_gbs",
            "decode_gbs_neg",
            "decode_gbs_pos",
            "source",
            "note",
        ],
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Baseline Comparison Table",
        "",
        "Throughput is in GB/s. Parentheses report negative/positive fluctuation around the central value.",
        "",
        "| Method | Ratio (x) | Encode GB/s | Decode GB/s | Source / caveat |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        note = r["source"]
        if r.get("note"):
            note += f"; {r['note']}"
        lines.append(
            f"| {r['label']} | {fmt_num(r['ratio'])} | "
            f"{fmt_pm(r['encode_gbs'], r['encode_gbs_neg'], r['encode_gbs_pos'])} | "
            f"{fmt_pm(r['decode_gbs'], r['decode_gbs_neg'], r['decode_gbs_pos'])} | {note} |"
        )
    path.write_text("\n".join(lines) + "\n")


def latex_escape(text):
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def write_latex(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Baseline codec comparison. Throughput is reported in GB/s. Subscripts and superscripts denote negative and positive fluctuation around the central value.}",
        "\\label{tab:baseline_comparison}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccc p{0.31\\linewidth}}",
        "\\toprule",
        "Method & Ratio & Encode & Decode & Source / caveat \\\\",
        "\\midrule",
    ]
    for r in rows:
        note = r["source"]
        if r.get("note"):
            note += f"; {r['note']}"
        lines.append(
            f"{latex_escape(r['label'])} & {fmt_num(r['ratio'])} & "
            f"{fmt_latex_pm(r['encode_gbs'], r['encode_gbs_neg'], r['encode_gbs_pos'])} & "
            f"{fmt_latex_pm(r['decode_gbs'], r['decode_gbs_neg'], r['decode_gbs_pos'])} & "
            f"{latex_escape(note)} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--repeats", type=Path, default=DEFAULT_REPEATS)
    parser.add_argument("--additional", type=Path, default=DEFAULT_ADDITIONAL)
    parser.add_argument("--zipserv-encode", type=Path, default=DEFAULT_ZIPSERV_ENC)
    parser.add_argument("--splitzip", type=Path, default=DEFAULT_SPLITZIP)
    parser.add_argument("--splitzip-kernel", type=Path, default=DEFAULT_SPLITZIP_KERNEL)
    parser.add_argument("--falcon", type=Path, default=DEFAULT_FALCON)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD)
    parser.add_argument("--tex-output", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--include-dietgpu", action="store_true")
    args = parser.parse_args()

    rows = build_rows(args)
    write_json(args.json_output, rows)
    write_markdown(args.md_output, rows)
    write_latex(args.tex_output, rows)
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.md_output}")
    print(f"Wrote {args.tex_output}")


if __name__ == "__main__":
    main()
