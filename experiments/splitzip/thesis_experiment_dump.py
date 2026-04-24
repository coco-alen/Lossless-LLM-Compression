import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE_JSON = ROOT / "experiments" / "splitzip" / "codec_ablation_results_with_dietgpu_nvcomp.json"
DEFAULT_JSON_OUT = ROOT / "experiments" / "splitzip" / "thesis_experiment_data.json"
DEFAULT_MD_OUT = ROOT / "experiments" / "splitzip" / "thesis_experiment_data.md"

SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
BREAKDOWN_SEQ_LENS = [2048, 8192, 16384, 32768]
TRANSPORT_MODES = [
    {"name": "CPU-RDMA", "bandwidth_gbs": 47.0},
    {"name": "RoCE 4x200G", "bandwidth_gbs": 87.0},
]
TRANSFER_MODE_FOR_BREAKDOWN = "RoCE 4x200G"
MODELS = [
    {
        "display_name": "Llama-3-8B",
        "hf_name": "NousResearch/Meta-Llama-3-8B",
    },
    {
        "display_name": "Qwen3-30B-A3B",
        "hf_name": "Qwen/Qwen3-30B-A3B",
    },
    {
        "display_name": "Qwen3-32B",
        "hf_name": "Qwen/Qwen3-32B",
    },
]
LINE_CHART_MODELS = {"Llama-3-8B", "Qwen3-30B-A3B"}


def raw_bytes_for_shape(shape):
    return math.prod(shape) * 2


def build_token_batch(tokenizer, target_len, salt=""):
    base = (
        "Disaggregated prefill decode serving moves KV cache tensors between workers. "
        "Lossless compression can reduce transfer time if the codec preserves exact BF16 values "
        "while keeping encode and decode throughput high. "
        + salt
    )
    ids = tokenizer(base, add_special_tokens=False)["input_ids"]
    if not ids:
        raise RuntimeError("Tokenizer returned no ids for the base prompt")
    repeated = (ids * ((target_len + len(ids) - 1) // len(ids)))[:target_len]
    return torch.tensor([repeated], dtype=torch.long)


def kv_to_rows(key, value):
    key_rows = key.detach().squeeze(0).permute(1, 0, 2).contiguous().flatten(1)
    value_rows = value.detach().squeeze(0).permute(1, 0, 2).contiguous().flatten(1)
    return torch.cat([key_rows, value_rows], dim=1).to(torch.bfloat16).cpu()


def assemble_row_prefix(blocks, target_rows, start_idx=0):
    pieces = []
    remaining = target_rows
    idx = start_idx
    n_blocks = len(blocks)
    while remaining > 0:
        block = blocks[idx % n_blocks]
        take = min(remaining, block.shape[0])
        pieces.append(block[:take])
        remaining -= take
        idx += 1
    return torch.cat(pieces, dim=0).contiguous()


def pick_splitzip_iters(numel):
    size_mb = numel * 2 / (1024 * 1024)
    if size_mb <= 1:
        return 8, 50
    if size_mb <= 8:
        return 5, 30
    if size_mb <= 32:
        return 3, 16
    if size_mb <= 96:
        return 2, 10
    return 1, 6


def bench_cuda(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def load_model_activation_blocks(model_name, device, prompt_tokens=1024):
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise RuntimeError(f"Expected CUDA device, got {device}")
        device_map = {"": device.index if device.index is not None else 0}
        device_str = str(device)
    else:
        device_map = device
        device_str = str(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    input_ids = build_token_batch(tokenizer, prompt_tokens, salt=f" model={model_name}").to(device_str)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=True,
            return_dict=True,
        )

    pkv = outputs.past_key_values
    if hasattr(pkv, "to_legacy_cache"):
        pkv = pkv.to_legacy_cache()

    blocks = []
    for key, value in pkv:
        blocks.append(kv_to_rows(key, value))

    config = model.config
    meta = {
        "num_hidden_layers": int(getattr(config, "num_hidden_layers")),
        "num_attention_heads": int(getattr(config, "num_attention_heads")),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads")),
        "head_dim": int(getattr(config, "head_dim")),
        "hidden_size": int(getattr(config, "hidden_size")),
        "model_type": str(getattr(config, "model_type")),
        "prompt_tokens": int(input_ids.shape[1]),
        "block_rows_total": int(sum(block.shape[0] for block in blocks)),
        "block_width": int(blocks[0].shape[1]),
    }

    del outputs
    del model
    torch.cuda.empty_cache()
    return blocks, meta


def measure_dma_time(size_bytes, direction, device, cache):
    key = (int(size_bytes), direction)
    if key in cache:
        return cache[key]

    gpu_buf = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    cpu_pin = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=True)
    if direction == "d2h":
        op = lambda: cpu_pin.copy_(gpu_buf)
    elif direction == "h2d":
        op = lambda: gpu_buf.copy_(cpu_pin)
    else:
        raise ValueError(direction)

    if size_bytes <= 4 * 1024 * 1024:
        warmup, iters = 8, 80
    elif size_bytes <= 32 * 1024 * 1024:
        warmup, iters = 5, 40
    else:
        warmup, iters = 3, 20

    for _ in range(warmup):
        op()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        op()
        torch.cuda.synchronize()
    avg_s = (time.perf_counter() - t0) / iters
    cache[key] = avg_s
    del gpu_buf, cpu_pin
    return avg_s


def measure_splitzip(codec, cpu_matrix, device):
    gpu_matrix = cpu_matrix.to(device=device, dtype=torch.bfloat16, non_blocking=False).contiguous().view(-1)
    warmup, iters = pick_splitzip_iters(gpu_matrix.numel())

    encoded = codec.encode(gpu_matrix)
    decoded = codec.decode(*encoded)
    ok = torch.equal(gpu_matrix.view(torch.int16), decoded.view(torch.int16))
    if not ok:
        raise RuntimeError("SplitZip round-trip failed")

    encode_s = bench_cuda(lambda: codec.encode(gpu_matrix), warmup, iters)
    encoded = codec.encode(gpu_matrix)
    decode_s = bench_cuda(lambda: codec.decode(*encoded), warmup, iters)

    pk, sm, esc_pos, esc_val, n_out, n_esc = encoded
    comp_bytes = pk.numel() + sm.numel() + esc_pos.numel() * 4 + esc_val.numel()
    raw_bytes = gpu_matrix.numel() * 2

    out = {
        "ok": True,
        "raw_bytes": int(raw_bytes),
        "compressed_bytes": int(comp_bytes),
        "ratio": raw_bytes / comp_bytes,
        "encode_s": encode_s,
        "decode_s": decode_s,
        "encode_gbs": raw_bytes / encode_s / 1e9,
        "decode_gbs": raw_bytes / decode_s / 1e9,
        "escapes": int(n_esc),
    }

    del gpu_matrix, decoded, encoded
    torch.cuda.empty_cache()
    return out


def simulate_transport(raw_bytes, comp_bytes, enc_s, dec_s, n_layers, raw_d2h_s, raw_h2d_s, comp_d2h_s, comp_h2d_s, net_gbs):
    net_s_raw = raw_bytes / (net_gbs * 1e9)
    net_s_comp = comp_bytes / (net_gbs * 1e9)

    raw_stages = [raw_d2h_s, net_s_raw, raw_h2d_s]
    raw_pipe = sum(raw_stages) + max(raw_stages) * (n_layers - 1)

    sender = max(enc_s, comp_d2h_s)
    receiver = max(comp_h2d_s, dec_s)
    split_stages = [sender, net_s_comp, receiver]
    split_pipe = sum(split_stages) + max(split_stages) * (n_layers - 1)

    split_seq_encode = enc_s * n_layers
    split_seq_transfer = (comp_d2h_s + net_s_comp + comp_h2d_s) * n_layers
    split_seq_decode = dec_s * n_layers
    raw_seq_transfer = (raw_d2h_s + net_s_raw + raw_h2d_s) * n_layers

    return {
        "raw_pipe_s": raw_pipe,
        "splitzip_pipe_s": split_pipe,
        "speedup": raw_pipe / split_pipe,
        "raw_network_s_per_layer": net_s_raw,
        "split_network_s_per_layer": net_s_comp,
        "splitzip_breakdown_sequential_s": {
            "encode": split_seq_encode,
            "transfer": split_seq_transfer,
            "decode": split_seq_decode,
            "total": split_seq_encode + split_seq_transfer + split_seq_decode,
        },
        "native_transfer_sequential_s": raw_seq_transfer,
    }


def load_baseline_summary():
    data = json.loads(BASELINE_JSON.read_text())
    target = None
    for entry in data["shapes"]:
        if tuple(entry["shape"]) == (32768, 4096):
            target = entry
            break
    if target is None:
        raise RuntimeError("Did not find 32768x4096 row in baseline JSON")

    rows = []
    for method in ["dietgpu", "nvcomp_lz4", "dfloat11", "zipserv", "splitzip"]:
        r = target["results"][method]
        rows.append(
            {
                "method": method,
                "ratio": r["ratio"],
                "encode_gbs": r["encode_gbs"],
                "decode_gbs": r["decode_gbs"],
                "source": "measured",
            }
        )

    rows.append(
        {
            "method": "zipnn",
            "ratio": 1.0 / 0.66,
            "encode_gbs": 1.15,
            "decode_gbs": 1.65,
            "source": "reported",
            "note": "User-provided ZipNN numbers; compressed-size fraction 66%.",
        }
    )
    return {
        "shape": [32768, 4096],
        "activation_model": data["activation_model"],
        "rows": rows,
        "notes": data["notes"],
    }


def format_float(x, digits=3):
    return f"{x:.{digits}f}"


def make_markdown(data):
    lines = []
    lines.append("# Thesis Experiment Data")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Baseline codec comparison uses the measured `32768 x 4096` real-activation BF16 tensor from `codec_ablation_results_with_dietgpu_nvcomp.json`.")
    lines.append("- ZipNN is included from the reported numbers already provided in the project discussion; it was not rerun locally in this dump.")
    lines.append("- Transfer sweeps use real BF16 KV blocks extracted from one 1024-token forward per model, then assembled to the target sequence length.")
    lines.append("- The transfer line-chart tables below use a pipelined staged-transfer model: native = raw BF16 DMA/network/DMA pipeline, SplitZip = encode + compressed DMA/network/DMA + decode.")
    lines.append("- The Qwen3-32B breakdown table is additive sequential accounting so the stacked bars sum cleanly; it is separate from the pipelined wall-clock tables above.")
    lines.append("")

    lines.append("## 1. Baseline Comparison")
    lines.append("")
    lines.append("| Method | Ratio (x) | Encode GB/s | Decode GB/s | Source |")
    lines.append("| --- | ---: | ---: | ---: | --- |")
    pretty = {
        "dietgpu": "DietGPU",
        "nvcomp_lz4": "nvCOMP LZ4",
        "dfloat11": "DFloat11",
        "zipserv": "ZipServ",
        "splitzip": "SplitZip",
        "zipnn": "ZipNN",
    }
    for row in data["baseline"]["rows"]:
        lines.append(
            f"| {pretty[row['method']]} | {format_float(row['ratio'])} | "
            f"{format_float(row['encode_gbs'])} | {format_float(row['decode_gbs'])} | {row['source']} |"
        )
    lines.append("")

    lines.append("## 2. Transfer Time vs Sequence Length")
    lines.append("")
    for model_name, model_block in data["transfer"].items():
        if model_name not in LINE_CHART_MODELS:
            continue
        lines.append(f"### {model_name}")
        lines.append("")
        meta = model_block["model_meta"]
        lines.append(
            f"- Layers: {meta['num_hidden_layers']}, KV heads: {meta['num_key_value_heads']}, "
            f"head dim: {meta['head_dim']}, block width: {meta['block_width']}"
        )
        lines.append("")
        for mode_name, mode_rows in model_block["modes"].items():
            lines.append(f"#### {mode_name}")
            lines.append("")
            lines.append("| Seq Len | Native ms | SplitZip ms | Speedup (x) | Ratio (x) | Enc GB/s | Dec GB/s |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for row in mode_rows:
                lines.append(
                    f"| {row['seq_len']} | {format_float(row['native_ms'])} | {format_float(row['splitzip_ms'])} | "
                    f"{format_float(row['speedup'])} | {format_float(row['ratio'])} | "
                    f"{format_float(row['encode_gbs'])} | {format_float(row['decode_gbs'])} |"
                )
            lines.append("")

    lines.append("## 3. Qwen3-32B Transmission Breakdown")
    lines.append("")
    lines.append(f"Transport mode: {data['breakdown']['transport_mode']}")
    lines.append("")
    lines.append("| Seq Len | Native transfer ms | SplitZip encode ms | SplitZip transfer ms | SplitZip decode ms | SplitZip total ms | Encode % | Transfer % | Decode % |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in data["breakdown"]["rows"]:
        lines.append(
            f"| {row['seq_len']} | {format_float(row['native_transfer_ms'])} | "
            f"{format_float(row['splitzip_encode_ms'])} | {format_float(row['splitzip_transfer_ms'])} | "
            f"{format_float(row['splitzip_decode_ms'])} | {format_float(row['splitzip_total_ms'])} | "
            f"{format_float(row['encode_pct'])} | {format_float(row['transfer_pct'])} | {format_float(row['decode_pct'])} |"
        )
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- JSON: `{data['artifacts']['json']}`")
    lines.append(f"- This Markdown: `{data['artifacts']['markdown']}`")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    from experiments.splitzip.lossless_fast import FastLosslessCodec

    baseline = load_baseline_summary()
    dma_cache = {}

    transfer_data = {}
    breakdown_rows = []

    for model_spec in MODELS:
        display_name = model_spec["display_name"]
        print(f"[1/3] Loading real KV blocks for {display_name}", flush=True)
        blocks, meta = load_model_activation_blocks(model_spec["hf_name"], device)

        codec = FastLosslessCodec(str(device))
        calibration_rows = min(meta["block_rows_total"], 32768)
        calibration_matrix = assemble_row_prefix(blocks, calibration_rows)
        calibration_gpu = calibration_matrix.to(device=device, dtype=torch.bfloat16).contiguous().view(-1)
        t0 = time.perf_counter()
        codec.calibrate(calibration_gpu)
        torch.cuda.synchronize()
        calibration_s = time.perf_counter() - t0
        del calibration_gpu
        torch.cuda.empty_cache()

        mode_tables = {mode["name"]: [] for mode in TRANSPORT_MODES}
        for seq_len in SEQ_LENS:
            print(f"  measuring {display_name} seq_len={seq_len}", flush=True)
            cpu_matrix = assemble_row_prefix(blocks, seq_len)
            sz = measure_splitzip(codec, cpu_matrix, device)

            raw_d2h_s = measure_dma_time(sz["raw_bytes"], "d2h", device, dma_cache)
            raw_h2d_s = measure_dma_time(sz["raw_bytes"], "h2d", device, dma_cache)
            comp_d2h_s = measure_dma_time(sz["compressed_bytes"], "d2h", device, dma_cache)
            comp_h2d_s = measure_dma_time(sz["compressed_bytes"], "h2d", device, dma_cache)

            for mode in TRANSPORT_MODES:
                sim = simulate_transport(
                    raw_bytes=sz["raw_bytes"],
                    comp_bytes=sz["compressed_bytes"],
                    enc_s=sz["encode_s"],
                    dec_s=sz["decode_s"],
                    n_layers=meta["num_hidden_layers"],
                    raw_d2h_s=raw_d2h_s,
                    raw_h2d_s=raw_h2d_s,
                    comp_d2h_s=comp_d2h_s,
                    comp_h2d_s=comp_h2d_s,
                    net_gbs=mode["bandwidth_gbs"],
                )
                mode_tables[mode["name"]].append(
                    {
                        "seq_len": seq_len,
                        "native_ms": sim["raw_pipe_s"] * 1000,
                        "splitzip_ms": sim["splitzip_pipe_s"] * 1000,
                        "speedup": sim["speedup"],
                        "ratio": sz["ratio"],
                        "encode_gbs": sz["encode_gbs"],
                        "decode_gbs": sz["decode_gbs"],
                        "raw_bytes_per_layer": sz["raw_bytes"],
                        "compressed_bytes_per_layer": sz["compressed_bytes"],
                    }
                )

                if display_name == "Qwen3-32B" and seq_len in BREAKDOWN_SEQ_LENS and mode["name"] == TRANSFER_MODE_FOR_BREAKDOWN:
                    bd = sim["splitzip_breakdown_sequential_s"]
                    total = bd["total"]
                    breakdown_rows.append(
                        {
                            "seq_len": seq_len,
                            "native_transfer_ms": sim["native_transfer_sequential_s"] * 1000,
                            "splitzip_encode_ms": bd["encode"] * 1000,
                            "splitzip_transfer_ms": bd["transfer"] * 1000,
                            "splitzip_decode_ms": bd["decode"] * 1000,
                            "splitzip_total_ms": total * 1000,
                            "encode_pct": bd["encode"] / total * 100,
                            "transfer_pct": bd["transfer"] / total * 100,
                            "decode_pct": bd["decode"] / total * 100,
                        }
                    )

        transfer_data[display_name] = {
            "model_meta": meta | {"calibration_s": calibration_s},
            "modes": mode_tables,
        }
        del codec
        torch.cuda.empty_cache()

    out = {
        "baseline": baseline,
        "transfer": transfer_data,
        "breakdown": {
            "model": "Qwen3-32B",
            "transport_mode": TRANSFER_MODE_FOR_BREAKDOWN,
            "rows": breakdown_rows,
        },
        "artifacts": {
            "json": str(Path(args.json_out).resolve()),
            "markdown": str(Path(args.md_out).resolve()),
        },
    }

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.write_text(json.dumps(out, indent=2))
    md_path.write_text(make_markdown(out))
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
