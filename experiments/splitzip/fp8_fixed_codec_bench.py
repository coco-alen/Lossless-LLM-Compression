"""
FP8 fixed-width SplitZip benchmark.

This is a practical counterpart to fp8_kv_profile.py. The profiler estimates
entropy-limited ratios; this file measures a GPU-native fixed-width codec for
native FP8 cache bytes:

  E4M3: sign+mantissa stored as 4 raw bits, exponent coded as top-7 plus escape.
  E5M2: sign+mantissa stored as 3 raw bits, exponent coded as top-7 plus escape.

The codec is lossless with respect to the native FP8 byte representation.
"""

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _enc_fp8_top7(raw, lut, code_pk, sm_pk, n: tl.constexpr,
                  FMT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    elem = offs * 8
    valid = elem + 7 < n

    r0 = tl.load(raw + elem, mask=valid, other=0).to(tl.int32)
    r1 = tl.load(raw + elem + 1, mask=valid, other=0).to(tl.int32)
    r2 = tl.load(raw + elem + 2, mask=valid, other=0).to(tl.int32)
    r3 = tl.load(raw + elem + 3, mask=valid, other=0).to(tl.int32)
    r4 = tl.load(raw + elem + 4, mask=valid, other=0).to(tl.int32)
    r5 = tl.load(raw + elem + 5, mask=valid, other=0).to(tl.int32)
    r6 = tl.load(raw + elem + 6, mask=valid, other=0).to(tl.int32)
    r7 = tl.load(raw + elem + 7, mask=valid, other=0).to(tl.int32)

    if FMT == 0:
        e0 = (r0 >> 3) & 0x0F
        e1 = (r1 >> 3) & 0x0F
        e2 = (r2 >> 3) & 0x0F
        e3 = (r3 >> 3) & 0x0F
        e4 = (r4 >> 3) & 0x0F
        e5 = (r5 >> 3) & 0x0F
        e6 = (r6 >> 3) & 0x0F
        e7 = (r7 >> 3) & 0x0F

        s0 = ((r0 >> 4) & 0x08) | (r0 & 0x07)
        s1 = ((r1 >> 4) & 0x08) | (r1 & 0x07)
        s2 = ((r2 >> 4) & 0x08) | (r2 & 0x07)
        s3 = ((r3 >> 4) & 0x08) | (r3 & 0x07)
        s4 = ((r4 >> 4) & 0x08) | (r4 & 0x07)
        s5 = ((r5 >> 4) & 0x08) | (r5 & 0x07)
        s6 = ((r6 >> 4) & 0x08) | (r6 & 0x07)
        s7 = ((r7 >> 4) & 0x08) | (r7 & 0x07)

        sm_base = offs * 4
        tl.store(sm_pk + sm_base, ((s0 << 4) | s1).to(tl.uint8), mask=valid)
        tl.store(sm_pk + sm_base + 1, ((s2 << 4) | s3).to(tl.uint8), mask=valid)
        tl.store(sm_pk + sm_base + 2, ((s4 << 4) | s5).to(tl.uint8), mask=valid)
        tl.store(sm_pk + sm_base + 3, ((s6 << 4) | s7).to(tl.uint8), mask=valid)
    else:
        e0 = (r0 >> 2) & 0x1F
        e1 = (r1 >> 2) & 0x1F
        e2 = (r2 >> 2) & 0x1F
        e3 = (r3 >> 2) & 0x1F
        e4 = (r4 >> 2) & 0x1F
        e5 = (r5 >> 2) & 0x1F
        e6 = (r6 >> 2) & 0x1F
        e7 = (r7 >> 2) & 0x1F

        s0 = ((r0 >> 5) & 0x04) | (r0 & 0x03)
        s1 = ((r1 >> 5) & 0x04) | (r1 & 0x03)
        s2 = ((r2 >> 5) & 0x04) | (r2 & 0x03)
        s3 = ((r3 >> 5) & 0x04) | (r3 & 0x03)
        s4 = ((r4 >> 5) & 0x04) | (r4 & 0x03)
        s5 = ((r5 >> 5) & 0x04) | (r5 & 0x03)
        s6 = ((r6 >> 5) & 0x04) | (r6 & 0x03)
        s7 = ((r7 >> 5) & 0x04) | (r7 & 0x03)

        sm_packed = (s0 << 21) | (s1 << 18) | (s2 << 15) | (s3 << 12) | \
                    (s4 << 9) | (s5 << 6) | (s6 << 3) | s7
        sm_base = offs * 3
        tl.store(sm_pk + sm_base, ((sm_packed >> 16) & 0xFF).to(tl.uint8), mask=valid)
        tl.store(sm_pk + sm_base + 1, ((sm_packed >> 8) & 0xFF).to(tl.uint8), mask=valid)
        tl.store(sm_pk + sm_base + 2, (sm_packed & 0xFF).to(tl.uint8), mask=valid)

    c0 = tl.load(lut + e0, mask=valid, other=7).to(tl.int32)
    c1 = tl.load(lut + e1, mask=valid, other=7).to(tl.int32)
    c2 = tl.load(lut + e2, mask=valid, other=7).to(tl.int32)
    c3 = tl.load(lut + e3, mask=valid, other=7).to(tl.int32)
    c4 = tl.load(lut + e4, mask=valid, other=7).to(tl.int32)
    c5 = tl.load(lut + e5, mask=valid, other=7).to(tl.int32)
    c6 = tl.load(lut + e6, mask=valid, other=7).to(tl.int32)
    c7 = tl.load(lut + e7, mask=valid, other=7).to(tl.int32)

    code_packed = (c0 << 21) | (c1 << 18) | (c2 << 15) | (c3 << 12) | \
                  (c4 << 9) | (c5 << 6) | (c6 << 3) | c7
    code_base = offs * 3
    tl.store(code_pk + code_base, ((code_packed >> 16) & 0xFF).to(tl.uint8), mask=valid)
    tl.store(code_pk + code_base + 1, ((code_packed >> 8) & 0xFF).to(tl.uint8), mask=valid)
    tl.store(code_pk + code_base + 2, (code_packed & 0xFF).to(tl.uint8), mask=valid)


@triton.jit
def _dec_fp8_top7(code_pk, sm_pk, dlut, out, n: tl.constexpr,
                  FMT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    elem = offs * 8
    valid = elem + 7 < n

    code_base = offs * 3
    cb0 = tl.load(code_pk + code_base, mask=valid, other=0).to(tl.int32)
    cb1 = tl.load(code_pk + code_base + 1, mask=valid, other=0).to(tl.int32)
    cb2 = tl.load(code_pk + code_base + 2, mask=valid, other=0).to(tl.int32)
    code_packed = (cb0 << 16) | (cb1 << 8) | cb2

    c0 = (code_packed >> 21) & 7
    c1 = (code_packed >> 18) & 7
    c2 = (code_packed >> 15) & 7
    c3 = (code_packed >> 12) & 7
    c4 = (code_packed >> 9) & 7
    c5 = (code_packed >> 6) & 7
    c6 = (code_packed >> 3) & 7
    c7 = code_packed & 7

    e0 = tl.load(dlut + c0, mask=valid, other=0).to(tl.int32)
    e1 = tl.load(dlut + c1, mask=valid, other=0).to(tl.int32)
    e2 = tl.load(dlut + c2, mask=valid, other=0).to(tl.int32)
    e3 = tl.load(dlut + c3, mask=valid, other=0).to(tl.int32)
    e4 = tl.load(dlut + c4, mask=valid, other=0).to(tl.int32)
    e5 = tl.load(dlut + c5, mask=valid, other=0).to(tl.int32)
    e6 = tl.load(dlut + c6, mask=valid, other=0).to(tl.int32)
    e7 = tl.load(dlut + c7, mask=valid, other=0).to(tl.int32)

    if FMT == 0:
        sm_base = offs * 4
        p0 = tl.load(sm_pk + sm_base, mask=valid, other=0).to(tl.int32)
        p1 = tl.load(sm_pk + sm_base + 1, mask=valid, other=0).to(tl.int32)
        p2 = tl.load(sm_pk + sm_base + 2, mask=valid, other=0).to(tl.int32)
        p3 = tl.load(sm_pk + sm_base + 3, mask=valid, other=0).to(tl.int32)

        s0 = (p0 >> 4) & 0x0F
        s1 = p0 & 0x0F
        s2 = (p1 >> 4) & 0x0F
        s3 = p1 & 0x0F
        s4 = (p2 >> 4) & 0x0F
        s5 = p2 & 0x0F
        s6 = (p3 >> 4) & 0x0F
        s7 = p3 & 0x0F

        o0 = ((s0 & 0x08) << 4) | (e0 << 3) | (s0 & 0x07)
        o1 = ((s1 & 0x08) << 4) | (e1 << 3) | (s1 & 0x07)
        o2 = ((s2 & 0x08) << 4) | (e2 << 3) | (s2 & 0x07)
        o3 = ((s3 & 0x08) << 4) | (e3 << 3) | (s3 & 0x07)
        o4 = ((s4 & 0x08) << 4) | (e4 << 3) | (s4 & 0x07)
        o5 = ((s5 & 0x08) << 4) | (e5 << 3) | (s5 & 0x07)
        o6 = ((s6 & 0x08) << 4) | (e6 << 3) | (s6 & 0x07)
        o7 = ((s7 & 0x08) << 4) | (e7 << 3) | (s7 & 0x07)
    else:
        sm_base = offs * 3
        sb0 = tl.load(sm_pk + sm_base, mask=valid, other=0).to(tl.int32)
        sb1 = tl.load(sm_pk + sm_base + 1, mask=valid, other=0).to(tl.int32)
        sb2 = tl.load(sm_pk + sm_base + 2, mask=valid, other=0).to(tl.int32)
        sm_packed = (sb0 << 16) | (sb1 << 8) | sb2

        s0 = (sm_packed >> 21) & 7
        s1 = (sm_packed >> 18) & 7
        s2 = (sm_packed >> 15) & 7
        s3 = (sm_packed >> 12) & 7
        s4 = (sm_packed >> 9) & 7
        s5 = (sm_packed >> 6) & 7
        s6 = (sm_packed >> 3) & 7
        s7 = sm_packed & 7

        o0 = ((s0 & 0x04) << 5) | (e0 << 2) | (s0 & 0x03)
        o1 = ((s1 & 0x04) << 5) | (e1 << 2) | (s1 & 0x03)
        o2 = ((s2 & 0x04) << 5) | (e2 << 2) | (s2 & 0x03)
        o3 = ((s3 & 0x04) << 5) | (e3 << 2) | (s3 & 0x03)
        o4 = ((s4 & 0x04) << 5) | (e4 << 2) | (s4 & 0x03)
        o5 = ((s5 & 0x04) << 5) | (e5 << 2) | (s5 & 0x03)
        o6 = ((s6 & 0x04) << 5) | (e6 << 2) | (s6 & 0x03)
        o7 = ((s7 & 0x04) << 5) | (e7 << 2) | (s7 & 0x03)

    tl.store(out + elem, o0.to(tl.uint8), mask=valid)
    tl.store(out + elem + 1, o1.to(tl.uint8), mask=valid)
    tl.store(out + elem + 2, o2.to(tl.uint8), mask=valid)
    tl.store(out + elem + 3, o3.to(tl.uint8), mask=valid)
    tl.store(out + elem + 4, o4.to(tl.uint8), mask=valid)
    tl.store(out + elem + 5, o5.to(tl.uint8), mask=valid)
    tl.store(out + elem + 6, o6.to(tl.uint8), mask=valid)
    tl.store(out + elem + 7, o7.to(tl.uint8), mask=valid)


@triton.jit
def _fix_fp8_escapes(esc_pos, esc_val, out, n_esc, FMT: tl.constexpr,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_esc
    pos = tl.load(esc_pos + offs, mask=mask, other=0)
    exp = tl.load(esc_val + offs, mask=mask, other=0).to(tl.int32)
    raw = tl.load(out + pos, mask=mask, other=0).to(tl.int32)
    if FMT == 0:
        fixed = (raw & 0x87) | (exp << 3)
    else:
        fixed = (raw & 0x83) | (exp << 2)
    tl.store(out + pos, fixed.to(tl.uint8), mask=mask)


def bench(fn, warmup=30, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def build_codebook(raw, fmt, device, lossless, strategy="freq"):
    if fmt == "e4m3":
        exponents = ((raw >> 3) & 0x0F).to(torch.uint8)
        lut_size = 16
    else:
        exponents = ((raw >> 2) & 0x1F).to(torch.uint8)
        lut_size = 32

    vals, counts = torch.unique(exponents, return_counts=True)
    order = torch.argsort(counts, descending=True)
    vals_list = [vals[i].item() for i in range(vals.numel())]
    ranked_vals = [vals[order[i]].item() for i in range(vals.numel())]

    if lossless:
        selected = ranked_vals[:min(7, len(ranked_vals))]
    elif strategy == "extreme":
        selected = []
        if vals_list:
            selected.extend([min(vals_list), max(vals_list)])
        for value in ranked_vals:
            if value not in selected:
                selected.append(value)
            if len(selected) == 8:
                break
    elif strategy == "uniform":
        lo = min(vals_list)
        hi = max(vals_list)
        if lo == hi:
            selected = [lo]
        else:
            selected = sorted(set(round(lo + (hi - lo) * i / 7) for i in range(8)))
        for value in ranked_vals:
            if value not in selected:
                selected.append(value)
            if len(selected) == 8:
                break
    else:
        selected = ranked_vals[:min(8, len(ranked_vals))]

    dlut = torch.zeros(8, dtype=torch.uint8, device=device)
    for i, value in enumerate(selected[:8]):
        dlut[i] = value

    if lossless:
        lut = torch.full((lut_size,), 7, dtype=torch.uint8, device=device)
        for i, value in enumerate(selected[:7]):
            lut[value] = i
        dlut[7] = dlut[0]
    else:
        lut = torch.zeros((lut_size,), dtype=torch.uint8, device=device)
        for exp in range(lut_size):
            best_code = 0
            best_dist = 999
            for code, top_exp in enumerate(selected[:8]):
                dist = abs(exp - top_exp)
                if dist < best_dist:
                    best_dist = dist
                    best_code = code
            lut[exp] = best_code

    selected_set = set(selected[:7] if lossless else selected[:8])
    covered = 0
    for i, value in enumerate(vals_list):
        if value in selected_set:
            covered += counts[i].item()
    coverage = covered / counts.sum().item()
    entropy = compute_entropy(exponents)
    return lut, dlut, coverage, entropy


def compute_entropy(x):
    vals, counts = torch.unique(x, return_counts=True)
    probs = counts.float() / counts.sum()
    return -(probs * torch.log2(probs)).sum().item()


class FP8FixedCodec:
    def __init__(self, fmt, raw, lossless=True, strategy="freq", block=128):
        self.fmt = fmt
        self.fmt_id = 0 if fmt == "e4m3" else 1
        self.lossless = lossless
        self.device = raw.device
        self.n = raw.numel()
        if self.n % 8 != 0:
            raise ValueError("FP8 benchmark expects numel to be divisible by 8")
        self.block = block
        self.groups = self.n // 8
        self.code_bytes = self.groups * 3
        self.sm_bytes = self.groups * (4 if fmt == "e4m3" else 3)
        self.lut, self.dlut, self.coverage, self.exp_entropy = build_codebook(
            raw, fmt, self.device, lossless, strategy)
        self.code_pk = torch.empty(self.code_bytes, dtype=torch.uint8, device=self.device)
        self.sm_pk = torch.empty(self.sm_bytes, dtype=torch.uint8, device=self.device)
        self.out = torch.empty_like(raw)
        self.grid = ((self.groups + block - 1) // block,)

    def encode_core(self, raw):
        _enc_fp8_top7[self.grid](raw, self.lut, self.code_pk, self.sm_pk, self.n,
                                 FMT=self.fmt_id, BLOCK=self.block)
        return self.code_pk, self.sm_pk

    def collect_escapes(self, raw):
        if not self.lossless:
            empty_pos = torch.empty(0, dtype=torch.int32, device=self.device)
            empty_val = torch.empty(0, dtype=torch.uint8, device=self.device)
            return empty_pos, empty_val
        if self.fmt == "e4m3":
            exponents = ((raw >> 3) & 0x0F).to(torch.uint8)
        else:
            exponents = ((raw >> 2) & 0x1F).to(torch.uint8)
        codes = self.lut[exponents.long()]
        esc_pos = (codes == 7).nonzero(as_tuple=True)[0].to(torch.int32)
        esc_val = exponents[esc_pos.long()] if esc_pos.numel() else torch.empty(
            0, dtype=torch.uint8, device=self.device)
        return esc_pos, esc_val

    def encode(self, raw):
        self.encode_core(raw)
        esc_pos, esc_val = self.collect_escapes(raw)
        return self.code_pk, self.sm_pk, esc_pos, esc_val

    def decode_core(self):
        _dec_fp8_top7[self.grid](self.code_pk, self.sm_pk, self.dlut, self.out, self.n,
                                 FMT=self.fmt_id, BLOCK=self.block)
        return self.out

    def decode(self, esc_pos, esc_val):
        self.decode_core()
        if esc_pos.numel() > 0:
            grid = ((esc_pos.numel() + 255) // 256,)
            _fix_fp8_escapes[grid](esc_pos, esc_val, self.out, esc_pos.numel(),
                                   FMT=self.fmt_id, BLOCK=256)
        return self.out

    def compressed_bytes(self, n_esc):
        return self.code_bytes + self.sm_bytes + n_esc * 5


def pipeline_speedup(raw_bytes, comp_bytes, enc_s, dec_s, bw_gbs=87.0,
                     n_layers=80):
    raw_total = raw_bytes * n_layers / (bw_gbs * 1e9)
    xfer = comp_bytes / (bw_gbs * 1e9)
    stage = max(enc_s, xfer, dec_s)
    total = enc_s + stage * n_layers + dec_s
    return raw_total / total


def make_raw_fp8(fmt, n, device, scale):
    bf16 = (torch.randn(n, dtype=torch.bfloat16, device=device) * scale).contiguous()
    if fmt == "e4m3":
        return bf16.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    return bf16.to(torch.float8_e5m2).view(torch.uint8).contiguous()


def raw_fp8_from_bf16(fmt, bf16_flat, size_mb):
    if fmt == "e4m3":
        raw = bf16_flat.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    else:
        raw = bf16_flat.to(torch.float8_e5m2).view(torch.uint8).contiguous()
    target = size_mb * 1024 * 1024
    repeats = (target + raw.numel() - 1) // raw.numel()
    raw = raw.repeat(repeats)[:target].contiguous()
    return raw[: raw.numel() - (raw.numel() % 8)].contiguous()


def load_model_kv_bf16(model_name, max_new_tokens, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompt = ("The theory of general relativity describes gravity as the curvature of "
              "spacetime caused by mass and energy. Einstein's field equations relate "
              "the geometry of spacetime to the distribution of matter within it.")
    print(f"Loading model KV source: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            use_cache=True,
        )
    pieces = []
    for key, value in outputs.past_key_values:
        pieces.append(key.detach().contiguous().view(-1))
        pieces.append(value.detach().contiguous().view(-1))
    flat = torch.cat(pieces).contiguous()
    print(f"Model KV source: {flat.numel() * 2 / 1e6:.2f} MB BF16, "
          f"seq_len={outputs.sequences.shape[1]}, layers={len(outputs.past_key_values)}")
    del model
    torch.cuda.empty_cache()
    return flat


def fp8_value_error(fmt, raw, decoded):
    if fmt == "e4m3":
        orig = raw.view(torch.float8_e4m3fn).to(torch.bfloat16).float()
        dec = decoded.view(torch.float8_e4m3fn).to(torch.bfloat16).float()
    else:
        orig = raw.view(torch.float8_e5m2).to(torch.bfloat16).float()
        dec = decoded.view(torch.float8_e5m2).to(torch.bfloat16).float()
    err = (orig - dec).abs()
    return err.max().item(), err.pow(2).mean().sqrt().item()


def run_case(fmt, mode, raw, block, iters):
    lossless = mode == "lossless-top7"
    if mode.endswith("-ext"):
        strategy = "extreme"
    elif mode.endswith("-unif"):
        strategy = "uniform"
    else:
        strategy = "freq"
    codec = FP8FixedCodec(fmt, raw, lossless, strategy, block)

    codec.encode(raw)
    esc_pos, esc_val = codec.collect_escapes(raw)
    decoded = codec.decode(esc_pos, esc_val)
    correct = torch.equal(raw, decoded)
    mismatches = (raw != decoded).sum().item()
    max_err, rmse = fp8_value_error(fmt, raw, decoded)
    n_esc = esc_pos.numel()
    comp_bytes = codec.compressed_bytes(n_esc)
    ratio = raw.numel() / comp_bytes

    enc_core_s = bench(lambda: codec.encode_core(raw), iters=iters)
    enc_full_s = bench(lambda: codec.encode(raw), iters=max(20, iters // 4))
    codec.encode_core(raw)
    dec_core_s = bench(lambda: codec.decode_core(), iters=iters)
    dec_full_s = bench(lambda: codec.decode(esc_pos, esc_val), iters=iters)

    raw_bytes = raw.numel()
    return {
        "fmt": fmt,
        "mode": mode,
        "size_mb": raw.numel() / 1024 / 1024,
        "coverage": codec.coverage,
        "exp_entropy": codec.exp_entropy,
        "ratio": ratio,
        "total_vs_bf16": ratio * 2.0,
        "n_esc": n_esc,
        "esc_rate": n_esc / raw.numel(),
        "correct": correct,
        "mismatch_rate": mismatches / raw.numel(),
        "max_err": max_err,
        "rmse": rmse,
        "enc_core_gbs": raw_bytes / enc_core_s / 1e9,
        "enc_full_gbs": raw_bytes / enc_full_s / 1e9,
        "dec_core_gbs": raw_bytes / dec_core_s / 1e9,
        "dec_full_gbs": raw_bytes / dec_full_s / 1e9,
        "sp_fp8_87": pipeline_speedup(raw_bytes, comp_bytes, enc_full_s, dec_full_s, 87.0),
        "sp_bf16_87": pipeline_speedup(raw_bytes * 2, comp_bytes, enc_full_s, dec_full_s, 87.0),
        "sp_fp8_190": pipeline_speedup(raw_bytes, comp_bytes, enc_full_s, dec_full_s, 190.0),
        "sp_bf16_190": pipeline_speedup(raw_bytes * 2, comp_bytes, enc_full_s, dec_full_s, 190.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mb", type=int, default=134,
                        help="Native FP8 layer size in MB. 134 MB corresponds to 268 MB BF16.")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--model", default=None,
                        help="Optional model name. If set, tile real generated KV bytes to --size-mb.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Native FP8 size: {args.size_mb} MB, BF16-equivalent: {args.size_mb * 2} MB")
    device = "cuda"
    model_bf16 = None
    if args.model:
        model_bf16 = load_model_kv_bf16(args.model, args.max_new_tokens, device)
    print()
    print(f"{'Fmt':<6} {'Mode':<14} {'Cov':>7} {'Hexp':>6} {'Ratio':>8} {'vsBF16':>8} "
          f"{'Esc%':>8} {'Mis%':>8} {'MaxErr':>8} {'EncFull':>8} {'DecFull':>8} "
          f"{'FP8@87':>8} {'BF16@87':>8} {'OK':>4}")
    print("-" * 132)
    for fmt in ("e4m3", "e5m2"):
        if model_bf16 is None:
            raw = make_raw_fp8(fmt, args.size_mb * 1024 * 1024, device, args.scale)
        else:
            raw = raw_fp8_from_bf16(fmt, model_bf16, args.size_mb)
        for mode in ("lossless-top7", "near-top8", "near-top8-ext", "near-top8-unif"):
            r = run_case(fmt, mode, raw, args.block, args.iters)
            print(f"{fmt:<6} {mode:<14} {r['coverage']*100:>6.2f}% {r['exp_entropy']:>6.3f} "
              f"{r['ratio']:>7.3f}x {r['total_vs_bf16']:>7.3f}x "
              f"{r['esc_rate']*100:>7.3f}% {r['mismatch_rate']*100:>7.3f}% "
              f"{r['max_err']:>8.3g} {r['enc_full_gbs']:>7.0f} "
              f"{r['dec_full_gbs']:>7.0f} {r['sp_fp8_87']:>7.3f}x "
              f"{r['sp_bf16_87']:>7.3f}x {'PASS' if r['correct'] else 'FAIL':>4}")

    print("\nNotes:")
    print("  lossless-top7 stores top-7 exponent codes plus explicit escapes.")
    print("  near-top8 maps rare exponents to the nearest top-8 exponent and has no escape pass.")
    print("  EncFull includes PyTorch escape collection for lossless-top7; near-top8 is just the Triton pack kernel.")
    print("  FP8@87 compares compressed native FP8 transfer to raw native FP8 transfer.")
    print("  BF16@87 compares compressed native FP8 transfer to raw BF16 transfer.")


if __name__ == "__main__":
    main()
