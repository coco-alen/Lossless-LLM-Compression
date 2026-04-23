"""
Mooncake local-loopback latency summary for BF16, FP8 E4M3, and FP8 E5M2.

This script measures Mooncake TransferEngine TCP loopback for the native and
compressed payload sizes used in the current SplitZip experiments, then composes
full-path latency with measured GPU codec throughput:

    full = encode + Mooncake transfer(compressed payload) + decode

The codec timings are expressed as GB/s constants so the transport measurement
can be rerun independently of GPU contention.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time

import mooncake.engine as mte


@dataclass(frozen=True)
class FormatCase:
    name: str
    native_mb: float
    ratio: float
    enc_gbs: float
    dec_gbs: float
    notes: str


BF16_MB = 402.7
CASES = [
    FormatCase(
        name="BF16 + SplitZip",
        native_mb=BF16_MB,
        ratio=1.3010,
        enc_gbs=257.0,
        dec_gbs=471.0,
        notes="BF16 exact, optimized Triton escape compaction",
    ),
    FormatCase(
        name="E4M3 + SplitZip",
        native_mb=BF16_MB / 2,
        ratio=1.059,
        enc_gbs=130.0,
        dec_gbs=223.0,
        notes="native-FP8 exact top-8 compact escapes",
    ),
    FormatCase(
        name="E5M2 + SplitZip",
        native_mb=BF16_MB / 2,
        ratio=1.214,
        enc_gbs=133.0,
        dec_gbs=233.0,
        notes="native-FP8 exact top-8 compact escapes",
    ),
]


def init_engine(name: str, port: int):
    engine = mte.TransferEngine()
    ret = engine.initialize(f"localhost:{port}", "localhost:2379", "tcp", "cpu")
    if ret != 0:
        raise RuntimeError(f"failed to initialize {name} TransferEngine: ret={ret}")
    return engine


def transfer_median_ms(src, dst, target: str, size_bytes: int, warmup=3, runs=9):
    src_buf = src.allocate_managed_buffer(size_bytes)
    dst_buf = dst.allocate_managed_buffer(size_bytes)
    # Avoid allocating a huge random buffer; content does not affect transport latency.
    src.write_bytes_to_buffer(src_buf, bytes(size_bytes), size_bytes)
    for _ in range(warmup):
        src.transfer_sync_write(target, src_buf, dst_buf, size_bytes)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        src.transfer_sync_write(target, src_buf, dst_buf, size_bytes)
        times.append((time.perf_counter() - t0) * 1000)
    src.free_managed_buffer(src_buf, size_bytes)
    dst.free_managed_buffer(dst_buf, size_bytes)
    return statistics.median(times), times


def mb_to_bytes(mb: float):
    return int(round(mb * 1_000_000))


def main():
    prefill = init_engine("prefill", 23456)
    decode = init_engine("decode", 23457)
    target = "localhost:23457"

    measured = {}
    sizes = sorted(
        {round(c.native_mb, 3) for c in CASES}
        | {round(c.native_mb / c.ratio, 3) for c in CASES},
        reverse=True,
    )

    print("Mooncake local TCP loopback transfer measurements")
    print(f"{'Payload MB':>10} {'Median ms':>10} {'GB/s':>8}")
    print("-" * 34)
    for size_mb in sizes:
        median_ms, _ = transfer_median_ms(prefill, decode, target, mb_to_bytes(size_mb))
        measured[size_mb] = median_ms
        print(f"{size_mb:>10.1f} {median_ms:>10.2f} {size_mb / median_ms:>8.2f}")

    print()
    print(
        f"{'Format':<18} {'Native':>8} {'Comp':>8} {'Ratio':>7} "
        f"{'Raw':>8} {'Enc':>7} {'Xfer':>8} {'Dec':>7} {'Full':>8} "
        f"{'vsNative':>9} {'vsBF16':>8}"
    )
    print("-" * 114)

    bf16_raw_ms = measured[round(BF16_MB, 3)]
    for case in CASES:
        comp_mb = case.native_mb / case.ratio
        native_key = round(case.native_mb, 3)
        comp_key = round(comp_mb, 3)
        raw_ms = measured[native_key]
        xfer_ms = measured[comp_key]
        enc_ms = case.native_mb / case.enc_gbs
        dec_ms = case.native_mb / case.dec_gbs
        full_ms = enc_ms + xfer_ms + dec_ms
        print(
            f"{case.name:<18} {case.native_mb:>7.1f} {comp_mb:>7.1f} "
            f"{case.ratio:>6.3f}x {raw_ms:>7.2f} {enc_ms:>6.2f} "
            f"{xfer_ms:>7.2f} {dec_ms:>6.2f} {full_ms:>7.2f} "
            f"{raw_ms / full_ms:>8.3f}x {bf16_raw_ms / full_ms:>7.3f}x"
        )

    print()
    print("Codec timing assumptions:")
    for case in CASES:
        print(f"  {case.name}: enc={case.enc_gbs:.0f} GB/s, "
              f"dec={case.dec_gbs:.0f} GB/s; {case.notes}")


if __name__ == "__main__":
    main()
