# SGLang Integration Notes

The SGLang PD path transfers KV cache blocks in `python/sglang/srt/disaggregation/mooncake/conn.py`, primarily through `MooncakeKVManager._send_kvcache_generic()` and `_transfer_data()`.

## Why a Python Wrapper Alone Is Not Enough

`_transfer_data()` runs on the sender side and asks Mooncake to write bytes directly into registered remote decode-side KV addresses.  SplitZip cannot simply compress these bytes before the call and write them into the same destination address, because the destination tensor expects decompressed BF16 layout.  A correct integration needs three operations:

1. Encode the sender-side BF16 KV block into sender-side compressed scratch buffers.
2. Transfer the compressed scratch buffers to decode-side compressed scratch buffers.
3. Launch decode-side SplitZip decompression to reconstruct BF16 into the final KV cache addresses before the request is marked ready.

Step 3 must execute in the decode worker process.  Therefore, the integration point is not a pure replacement of `_transfer_data()`; it needs a small protocol extension so the decode worker registers compressed scratch space and triggers decompression after Mooncake reports transfer completion.

## Proposed Hook Points

- Sender side: wrap the transfer-block construction in `_send_kvcache_generic()` and encode each contiguous BF16 transfer block into a compact `{packed_codes, sign_mantissa, counts, local_pos, escape_values}` payload.
- Bootstrap metadata: include compressed scratch addresses, compressed lengths, original lengths, chunk size, and codebook identifier for each block.
- Receiver side: after Mooncake transfer completion, call the SplitZip decode kernel with the received compressed buffers and final KV destination pointer.
- Benchmark flag: gate the path with `SPLITZIP_SGLANG_ENABLE=1` so native and SplitZip runs share the same launch scripts.

`sglang_sweep.py` already emits the native and SplitZip launch plan with this flag.  The actual SGLang source patch should be applied in `/data02/home/yilian2/project/sglang-dev` after choosing the compressed scratch allocator strategy.
