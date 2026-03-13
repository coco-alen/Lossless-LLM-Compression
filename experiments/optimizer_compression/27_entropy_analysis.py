"""
Experiment 27: Detailed entropy analysis of FP32 optimizer state bytes.

Measure per-byte-plane entropy, unique counts, and theoretical compression limits
to determine the best compression strategy.
"""

import torch
import gc
import math
from transformers import AutoModelForCausalLM


def entropy_bits(counts):
    """Shannon entropy in bits from count tensor."""
    total = counts.sum().item()
    if total == 0:
        return 0
    probs = counts[counts > 0].float() / total
    return -(probs * probs.log2()).sum().item()


def analyze_states(model_name="Qwen/Qwen3-0.6B"):
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Do a few training steps
    for s in range(5):
        torch.manual_seed(s + 100)
        ids = torch.randint(100, 10000, (2, 128), device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model(input_ids=ids, labels=ids).loss.backward()
        opt.step(); opt.zero_grad()

    # Gather all m and v into flat tensors
    params = []
    for group in opt.param_groups:
        for p in group['params']:
            if p in opt.state and 'exp_avg' in opt.state[p]:
                params.append(p)

    total_n = sum(p.numel() for p in params)
    print(f"Total params: {total_n:,} ({total_n*4/1024**2:.0f} MB per state)")
    print(f"Total optimizer memory: {total_n*4*2/1024**2:.0f} MB\n")

    for state_name, key in [("m (exp_avg)", "exp_avg"), ("v (exp_avg_sq)", "exp_avg_sq")]:
        flat = torch.empty(total_n, dtype=torch.float32, device='cuda')
        offset = 0
        for p in params:
            n = p.numel()
            flat[offset:offset+n] = opt.state[p][key].flatten()
            offset += n

        int32 = flat.view(torch.int32)
        bytes_view = int32.view(torch.uint8).reshape(total_n, 4)

        print(f"=== {state_name} ===")
        total_entropy = 0
        for byte_idx in range(4):
            byte_data = bytes_view[:, byte_idx]
            counts = torch.bincount(byte_data.to(torch.int32), minlength=256)
            n_unique = (counts > 0).sum().item()
            h = entropy_bits(counts)
            total_entropy += h

            # Top 5 most common values
            top5_vals = counts.argsort(descending=True)[:5]
            top5_pcts = [(counts[v].item() / total_n * 100) for v in top5_vals]

            print(f"  byte{byte_idx}: {n_unique} unique, entropy={h:.3f} bits, "
                  f"fixed={math.ceil(math.log2(max(n_unique, 2)))} bits")
            top5_str = ", ".join(f"0x{v.item():02X}({p:.1f}%)" for v, p in zip(top5_vals, top5_pcts))
            print(f"    top5: {top5_str}")

        print(f"  TOTAL entropy: {total_entropy:.3f} bits/value (of 32)")
        print(f"  Theoretical limit: {total_entropy/32*100:.1f}%")
        print(f"  Theoretical savings: {(32-total_entropy)/32*100:.1f}% = "
              f"{total_n*4*(32-total_entropy)/32/1024**2:.0f} MB")
        print()

        # High-16 analysis
        high16 = ((int32 >> 16) & 0xFFFF).to(torch.int32)
        counts16 = torch.bincount(high16, minlength=65536)
        n_unique_16 = (counts16 > 0).sum().item()
        h16 = entropy_bits(counts16)
        print(f"  high16: {n_unique_16} unique, entropy={h16:.3f} bits, "
              f"fixed={math.ceil(math.log2(max(n_unique_16, 2)))} bits")
        print(f"  If high16 Huffman + low16 raw: {(h16+16)/32*100:.1f}% = "
              f"saves {total_n*4*(16-h16)/32/1024**2:.0f} MB")

        # Byte3+byte2 joint analysis
        b3 = bytes_view[:, 3]
        b2 = bytes_view[:, 2]
        joint = b3.to(torch.int32) * 256 + b2.to(torch.int32)
        counts_joint = torch.bincount(joint, minlength=65536)
        h_joint = entropy_bits(counts_joint)
        n_joint = (counts_joint > 0).sum().item()
        print(f"  byte3×byte2: {n_joint} unique, joint entropy={h_joint:.3f} bits")
        print(f"  vs separate: {entropy_bits(torch.bincount(b3.to(torch.int32), minlength=256)):.3f} + "
              f"{entropy_bits(torch.bincount(b2.to(torch.int32), minlength=256)):.3f} = "
              f"{entropy_bits(torch.bincount(b3.to(torch.int32), minlength=256)) + entropy_bits(torch.bincount(b2.to(torch.int32), minlength=256)):.3f} bits")
        print()

        del flat, int32, bytes_view
        gc.collect(); torch.cuda.empty_cache()

    del model, opt
    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    analyze_states()
