"""
GPT-2 to gpt-oss, One Change at a Time scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""GPT-2 to gpt-oss, one change at a time.

Story: a character-level GPT-2 is trained on children's stories, then modernized one
change at a time in the order of Sebastian Raschka's "From GPT-2 to gpt-oss": dropout
off, rotary positions, RMSNorm, SwiGLU, grouped-query attention, sliding windows on
alternating layers, attention sinks, a mixture of experts, and MXFP4 experts. Every
rung trains with the same seed, data order and step budget, so each row shows what one
change does to quality, parameters and the KV cache. Last, YaRN stretches a RoPE model
trained on 64-character windows to 256 characters without any training.
"""
import time
import torch


def main() -> None:
    t0 = time.time()
    torch.manual_seed(0)
    text = fetch_stories(1_100_000)
    tok = CharTokenizer(text)
    data = tok.encode(text)
    train_data, val_data = data[:1_000_000], data[1_000_000:]
    steps = 300

    cfg = dict(vocab_size=len(tok), d_model=64, n_layers=2, n_heads=4, n_kv_heads=4, block_size=64,
               pos="learned", norm="layer", mlp="gelu", hidden=160, n_experts=8, top_k=2,
               window=None, sinks=False, dropout=0.1, rope_base=10000.0)
    ladder = [("GPT-2 baseline", {}),
              ("- dropout", {"dropout": 0.0}),
              ("+ RoPE", {"pos": "rope"}),
              ("+ RMSNorm", {"norm": "rms"}),
              ("+ SwiGLU (clamped)", {"mlp": "swiglu"}),
              ("+ GQA (4 q : 1 kv)", {"n_kv_heads": 1}),
              ("+ sliding window 16", {"window": 16}),
              ("+ attention sinks", {"sinks": True}),
              ("+ MoE 8 experts, top-2", {"mlp": "moe", "hidden": 96})]

    print("1. The corpus")
    print(f"   {len(text):,} characters of TinyStories, vocabulary {len(tok)}, {len(train_data):,} for training and {len(val_data):,} held out")
    print(f"   every model: d_model 64, 2 layers, 4 heads, {steps} steps of 32 x 64-character windows, seed 0")

    print("\n2. The ladder (bits per character on held-out text; KV cache for one 1,024-token sequence in bfloat16)")
    print(f"   {'model':26s} {'bpc':>6s} {'change':>7s} {'total':>8s} {'active':>8s} {'KV bytes':>9s}")
    rows, prev, rope_model = [], None, None
    for name, change in ladder:
        cfg.update(change)
        row = ablation_row(name, cfg, train_data, val_data, steps)
        delta = "" if prev is None else f"{row['bpc'] - prev:+.3f}"
        print(f"   {name:26s} {row['bpc']:6.3f} {delta:>7s} {row['total']:8,d} {row['active']:8,d} {row['kv_bytes']:9,d}")
        prev = row["bpc"]
        rows.append(row)
        if name == "+ RMSNorm":
            rope_model = row["model"]
    moe_model = rows[-1]["model"]
    before = rows[-1]["bpc"]
    n_q = quantize_experts_mxfp4(moe_model)
    after = evaluate_bpc(moe_model, val_data, cfg["block_size"])
    print(f"   {'+ MXFP4 experts':26s} {after:6.3f} {after - before:+7.3f}   ({n_q:,} expert weights at 4.25 bits instead of 16)")
    first, last = rows[0], rows[-1]
    print(f"   GPT-2 -> gpt-oss style: {first['bpc']:.3f} -> {after:.3f} bpc, KV cache {first['kv_bytes'] / last['kv_bytes']:.1f}x smaller, "
          f"{last['total'] / last['active']:.1f}x more total than active parameters")
    print("   Differences below about 0.05 bpc are within seed-to-seed noise at this scale; the parameter and KV columns are exact.")

    print("\n3. Context extension with YaRN (the RoPE + RMSNorm model, trained on 64-character windows)")
    for factor, length in ((2.0, 128), (4.0, 256)):
        r = context_extension(rope_model, val_data, length, factor, 64)
        print(f"   {length} characters: train-length {r['train_len']:.3f} | plain RoPE {r['plain']:.3f} | YaRN x{factor:.0f} {r['yarn']:.3f}")
    print("   Plain RoPE degrades once windows are much longer than training; YaRN's rescaled frequencies recover part of it without training.")

    print("\n4. The same arithmetic at gpt-oss-120b's scale (131,072-token context, bfloat16 KV cache)")
    for label, n_kv, windows in (("GPT-2 style: 64 KV heads, full attention", 64, [None] * 36),
                                 ("GQA: 8 KV heads, full attention", 8, [None] * 36),
                                 ("gpt-oss: 8 KV heads, 128-token windows on alternating layers", 8, [128, None] * 18)):
        print(f"   {label:62s} {kv_cache_bytes(n_kv, 64, windows, 131072) / 1e9:6.2f} GB per sequence")
    print(f"\nTotal time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

