# GPT-2 to gpt-oss, One Change at a Time

Six years separate GPT-2 from OpenAI's open-weight gpt-oss, and almost every part of the transformer block changed in between. This project rebuilds that history at a scale you can train on a CPU. You start with a character-level GPT-2 trained on children's stories: learned absolute positions, LayerNorm, a GELU feed-forward layer, full multi-head attention and dropout. Then you apply the changes Sebastian Raschka walks through in 'From GPT-2 to gpt-oss', each as its own module: dropout removed, rotary position embeddings with the YaRN context extension gpt-oss ships with, the clamped SwiGLU feed-forward layer, a top-k mixture of experts, RMSNorm, grouped-query attention, sliding-window attention on every other layer and the learned attention sinks that gpt-oss adds to each head. Every module follows OpenAI's reference implementation. Finally you quantize the experts to MXFP4, the 4-bit format that lets gpt-oss-120b fit on one GPU, and run the whole ladder at a matched training budget: bits per character, total and active parameters, and the KV-cache bytes each change saves, side by side.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** fetch_stories
- [x] **2.** CharTokenizer
- [x] **3.** get_batch
- [x] **4.** CausalSelfAttention
- [x] **5.** GPT2MLP
- [x] **6.** GPT2
- [x] **7.** evaluate_bpc
- [x] **8.** train_lm
- [x] **9.** set_dropout
- [x] **10.** rope_inv_freq
- [x] **11.** apply_rope
- [x] **12.** yarn_inv_freq
- [x] **13.** swiglu_oss
- [x] **14.** SwiGLUMLP
- [x] **15.** route_topk
- [x] **16.** MoEMLP
- [x] **17.** RMSNorm
- [x] **18.** attention_mask
- [x] **19.** OssAttention
- [x] **20.** kv_cache_bytes
- [x] **21.** TinyLM
- [x] **22.** param_report
- [x] **23.** mxfp4_fake_quant
- [x] **24.** quantize_experts_mxfp4
- [x] **25.** context_extension
- [x] **26.** ablation_row

## Results

```
1. The corpus
   1,100,000 characters of TinyStories, vocabulary 83, 1,000,000 for training and 100,000 held out
   every model: d_model 64, 2 layers, 4 heads, 300 steps of 32 x 64-character windows, seed 0

2. The ladder (bits per character on held-out text; KV cache for one 1,024-token sequence in bfloat16)
   model                         bpc  change    total   active  KV bytes
   GPT-2 baseline              2.859          109,504  109,504   524,288
   - dropout                   2.720  -0.139  109,504  109,504   524,288
   + RoPE                      2.352  -0.368  105,408  105,408   524,288
   + RMSNorm                   2.360  +0.008  105,088  105,088   524,288
   + SwiGLU (clamped)          2.275  -0.085  101,120  101,120   524,288
   + GQA (4 q : 1 kv)          2.321  +0.045   88,640   88,640   131,072
   + sliding window 16         2.251  -0.069   88,640   88,640    66,560
   + attention sinks           2.285  +0.034   88,648   88,648    66,560
   + MoE 8 experts, top-2      2.191  -0.094  326,488  102,232    66,560
   + MXFP4 experts             2.194  +0.003   (294,912 expert weights at 4.25 bits instead of 16)
   GPT-2 -> gpt-oss style: 2.859 -> 2.194 bpc, KV cache 7.9x smaller, 3.2x more total than active parameters
   Differences below about 0.05 bpc are within seed-to-seed noise at this scale; the parameter and KV columns are exact.

3. Context extension with YaRN (the RoPE + RMSNorm model, trained on 64-character windows)
   128 characters: train-length 2.360 | plain RoPE 2.386 | YaRN x2 2.435
   256 characters: train-length 2.360 | plain RoPE 2.959 | YaRN x4 2.757
   Plain RoPE degrades once windows are much longer than training; YaRN's rescaled frequencies recover part of it without training.

4. The same arithmetic at gpt-oss-120b's scale (131,072-token context, bfloat16 KV cache)
   GPT-2 style: 64 KV heads, full attention                        77.31 GB per sequence
   GQA: 8 KV heads, full attention                                  9.66 GB per sequence
   gpt-oss: 8 KV heads, 128-token windows on alternating layers     4.84 GB per sequence

Total time 101 s
```
