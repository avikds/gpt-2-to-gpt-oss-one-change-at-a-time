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

---

Built on Deep-ML.
