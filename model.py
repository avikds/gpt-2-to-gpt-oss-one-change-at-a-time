"""
GPT-2 to gpt-oss, One Change at a Time

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - fetch_stories
import os
import tempfile
import urllib.request

STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"


def fetch_stories(max_chars):
    # Store the downloaded dataset in a temporary cache.
    cache_path = os.path.join(
        tempfile.gettempdir(),
        "gptoss_tinystories_valid.txt"
    )

    # Download the validation dataset only if it is not already cached.
    if not os.path.exists(cache_path):
        urllib.request.urlretrieve(STORIES_URL, cache_path)

    # Read the dataset and return the first max_chars characters.
    with open(cache_path, "r", encoding="utf-8", errors="ignore") as f:
        stories = f.read()

    return stories[:max_chars]

# Step 2 - CharTokenizer
import torch

class CharTokenizer:
    def __init__(self, text):
        # Build a sorted list of unique characters.
        self.itos = sorted(set(text))

        # Map each character to its integer index.
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self):
        # Return the vocabulary size.
        return len(self.itos)

    def encode(self, s):
        # Convert characters to IDs, silently skipping unknown characters.
        ids = [self.stoi[ch] for ch in s if ch in self.stoi]

        # Return a 1-D tensor of type torch.long.
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        # Support both Python lists and 1-D PyTorch tensors.
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Convert token IDs back to a string.
        return "".join(self.itos[int(i)] for i in ids)

# Step 3 - get_batch
def get_batch(data, block_size, batch_size, generator=None):
    # Randomly sample starting positions for training windows.
    starts = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,),
        generator=generator
    )

    # Input windows: characters from position s to s + block_size - 1.
    x = torch.stack([
        data[int(s):int(s) + block_size]
        for s in starts
    ]).long()

    # Target windows: the same sequences shifted one character to the right.
    y = torch.stack([
        data[int(s) + 1:int(s) + block_size + 1]
        for s in starts
    ]).long()

    return x, y

# Step 4 - CausalSelfAttention
import math
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()

        # Ensure the model dimension can be divided evenly across attention heads.
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Project the input into queries, keys, and values in a single operation.
        self.qkv = nn.Linear(d_model, 3 * d_model)

        # Project the concatenated attention heads back to d_model dimensions.
        self.proj = nn.Linear(d_model, d_model)

        # Dropout applied to attention weights and the residual output.
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x has shape (B, T, d_model).
        B, T, C = x.shape

        # Compute Q, K, V and split them into individual attention heads.
        # Shape after reshape: (B, T, 3, n_heads, head_dim).
        # Shape after permute: (3, B, n_heads, T, head_dim).
        qkv = self.qkv(x).reshape(
            B, T, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention scores.
        # Result shape: (B, n_heads, T, T).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create a causal mask that blocks access to future positions.
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=1
        )

        # Set future-position scores to negative infinity.
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Normalize scores into attention weights.
        weights = torch.softmax(scores, dim=-1)

        # Apply dropout to attention weights.
        weights = self.attn_drop(weights)

        # Apply attention weights to values.
        # Shape: (B, n_heads, T, head_dim).
        out = weights @ v

        # Merge attention heads back into (B, T, d_model).
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection and residual dropout.
        out = self.proj(out)
        out = self.resid_drop(out)

        return out

# Step 5 - GPT2MLP
import torch.nn.functional as F

class GPT2MLP(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()

        # Expand the hidden dimension from d_model to 4 * d_model.
        self.fc = nn.Linear(d_model, 4 * d_model)

        # Project the expanded representation back to d_model.
        self.proj = nn.Linear(4 * d_model, d_model)

        # Apply dropout to the projected output.
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # Apply the first linear layer, GPT-2's tanh-approximated GELU,
        # the output projection, and dropout in sequence.
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.proj(x)
        x = self.drop(x)

        return x

# Step 6 - GPT2
class GPT2Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = GPT2MLP(d_model, dropout)

    def forward(self, x):
        # Pre-norm attention with a residual connection.
        x = x + self.attn(self.ln_1(x))

        # Pre-norm feed-forward network with a residual connection.
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        block_size,
        dropout=0.1
    ):
        super().__init__()

        self.block_size = block_size

        # Token embeddings and learned absolute positional embeddings.
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)

        self.drop = nn.Dropout(dropout)

        # Transformer blocks.
        self.blocks = nn.ModuleList([
            GPT2Block(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        # Output projection without a separate bias.
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie the output weights to the token embeddings before initialization.
        self.head.weight = self.wte.weight

        # Initialize Linear and Embedding weights using GPT-2's specified
        # normal distribution; initialize Linear biases to zero.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx):
        # idx shape: (B, T)
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.block_size}"
            )

        # Add token and learned positional embeddings.
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        # Apply all transformer blocks.
        for block in self.blocks:
            x = block(x)

        # Normalize and produce vocabulary logits.
        x = self.ln_f(x)
        logits = self.head(x)

        return logits

# Step 7 - evaluate_bpc
def evaluate_bpc(model, data, block_size, batch_size=16, n_batches=8, seed=0):
    # Remember the original training/evaluation mode.
    was_training = model.training

    # Switch to evaluation mode to disable dropout.
    model.eval()

    # Use a fixed random seed to make evaluation batches reproducible.
    generator = torch.Generator().manual_seed(seed)
    losses = []

    try:
        # Disable gradient tracking during evaluation.
        with torch.no_grad():
            for _ in range(n_batches):
                # Sample a batch of input sequences and next-character targets.
                x, y = get_batch(
                    data,
                    block_size,
                    batch_size,
                    generator=generator
                )

                # Compute model logits: (batch_size, block_size, vocab_size).
                logits = model(x)

                # Compute mean cross-entropy in nats over all batch positions.
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1)
                )

                losses.append(loss.item())

    finally:
        # Restore the model's original mode, even if evaluation raises an error.
        model.train(was_training)

    # Average cross-entropy across batches and convert nats to bits.
    mean_loss = sum(losses) / len(losses)

    # For character-level tokenization, this is bits per character (BPC).
    return float(mean_loss / math.log(2))

# Step 8 - train_lm
def train_lm(model, data, steps, block_size, batch_size=32, lr=3e-3, seed=0):
    # Seed PyTorch for reproducible initialization of random operations.
    torch.manual_seed(seed)

    # Use a dedicated generator to make training batches reproducible.
    g = torch.Generator().manual_seed(seed)

    # AdamW optimizer with the specified GPT-2 baseline training settings.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Store the training loss from every optimization step.
    losses = []

    # Enable training mode, including dropout.
    model.train()

    for _ in range(steps):
        # Sample a batch of input sequences and next-character targets.
        x, y = get_batch(
            data,
            block_size,
            batch_size,
            generator=g
        )

        # Clear gradients from the previous optimization step.
        optimizer.zero_grad(set_to_none=True)

        # Forward pass: logits have shape (batch_size, block_size, vocab_size).
        logits = model(x)

        # Compute mean cross-entropy across all batch positions.
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        # Backpropagate the loss.
        loss.backward()

        # Clip the global gradient norm to 1.0 for training stability.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update model parameters.
        optimizer.step()

        # Record the loss as a Python float.
        losses.append(float(loss.item()))

    return losses

# Step 9 - set_dropout
def set_dropout(model, p):
    # Update the dropout probability for every dropout module in the model.
    count = 0

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = p
            count += 1

    # Return the number of dropout modules updated.
    return count

# Step 10 - rope_inv_freq
def rope_inv_freq(head_dim, base):
    # Indices for each pair of dimensions.
    i = torch.arange(head_dim // 2, dtype=torch.float32)

    # Compute rotary frequencies: base^(-2i / head_dim).
    inv_freq = torch.pow(
        torch.tensor(float(base), dtype=torch.float32),
        -2.0 * i / head_dim
    )

    return inv_freq

# Step 11 - apply_rope
def apply_rope(x, inv_freq, positions, concentration=1.0):
    # Compute rotary angles in float32 on the same device as x.
    positions = positions.to(device=x.device, dtype=torch.float32)
    inv_freq = inv_freq.to(device=x.device, dtype=torch.float32)

    # Outer product: (T,) x (head_dim // 2,)
    # produces angles of shape (T, head_dim // 2).
    angles = torch.outer(positions, inv_freq)

    # Compute cosine and sine, applying the concentration factor.
    cos = (torch.cos(angles) * concentration).to(dtype=x.dtype)
    sin = (torch.sin(angles) * concentration).to(dtype=x.dtype)

    # Split the last dimension into two halves (GPT-OSS half-split layout).
    x1, x2 = torch.chunk(x, 2, dim=-1)

    # Rotate each pair of dimensions according to its position.
    rotated_first = x1 * cos - x2 * sin
    rotated_second = x2 * cos + x1 * sin

    # Concatenate the rotated halves to restore the original shape.
    return torch.cat((rotated_first, rotated_second), dim=-1)

# Step 12 - yarn_inv_freq
def yarn_inv_freq(head_dim, base, factor, initial_context, ntk_alpha=1.0, ntk_beta=32.0):
    # Reciprocal of the standard RoPE inverse frequencies.
    freq = torch.pow(
        torch.tensor(float(base), dtype=torch.float32),
        torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    )

    # No context extension: use standard RoPE frequencies.
    if factor <= 1:
        return 1.0, 1.0 / freq

    # YaRN attention concentration factor.
    concentration = 0.1 * math.log(factor) + 1.0

    # Compute the low and high correction dimensions for NTK-by-parts.
    d_half = head_dim // 2

    low = (
        d_half
        * math.log(initial_context / (ntk_beta * 2 * math.pi))
        / math.log(base)
    )

    high = (
        d_half
        * math.log(initial_context / (ntk_alpha * 2 * math.pi))
        / math.log(base)
    )

    assert 0 < low < high < d_half - 1

    # Interpolated frequencies stretch the context; extrapolated
    # frequencies retain the original RoPE behavior.
    interpolation = 1.0 / (factor * freq)
    extrapolation = 1.0 / freq

    # Build a smooth transition mask between the two frequency ranges.
    ramp = (
        torch.arange(d_half, dtype=torch.float32) - low
    ) / (high - low)

    mask = 1.0 - ramp.clamp(0.0, 1.0)

    # Blend interpolated and extrapolated frequencies.
    inv_freq = interpolation * (1.0 - mask) + extrapolation * mask

    return float(concentration), inv_freq

# Step 13 - swiglu_oss
def swiglu_oss(x, alpha=1.702, limit=7.0):
    # Split the interleaved input into gate and linear components.
    x_glu = x[..., ::2]
    x_linear = x[..., 1::2]

    # Clamp the gate from above and the linear component on both sides.
    x_glu = torch.clamp(x_glu, max=limit)
    x_linear = torch.clamp(x_linear, min=-limit, max=limit)

    # Apply gpt-oss's gated activation.
    return x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + 1)

# Step 14 - SwiGLUMLP
class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, hidden):
        super().__init__()

        # Project to 2 * hidden dimensions for interleaved gate and linear values.
        self.fc1 = nn.Linear(d_model, 2 * hidden)

        # Project the activated hidden representation back to d_model.
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        # Apply the first projection, gpt-oss SwiGLU activation,
        # and final projection.
        x = self.fc1(x)
        x = swiglu_oss(x)
        x = self.fc2(x)

        return x

# Step 15 - route_topk
def route_topk(router_logits, k):
    # Sort expert scores in descending order using a stable sort.
    # Equal scores retain their original order, so lower expert indices
    # come first when scores are tied.
    sorted_logits, sorted_indices = torch.sort(
        router_logits,
        dim=-1,
        descending=True,
        stable=True
    )

    # Keep only the top-k expert scores and their indices.
    topk_logits = sorted_logits[..., :k]
    indices = sorted_indices[..., :k]

    # Normalize the selected scores using softmax over the top-k experts.
    weights = torch.softmax(topk_logits, dim=-1)

    return weights, indices

# Step 16 - MoEMLP
class MoEMLP(nn.Module):
    def __init__(self, d_model, n_experts, k, hidden):
        super().__init__()

        # Number of experts selected for each token.
        self.k = k

        # Router produces one score per expert for each token.
        self.gate = nn.Linear(d_model, n_experts)

        # Create the expert feed-forward networks.
        self.experts = nn.ModuleList([
            SwiGLUMLP(d_model, hidden)
            for _ in range(n_experts)
        ])

    def forward(self, x):
        # Preserve the original shape so the output can be restored.
        original_shape = x.shape

        # Flatten all leading dimensions into a single token dimension.
        tokens = x.reshape(-1, original_shape[-1])

        # Select the top-k experts and normalize their routing weights.
        router_logits = self.gate(tokens)
        weights, indices = route_topk(router_logits, self.k)

        # Initialize the output for all tokens.
        output = torch.zeros_like(tokens)

        # Process only the tokens assigned to each expert.
        for expert_idx, expert in enumerate(self.experts):
            # Locate token and top-k slot pairs assigned to this expert.
            token_idx, slot_idx = torch.where(indices == expert_idx)

            # Skip experts that receive no tokens.
            if token_idx.numel() == 0:
                continue

            # Run this expert only on its assigned tokens.
            expert_input = tokens.index_select(0, token_idx)
            expert_output = expert(expert_input)

            # Weight each expert output by its routing probability.
            routing_weights = weights[token_idx, slot_idx].unsqueeze(-1)
            weighted_output = expert_output * routing_weights

            # Accumulate weighted outputs for the corresponding tokens.
            output.index_add_(0, token_idx, weighted_output)

        # Restore the original input shape.
        return output.reshape(original_shape)

# Step 17 - RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        # Learnable scale parameter, initialized to ones.
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # Preserve the input dtype for the final output.
        input_dtype = x.dtype

        # Perform normalization calculations in float32 for numerical stability.
        t = x.to(torch.float32)

        # Compute the root-mean-square normalization factor.
        rms_inv = torch.rsqrt(
            torch.mean(t * t, dim=-1, keepdim=True) + self.eps
        )

        # Normalize and apply the learnable scale, then restore input dtype.
        output = t * rms_inv * self.scale

        return output.to(input_dtype)

# Step 18 - attention_mask
def attention_mask(T, window=None):
    # Query positions: (T, 1); key positions: (1, T).
    positions = torch.arange(T)
    query_positions = positions.unsqueeze(1)
    key_positions = positions.unsqueeze(0)

    # Causal mask: each query can attend only to itself and earlier keys.
    allowed = key_positions <= query_positions

    # Sliding-window mask: retain only the last `window` positions,
    # including the current position.
    if window is not None:
        allowed = allowed & (key_positions > query_positions - window)

    return allowed

# Step 19 - OssAttention
class OssAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, window=None, sinks=False):
        super().__init__()

        # Grouped-query attention requires query heads to divide evenly
        # into groups sharing key/value heads.
        assert n_heads % n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads"
        )
        assert d_model % n_heads == 0, (
            "d_model must be divisible by n_heads"
        )

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.window = window

        # Query projection uses all attention heads.
        self.q = nn.Linear(d_model, n_heads * self.head_dim)

        # Key and value projections use fewer heads for GQA.
        self.k = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v = nn.Linear(d_model, n_kv_heads * self.head_dim)

        # Output projection merges the attention heads.
        self.out = nn.Linear(n_heads * self.head_dim, d_model)

        # One learned attention-sink logit per query head, when enabled.
        self.sinks = (
            nn.Parameter(torch.zeros(n_heads)) if sinks else None
        )

    def forward(self, x, inv_freq=None, concentration=1.0):
        # Input shape: (B, T, d_model).
        B, T, _ = x.shape

        # Project queries, keys, and values, then split into heads.
        q = self.q(x).reshape(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = self.k(x).reshape(
            B, T, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        v = self.v(x).reshape(
            B, T, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply rotary positional embeddings when inverse frequencies
        # are provided. Positions are shared across the batch.
        if inv_freq is not None:
            positions = torch.arange(T, device=x.device)

            q = apply_rope(
                q, inv_freq, positions, concentration
            )
            k = apply_rope(
                k, inv_freq, positions, concentration
            )

        # Expand each key/value head across its associated query heads.
        repeats = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

        # Scaled dot-product attention scores:
        # (B, n_heads, T, T).
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Apply the causal mask, optionally restricting attention
        # to the current position and the preceding window - 1 positions.
        allowed = attention_mask(T, self.window).to(device=x.device)
        scores = scores.masked_fill(
            ~allowed.unsqueeze(0).unsqueeze(0),
            float("-inf")
        )

        if self.sinks is not None:
            # Append one learned sink logit per attention head to every
            # query row. The sink has no associated value vector.
            sink_logits = self.sinks.to(
                device=scores.device, dtype=scores.dtype
            ).view(1, self.n_heads, 1, 1).expand(B, -1, T, -1)

            scores_with_sink = torch.cat(
                (scores, sink_logits), dim=-1
            )

            # Normalize over the real keys and the extra sink column.
            weights = torch.softmax(scores_with_sink, dim=-1)

            # Discard the sink weight: it consumes probability mass
            # but contributes no value to the weighted sum.
            weights = weights[..., :T]
        else:
            weights = torch.softmax(scores, dim=-1)

        # Weighted sum of values: (B, n_heads, T, head_dim).
        out = weights @ v

        # Merge heads and project back to d_model.
        out = out.transpose(1, 2).contiguous().reshape(
            B, T, self.n_heads * self.head_dim
        )

        return self.out(out)

# Step 20 - kv_cache_bytes
def kv_cache_bytes(n_kv_heads, head_dim, windows, context_len, bytes_per_value=2):
    total_bytes = 0

    # Calculate the KV-cache memory required by each layer.
    for window in windows:
        # Full attention caches the entire context; sliding-window
        # attention caches at most the configured window size.
        cached_tokens = (
            context_len if window is None
            else min(context_len, window)
        )

        # Both keys and values must be stored for every cached token.
        layer_bytes = (
            2
            * n_kv_heads
            * head_dim
            * cached_tokens
            * bytes_per_value
        )

        total_bytes += layer_bytes

    return int(total_bytes)

# Step 21 - TinyLM
class TinyBlock(nn.Module):
    def __init__(self, cfg, window):
        super().__init__()

        d_model = cfg["d_model"]
        dropout = cfg["dropout"]

        # Configurable normalization.
        if cfg["norm"] == "layer":
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        elif cfg["norm"] == "rms":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            raise ValueError(f"Unsupported normalization: {cfg['norm']}")

        # Grouped-query attention with optional sliding window and sinks.
        self.attn = OssAttention(
            d_model,
            cfg["n_heads"],
            cfg["n_kv_heads"],
            window=window,
            sinks=cfg["sinks"],
        )

        # Configurable feed-forward layer.
        if cfg["mlp"] == "gelu":
            self.mlp = GPT2MLP(d_model)
        elif cfg["mlp"] == "swiglu":
            self.mlp = SwiGLUMLP(d_model, cfg["hidden"])
        elif cfg["mlp"] == "moe":
            self.mlp = MoEMLP(
                d_model,
                cfg["n_experts"],
                cfg["top_k"],
                cfg["hidden"],
            )
        else:
            raise ValueError(f"Unsupported MLP type: {cfg['mlp']}")

        # Residual dropout.
        self.drop = nn.Dropout(dropout)

    def forward(self, x, inv_freq, concentration):
        # Pre-normalized attention with a residual connection.
        x = x + self.drop(
            self.attn(self.norm1(x), inv_freq, concentration)
        )

        # Pre-normalized feed-forward layer with a residual connection.
        x = x + self.drop(self.mlp(self.norm2(x)))

        return x


class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = dict(cfg)

        vocab_size = cfg["vocab_size"]
        d_model = cfg["d_model"]
        n_layers = cfg["n_layers"]
        block_size = cfg["block_size"]

        # Alternate sliding-window and full-attention layers.
        if cfg["window"] is None:
            self.windows = [None] * n_layers
        else:
            self.windows = [
                cfg["window"] if i % 2 == 0 else None
                for i in range(n_layers)
            ]

        self.wte = nn.Embedding(vocab_size, d_model)

        # Learned positional embeddings are only needed for learned mode.
        if cfg["pos"] == "learned":
            self.wpe = nn.Embedding(block_size, d_model)
        elif cfg["pos"] == "rope":
            self.wpe = None
        else:
            raise ValueError(f"Unsupported position type: {cfg['pos']}")

        self.drop = nn.Dropout(cfg["dropout"])

        self.blocks = nn.ModuleList([
            TinyBlock(cfg, window)
            for window in self.windows
        ])

        # Final normalization.
        if cfg["norm"] == "layer":
            self.norm_f = nn.LayerNorm(d_model)
        else:
            self.norm_f = RMSNorm(d_model)

        # Tied token embedding and output projection weights.
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.wte.weight

        # RoPE frequencies are a non-persistent buffer.
        self.register_buffer(
            "inv_freq",
            rope_inv_freq(
                d_model // cfg["n_heads"],
                cfg["rope_base"],
            ),
            persistent=False,
        )

        self.concentration = 1.0

        # GPT-2-style initialization.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_rope(self, inv_freq, concentration=1.0):
        # Update rotary frequencies and the attention concentration factor.
        self.inv_freq = inv_freq.to(
            device=self.wte.weight.device,
            dtype=torch.float32,
        )
        self.concentration = float(concentration)

    def forward(self, idx):
        # idx shape: (B, T).
        B, T = idx.shape

        # FIX: enforce block_size only for learned absolute positions.
        # RoPE supports sequence lengths greater than block_size.
        if self.wpe is not None and T > self.cfg["block_size"]:
            raise ValueError(
                f"Sequence length {T} exceeds block_size "
                f"{self.cfg['block_size']}"
            )

        # Token embeddings.
        x = self.wte(idx)

        # Add learned positions only when a position table exists.
        if self.wpe is not None:
            positions = torch.arange(T, device=idx.device)
            x = x + self.wpe(positions)

        x = self.drop(x)

        # Supply rotary frequencies only for RoPE models.
        inv_freq = self.inv_freq if self.cfg["pos"] == "rope" else None

        # Transformer blocks.
        for block in self.blocks:
            x = block(x, inv_freq, self.concentration)

        # Final normalization and vocabulary logits.
        x = self.norm_f(x)
        return self.head(x)

# Step 22 - param_report
def param_report(model):
    # Count all unique parameters; tied weights are counted only once.
    total = sum(p.numel() for p in model.parameters())

    # Dense models use all parameters during each token's forward pass.
    active = total

    # Subtract the parameters of experts that are not selected per token.
    for module in model.modules():
        if isinstance(module, MoEMLP):
            n_experts = len(module.experts)
            k = module.k

            # Each expert has the same architecture and parameter count.
            if n_experts > 0:
                expert_params = sum(
                    p.numel() for p in module.experts[0].parameters()
                )

                active -= (n_experts - k) * expert_params

    # Count token embedding parameters.
    embedding = model.wte.weight.numel()

    return {
        "total": int(total),
        "active": int(active),
        "embedding": int(embedding),
    }

