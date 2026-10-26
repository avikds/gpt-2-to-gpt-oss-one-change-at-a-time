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

