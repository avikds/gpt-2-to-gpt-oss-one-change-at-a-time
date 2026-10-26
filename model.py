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

