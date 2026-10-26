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

