"""Warm the Hugging Face cache at image build time.

Without this, every cold start of the Space downloads the ESM-2 backbone, and
each gunicorn worker does so independently. Baking it into a Docker layer makes
boots fast and offline-safe.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import ESM_CHECKPOINT, load_tokenizer  # noqa: E402
from transformers import EsmModel  # noqa: E402

if __name__ == "__main__":
    print(f"[Preload] Fetching {ESM_CHECKPOINT} into {os.environ.get('HF_HOME', 'default cache')}...")
    load_tokenizer()
    EsmModel.from_pretrained(ESM_CHECKPOINT)
    print("[Preload] Backbone cached.")
