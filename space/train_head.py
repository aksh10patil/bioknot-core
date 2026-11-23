"""Train the classifier head, used as a first-boot fallback inside the Space.

Mirrors `training/train.py` in the main repo.

MUST be run as a subprocess, never imported by `app.py`. `app.py` imports
`spaces`, which patches torch to report CUDA as the active accelerator; the Adam
step then runs `_accelerator_graph_capture_health_check()`, calls into CUDA from
the CPU process, and raises `no CUDA-capable device is detected`. A fresh
interpreter that never imports `spaces` sees plain CPU torch and trains fine.
"""

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bioknot_model import MAX_TOKENS, NUM_CLASSES, BioKnotClassifier, load_tokenizer

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(MODULE_DIR, "data", "bioknot_dataset.json")


class _KnotDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["sequence"], self.data[idx]["label"]


def _collate(tokenizer):
    def collate(batch):
        sequences = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tokenizer(
            sequences,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding="longest",
        )
        return enc["input_ids"], enc["attention_mask"], labels

    return collate


def train(output_path, epochs=30, batch_size=2, lr=1e-3, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = load_tokenizer()
    model = BioKnotClassifier(num_classes=NUM_CLASSES)
    dataset = _KnotDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate(tokenizer))

    counts = {}
    for _, label in dataset:
        counts[label] = counts.get(label, 0) + 1

    total_samples = len(dataset)
    class_weights = torch.zeros(NUM_CLASSES, dtype=torch.float)
    for c in range(NUM_CLASSES):
        count = counts.get(c, 0)
        class_weights[c] = (total_samples / (NUM_CLASSES * count)) if count > 0 else 1.0

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    esm_params = [p for p in model.esm.parameters() if p.requires_grad]
    head_params = list(model.conv1.parameters()) + list(model.fc.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": esm_params, "lr": 2e-5, "weight_decay": 0.01},
            {"params": head_params, "lr": 5e-4, "weight_decay": 0.01},
        ]
    )

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for input_ids, attention_mask, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(input_ids, attention_mask), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"[train_head] epoch {epoch:02d}/{epochs} loss {total / len(loader):.4f}", flush=True)

    torch.save(model.state_dict(), output_path)
    print(f"[train_head] saved {output_path}", flush=True)
    return output_path


if __name__ == "__main__":
    if "spaces" in sys.modules:
        raise RuntimeError("train_head must run in an interpreter that has not imported `spaces`.")
    train(sys.argv[1] if len(sys.argv) > 1 else os.path.join(MODULE_DIR, "bioknot_native.pt"))
