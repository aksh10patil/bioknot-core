"""Fine-tune the BioKnot topological classifier head on the curated dataset.

Run from the repository root:

    python -m training.train

This is executed during the Docker build so the image ships with a deterministic
checkpoint rather than a randomly initialised head.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import (  # noqa: E402
    CLASS_MAP,
    MAX_TOKENS,
    NUM_CLASSES,
    BioKnotClassifier,
    load_tokenizer,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(REPO_ROOT, "data", "bioknot_dataset.json")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "bioknot_native.pt")


class ProteinKnotDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["sequence"], item["label"]


def make_collate_fn(tokenizer):
    """Pad to the longest sequence in the batch rather than to MAX_TOKENS.

    The curated sequences are 76-410 AA, so padding every one to 1024 tokens
    wasted roughly 3x the compute for identical results.
    """

    def collate(batch):
        sequences = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        encoding = tokenizer(
            sequences,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding="longest",
        )
        return encoding["input_ids"], encoding["attention_mask"], labels

    return collate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    parser = argparse.ArgumentParser(description="Train the BioKnot native classifier.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print("[Training Engine] Initialising ESM-2 fine-tuning pipeline...")
    tokenizer = load_tokenizer()
    model = BioKnotClassifier(num_classes=NUM_CLASSES)

    dataset = ProteinKnotDataset(args.dataset)
    counts = {}
    for _, label in dataset:
        counts[label] = counts.get(label, 0) + 1

    print(f"[Training Engine] Dataset: {len(dataset)} sequences from {args.dataset}")
    for label in sorted(counts):
        print(f"    class {label} ({CLASS_MAP.get(label, '?')}): {counts[label]}")

    if len(dataset) < 100:
        print(
            f"\n[Training Engine] WARNING: {len(dataset)} training sequences is far too few to "
            "generalise. The resulting checkpoint memorises the training set and should be "
            "treated as a demonstration artefact, not a validated classifier.\n"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    print(f"[Training Engine] Fine-tuning across {args.epochs} epochs...")
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for input_ids, attention_mask, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0 or epoch == 1:
            print(f" -> Epoch {epoch:02d}/{args.epochs} | Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\n[Training Engine] Complete. Weights saved to {args.output}")


if __name__ == "__main__":
    main()
