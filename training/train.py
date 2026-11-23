"""Fine-tune the BioKnot topological classifier on MMseqs2 cluster splits.

Applies:
- Device acceleration (CUDA -> MPS -> CPU).
- Support for configurable ESM-2 backbones (--backbone).
- MMseqs2 cluster-stratified 5-fold cross validation.
- WeightedRandomSampler with smoothed inverse-frequency weights to prevent minority collapse.
- Focal Loss (gamma=2.0) with class weighting and label smoothing.
- Differential learning rates (2e-5 for unfrozen ESM layers, 5e-4 for Conv1D and Linear head).
- CosineAnnealingLR schedule.
- Gradient clipping (max_norm=1.0).
- Early stopping based strictly on validation Macro-F1.
- Post-hoc temperature scaling for probability calibration.
"""

import argparse
import json
import os
import random
import sys

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import (
    CLASS_MAP,
    DEFAULT_BACKBONE,
    DEFAULT_WEIGHTS_PATH,
    MAX_TOKENS,
    NUM_CLASSES,
    BioKnotClassifier,
    get_device,
    load_tokenizer,
)
from data.cluster_splits import SPLITS_PATH, create_splits

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATASET = os.path.join(REPO_ROOT, "data", "bioknot_dataset.json")


class ProteinKnotDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[str, int, str]:
        item = self.records[idx]
        return item["sequence"], item["label"], item["id"]


def make_collate_fn(tokenizer):
    def collate(batch):
        sequences = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        ids = [b[2] for b in batch]
        encoding = tokenizer(
            sequences,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
            padding="longest",
        )
        return encoding["input_ids"], encoding["attention_mask"], labels, ids

    return collate


class FocalLoss(nn.Module):
    """Focal Loss with class weights and label smoothing."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - target_probs) ** self.gamma
        if self.alpha is not None:
            alpha_factor = self.alpha[targets]
            loss = -alpha_factor * focal_weight * target_log_probs
        else:
            loss = -focal_weight * target_log_probs

        if self.label_smoothing > 0.0:
            smooth_loss = -log_probs.mean(dim=-1)
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth_loss

        return loss.mean()


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling fitted on validation logits."""

    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 100) -> float:
        val_logits = val_logits.cpu()
        val_labels = val_labels.cpu()
        self.to("cpu")
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_val(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []
    all_ids = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for input_ids, attention_mask, labels, seq_ids in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())
            all_ids.extend(seq_ids)

    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    avg_loss = total_loss / max(1, len(dataloader))
    concat_logits = torch.cat(all_logits, dim=0)
    return avg_loss, macro_f1, concat_logits, np.array(all_targets), all_ids


def train_single_fold(
    fold_idx: int,
    train_records: List[Dict],
    val_records: List[Dict],
    backbone: str,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    device: torch.device,
    save_path: Optional[str] = None,
) -> Tuple[BioKnotClassifier, float, torch.Tensor, np.ndarray, List[str], float]:
    """Train on one fold with WeightedRandomSampler, Focal Loss, and differential LRs."""
    set_seed(seed + fold_idx)
    tokenizer = load_tokenizer(backbone)
    model = BioKnotClassifier(num_classes=NUM_CLASSES, backbone=backbone).to(device)

    train_labels = [r["label"] for r in train_records]
    train_counts = Counter(train_labels)

    # Smoothed inverse frequency class weights
    total_train = len(train_records)
    class_weights = torch.zeros(NUM_CLASSES, dtype=torch.float)
    for c in range(NUM_CLASSES):
        count = train_counts.get(c, 0)
        # Smoothed square-root inverse weight
        class_weights[c] = (total_train / (NUM_CLASSES * max(count, 1))) ** 0.5
    class_weights = class_weights / class_weights.mean()

    # Per-sample weights for balanced batch sampling
    sample_weights = [class_weights[lbl].item() for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_records),
        replacement=True,
    )

    train_loader = DataLoader(
        ProteinKnotDataset(train_records),
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=make_collate_fn(tokenizer),
    )
    val_loader = DataLoader(
        ProteinKnotDataset(val_records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
    )

    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0, label_smoothing=0.05)

    esm_params = [p for p in model.esm.parameters() if p.requires_grad]
    head_params = list(model.conv1.parameters()) + list(model.fc.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": esm_params, "lr": 2e-5, "weight_decay": 0.01},
            {"params": head_params, "lr": 5e-4, "weight_decay": 0.01},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    patience_counter = 0
    best_state_dict = None
    best_val_logits = None
    best_val_targets = None
    best_val_ids = None

    def clear_device_cache():
        if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[Fold {fold_idx}] Fine-tuning with backbone {backbone} ({len(train_records)} train, {len(val_records)} val)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for step, (input_ids, attention_mask, labels, _) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            if (step + 1) % 10 == 0:
                clear_device_cache()

        scheduler.step()
        clear_device_cache()
        avg_train_loss = train_loss / max(1, len(train_loader))
        val_loss, val_macro_f1, val_logits, val_targets, val_ids = evaluate_val(model, val_loader, device)
        clear_device_cache()

        print(
            f"  [Fold {fold_idx}] Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_logits = val_logits
            best_val_targets = val_targets
            best_val_ids = val_ids
            if save_path:
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                torch.save(best_state_dict, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [Fold {fold_idx}] Early stopping triggered after {patience} epochs without Macro-F1 improvement.")
                break

    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Post-hoc temperature scaling calibration
    scaler = TemperatureScaler()
    temperature = scaler.fit(best_val_logits, torch.tensor(best_val_targets, dtype=torch.long))
    print(f"  [Fold {fold_idx}] Best Val Macro-F1: {best_val_f1:.4f} | Fitted Temperature: {temperature:.3f}")

    return model, best_val_f1, best_val_logits, best_val_targets, best_val_ids, temperature


def main():
    parser = argparse.ArgumentParser(description="Train BioKnot native classifier with MMseqs2 5-fold CV.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to bioknot_dataset.json")
    parser.add_argument("--splits", default=SPLITS_PATH, help="Path to splits.json")
    parser.add_argument("--output", default=DEFAULT_WEIGHTS_PATH, help="Path to save main bioknot_native.pt")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, help="ESM-2 backbone checkpoint")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fold", type=int, default=-1, help="Specific fold to train (0-4), or -1 for all 5 folds")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[Training Engine] Using device: {device} | Backbone: {args.backbone}")

    if not os.path.exists(args.splits):
        print(f"[Training Engine] Splits file {args.splits} not found. Generating MMseqs2 splits...")
        create_splits(seed=args.seed)

    with open(args.dataset, "r") as f:
        all_records = json.load(f)
    rec_by_id = {r["id"]: r for r in all_records}

    with open(args.splits, "r") as f:
        splits = json.load(f)

    folds = splits.get("folds", [])
    if not folds:
        raise ValueError("No 'folds' array found in splits.json. Re-run data/cluster_splits.py.")

    ckpt_dir = os.path.join(REPO_ROOT, "eval", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    target_folds = range(len(folds)) if args.fold == -1 else [args.fold]
    best_overall_f1 = -1.0
    best_overall_state = None

    for f_idx in target_folds:
        f_info = folds[f_idx]
        train_records = [rec_by_id[k] for k in f_info["train"] if k in rec_by_id]
        val_records = [rec_by_id[k] for k in f_info["val"] if k in rec_by_id]

        fold_ckpt = os.path.join(ckpt_dir, f"model_fold_{f_idx}.pt")
        model, val_f1, val_logits, val_targets, val_ids, temp = train_single_fold(
            fold_idx=f_idx,
            train_records=train_records,
            val_records=val_records,
            backbone=args.backbone,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seed=args.seed,
            device=device,
            save_path=fold_ckpt,
        )

        # Save fold out-of-fold predictions
        oof_fold_path = os.path.join(ckpt_dir, f"oof_fold_{f_idx}.pt")
        torch.save({
            "logits": val_logits,
            "targets": val_targets,
            "ids": val_ids,
            "temperature": temp,
            "macro_f1": val_f1,
        }, oof_fold_path)

        if val_f1 > best_overall_f1:
            best_overall_f1 = val_f1
            best_overall_state = torch.load(fold_ckpt, map_location="cpu")

    # Save the highest-performing fold checkpoint to the main serving weights path
    if best_overall_state is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        torch.save(best_overall_state, args.output)
        print(f"\n[Training Engine] Saved champion checkpoint (Macro-F1: {best_overall_f1:.4f}) to {args.output}")


if __name__ == "__main__":
    main()
