#!/usr/bin/env python3
"""Comprehensive evaluation suite for BioKnot on MMseqs2 5-fold cross-validation.

Computes:
1. 4-Class Out-of-Fold Metrics:
   - Macro-F1, Weighted-F1, Matthews Correlation Coefficient (MCC).
   - Expected Calibration Error (ECE) with post-hoc temperature scaling.
   - Full 4x4 Confusion Matrix.
   - Per-class Precision, Recall, F1, ROC-AUC, PR-AUC, and Support.
2. Binary Pre-Screening Metrics (Path A: Unknot vs. Any Knot):
   - ROC-AUC, PR-AUC, MCC.
   - Specificity at 90% and 95% knotting recall.
   - Estimated compute saved by routing low-risk unknots to early exit before ESMFold.
3. Baseline Ablation across identical 5 folds:
   - Logistic Regression on frozen mean-pooled ESM-2 embeddings out-of-fold.

Saves structured results to eval/metrics.json.
"""

import argparse
import json
import os
import sys

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoModel

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
from training.train import train_single_fold

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)
DATASET_PATH = os.path.join(REPO_ROOT, "data", "bioknot_dataset.json")
METRICS_PATH = os.path.join(EVAL_DIR, "metrics.json")
CKPT_DIR = os.path.join(EVAL_DIR, "checkpoints")


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == targets

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def extract_frozen_embeddings(
    sequences: List[str],
    tokenizer,
    esm_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Extract frozen mean-pooled ESM-2 embeddings for linear probe ablation."""
    esm_model.eval()
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            chunk = sequences[i : i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = esm_model(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            all_embeddings.append(pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def compute_binary_screening_metrics(
    y_true_multi: np.ndarray,
    probs_multi: np.ndarray,
) -> Dict:
    """Compute Path A binary pre-screening metrics (Unknot vs. Any Knot)."""
    # 0 = Unknot, 1/2/3 = Knot
    y_true_knot = (y_true_multi > 0).astype(int)
    # Knot probability = sum of probabilities of knotted classes (1, 2, 3)
    p_knot = probs_multi[:, 1:].sum(axis=1)

    roc_auc = float(roc_auc_score(y_true_knot, p_knot))
    p_curve, r_curve, _ = precision_recall_curve(y_true_knot, p_knot)
    pr_auc = float(auc(r_curve, p_curve))

    # Standard threshold 0.5 for binary classification
    y_pred_binary = (p_knot >= 0.5).astype(int)
    mcc = float(matthews_corrcoef(y_true_knot, y_pred_binary))

    fpr, tpr, thresholds = roc_curve(y_true_knot, p_knot)

    # Specificity = 1 - FPR
    specificities = 1.0 - fpr

    # Specificity at 90% knotting recall
    idx_90 = np.where(tpr >= 0.90)[0]
    if len(idx_90) > 0:
        spec_90 = float(np.max(specificities[idx_90]))
        thresh_90 = float(thresholds[idx_90[np.argmax(specificities[idx_90])]])
    else:
        spec_90 = 0.0
        thresh_90 = 0.5

    # Specificity at 95% knotting recall
    idx_95 = np.where(tpr >= 0.95)[0]
    if len(idx_95) > 0:
        spec_95 = float(np.max(specificities[idx_95]))
        thresh_95 = float(thresholds[idx_95[np.argmax(specificities[idx_95])]])
    else:
        spec_95 = 0.0
        thresh_95 = 0.5

    n_total = len(y_true_knot)
    n_unknots = int(np.sum(y_true_knot == 0))

    # Compute saved = fraction of proteins that exit early because p_knot < threshold
    saved_90_unknots = float(spec_90)
    saved_90_overall = float(np.mean(p_knot < thresh_90))

    saved_95_unknots = float(spec_95)
    saved_95_overall = float(np.mean(p_knot < thresh_95))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": mcc,
        "operating_points": {
            "at_90_percent_knot_recall": {
                "knot_recall": 0.90,
                "unknot_specificity": spec_90,
                "decision_threshold": thresh_90,
                "unknot_compute_saved_percent": spec_90 * 100.0,
                "overall_compute_saved_percent": saved_90_overall * 100.0,
            },
            "at_95_percent_knot_recall": {
                "knot_recall": 0.95,
                "unknot_specificity": spec_95,
                "decision_threshold": thresh_95,
                "unknot_compute_saved_percent": spec_95 * 100.0,
                "overall_compute_saved_percent": saved_95_overall * 100.0,
            },
        },
        "description": "Path A early-exit pre-screen: unknotted sequences bypass ESMFold 3D structure generation",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BioKnot on MMseqs2 5-fold cross-validation.")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, help="ESM-2 backbone model")
    parser.add_argument("--retrain", action="store_true", help="Force retraining of 5 folds even if checkpoints exist")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    device = get_device()
    print("==========================================================================")
    print("     BIOKNOT-CORE: MMSEQS2 5-FOLD STRATIFIED GROUP CV EVALUATION SUITE    ")
    print("==========================================================================")
    print(f"Device   : {device}")
    print(f"Backbone : {args.backbone}")

    # Ensure splits exist
    if not os.path.exists(SPLITS_PATH):
        print("[Eval] Generating MMseqs2 cluster splits...")
        create_splits(seed=args.seed)

    with open(DATASET_PATH, "r") as f:
        records = json.load(f)
    rec_by_id = {r["id"]: r for r in records}

    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)

    folds = splits.get("folds", [])
    if not folds:
        raise ValueError("splits.json does not contain valid 5-fold CV splits.")

    os.makedirs(CKPT_DIR, exist_ok=True)
    tokenizer = load_tokenizer(args.backbone)

    oof_predictions = {}
    champion_f1 = -1.0
    champion_ckpt = None

    print(f"\n--- Stage 1: Fine-Tuned BioKnotClassifier 5-Fold Cross Validation ---")
    for f_info in folds:
        f_idx = f_info["fold"]
        fold_ckpt = os.path.join(CKPT_DIR, f"model_fold_{f_idx}.pt")
        oof_save_path = os.path.join(CKPT_DIR, f"oof_fold_{f_idx}.pt")

        train_records = [rec_by_id[k] for k in f_info["train"] if k in rec_by_id]
        val_records = [rec_by_id[k] for k in f_info["val"] if k in rec_by_id]

        # Check if we should retrain or load existing fold checkpoint
        if args.retrain or not os.path.exists(fold_ckpt) or not os.path.exists(oof_save_path):
            print(f"[Fold {f_idx}] Training model on fold {f_idx}...")
            model, val_f1, val_logits, val_targets, val_ids, temp = train_single_fold(
                fold_idx=f_idx,
                train_records=train_records,
                val_records=val_records,
                backbone=args.backbone,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=5,
                seed=args.seed,
                device=device,
                save_path=fold_ckpt,
            )
            torch.save({
                "logits": val_logits,
                "targets": val_targets,
                "ids": val_ids,
                "temperature": temp,
                "macro_f1": val_f1,
            }, oof_save_path)
        else:
            print(f"[Fold {f_idx}] Loading cached fold predictions from {oof_save_path}...")
            oof_data = torch.load(oof_save_path, map_location="cpu")
            val_logits = oof_data["logits"]
            val_targets = oof_data["targets"]
            val_ids = oof_data["ids"]
            temp = oof_data.get("temperature", 1.0)
            val_f1 = oof_data.get("macro_f1", 0.0)

        # Calibrated softmax probabilities via temperature scaling
        scaled_logits = val_logits / max(temp, 0.1)
        scaled_probs = torch.softmax(scaled_logits, dim=-1).numpy()

        for s_id, prob, target in zip(val_ids, scaled_probs, val_targets):
            oof_predictions[s_id] = {
                "prob": prob,
                "target": target,
                "pred": int(np.argmax(prob)),
            }

        if val_f1 > champion_f1:
            champion_f1 = val_f1
            champion_ckpt = fold_ckpt

    # Update bioknot_native.pt if champion checkpoint exists
    if champion_ckpt and os.path.exists(champion_ckpt):
        torch.save(torch.load(champion_ckpt, map_location="cpu"), DEFAULT_WEIGHTS_PATH)
        print(f"[Eval] Synced champion model (Fold Val Macro-F1: {champion_f1:.4f}) to {DEFAULT_WEIGHTS_PATH}")

    # Gather ordered out-of-fold predictions for all dataset records
    y_true_all = []
    y_pred_all = []
    probs_all = []

    for r in records:
        s_id = r["id"]
        res = oof_predictions[s_id]
        y_true_all.append(res["target"])
        y_pred_all.append(res["pred"])
        probs_all.append(res["prob"])

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    y_probs = np.vstack(probs_all)

    # 1. 4-Class Out-of-Fold Overall Metrics
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    ece = compute_ece(y_probs, y_true)
    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()

    # Per-class metrics
    per_class = {}
    for c in range(NUM_CLASSES):
        class_name = CLASS_MAP.get(c, f"Class {c}")
        y_true_binary = (y_true == c).astype(int)
        y_pred_binary = (y_pred == c).astype(int)
        y_score_c = y_probs[:, c]

        prec = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
        rec = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
        f1_c = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

        try:
            if len(np.unique(y_true_binary)) > 1:
                roc_auc = float(roc_auc_score(y_true_binary, y_score_c))
                p_curve, r_curve, _ = precision_recall_curve(y_true_binary, y_score_c)
                pr_auc = float(auc(r_curve, p_curve))
            else:
                roc_auc = 0.5
                pr_auc = 0.0
        except Exception:
            roc_auc = 0.5
            pr_auc = 0.0

        per_class[class_name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1_c,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "support": int(np.sum(y_true_binary)),
        }

    # 2. Binary Pre-Screening Metrics (Path A: Unknot vs. Any Knot)
    binary_metrics = compute_binary_screening_metrics(y_true, y_probs)

    # 3. Baseline Ablation: Frozen ESM-2 Linear Probe on exact same 5 folds
    print("\n--- Stage 2: Frozen ESM-2 Linear Probe Ablation (Identical 5 Folds) ---")
    base_esm = AutoModel.from_pretrained(args.backbone).to(device)
    base_esm.eval()

    all_seqs = [r["sequence"] for r in records]
    all_embeddings = extract_frozen_embeddings(all_seqs, tokenizer, base_esm, device)
    id_to_idx = {r["id"]: i for i, r in enumerate(records)}

    baseline_oof_preds = [None] * len(records)
    for f_info in folds:
        train_indices = [id_to_idx[k] for k in f_info["train"] if k in id_to_idx]
        val_indices = [id_to_idx[k] for k in f_info["val"] if k in id_to_idx]

        X_tr = all_embeddings[train_indices]
        y_tr = y_true[train_indices]
        X_v = all_embeddings[val_indices]

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
        clf.fit(X_tr, y_tr)
        val_preds = clf.predict(X_v)

        for idx, pred in zip(val_indices, val_preds):
            baseline_oof_preds[idx] = int(pred)

    baseline_y_pred = np.array(baseline_oof_preds)
    base_macro_f1 = float(f1_score(y_true, baseline_y_pred, average="macro", zero_division=0))
    base_weighted_f1 = float(f1_score(y_true, baseline_y_pred, average="weighted", zero_division=0))
    base_mcc = float(matthews_corrcoef(y_true, baseline_y_pred))

    base_per_class = {}
    for c in range(NUM_CLASSES):
        class_name = CLASS_MAP.get(c, f"Class {c}")
        y_true_b = (y_true == c).astype(int)
        y_pred_b = (baseline_y_pred == c).astype(int)
        base_per_class[class_name] = {
            "f1": float(f1_score(y_true_b, y_pred_b, zero_division=0)),
            "precision": float(precision_score(y_true_b, y_pred_b, zero_division=0)),
            "recall": float(recall_score(y_true_b, y_pred_b, zero_division=0)),
            "support": int(np.sum(y_true_b)),
        }

    metrics_payload = {
        "dataset": {
            "total_records": len(records),
            "clustering": splits.get("clustering_method", "mmseqs easy-cluster"),
            "mmseqs_version": splits.get("mmseqs_version", "unknown"),
            "mmseqs_flags": splits.get("mmseqs_flags", "--min-seq-id 0.3 -c 0.8 --cov-mode 0"),
            "total_clusters": splits.get("stats", {}).get("total_clusters", 470),
            "evaluation_strategy": "5-fold StratifiedGroupKFold (out-of-fold)",
            "n_folds": len(folds),
        },
        "model_performance": {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "mcc": mcc,
            "expected_calibration_error": ece,
            "per_class": per_class,
            "confusion_matrix": conf_mat,
        },
        "binary_prescreening": binary_metrics,
        "baseline_ablation": {
            "description": "Linear probe (Logistic Regression) on frozen mean-pooled ESM-2 embeddings across identical 5 folds",
            "macro_f1": base_macro_f1,
            "weighted_f1": base_weighted_f1,
            "mcc": base_mcc,
            "per_class": base_per_class,
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n==========================================================================")
    print("BIOKNOT-CORE 5-FOLD OUT-OF-FOLD EVALUATION SUMMARY:")
    print("==========================================================================")
    print(f"BioKnotClassifier (Fine-Tuned):")
    print(f"  Macro-F1    : {macro_f1:.4f}")
    print(f"  Weighted-F1 : {weighted_f1:.4f}")
    print(f"  MCC         : {mcc:.4f}")
    print(f"  ECE         : {ece:.4f}")
    print("\nFrozen ESM-2 Linear Probe Baseline:")
    print(f"  Macro-F1    : {base_macro_f1:.4f}")
    print(f"  Weighted-F1 : {base_weighted_f1:.4f}")
    print(f"  MCC         : {base_mcc:.4f}")
    print("\nPer-Class Breakdown (Out-of-Fold Fine-Tuned):")
    for c_name, vals in per_class.items():
        print(
            f"  {c_name:<20}: F1={vals['f1']:.4f} | Prec={vals['precision']:.4f} | "
            f"Rec={vals['recall']:.4f} | ROC-AUC={vals['roc_auc']:.4f} | PR-AUC={vals['pr_auc']:.4f} (N={vals['support']})"
        )
    print(f"\n4x4 Confusion Matrix (Rows=True, Cols=Pred):\n{np.array(conf_mat)}")
    print("\nBinary Pre-Screening Metrics (Path A: Unknot vs Knot):")
    print(f"  ROC-AUC     : {binary_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC      : {binary_metrics['pr_auc']:.4f}")
    print(f"  MCC         : {binary_metrics['mcc']:.4f}")
    op90 = binary_metrics["operating_points"]["at_90_percent_knot_recall"]
    op95 = binary_metrics["operating_points"]["at_95_percent_knot_recall"]
    print(f"  At 90% Knot Recall -> Unknot Specificity: {op90['unknot_specificity']*100:.2f}% | Overall Compute Saved: {op90['overall_compute_saved_percent']:.2f}%")
    print(f"  At 95% Knot Recall -> Unknot Specificity: {op95['unknot_specificity']*100:.2f}% | Overall Compute Saved: {op95['overall_compute_saved_percent']:.2f}%")
    print(f"\nSaved updated metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
