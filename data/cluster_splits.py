#!/usr/bin/env python3
"""Sequence-identity clustering and homology-aware train/val/test splits (<= 30% identity).

Clusters all sequences at <= 30% sequence identity using MMseqs2:
  `mmseqs easy-cluster <fasta> <prefix> <tmp> --min-seq-id 0.3 -c 0.8 --cov-mode 0`

Partitions the dataset into 5 folds using `StratifiedGroupKFold` grouped by MMseqs2
cluster ID, guaranteeing zero homologous cluster leakage across folds while ensuring
all 1,078 sequences (including all 58 Figure-Eight and 24 Complex chains) are evaluated
out-of-fold. Persists splits and cluster metadata to `data/splits.json`.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(DATA_DIR, "bioknot_dataset.json")
SPLITS_PATH = os.path.join(DATA_DIR, "splits.json")


def get_mmseqs_version():
    """Query mmseqs executable version string."""
    try:
        res = subprocess.run(["mmseqs", "version"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def cluster_sequences_mmseqs(records, min_seq_id=0.30, coverage=0.80, cov_mode=0):
    """Cluster sequences at <= 30% identity using mmseqs easy-cluster."""
    mmseqs_bin = shutil.which("mmseqs")
    if not mmseqs_bin:
        raise RuntimeError(
            "mmseqs was not found on PATH. Please install MMseqs2 (e.g. 'brew install mmseqs2' "
            "or 'conda install -c bioconda mmseqs2') or pass --allow-kmer-fallback to explicitly "
            "allow fallback clustering."
        )

    version = get_mmseqs_version()
    flags = f"--min-seq-id {min_seq_id} -c {coverage} --cov-mode {cov_mode}"
    print(f"[Clustering] Found MMseqs2 ({version}) at: {mmseqs_bin}")
    print(f"[Clustering] Running: mmseqs easy-cluster <fasta> <prefix> <tmp> {flags}")

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, "bioknot_input.fasta")
        with open(fasta_path, "w") as f:
            for r in records:
                f.write(f">{r['id']}\n{r['sequence']}\n")

        prefix = os.path.join(tmpdir, "mmseqs_clu")
        cmd = [
            mmseqs_bin,
            "easy-cluster",
            fasta_path,
            prefix,
            tmpdir,
            "--min-seq-id",
            str(min_seq_id),
            "-c",
            str(coverage),
            "--cov-mode",
            str(cov_mode),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"mmseqs easy-cluster failed with code {result.returncode}:\n{result.stderr}")

        tsv_path = f"{prefix}_cluster.tsv"
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"Expected mmseqs cluster output not found at: {tsv_path}")

        clusters_map = defaultdict(list)
        seq_to_rep = {}
        with open(tsv_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rep, member = parts[0], parts[1]
                    clusters_map[rep].append(member)
                    seq_to_rep[member] = rep

    clusters = list(clusters_map.values())
    meta = {
        "method": "mmseqs easy-cluster",
        "version": version,
        "flags": flags,
        "min_seq_id": min_seq_id,
        "coverage": coverage,
        "cov_mode": cov_mode,
        "seq_to_rep": seq_to_rep,
    }
    return clusters, meta


def cluster_sequences_fallback(records, min_seq_id=0.30):
    """Fallback k-mer graph-component clustering if explicitly allowed."""
    print("[Clustering] [WARN] Using k-mer graph-component fallback clustering at <= 30% sequence identity.")
    n = len(records)
    k = 5
    kmers = [
        set(records[i]["sequence"][j : j + k] for j in range(len(records[i]["sequence"]) - k + 1))
        for i in range(n)
    ]

    adj = defaultdict(list)
    for i in range(n):
        ki = kmers[i]
        len_ki = len(ki)
        for j in range(i + 1, n):
            kj = kmers[j]
            inter = len(ki & kj)
            sim = inter / min(len_ki, len(kj))
            if sim >= min_seq_id:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    clusters = []
    seq_to_rep = {}
    for i in range(n):
        if not visited[i]:
            comp = []
            queue = [i]
            visited[i] = True
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            rep_id = records[comp[0]]["id"]
            members = [records[idx]["id"] for idx in comp]
            for m in members:
                seq_to_rep[m] = rep_id
            clusters.append(members)

    meta = {
        "method": "k-mer graph-component (fallback, <=30% identity)",
        "version": "fallback",
        "flags": "k=5 graph component",
        "min_seq_id": min_seq_id,
        "coverage": 0.8,
        "cov_mode": 0,
        "seq_to_rep": seq_to_rep,
    }
    return clusters, meta


def create_splits(seed=42, allow_kmer_fallback=False, n_splits=5):
    """Generate 5-fold StratifiedGroupKFold splits grouped by sequence clusters."""
    random.seed(seed)
    np.random.seed(seed)

    with open(DATASET_PATH, "r") as f:
        records = json.load(f)

    id_to_record = {r["id"]: r for r in records}

    if shutil.which("mmseqs") is None:
        if not allow_kmer_fallback:
            raise RuntimeError(
                "mmseqs is not installed on PATH. Falling back to k-mer clustering is disabled by default "
                "to prevent silent degradation of research evaluation. Install mmseqs2 or pass --allow-kmer-fallback."
            )
        clusters, meta = cluster_sequences_fallback(records, min_seq_id=0.30)
    else:
        clusters, meta = cluster_sequences_mmseqs(records, min_seq_id=0.30, coverage=0.80, cov_mode=0)

    seq_to_cluster = meta["seq_to_rep"]
    print(f"[Clustering] Partitioned {len(records)} sequences into {len(clusters)} clusters.")

    # Prepare inputs for StratifiedGroupKFold
    seq_ids = np.array([r["id"] for r in records])
    labels = np.array([r["label"] for r in records])
    group_ids = np.array([seq_to_cluster[r["id"]] for r in records])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    seen_val_ids = set()

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(seq_ids, labels, group_ids)):
        fold_train_ids = sorted(seq_ids[train_idx].tolist())
        fold_val_ids = sorted(seq_ids[val_idx].tolist())

        train_clusters = set(group_ids[train_idx])
        val_clusters = set(group_ids[val_idx])

        # Strict cluster leakage assertion
        leakage = train_clusters & val_clusters
        assert len(leakage) == 0, f"Fold {fold_idx}: Homologous cluster leakage detected! Overlap: {len(leakage)}"

        # Validate disjoint out-of-fold sets
        assert len(seen_val_ids & set(fold_val_ids)) == 0, f"Fold {fold_idx}: Overlapping validation IDs across folds!"
        seen_val_ids.update(fold_val_ids)

        train_label_dist = dict(Counter(labels[train_idx]))
        val_label_dist = dict(Counter(labels[val_idx]))

        folds.append({
            "fold": fold_idx,
            "train": fold_train_ids,
            "val": fold_val_ids,
            "train_clusters": len(train_clusters),
            "val_clusters": len(val_clusters),
            "train_size": len(fold_train_ids),
            "val_size": len(fold_val_ids),
            "train_label_distribution": {int(k): int(v) for k, v in train_label_dist.items()},
            "val_label_distribution": {int(k): int(v) for k, v in val_label_dist.items()},
        })

    # Assert complete coverage across all sequences
    assert len(seen_val_ids) == len(records), f"Folds did not cover all sequences: {len(seen_val_ids)} vs {len(records)}"

    payload = {
        "clustering_method": meta["method"],
        "mmseqs_version": meta["version"],
        "mmseqs_flags": meta["flags"],
        "seed": seed,
        "n_splits": n_splits,
        "stats": {
            "total_sequences": len(records),
            "total_clusters": len(clusters),
        },
        "folds": folds,
        # Backward-compatible references (using Fold 0)
        "train": folds[0]["train"],
        "val": folds[0]["val"],
        "test": folds[0]["val"],
    }

    with open(SPLITS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n[Splits Generator] 5-Fold StratifiedGroupKFold Summary:")
    print(f"  Clustering Method : {meta['method']}")
    print(f"  MMseqs2 Version   : {meta['version']}")
    print(f"  MMseqs2 Flags     : {meta['flags']}")
    print(f"  Total Sequences   : {len(records)}")
    print(f"  Total Clusters    : {len(clusters)}")
    for f_info in folds:
        print(
            f"  Fold {f_info['fold']}: Train={f_info['train_size']} ({f_info['train_clusters']} clu) | "
            f"Val={f_info['val_size']} ({f_info['val_clusters']} clu) | Val Dist={f_info['val_label_distribution']}"
        )
    print(f"[Splits Generator] Zero cluster leakage verified across all {n_splits} folds.")
    print(f"[Splits Generator] Saved verified splits to {SPLITS_PATH}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Create MMseqs2 30% sequence identity 5-fold splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--allow-kmer-fallback",
        action="store_true",
        help="Allow k-mer graph-component fallback clustering if mmseqs is missing",
    )
    args = parser.parse_args()
    create_splits(seed=args.seed, allow_kmer_fallback=args.allow_kmer_fallback)


if __name__ == "__main__":
    main()
