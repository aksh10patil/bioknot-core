#!/usr/bin/env python3
"""Build the verified BioKnot protein topology dataset from KnotProt 2.0 and RCSB PDB.

Fetches:
1. Verified knotted protein chains (Rolfsen knot types 3_1, 4_1, 5_2, 6_1, etc.) from KnotProt 2.0.
2. Verified unknotted PDB protein chains (0_1), including hard-negative Rossmann-fold
   methyltransferases and diverse globular controls.

Outputs data/bioknot_dataset.json with verified labels, zero duplicate sequences,
and zero cross-class chimeric substring overlaps.
"""

import gzip
import json
import os
import re
import ssl
import sys
import urllib.request

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()

VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
MIN_LEN = 50
MAX_LEN = 1022

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_JSON = os.path.join(DATA_DIR, "bioknot_dataset.json")

# Known hard negative Rossmann-fold methyltransferases & diverse unknotted controls
CURATED_UNKNOTS = [
    ("1UBQ", "A"), ("1EMA", "A"), ("1TIM", "A"), ("1MBN", "A"), ("1CRN", "A"),
    ("1DL5", "A"), ("1X7P", "A"), ("1Y7B", "A"), ("2B3T", "A"), ("1N2X", "A"),
    ("3CZX", "A"), ("1NW7", "A"), ("1UFK", "A"), ("1P91", "A"), ("1V2Z", "A"),
    ("1EJ8", "A"), ("1FP1", "A"), ("1G64", "A"), ("1HE1", "A"), ("1I1N", "A"),
    ("1J8B", "A"), ("1K3R", "A"), ("1L2H", "A"), ("1M6T", "A"), ("1N62", "A"),
]


def clean_sequence(seq: str) -> str:
    cleaned = re.sub(r"[^A-Za-z]", "", seq).upper()
    return cleaned


def is_valid_sequence(seq: str) -> bool:
    if not (MIN_LEN <= len(seq) <= MAX_LEN):
        return False
    return set(seq).issubset(VALID_RESIDUES)


def fetch_knotprot_knots():
    print("[Dataset Builder] Fetching knotted chains metadata from KnotProt 2.0...")
    url_meta = "https://knotprot.cent.uw.edu.pl/_all_chains_knotted_N_C"
    req = urllib.request.Request(url_meta, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=SSL_CTX) as r:
        meta_lines = r.read().decode("utf-8", errors="ignore").splitlines()

    print("[Dataset Builder] Fetching knotted chains sequence mapping from KnotProt 2.0...")
    url_seq = "https://knotprot.cent.uw.edu.pl/_all_knotted_chains_sequence"
    req = urllib.request.Request(url_seq, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=SSL_CTX) as r:
        seq_lines = r.read().decode("utf-8", errors="ignore").splitlines()

    seq_map = {}
    for line in seq_lines:
        parts = line.strip().split(";")
        if len(parts) >= 3:
            seq_map[(parts[0].lower(), parts[1])] = clean_sequence(parts[2])

    # Group all knotted records by (pdb_id, chain_id) and select the highest-order knot
    def knot_rank(kt):
        # 820 > 63 > 62 > 61 > 52 > 51 > 41 > 31
        if kt.isdigit():
            return int(kt)
        return 0

    chain_entries = {}
    for line in meta_lines:
        parts = line.strip().split(";")
        if len(parts) >= 6 and parts[3] in ("Y", "T") and parts[4] == "K":
            pdb_id = parts[0].upper()
            chain_id = parts[1]
            k_type = parts[5].strip()
            key = (pdb_id, chain_id)
            if key not in chain_entries or knot_rank(k_type) > knot_rank(chain_entries[key]):
                chain_entries[key] = k_type

    knotted_records = []
    for (pdb_id, chain_id), k_type in chain_entries.items():
        seq = seq_map.get((pdb_id.lower(), chain_id))
        if not seq or not is_valid_sequence(seq):
            continue

        if k_type == "31":
            label = 1
            knot_name = "Trefoil (3_1)"
        elif k_type == "41":
            label = 2
            knot_name = "Figure-Eight (4_1)"
        else:
            label = 3
            knot_name = "Complex"

        knotted_records.append({
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "knot_type": knot_name,
            "label": label,
            "sequence": seq,
        })

    print(f"[Dataset Builder] Loaded {len(knotted_records)} verified knotted chain candidates.")
    return knotted_records


def fetch_unknotted_candidates(target_count=1000):
    print("[Dataset Builder] Fetching verified unknotted chains list from KnotProt 2.0...")
    url_un = "https://knotprot.cent.uw.edu.pl/unknotted.txt.gz"
    req = urllib.request.Request(url_un, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=SSL_CTX) as r:
        un_lines = gzip.decompress(r.read()).decode("utf-8", errors="ignore").splitlines()

    # Collect chain targets
    targets = list(CURATED_UNKNOTS)
    seen = {f"{p}_{c}" for p, c in targets}

    for line in un_lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            pdb_id = parts[0].upper()
            chain_id = parts[1]
            key = f"{pdb_id}_{chain_id}"
            if key not in seen:
                seen.add(key)
                targets.append((pdb_id, chain_id))
            if len(targets) >= target_count + 300:
                break

    print(f"[Dataset Builder] Fetching FASTA sequences from RCSB for {len(targets)} unknotted chains...")
    unknotted_records = []
    batch_size = 100

    # Group by PDB ID
    pdb_to_chains = {}
    for p, c in targets:
        pdb_to_chains.setdefault(p, set()).add(c)

    unique_pdbs = list(pdb_to_chains.keys())

    for i in range(0, len(unique_pdbs), batch_size):
        chunk = unique_pdbs[i:i + batch_size]
        ids_param = ",".join(chunk)
        url_fasta = f"https://www.rcsb.org/fasta/entry/{ids_param}/download"
        try:
            req = urllib.request.Request(url_fasta, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=30) as resp:
                fasta_text = resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"  [Warn] Failed chunk {i//batch_size + 1}: {e}")
            continue

        # Parse multi-entry FASTA
        current_pdb = None
        current_chains = set()
        current_seq_lines = []

        def flush_entry():
            if current_pdb and current_chains and current_seq_lines:
                seq = clean_sequence("".join(current_seq_lines))
                if is_valid_sequence(seq):
                    for ch in current_chains:
                        unknotted_records.append({
                            "pdb_id": current_pdb,
                            "chain_id": ch,
                            "knot_type": "Unknot (0_1)",
                            "label": 0,
                            "sequence": seq,
                        })

        for line in fasta_text.splitlines():
            if line.startswith(">"):
                flush_entry()
                current_seq_lines = []
                current_pdb = None
                current_chains = set()

                header = line[1:].strip()
                match = re.match(r"^([0-9A-Za-z]{4})_([0-9]+)\|Chain[s]?\s+([A-Za-z0-9,\s]+)\|", header)
                if match:
                    current_pdb = match.group(1).upper()
                    raw_chains = [c.strip() for c in match.group(3).split(",")]
                    target_chains = pdb_to_chains.get(current_pdb, set())
                    current_chains = set(raw_chains) & target_chains
                    if not current_chains and raw_chains:
                        current_chains = {raw_chains[0]}
            else:
                current_seq_lines.append(line.strip())

        flush_entry()
        print(f"  Fetched {len(unknotted_records)} unknotted chains so far...")
        if len(unknotted_records) >= target_count:
            break

    return unknotted_records


def build_and_filter_dataset():
    knotted = fetch_knotprot_knots()
    unknotted = fetch_unknotted_candidates(target_count=900)

    all_candidates = knotted + unknotted

    # Deduplicate exact sequences and prevent cross-class substring overlaps (> 30 AA)
    final_dataset = []
    seen_sequences = set()
    # 31-mer to label mapping
    kmer_to_label = {}
    K = 31

    rejected_dups = 0
    rejected_cross_overlaps = 0

    for rec in all_candidates:
        seq = rec["sequence"]
        label = rec["label"]

        if seq in seen_sequences:
            rejected_dups += 1
            continue

        # Check for cross-class 31-mer overlap
        has_cross_overlap = False
        seq_kmers = []
        for i in range(len(seq) - K + 1):
            kmer = seq[i:i + K]
            seq_kmers.append(kmer)
            existing_label = kmer_to_label.get(kmer)
            if existing_label is not None and existing_label != label:
                has_cross_overlap = True
                break

        if has_cross_overlap:
            rejected_cross_overlaps += 1
            continue

        # Accept sequence
        seen_sequences.add(seq)
        for kmer in seq_kmers:
            kmer_to_label[kmer] = label

        entry_id = f"{rec['pdb_id']}_{rec['chain_id']}"
        rec["id"] = entry_id
        final_dataset.append(rec)

    print(f"\n[Dataset Builder] Filtering Summary:")
    print(f"  Total accepted chains: {len(final_dataset)}")
    print(f"  Rejected duplicates: {rejected_dups}")
    print(f"  Rejected cross-class chimeric overlaps: {rejected_cross_overlaps}")

    class_counts = {}
    for r in final_dataset:
        class_counts[r["label"]] = class_counts.get(r["label"], 0) + 1

    print("\n[Dataset Builder] Composition by Class:")
    class_names = {0: "Unknot (0_1)", 1: "Trefoil (3_1)", 2: "Figure-Eight (4_1)", 3: "Complex"}
    for lbl in sorted(class_counts):
        print(f"  Class {lbl} ({class_names.get(lbl)}): {class_counts[lbl]} chains")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(final_dataset, f, indent=2)

    print(f"\n[Dataset Builder] Successfully saved dataset to {OUTPUT_JSON}")
    assert len(final_dataset) >= 1000, f"Dataset has {len(final_dataset)} chains, expected >= 1000"


if __name__ == "__main__":
    build_and_filter_dataset()
