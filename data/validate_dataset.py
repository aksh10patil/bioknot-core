#!/usr/bin/env python3
"""Dataset integrity validator for BioKnot.

Asserts:
1. Zero duplicate sequences.
2. Zero cross-class exact substring overlaps > 30 AA (using 31-mer index).
3. Valid pdb_id and chain_id on 100% of records.
4. Valid amino acid characters (VALID_RESIDUES) and length within [50, 1022].
"""

import json
import os
import re
import sys

VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
MIN_LEN = 50
MAX_LEN = 1022

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(DATA_DIR, "bioknot_dataset.json")


def validate_dataset(json_path=None):
    json_path = json_path or DEFAULT_DATASET
    print(f"[Validator] Validating dataset integrity: {json_path}")

    assert os.path.exists(json_path), f"Dataset file does not exist: {json_path}"

    with open(json_path, "r") as f:
        data = json.load(f)

    assert isinstance(data, list), "Dataset must be a JSON array of records"
    assert len(data) >= 1000, f"Dataset has only {len(data)} records; minimum required is 1000"

    print(f"  Total records: {len(data)}")

    seen_sequences = {}
    kmer_to_class = {}
    K = 31  # Check for overlaps > 30 AA

    for idx, rec in enumerate(data):
        assert "pdb_id" in rec and rec["pdb_id"], f"Record {idx} missing pdb_id"
        assert "chain_id" in rec and rec["chain_id"], f"Record {idx} missing chain_id"
        assert "label" in rec and rec["label"] in (0, 1, 2, 3), f"Record {idx} invalid label: {rec.get('label')}"
        assert "sequence" in rec and rec["sequence"], f"Record {idx} missing sequence"

        pdb_id = rec["pdb_id"]
        chain_id = rec["chain_id"]
        label = rec["label"]
        seq = rec["sequence"]

        # 1. PDB ID and Chain ID format
        assert re.match(r"^[0-9A-Za-z]{4}$", pdb_id), f"Invalid PDB ID format: '{pdb_id}' at record {idx}"
        assert re.match(r"^[0-9A-Za-z]+$", chain_id), f"Invalid chain ID format: '{chain_id}' at record {idx}"

        # 2. Length and Residue Validity
        assert MIN_LEN <= len(seq) <= MAX_LEN, (
            f"Record {idx} ({pdb_id}_{chain_id}) length {len(seq)} outside allowed range [{MIN_LEN}, {MAX_LEN}]"
        )
        invalid_chars = set(seq) - VALID_RESIDUES
        assert not invalid_chars, f"Record {idx} contains invalid residues: {invalid_chars}"

        # 3. Exact Duplicate Sequences Check
        if seq in seen_sequences:
            prev_idx, prev_id = seen_sequences[seq]
            raise AssertionError(f"Duplicate sequence found between record {idx} ({pdb_id}_{chain_id}) and {prev_idx} ({prev_id})")
        seen_sequences[seq] = (idx, f"{pdb_id}_{chain_id}")

        # 4. Cross-class exact substring overlaps > 30 AA
        for i in range(len(seq) - K + 1):
            kmer = seq[i:i + K]
            if kmer in kmer_to_class:
                prev_label, prev_id = kmer_to_class[kmer]
                if prev_label != label:
                    raise AssertionError(
                        f"Cross-class substring overlap > 30 AA detected between "
                        f"class {label} ({pdb_id}_{chain_id}) and class {prev_label} ({prev_id})"
                    )
            else:
                kmer_to_class[kmer] = (label, f"{pdb_id}_{chain_id}")

    print("  [PASS] Zero duplicate sequences.")
    print("  [PASS] Zero cross-class exact substring overlaps > 30 AA.")
    print("  [PASS] Valid pdb_id and chain_id on 100% of records.")
    print(f"  [PASS] All {len(data)} records conform to valid residue and length constraints.")
    print("[Validator] Dataset integrity 100% verified!")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    validate_dataset(path)
