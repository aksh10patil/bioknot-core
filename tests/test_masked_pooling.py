"""Verification test for masked residue pooling batch invariance."""

import pytest
import torch
from app.model import BioKnotClassifier, load_tokenizer

def test_masked_pooling_batch_invariance():
    tokenizer = load_tokenizer()
    model = BioKnotClassifier()
    model.eval()

    # Fixed 100-AA test sequence
    seq_100 = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG" + "A" * 24
    assert len(seq_100) == 100

    # Long 510-AA sequence (so total tokens with <cls> and <eos> is 512)
    seq_510 = "M" + "A" * 508 + "G"
    assert len(seq_510) == 510

    # 1. Single sequence inference (no padding)
    single_enc = tokenizer(seq_100, return_tensors="pt")
    with torch.no_grad():
        logits_single = model(single_enc["input_ids"], single_enc["attention_mask"])

    # 2. Batched sequence inference padded to 512 tokens
    batch_enc = tokenizer([seq_100, seq_510], return_tensors="pt", padding="longest")
    assert batch_enc["input_ids"].shape[1] == 512

    with torch.no_grad():
        logits_batch = model(batch_enc["input_ids"], batch_enc["attention_mask"])

    diff = (logits_single[0] - logits_batch[0]).abs().max().item()
    print(f"\n[Test] Maximum logit absolute difference between single and padded batch: {diff:.2e}")
    assert diff < 1e-5, f"Masked pooling failed: max diff {diff} exceeds tolerance 1e-5"
