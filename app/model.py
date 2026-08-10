"""Shared definition of the BioKnot native topological classifier.

This module is the single source of truth for the architecture. `training/train.py`
builds and saves weights against it and `app/server.py` loads them back, so a
checkpoint can never silently disagree with the serving code.
"""

import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

ESM_CHECKPOINT = "facebook/esm2_t6_8M_UR50D"
NUM_CLASSES = 4
MAX_TOKENS = 1024

CLASS_MAP = {
    0: "Unknot (0_1)",
    1: "Trefoil (3_1)",
    2: "Figure-Eight (4_1)",
    3: "Complex",
}

# The 20 standard amino acids plus the ambiguity codes ESM-2 tokenises.
VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS_PATH = os.path.join(REPO_ROOT, "bioknot_native.pt")


class BioKnotClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(BioKnotClassifier, self).__init__()
        self.esm = EsmModel.from_pretrained(ESM_CHECKPOINT)

        for param in self.esm.parameters():
            param.requires_grad = False

        for layer in self.esm.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        hidden_dim = self.esm.config.hidden_size
        self.conv1 = nn.Conv1d(hidden_dim, 256, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        seq_embeddings = outputs.last_hidden_state

        x = seq_embeddings.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits


def load_tokenizer():
    return AutoTokenizer.from_pretrained(ESM_CHECKPOINT)


def load_classifier(weights_path=None):
    """Build the classifier and load fine-tuned weights.

    Returns `(model, loaded)`. `loaded` is False when no checkpoint was found, in
    which case the classification head is randomly initialised and its output is
    meaningless -- callers must treat that as an error state rather than serving
    the predictions. See `app/server.py`.
    """
    weights_path = weights_path or os.environ.get("BIOKNOT_WEIGHTS", DEFAULT_WEIGHTS_PATH)
    model = BioKnotClassifier(num_classes=NUM_CLASSES)

    loaded = False
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        loaded = True

    model.eval()
    return model, loaded


def sanitise_sequence(raw):
    """Strip FASTA headers and whitespace, upper-case, and validate residues.

    Returns `(sequence, error)`.
    """
    if not isinstance(raw, str):
        return None, "Sequence must be a string."

    lines = [ln for ln in raw.strip().splitlines() if not ln.startswith(">")]
    sequence = "".join(lines)
    sequence = "".join(sequence.split()).upper()

    if not sequence:
        return None, "No sequence provided."

    invalid = sorted(set(sequence) - VALID_RESIDUES)
    if invalid:
        return None, f"Sequence contains non-amino-acid characters: {', '.join(invalid)}"

    return sequence, None
