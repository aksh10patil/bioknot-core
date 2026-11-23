"""Shared definition of the BioKnot native topological classifier.

GENERATED FILE -- do not edit here. Synced from app/model.py by scripts/sync_space.sh.

This module is the single source of truth for the architecture. `training/train.py`
builds and saves weights against it and `app/server.py` loads them back, so a
checkpoint can never silently disagree with the serving code.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

DEFAULT_BACKBONE = "facebook/esm2_t6_8M_UR50D"
ESM_CHECKPOINT = DEFAULT_BACKBONE  # Backward-compatible alias
NUM_CLASSES = 4
MAX_TOKENS = 1024
MAX_RESIDUES = MAX_TOKENS - 2  # 1022 (accounts for leading <cls> and trailing <eos>)

CLASS_MAP = {
    0: "Unknot (0_1)",
    1: "Trefoil (3_1)",
    2: "Figure-Eight (4_1)",
    3: "Complex",
}

# The 20 standard amino acids plus the ambiguity codes ESM-2 tokenises.
VALID_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(MODULE_DIR, "bioknot_native.pt")


def get_device() -> torch.device:
    """Select accelerated device: CUDA -> MPS (Apple Silicon) -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BioKnotClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, backbone: str = DEFAULT_BACKBONE):
        super(BioKnotClassifier, self).__init__()
        self.backbone_name = backbone
        self.esm = EsmModel.from_pretrained(backbone)

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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        seq_embeddings = outputs.last_hidden_state

        # Construct residue-only float mask: zero out <pad>, leading <cls> (idx 0), and trailing <eos>
        residue_mask = attention_mask.clone().float()
        residue_mask[:, 0] = 0.0
        lengths = attention_mask.sum(dim=1).long()
        for b, l in enumerate(lengths):
            if l > 0:
                residue_mask[b, l - 1] = 0.0

        # Multiply seq_embeddings by residue_mask.unsqueeze(-1) before transposing into conv1
        seq_embeddings = seq_embeddings * residue_mask.unsqueeze(-1)
        x = seq_embeddings.transpose(1, 2)
        x = self.relu(self.conv1(x))

        # Multiply activations by residue_mask.unsqueeze(1) before pooling
        mask_2d = residue_mask.unsqueeze(1)
        x = x * mask_2d
        x = self.dropout(x)

        # Masked mean pooling
        pooled = (x * mask_2d).sum(dim=-1) / mask_2d.sum(dim=-1).clamp(min=1.0)
        logits = self.fc(pooled)
        return logits


def load_tokenizer(backbone: str = DEFAULT_BACKBONE):
    return AutoTokenizer.from_pretrained(backbone)


def load_classifier(
    weights_path: Optional[str] = None,
    backbone: str = DEFAULT_BACKBONE,
    device: Optional[torch.device] = None,
) -> Tuple[BioKnotClassifier, bool]:
    """Build the classifier and load fine-tuned weights.

    Returns `(model, loaded)`. `loaded` is False when no checkpoint was found.
    """
    weights_path = weights_path or os.environ.get("BIOKNOT_WEIGHTS", DEFAULT_WEIGHTS_PATH)
    model = BioKnotClassifier(num_classes=NUM_CLASSES, backbone=backbone)

    loaded = False
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        loaded = True

    if device is not None:
        model.to(device)

    model.eval()
    return model, loaded


def sanitise_sequence(raw: str) -> Tuple[Optional[str], Optional[str]]:
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
