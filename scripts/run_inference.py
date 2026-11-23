"""One-shot CLI inference against the fine-tuned classifier (no server needed).

    python scripts/run_inference.py [sequence | path/to/file.fasta]
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import (  # noqa: E402
    CLASS_MAP,
    MAX_RESIDUES,
    MAX_TOKENS,
    load_classifier,
    load_tokenizer,
    sanitise_sequence,
)

DEFAULT_SEQ = (
    "MSEFRAQWRNPGDVFSVLLIVGGDVILLALACVTGRVFTPVAFSFGWVAYAISAVVAAVGDNKLLARCTP"
    "EIPLKVINLKNGYNRDNKSWLLGRFMKDYNYWMPQDVRERLVRPPVRPDEEDRVAHDPFPARSSGLEPFP"
    "TNVTMGRSNAALCITVYRWVDGAKLGLPDRDWAWWSGFFMSILQLGVAAIPWGLHNNWNIFIATAAGTVM"
    "AYASASLPQWQIEKWTPERRKKDIAITQGNGNQHVIVILGADHGFDMEALAAARSPELGSTRIYTSILAF"
    "LWLVLLVTCCGIETDTWYLIAVGGLGMVQNLVVAGVPRKPAALGLPIELDRNLKDGQAAPNIFAEPKVMW"
    "SIMEFEKRFPGCGNALVKEFFPGDLLEWEVSWWASEDLAEREGLLSTQRRKAWDKAYEAAQEGR"
)


def main():
    raw = DEFAULT_SEQ
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        raw = open(arg).read() if os.path.exists(arg) else arg

    sequence, error = sanitise_sequence(raw)
    if error:
        sys.exit(f"[Inference Engine] {error}")

    tokenizer = load_tokenizer()
    model, loaded = load_classifier()
    if not loaded:
        sys.exit(
            "[Inference Engine] No checkpoint found. The classification head would be random "
            "and its output meaningless.\n"
            "Build one first:  python -m training.train"
        )

    evaluated = sequence[:MAX_RESIDUES]
    inputs = tokenizer(evaluated, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        pred_class = int(np.argmax(probs))

    print("\n" + "=" * 60)
    print("  FINE-TUNED NATIVE AI INFERENCE RESULT")
    print("=" * 60)
    print(f"  Sequence Length    : {len(sequence)} AA")
    print(f"  Predicted Topology : {CLASS_MAP[pred_class]}")
    print(f"  Confidence Score   : {probs[pred_class]:.4f}")
    print("  Class Probabilities:")
    for i, name in CLASS_MAP.items():
        print(f"    - {name:<18}: {probs[i]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
