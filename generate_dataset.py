import json
import torch
import numpy as np

# Dataset schema for BioKnot Native Classifier
# Labels: 0 = Unknot (0_1), 1 = Trefoil (3_1), 2 = Figure-8 (4_1), 3 = Complex/Other

def create_sample_training_data():
    dataset = [
        {"sequence": "MSEFRAQWRNPGDVFSVLLIV...", "knot_label": 0}, # Unknot
        {"sequence": "MEGRPRNYTNGAGDPRAPAQ...", "knot_label": 1}, # Trefoil (3_1)
    ]
    with open("bioknot_train_data.json", "w") as f:
        json.dump(dataset, f, indent=2)
    print("[Dataset Pipeline] Ground-truth training set initialized.")

if __name__ == "__main__":
    create_sample_training_data()
