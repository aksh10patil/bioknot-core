import os
import time
import requests
import torch
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, EsmModel
import torch.nn as nn

app = Flask(__name__, static_folder='.')
CORS(app)

# =====================================================================
# FINE-TUNED AI ARCHITECTURE DEFINITION
# =====================================================================

class BioKnotClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BioKnotClassifier, self).__init__()
        self.esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
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

print("[Backend Setup] Initializing ESM-2 Tokenizer & Fine-Tuned Model...")
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
native_model = BioKnotClassifier(num_classes=4)

WEIGHTS_PATH = os.path.expanduser("~/bioknot_wasm/bioknot_native.pt")
if os.path.exists(WEIGHTS_PATH):
    print(f"[Backend Setup] Loading fine-tuned weights from {WEIGHTS_PATH}")
    native_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
native_model.eval()

# =====================================================================
# UTILITIES: ESMAtlas API Fetcher with Exponential Backoff
# =====================================================================

def fetch_pdb_for_chunk(seq_chunk, retries=3, delay=2):
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                data=seq_chunk,
                headers={"Content-Type": "text/plain"},
                timeout=120
            )
            if response.status_code == 200:
                return response.text
        except Exception as e:
            print(f"[ESMFold Bridge] Attempt {attempt + 1}/{retries} failed: {e}")
        time.sleep(delay)
    return None

def extract_ca_coords(pdb_str):
    coords = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                coords.append([
                    float(line[30:38].strip()),
                    float(line[38:46].strip()),
                    float(line[46:54].strip())
                ])
            except ValueError:
                continue
    return np.array(coords)

# =====================================================================
# ROUTE 1: ESMFold WASM 3D Structure Pipeline
# =====================================================================

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_structure():
    data = request.json
    sequence = data.get("sequence", "").strip()

    if not sequence:
        return jsonify({"error": "No sequence provided"}), 400

    seq_len = len(sequence)
    print(f"\n[ESMFold Bridge] Request received for sequence length: {seq_len} AA")

    if seq_len <= 400:
        pdb_str = fetch_pdb_for_chunk(sequence)
        if not pdb_str:
            return jsonify({"error": "ESMFold API timed out or rate-limited. Retry request."}), 500
        coords_arr = extract_ca_coords(pdb_str)
    else:
        print(f"[ESMFold Bridge] Sequence > 400 AA. Executing Rigid Kabsch Domain Tiling...")
        window_size = 380
        stride = 200
        overlap = window_size - stride

        w1_seq = sequence[0:window_size]
        pdb1 = fetch_pdb_for_chunk(w1_seq)
        if not pdb1:
            return jsonify({"error": "ESMFold API failed on initial window."}), 500
        
        full_chain = extract_ca_coords(pdb1)

        for start in range(stride, seq_len, stride):
            end = min(start + window_size, seq_len)
            w_seq = sequence[start:end]
            
            pdb_next = fetch_pdb_for_chunk(w_seq)
            if not pdb_next:
                continue
            
            w_coords = extract_ca_coords(pdb_next)
            curr_overlap_len = min(overlap, len(full_chain) - start, len(w_coords))
            if curr_overlap_len < 5:
                full_chain = np.vstack([full_chain, w_coords])
                continue

            ref_overlap = full_chain[start : start + curr_overlap_len]
            target_overlap = w_coords[:curr_overlap_len]

            centroid_target = np.mean(target_overlap, axis=0)
            centroid_ref = np.mean(ref_overlap, axis=0)

            target_centered = target_overlap - centroid_target
            ref_centered = ref_overlap - centroid_ref

            H = np.dot(target_centered.T, ref_centered)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = np.dot(Vt.T, U.T)

            w_centered = w_coords - centroid_target
            aligned_w_coords = np.dot(w_centered, R.T) + centroid_ref
            full_chain = np.vstack([full_chain, aligned_w_coords[curr_overlap_len:]])

            if end == seq_len:
                break

        coords_arr = full_chain

    ca_coords = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in coords_arr]
    print(f"[ESMFold Bridge] Successfully extracted {len(ca_coords)} continuous C-alpha nodes.")
    
    return jsonify({
        "engine": "ESMFold + Rigid Kabsch Tiling",
        "sequence_length": seq_len,
        "ca_coords": ca_coords
    })

# =====================================================================
# ROUTE 2: Direct Fine-Tuned Sequence AI Predictor
# =====================================================================

@app.route('/predict_custom', methods=['POST'])
def predict_custom():
    data = request.json
    sequence = data.get("sequence", "").strip()

    if not sequence:
        return jsonify({"error": "No sequence provided"}), 400

    original_len = len(sequence)
    MAX_LEN = 1024
    if original_len > MAX_LEN:
        print(f"[Custom AI Engine] Truncating sequence length from {original_len} AA to {MAX_LEN} AA capacity.")
        sequence = sequence[:MAX_LEN]

    print(f"\n[Custom AI Engine] Running direct sequence inference ({len(sequence)} AA)...")
    
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = native_model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        pred_class = int(np.argmax(probs))

    class_map = {0: "Unknot (0_1)", 1: "Trefoil (3_1)", 2: "Figure-Eight (4_1)", 3: "Complex"}

    return jsonify({
        "engine": "BioKnot Fine-Tuned Native AI",
        "sequence_length": original_len,
        "evaluated_length": len(sequence),
        "predicted_knot": class_map.get(pred_class, "Unknown"),
        "confidence": float(probs[pred_class]),
        "probabilities": {class_map[i]: float(probs[i]) for i in range(4)}
    })

if __name__ == '__main__':
    print("====================================================")
    print("  BIOKNOT-CORE: HYBRID DUAL-ENGINE INFERENCE SUITE  ")
    print("====================================================")
    print("  1. ESMFold WASM Bridge : POST /predict            ")
    print("  2. Native AI Predictor : POST /predict_custom     ")
    print("====================================================")
    print("  Open in Browser: http://127.0.0.1:8000           ")
    print("====================================================")
    app.run(host='127.0.0.1', port=8000, debug=False)
