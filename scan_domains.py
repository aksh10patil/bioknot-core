import requests

url = "http://127.0.0.1:8000/predict_custom"

with open("combined_sequence.txt", "r") as f:
    full_seq = f.read().replace("\n", "").strip()

print(f"[Domain Scanner] Total Sequence Length: {len(full_seq)} AA")
print("=" * 70)
print(f"{'Domain Region':<20} | {'Predicted Topology':<20} | {'Confidence'}")
print("=" * 70)

window_size = 400
stride = 200

for start in range(0, len(full_seq) - window_size + 1, stride):
    end = start + window_size
    chunk = full_seq[start:end]
    
    try:
        res = requests.post(url, json={"sequence": chunk}, timeout=10).json()
        knot = res.get("predicted_knot", "Unknown")
        conf = res.get("confidence", 0.0)
        
        flag = " [KNOT DETECTED]" if knot != "Unknot (0_1)" else ""
        print(f"AA {start:05d} - {end:05d}    | {knot:<20} | {conf:.4f}{flag}")
    except Exception as e:
        print(f"AA {start:05d} - {end:05d}    | Error/Timeout ({e})")

print("=" * 70)
