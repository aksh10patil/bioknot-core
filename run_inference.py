import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

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

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
model = BioKnotClassifier(num_classes=4)

WEIGHTS_PATH = os.path.expanduser("~/bioknot_wasm/bioknot_native.pt")
if os.path.exists(WEIGHTS_PATH):
    print(f"[Inference Engine] Loading weights from {WEIGHTS_PATH}")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
model.eval()

# Sequence to evaluate
seq = "MSEFRAQWRNPGDVFSVLLIVGGDVILLALACVTGRVFTPVAFSFGWVAYAISAVVAAVGDNKLLARCTPEIPLKVINLKNGYNRDNKSWLLGRFMKDYNYWMPQDVRERLVRPPVRPDEEDRVAHDPFPARSSGLEPFPTNVTMGRSNAALCITVYRWVDGAKLGLPDRDWAWWSGFFMSILQLGVAAIPWGLHNNWNIFIATAAGTVMAYASASLPQWQIEKWTPERRKKDIAITQGNGNQHVIVILGADHGFDMEALAAARSPELGSTRIYTSILAFLWLVLLVTCCGIETDTWYLIAVGGLGMVQNLVVAGVPRKPAALGLPIELDRNLKDGQAAPNIFAEPKVMWSIMEFEKRFPGCGNALVKEFFPGDLLEWEVSWWASEDLAEREGLLSTQRRKAWDKAYEAAQEGR"

inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
with torch.no_grad():
    logits = model(inputs["input_ids"], inputs["attention_mask"])
    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    pred_class = int(probs.argmax())

class_map = {0: "Unknot (0_1)", 1: "Trefoil (3_1)", 2: "Figure-Eight (4_1)", 3: "Complex"}

print("\n" + "=" * 60)
print("  FINE-TUNED NATIVE AI INFERENCE RESULT")
print("=" * 60)
print(f"  Predicted Topology : {class_map[pred_class]}")
print(f"  Confidence Score   : {probs[pred_class]:.4f}")
print("  Class Probabilities:")
for i, name in class_map.items():
    print(f"    - {name:<18}: {probs[i]:.4f}")
print("=" * 60)
