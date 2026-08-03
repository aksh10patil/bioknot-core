import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, EsmModel

class BioKnotClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BioKnotClassifier, self).__init__()
        self.esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        # Freeze base layers, unfreeze top 2 transformer layers for fine-tuning
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
        seq_embeddings = outputs.last_hidden_state  # (batch, seq_len, 320)
        
        x = seq_embeddings.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        logits = self.fc(x)
        return logits

class ProteinKnotDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item["sequence"]
        label = item["label"]
        encoding = self.tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024, padding="max_length")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    print("[Training Engine] Initializing ESM-2 Fine-Tuning Pipeline...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    model = BioKnotClassifier(num_classes=4)
    
    dataset = ProteinKnotDataset("bioknot_dataset.json", tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    print("[Training Engine] Fine-tuning across 30 Epochs...")
    model.train()
    for epoch in range(1, 31):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f" -> Epoch {epoch:02d}/30 | Loss: {total_loss / len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "bioknot_native.pt")
    print("\n[Training Engine] Optimization complete! Fine-tuned weights saved to bioknot_native.pt")
