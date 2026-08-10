import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel

class BioKnotNativeClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(BioKnotNativeClassifier, self).__init__()
        # Load lightweight ESM-2 backbone for feature extraction (8M parameters)
        self.esm = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        # Freeze ESM-2 layers to train fast on CPU/GPU without heavy compute requirements
        for param in self.esm.parameters():
            param.requires_grad = False
            
        hidden_dim = self.esm.config.hidden_size # 320 for 8M model
        
        # 1D Convolutional & Classification Head
        self.conv1 = nn.Conv1d(hidden_dim, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask):
        # Extract embeddings from ESM-2
        with torch.no_grad():
            outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state # (batch, seq_len, 320)
        
        # Reshape for 1D Convolution: (batch, 320, seq_len)
        x = sequence_output.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1) # Global pooling across sequence dimension
        logits = self.classifier(x)
        return logits

print("[Model Architecture] BioKnotNativeClassifier initialized successfully.")
