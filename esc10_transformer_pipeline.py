
import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader
from esc10_cnn_pipeline import esc10_split, ESC10Dataset
from esc10_evaluation_extension import evaluate_model

# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        pe = self.pe[:, :x.size(1)].to(x.device) 
        return x

# Transformer Encoder Model
class TransformerAudioClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.pos_encoder = PositionalEncoding(d_model=input_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True  # Add this!
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)  # (B, T, F)
        x = self.linear(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)  # (B, F, T)
        return self.classifier(x)

# Training loop
def train_transformer_model(model, train_loader, val_loader, device='cpu', epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Validation Accuracy: {correct / total:.2%}")

# Execution entry point
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set, test_set, label_map = esc10_split("meta/esc50.csv", "audio", selected_fold=1)
    train_ds = ESC10Dataset(train_set)
    test_ds  = ESC10Dataset(test_set)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=8)

    input_dim = 64
    model = TransformerAudioClassifier(input_dim=input_dim, num_classes=len(label_map))
    train_transformer_model(model, train_loader, test_loader, device=device, epochs=200)

    # Evaluate after training
    evaluate_model(model, test_loader, label_map, device=device, log_file='transformer_eval_log.txt')
