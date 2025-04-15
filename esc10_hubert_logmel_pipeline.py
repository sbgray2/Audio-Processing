
import os
import torch
import torchaudio
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel, Wav2Vec2FeatureExtractor
from esc10_evaluation_extension import evaluate_model

# HuBERT Feature Extractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

# Dataset
class ESC10Dataset(Dataset):
    def __init__(self, data):
        self.X = [torch.tensor(x[0], dtype=torch.float32) for x in data]
        self.y = [x[1] for x in data]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Feature extraction
def extract_hubert_features(filepath, target_sr=16000):
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.squeeze()
    inputs = feature_extractor(waveform, sampling_rate=target_sr, return_tensors="pt")
    with torch.no_grad():
        outputs = hubert_model(input_values=inputs["input_values"])
        features = outputs.last_hidden_state.squeeze(0)
        return features.mean(dim=0).numpy()

# Dataset builder
def esc10_split(metadata_csv, audio_dir, selected_fold=1):
    df = pd.read_csv(metadata_csv)
    df = df[df['esc10'] == True]
    label_map = {label: i for i, label in enumerate(sorted(df['category'].unique()))}
    train, test = [], []
    i = 0
    for _, row in df.iterrows():
        path = os.path.join(audio_dir, row['filename'])
        label = label_map[row['category']]
        print(f"🔍 Processing {i+1}/{len(df)}: {path}")
        i += 1
        try:
            emb = extract_hubert_features(path)
            if row['fold'] == selected_fold:
                test.append((emb, label))
            else:
                train.append((emb, label))
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return train, test, label_map

# Training
def train_hubert_classifier(model, train_loader, val_loader, device='cpu', epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), torch.tensor(y).to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

        # Validation accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), torch.tensor(y).to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        print(f"Validation Accuracy: {correct / total:.2%}")

# Run entry point
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set, test_set, label_map = esc10_split("meta/esc50.csv", "audio", selected_fold=1)
    train_ds = ESC10Dataset(train_set)
    test_ds  = ESC10Dataset(test_set)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=8)

    input_dim = train_ds[0][0].shape[0]
    model = SimpleClassifier(input_dim=input_dim, num_classes=len(label_map))
    train_hubert_classifier(model, train_loader, test_loader, device=device, epochs=200)

    evaluate_model(model, test_loader, label_map, device=device, log_file="hubert_eval_log.txt")
