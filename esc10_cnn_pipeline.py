import pandas as pd
import os
import torch
import torchaudio
import torchaudio.transforms as T
def esc10_split(metadata_csv, audio_dir, selected_fold=1):
    df = pd.read_csv(metadata_csv)
    df = df[df['esc10'] == True]
    label_map = {label: idx for idx, label in enumerate(sorted(df['category'].unique()))}

    train_df = df[df['fold'] != selected_fold]
    test_df  = df[df['fold'] == selected_fold]

    def make_split(split_df):
        files = []
        for _, row in split_df.iterrows():
            path = os.path.join(audio_dir, row['filename'])
            label = label_map[row['category']]
            files.append((path, label))
        return files

    return make_split(train_df), make_split(test_df), label_map


# ## Step 2: Define ESC-10 Dataset Loader



class ESC10Dataset(torch.utils.data.Dataset):
    def __init__(self, file_label_pairs, sr=16000, duration=5, n_mels=64):
        self.files = file_label_pairs
        self.sample_rate = sr
        self.n_samples = sr * duration
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=512,
            n_mels=n_mels
        )
        self.db_transform = T.AmplitudeToDB()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        waveform, sr = torchaudio.load(path)
        waveform = waveform.mean(dim=0)
        if waveform.shape[0] < self.n_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.n_samples - waveform.shape[0]))
        else:
            waveform = waveform[:self.n_samples]

        mel = self.mel_transform(waveform)
        mel_db = self.db_transform(mel)
        return mel_db.unsqueeze(0), label


# ## Step 3: Define the CNN Model

import torch.nn as nn
import torch.nn.functional as F

class BirdCNN(nn.Module):
    def __init__(self, n_classes):
        super(BirdCNN, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # defer fc1 until input size is known
        self.fc1 = None
        self.fc2 = None

    def _init_fc_layers(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool(F.relu(self.conv2(out)))
        flat_dim = out.view(out.size(0), -1).shape[1]
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, self.n_classes)

    def forward(self, x):
        if self.fc1 is None:
            self._init_fc_layers(x)
            self.to(x.device)  # update device placement after layer init

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)



# ## Step 4: Training Loop

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

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

        if val_loader:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            print(f"Validation Accuracy: {correct / total:.2%}")