
import torch
from torch.utils.data import DataLoader
from esc10_cnn_pipeline import esc10_split, ESC10Dataset, BirdCNN, train_model
from esc10_evaluation_extension import evaluate_model

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load data
train_set, test_set, label_map = esc10_split("meta/esc50.csv", "audio", selected_fold=1)
train_ds = ESC10Dataset(train_set)
test_ds  = ESC10Dataset(test_set)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=8)

# Initialize model
model = BirdCNN(n_classes=len(label_map))

# Train model
train_model(model, train_loader, test_loader, epochs=200, device=device)

# Evaluate model
evaluate_model(model, test_loader, label_map, device=device)
