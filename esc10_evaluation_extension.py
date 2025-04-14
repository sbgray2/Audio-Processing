
import matplotlib
matplotlib.use('Agg')

import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, dataloader, label_map, device='cpu', log_file='evaluation_log.txt'):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    labels = list(label_map.keys())
    report = classification_report(y_true, y_pred, target_names=labels)

    with open(log_file, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report + "\n")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    with open(log_file, 'a') as f:
        f.write("Confusion matrix saved to confusion_matrix.png\n")
