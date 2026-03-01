# train.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pickle
from model import ASLSignLSTM
import random


# ── Dataset ──────────────────────────────────────────────────────────────────

class LandmarkDataset(Dataset):
    """expected format for input landmarks:
    [
      { "landmarks": np.array of shape (num_frames, 63), "label": "hello" },
      ...
    ]
    """
    def __init__(self, data, label_to_idx):
        self.samples = data
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        landmarks = torch.tensor(sample["landmarks"], dtype=torch.float32)
        label = torch.tensor(self.label_to_idx[sample["label"]], dtype=torch.long)
        return landmarks, label


def collate_fn(batch):
    """Pads variable-length sequences in a batch."""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return padded, labels, lengths


# ── Training loop ─────────────────────────────────────────────────────────────

def train(data_path="landmarks.pkl", epochs=200, lr=5e-4):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    labels = sorted(set(s["label"] for s in data))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    with open("label_map.pkl", "wb") as f:
        pickle.dump({"label_to_idx": label_to_idx, "idx_to_label": idx_to_label}, f)

    # Shuffle here, before the split
    random.shuffle(data)
    split = int(len(data) * 0.85)
    train_data, val_data = data[:split], data[split:]

    # diagnostics
    from collections import Counter
    print(f"Total samples: {len(data)}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Val labels: {dict(Counter(s['label'] for s in val_data))}")

    # Train/val split
    split = int(len(data) * 0.85)
    train_data, val_data = data[:split], data[split:]

    train_loader = DataLoader(
        LandmarkDataset(train_data, label_to_idx),
        batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        LandmarkDataset(val_data, label_to_idx),
        batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    model = ASLSignLSTM(num_classes=len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seqs, lbls, lengths in train_loader:
            optimizer.zero_grad()
            logits = model(seqs, lengths)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for seqs, lbls, lengths in val_loader:
                preds = model(seqs, lengths).argmax(dim=-1)
                correct += (preds == lbls).sum().item()
        
        acc = correct / len(val_data)
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(train_loader):.4f}  val_acc={acc:.2%}")

    torch.save(model.state_dict(), "asl_lstm.pt")
    print("Saved model to asl_lstm.pt")


if __name__ == "__main__":
    train()