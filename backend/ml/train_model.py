# Train pytorch model on ASL alphabet dataset

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from dotenv import load_dotenv
load_dotenv()

# Configuration
BATCH_SIZE    = 16
EPOCHS        = 8
LR_HEAD       = 1e-3
LR_FULL       = 1e-5
VAL_SPLIT     = 0.2
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH     = "ml/asl_model.pth"

# Build dataset from preprocessed skeleton images
def build_dataset():
    train_dir = "ml/asl_alphabet_skeleton"

    # Augment data with randomness to improve generalization
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = ImageFolder(train_dir, transform=transform_train)
    print(f"Dataset: {len(dataset)} images across {len(dataset.classes)} classes")
    return dataset

# Split dataset into training and validation sets and wrap in data loaders
def prepare_dataloaders(dataset):
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Classes : {dataset.classes}")
    print(f"Train   : {train_size} | Val: {val_size}")
    
    return train_loader, val_loader

# Train model
def run_epoch(loader, train=True, optimizer=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total



# Main
dataset = build_dataset()
train_loader, val_loader = prepare_dataloaders(dataset)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, len(dataset.classes))
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

print("-----------------------------------------")
print("Training classifier head only")
print("-----------------------------------------")
for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.classifier.parameters(), lr=LR_HEAD)

for epoch in range(EPOCHS // 2):
    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, train=True,  optimizer=optimizer)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)
    print(f"- Epoch {epoch+1:02d} | "
        f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
        f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} | "
        f"{time.time()-t0:.1f}s")

print("-----------------------------------------")
print("Fine tune full network")
print("-----------------------------------------")
for param in model.features.parameters():
    param.requires_grad = True

optimizer    = optim.Adam(model.parameters(), lr=LR_FULL)
best_val_acc = 0.0

for epoch in range(EPOCHS):
    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, train=True,  optimizer=optimizer)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)
    print(f"- Epoch {epoch+1:02d} | "
          f"Train loss: {train_loss:.4f} acc: {train_acc:.3f} | "
          f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} | "
          f"{time.time()-t0:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "gestures":    dataset.classes,
            "val_acc":     val_acc,
        }, SAVE_PATH)
        print(f"  - Saved best model (val_acc={val_acc:.3f})")

print(f"\nDone! Best val accuracy: {best_val_acc:.3f}")
print(f"Model saved to: {SAVE_PATH}")