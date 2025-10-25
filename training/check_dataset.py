"""
Check dataset for issues - class balance, data quality, etc
"""
import torch
import numpy as np
from dataset import get_data_loaders
from config import *
from collections import Counter

print("🔍 Dataset Quality Check\n")

# Load data
train_loader, val_loader, test_loader = get_data_loaders(DATA_DIR, batch_size=32)

print("="*60)
print("CLASS DISTRIBUTION")
print("="*60)

# Check train set
train_labels = []
for _, labels, _ in train_loader:
    train_labels.extend(labels.tolist())

train_counter = Counter(train_labels)
print(f"\nTrain set ({len(train_labels)} samples):")
for label in sorted(train_counter.keys()):
    count = train_counter[label]
    pct = count / len(train_labels) * 100
    emotion = EMOTION_LABELS[label]
    print(f"  {label}: {emotion:12s} - {count:3d} samples ({pct:5.1f}%)")

# Check if severely imbalanced
max_count = max(train_counter.values())
min_count = min(train_counter.values())
imbalance_ratio = max_count / min_count
print(f"\nImbalance ratio: {imbalance_ratio:.2f}x")
if imbalance_ratio > 3:
    print("⚠️  SEVERE CLASS IMBALANCE DETECTED!")
else:
    print("✅ Class balance looks reasonable")

# Check val set
val_labels = []
for _, labels, _ in val_loader:
    val_labels.extend(labels.tolist())

val_counter = Counter(val_labels)
print(f"\nVal set ({len(val_labels)} samples):")
for label in sorted(val_counter.keys()):
    count = val_counter[label]
    pct = count / len(val_labels) * 100
    emotion = EMOTION_LABELS[label]
    print(f"  {label}: {emotion:12s} - {count:3d} samples ({pct:5.1f}%)")

print()
print("="*60)
print("DATA QUALITY CHECK")
print("="*60)

# Check a few samples
all_mfccs = []
all_lengths = []

for i, (inputs, labels, lengths) in enumerate(train_loader):
    if i >= 10:  # Check first 10 batches
        break
    all_mfccs.append(inputs)
    all_lengths.extend(lengths.tolist())

all_mfccs = torch.cat(all_mfccs, dim=0)

print(f"\nShape: {all_mfccs.shape}")
print(f"Min/Max: {all_mfccs.min():.3f} / {all_mfccs.max():.3f}")
print(f"Mean/Std: {all_mfccs.mean():.3f} / {all_mfccs.std():.3f}")
print(f"Sequence lengths: min={min(all_lengths)}, max={max(all_lengths)}, avg={np.mean(all_lengths):.1f}")

if torch.isnan(all_mfccs).any():
    print("❌ DATA CONTAINS NaN VALUES!")
if torch.isinf(all_mfccs).any():
    print("❌ DATA CONTAINS INF VALUES!")
else:
    print("✅ No NaN or Inf values")

# Check if all same
if all_mfccs.std() < 0.01:
    print("❌ DATA HAS ALMOST NO VARIANCE - ALL SAMPLES ARE TOO SIMILAR!")
else:
    print("✅ Data has good variance")

print()
print("="*60)
print("SAMPLE PREDICTIONS (Untrained Model)")
print("="*60)

from model import create_model

model = create_model().to(DEVICE)
model.eval()

# Get one batch
inputs, labels, lengths = next(iter(train_loader))
inputs = inputs.to(DEVICE)
labels = labels.to(DEVICE)

with torch.no_grad():
    outputs, _ = model(inputs)
    _, predicted = torch.max(outputs, 1)

print(f"\nTrue labels:      {labels[:10].cpu().tolist()}")
print(f"Predicted labels: {predicted[:10].cpu().tolist()}")
print(f"Unique predictions: {torch.unique(predicted).cpu().tolist()}")
print(f"Num unique: {len(torch.unique(predicted))}")

if len(torch.unique(predicted)) == 1:
    print("\n❌ MODEL PREDICTS ONLY ONE CLASS - THIS IS THE PROBLEM!")
    print("   Possible causes:")
    print("   1. Learning rate too high (causing divergence)")
    print("   2. Initialization issue")
    print("   3. Gradient flow problem")
elif len(torch.unique(predicted)) < 4:
    print(f"\n⚠️  MODEL ONLY USES {len(torch.unique(predicted))} CLASSES")
else:
    print("\n✅ Model uses multiple classes (good for untrained)")
