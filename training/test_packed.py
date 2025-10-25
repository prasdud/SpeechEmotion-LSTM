"""
Test if PackedSequence fixes the variance issue
"""
import torch
import torch.nn as nn
from dataset import get_data_loaders
from model import create_model
from config import *

print("🔍 Testing PackedSequence Fix\n")

# Load data
train_loader, _, _ = get_data_loaders(DATA_DIR, batch_size=64)

# Get one batch
inputs, labels, lengths = next(iter(train_loader))
inputs_gpu = inputs.to(DEVICE)
labels_gpu = labels.to(DEVICE)

# Create model
model = create_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

print("="*60)
print("BEFORE FIX CHECK")
print("="*60)

model.eval()
with torch.no_grad():
    # Without lengths (old way)
    outputs_old, _ = model(inputs_gpu, lengths=None)
    last_old = model.lstm(inputs_gpu)[0][:, -1, :]
    
    print(f"Without packing:")
    print(f"  Output variance: {last_old.var(dim=0).mean():.8f}")
    print(f"  Unique predictions: {len(torch.unique(torch.argmax(outputs_old, dim=1)))}")

    # With lengths (new way)
    outputs_new, _ = model(inputs_gpu, lengths=lengths)
    
    print(f"\nWith packing:")
    print(f"  Unique predictions: {len(torch.unique(torch.argmax(outputs_new, dim=1)))}")

print()
print("="*60)
print("TRAINING TEST (10 iterations)")
print("="*60)

model.train()
for step in range(10):
    optimizer.zero_grad()
    outputs, _ = model(inputs_gpu, lengths=lengths)
    loss = criterion(outputs, labels_gpu)
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == labels_gpu).sum().item() / len(labels_gpu) * 100
    unique = len(torch.unique(predicted))
    
    print(f"Step {step+1:2d}: Loss={loss.item():.4f}, Acc={acc:5.1f}%, UniquePreds={unique}/8")

print()
if acc > 20:
    print("✅ LEARNING IS WORKING! Accuracy improving!")
elif acc > 15:
    print("⚠️  Some improvement, but slow")
else:
    print("❌ Still not learning properly")
