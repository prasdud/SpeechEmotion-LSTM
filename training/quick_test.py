"""
Quick sanity check - can model learn with reduced LR?
"""
import torch
import torch.nn as nn
from dataset import get_data_loaders
from model import create_model
from config import *

print("🔍 Testing if model can learn with new settings\n")
print(f"Learning Rate: {LEARNING_RATE}")
print(f"Mixed Precision: {USE_MIXED_PRECISION if 'USE_MIXED_PRECISION' in dir() else 'Not set'}\n")

# Load data
train_loader, _, _ = get_data_loaders(DATA_DIR, batch_size=32)

# Create model
model = create_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Get one batch
inputs, labels, _ = next(iter(train_loader))
inputs = inputs.to(DEVICE)
labels = labels.to(DEVICE)

print("="*60)
print("TRAINING TEST (10 iterations on same batch)")
print("="*60)

model.train()
for i in range(10):
    optimizer.zero_grad()
    outputs, _ = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == labels).sum().item() / len(labels) * 100
    
    print(f"Iter {i+1:2d}: Loss={loss.item():.4f}, Acc={acc:5.1f}%, Unique preds={len(torch.unique(predicted))}")

print()
if loss.item() < 2.0:
    print("✅ LOSS IS DECREASING - Model can learn!")
else:
    print("❌ LOSS STILL HIGH - May need further adjustment")

if len(torch.unique(predicted)) > 1:
    print("✅ Model using multiple classes")
else:
    print("❌ Model stuck on one class")
