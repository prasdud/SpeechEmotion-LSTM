"""
Debug training - check if model can learn anything
"""
import torch
import torch.nn as nn
from dataset import get_data_loaders
from model import create_model
from config import *

print("🔍 Debugging Training Setup\n")

# Load data
train_loader, val_loader, _ = get_data_loaders(DATA_DIR, batch_size=32)
print(f"✅ Data loaded: {len(train_loader.dataset)} train samples\n")

# Create model
model = create_model().to(DEVICE)
print(f"✅ Model created\n")

# Get one batch
inputs, labels, lengths = next(iter(train_loader))
inputs = inputs.to(DEVICE)
labels = labels.to(DEVICE)

print("="*60)
print("DATA INSPECTION")
print("="*60)
print(f"Batch size: {inputs.shape[0]}")
print(f"Input shape: {inputs.shape}")  # (batch, seq_len, 13)
print(f"Labels shape: {labels.shape}")
print(f"Label distribution: {torch.bincount(labels)}")
print(f"Min/Max input: {inputs.min():.3f} / {inputs.max():.3f}")
print(f"Input mean/std: {inputs.mean():.3f} / {inputs.std():.3f}")
print()

# Check for NaN or Inf
if torch.isnan(inputs).any():
    print("❌ INPUTS CONTAIN NaN!")
if torch.isinf(inputs).any():
    print("❌ INPUTS CONTAIN Inf!")

print("="*60)
print("MODEL FORWARD PASS")
print("="*60)

# Forward pass
model.eval()
with torch.no_grad():
    outputs, _ = model(inputs)
    
print(f"Output shape: {outputs.shape}")  # (batch, 8)
print(f"Output min/max: {outputs.min():.3f} / {outputs.max():.3f}")
print(f"Output mean/std: {outputs.mean():.3f} / {outputs.std():.3f}")

if torch.isnan(outputs).any():
    print("❌ OUTPUTS CONTAIN NaN!")
if torch.isinf(outputs).any():
    print("❌ OUTPUTS CONTAIN Inf!")

# Check predictions
_, predicted = torch.max(outputs, 1)
print(f"\nPredicted labels: {predicted}")
print(f"Prediction distribution: {torch.bincount(predicted)}")
print(f"Accuracy: {(predicted == labels).sum().item() / len(labels) * 100:.1f}%")
print()

# Check if model is predicting same class always
unique_preds = len(torch.unique(predicted))
if unique_preds == 1:
    print("❌ MODEL PREDICTS ONLY ONE CLASS!")
elif unique_preds < 4:
    print(f"⚠️  MODEL ONLY USES {unique_preds} OUT OF 8 CLASSES")
else:
    print(f"✅ Model uses {unique_preds} different classes")
print()

print("="*60)
print("GRADIENT CHECK")
print("="*60)

# Try one training step
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

outputs, _ = model(inputs)
loss = criterion(outputs, labels)
print(f"Loss before backward: {loss.item():.4f}")

loss.backward()

# Check gradients
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5

print(f"Total gradient norm: {total_norm:.4f}")

if total_norm == 0:
    print("❌ ZERO GRADIENTS - MODEL CAN'T LEARN!")
elif total_norm < 0.001:
    print("⚠️  VERY SMALL GRADIENTS - LEARNING WILL BE SLOW")
else:
    print("✅ Gradients look normal")
print()

# Apply optimizer step
optimizer.step()

# Forward again
with torch.no_grad():
    outputs_after, _ = model(inputs)
    loss_after = criterion(outputs_after, labels)
    
print(f"Loss after step: {loss_after.item():.4f}")
print(f"Loss change: {loss.item() - loss_after.item():.6f}")

if abs(loss.item() - loss_after.item()) < 1e-6:
    print("❌ LOSS NOT CHANGING - OPTIMIZER NOT WORKING!")
else:
    print("✅ Loss is changing - optimizer working")
