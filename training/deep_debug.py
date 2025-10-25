"""
Deep dive - why isn't the LSTM learning?
"""
import torch
import torch.nn as nn
import numpy as np
from dataset import get_data_loaders
from model import create_model
from config import *

print("🔍 LSTM Learning Investigation\n")

# Load data
train_loader, _, _ = get_data_loaders(DATA_DIR, batch_size=64)

# Get one batch
inputs, labels, lengths = next(iter(train_loader))
print("="*60)
print("DATA INSPECTION")
print("="*60)
print(f"Batch size: {inputs.shape[0]}")
print(f"Input shape: {inputs.shape}")
print(f"Sequence lengths: min={lengths.min()}, max={lengths.max()}, mean={lengths.float().mean():.1f}")
print(f"Labels: {labels.tolist()[:10]}")
print(f"Input stats: min={inputs.min():.2f}, max={inputs.max():.2f}, mean={inputs.mean():.2f}, std={inputs.std():.2f}")
print()

# Create model
model = create_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()

# Check LSTM output
print("="*60)
print("LSTM OUTPUT INSPECTION")
print("="*60)

inputs_gpu = inputs.to(DEVICE)
labels_gpu = labels.to(DEVICE)

model.eval()
with torch.no_grad():
    # Get LSTM intermediate output
    lstm_out, (h_n, c_n) = model.lstm(inputs_gpu)
    print(f"LSTM output shape: {lstm_out.shape}")
    print(f"LSTM output stats: min={lstm_out.min():.4f}, max={lstm_out.max():.4f}, mean={lstm_out.mean():.4f}, std={lstm_out.std():.4f}")
    
    # Get last timestep
    last_output = lstm_out[:, -1, :]
    print(f"\nLast timestep shape: {last_output.shape}")
    print(f"Last timestep stats: min={last_output.min():.4f}, max={last_output.max():.4f}, mean={last_output.mean():.4f}, std={last_output.std():.4f}")
    
    # Check if all outputs are similar
    output_variance = last_output.var(dim=0).mean()
    print(f"Output variance: {output_variance:.6f}")
    if output_variance < 0.001:
        print("❌ LSTM OUTPUTS HAVE ALMOST NO VARIANCE - All samples look the same to the model!")
    
    # Final predictions
    logits = model.fc(model.dropout(last_output))
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}, std={logits.std():.4f}")
    
    _, predicted = torch.max(logits, 1)
    print(f"Predictions: {predicted.cpu().tolist()[:10]}")
    print(f"Unique predictions: {torch.unique(predicted).cpu().tolist()}")

print()
print("="*60)
print("TRAINING STEP BY STEP")
print("="*60)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for step in range(5):
    optimizer.zero_grad()
    outputs, _ = model(inputs_gpu)
    loss = criterion(outputs, labels_gpu)
    loss.backward()
    
    # Check gradients
    total_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    optimizer.step()
    
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == labels_gpu).sum().item() / len(labels_gpu) * 100
    
    print(f"Step {step+1}: Loss={loss.item():.4f}, Acc={acc:5.1f}%, GradNorm={total_norm:.4f}, UniquePreds={len(torch.unique(predicted))}")

print()
print("="*60)
print("DIAGNOSIS")
print("="*60)

# Try without dropout
model_nodrop = create_model()
model_nodrop.dropout = nn.Identity()  # Disable dropout
model_nodrop = model_nodrop.to(DEVICE)

optimizer2 = torch.optim.Adam(model_nodrop.parameters(), lr=LEARNING_RATE)

print("\nTrying WITHOUT dropout:")
for step in range(5):
    optimizer2.zero_grad()
    outputs, _ = model_nodrop(inputs_gpu)
    loss = criterion(outputs, labels_gpu)
    loss.backward()
    optimizer2.step()
    
    _, predicted = torch.max(outputs, 1)
    acc = (predicted == labels_gpu).sum().item() / len(labels_gpu) * 100
    
    print(f"Step {step+1}: Loss={loss.item():.4f}, Acc={acc:5.1f}%, UniquePreds={len(torch.unique(predicted))}")
