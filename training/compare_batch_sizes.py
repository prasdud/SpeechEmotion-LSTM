"""
Quick comparison of different batch sizes
Tests 5 epochs with each batch size to see convergence
"""
import torch
from dataset import get_dataloaders
from model import create_model
from train import Trainer
import config

batch_sizes_to_test = [64, 128, 256, 512]
results = {}

print("Testing batch sizes on 5 epochs each...\n")

for batch_size in batch_sizes_to_test:
    print(f"{'='*60}")
    print(f"Testing BATCH_SIZE = {batch_size}")
    print(f"{'='*60}")
    
    # Create fresh model and data
    model = create_model()
    train_loader, val_loader, _ = get_dataloaders(
        config.DATA_DIR,
        batch_size=batch_size
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Updates per epoch: {len(train_loader)}")
    print()
    
    # Train for 5 epochs
    trainer = Trainer(model, train_loader, val_loader, max_epochs=5)
    history = trainer.train()
    
    # Store results
    results[batch_size] = {
        'final_val_acc': history['val_acc'][-1],
        'best_val_acc': max(history['val_acc']),
        'updates_per_epoch': len(train_loader)
    }
    
    print(f"\nBatch {batch_size} Results:")
    print(f"  Best Val Acc: {results[batch_size]['best_val_acc']:.1f}%")
    print(f"  Final Val Acc: {results[batch_size]['final_val_acc']:.1f}%")
    print()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Batch':<10} {'Updates/Epoch':<15} {'Best Acc':<12} {'Final Acc'}")
print("-"*60)
for bs in batch_sizes_to_test:
    r = results[bs]
    print(f"{bs:<10} {r['updates_per_epoch']:<15} {r['best_val_acc']:<12.1f} {r['final_val_acc']:.1f}%")

print("\n💡 Generally, more updates/epoch = better learning")
