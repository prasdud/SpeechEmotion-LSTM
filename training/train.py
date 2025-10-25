"""
Training script for Emotion LSTM
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import *
from model import create_model
from dataset import get_data_loaders


class Trainer:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Mixed precision training for faster GPU training
        self.scaler = torch.amp.GradScaler()
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # Maximize validation accuracy
            factor=0.5, 
            patience=5
        )
        
        # Training history
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{MAX_EPOCHS} [Train]')
        for batch_idx, (inputs, labels, lengths) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch}/{MAX_EPOCHS} [Val]')
            for inputs, labels, lengths in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc, all_preds, all_labels
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1],
            'val_loss': self.val_losses[-1],
            'val_acc': self.val_accs[-1],
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model at epoch {epoch} with val_acc: {self.val_accs[-1]:.2f}%")
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_history.png'))
        print(f"📊 Saved training history plot to {CHECKPOINT_DIR}/training_history.png")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        
        for epoch in range(1, MAX_EPOCHS + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(epoch)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{MAX_EPOCHS}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check if best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Early stopping
            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                print(f"   No improvement for {EARLY_STOPPING_PATIENCE} epochs")
                break
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        print("="*60 + "\n")
        
        # Plot training history
        self.plot_training_history()
        
        return self.best_val_acc


def main():
    """Main training function"""
    print("🚀 Emotion Recognition LSTM Training")
    print(f"Device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ ERROR: Data directory not found: {DATA_DIR}")
        print("Please download RAVDESS dataset and place it in data/RAVDESS/")
        return
    
    # Create data loaders
    print("\n📂 Loading dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE)
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_model()
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, DEVICE)
    
    # Train
    best_acc = trainer.train()
    
    # Load best model and save for production
    print("\n💾 Saving final model for production...")
    best_checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    # Save full model (for backend)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model, MODEL_SAVE_PATH)
    print(f"✅ Saved production model to: {MODEL_SAVE_PATH}")
    
    print("\n✨ Training pipeline complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"\nNext steps:")
    print(f"  1. Run test.py to evaluate on test set")
    print(f"  2. Run test_inference.py to verify backend compatibility")
    print(f"  3. Update frontend emotion mapping")


if __name__ == '__main__':
    main()
