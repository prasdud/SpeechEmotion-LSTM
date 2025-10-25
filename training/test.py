"""
Test script - Evaluate model on test set
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from config import *
from dataset import get_data_loaders


def plot_confusion_matrix(y_true, y_pred, save_path='checkpoints/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=list(EMOTION_LABELS.values()),
        yticklabels=list(EMOTION_LABELS.values())
    )
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Saved confusion matrix to {save_path}")


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    correct = 0
    total = 0
    
    print("\n🧪 Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels, lengths in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs, _ = model(inputs)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Accumulate results
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * correct / total
    
    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=list(EMOTION_LABELS.values()),
        digits=3
    ))
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds)
    
    # Per-class accuracy
    print(f"\n📊 Per-Class Accuracy:")
    for i, emotion in EMOTION_LABELS.items():
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            class_acc = 100. * (np.array(all_preds)[mask] == i).sum() / mask.sum()
            print(f"  {emotion:12s}: {class_acc:.2f}%")
    
    return accuracy, all_preds, all_labels, all_probs


def main():
    """Main test function"""
    print("🧪 Testing Emotion Recognition Model")
    
    # Check if model exists
    model_path = MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model not found!")
            print(f"   Looked for: {MODEL_SAVE_PATH}")
            print(f"   And: {model_path}")
            return
    
    # Load model
    print(f"\n📦 Loading model from {model_path}")
    if model_path.endswith('.pth') and 'checkpoint' in model_path:
        # Load from checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        from model import create_model
        model = create_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"   Val Acc: {checkpoint['val_acc']:.2f}%")
    else:
        # Load full model
        model = torch.load(model_path, map_location=DEVICE)
        print(f"   Loaded full model")
    
    model = model.to(DEVICE)
    model.eval()
    
    # Load test data
    print(f"\n📂 Loading test data...")
    _, _, test_loader = get_data_loaders(DATA_DIR, BATCH_SIZE)
    
    # Evaluate
    accuracy, preds, labels, probs = evaluate_model(model, test_loader, DEVICE)
    
    print(f"\n✅ Testing complete!")
    print(f"Final Test Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    main()
