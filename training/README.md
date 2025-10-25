# 🎓 Training Directory

This directory contains everything needed to train the LSTM emotion recognition model.

## 📁 Files

### Core Training Files
- **`config.py`** - All hyperparameters and configuration
- **`dataset.py`** - RAVDESS dataset loader (matches backend preprocessing)
- **`model.py`** - LSTM model architecture (backend-compatible)
- **`train.py`** - Training script with validation and checkpointing
- **`test.py`** - Evaluation on test set with metrics
- **`test_inference.py`** - Backend compatibility verification

### Documentation
- **`requirements.txt`** - Python dependencies
- **`CHECKLIST.md`** - Training progress tracker

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download RAVDESS dataset to ../data/RAVDESS/

# 3. Train model
python train.py

# 4. Test model
python test.py

# 5. Verify backend compatibility
python test_inference.py
```

## 🎯 What Makes This Training Special

### ✅ Backend Compatibility
The preprocessing in `dataset.py` EXACTLY matches your backend:
- Same sample rate (16kHz)
- Same frame size (25ms = 400 samples)
- Same hop length (10ms = 160 samples)  
- Same MFCC extraction (13 coefficients)
- Same normalization
- Same silent frame handling

This ensures **zero preprocessing mismatch** between training and inference!

### ✅ Inference Compatibility
The model in `model.py` supports BOTH:
- **Training mode:** Full sequences `(batch, seq_len, 13)`
- **Inference mode:** Frame-by-frame `(batch, 1, 13)` with hidden state

This matches your backend's frame-by-frame inference pattern!

### ✅ Production Ready
- Automatic best model selection
- Early stopping to prevent overfitting
- Learning rate scheduling
- Data augmentation for better generalization
- Comprehensive evaluation metrics

## 📊 Model Architecture

```
EmotionLSTM(
  (lstm): LSTM(13, 256, num_layers=2, batch_first=True, dropout=0.3)
  (dropout): Dropout(p=0.3)
  (fc): Linear(256, 8)
)
Total parameters: ~988K
```

## 🎭 Emotion Classes (RAVDESS)

0. Neutral 😐
1. Calm 😌
2. Happy 😊
3. Sad 😢
4. Angry 😠
5. Fearful 😨
6. Disgust 🤢
7. Surprised 😲

## ⚙️ Default Hyperparameters

Edit `config.py` to change these:

```python
# Data
NUM_MFCC = 13         # MUST match backend
SAMPLE_RATE = 16000   # MUST match backend

# Model
HIDDEN_SIZE = 256
NUM_LAYERS = 2
NUM_CLASSES = 8
DROPOUT = 0.3

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# Augmentation
USE_AUGMENTATION = True
AUGMENT_PROB = 0.3
```

## 📈 Expected Results

- **Training Accuracy:** 85-95%
- **Validation Accuracy:** 65-80%
- **Test Accuracy:** 60-75%
- **Training Time:** 30-60 minutes on GPU

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# In config.py
BATCH_SIZE = 16  # Reduce from 32
```

### Low Accuracy (<50%)
- Check data directory structure
- Increase training epochs
- Try different learning rate
- Reduce augmentation probability

### Backend Predictions Wrong
```bash
# Run compatibility test
python test_inference.py

# Compare preprocessing differences
```

## 📝 Training Output

After training completes, you'll have:

```
checkpoints/
├── best_model.pth              # Best model checkpoint
├── checkpoint_epoch_10.pth     # Regular checkpoints
├── checkpoint_epoch_20.pth
├── ...
├── training_history.png        # Training curves
└── confusion_matrix.png        # Test set confusion matrix

src/api/
└── model.pth                   # Production model (ready to use!)
```

## 🧪 Testing Commands

```bash
# Test model on test set
python test.py

# Verify backend compatibility
python test_inference.py

# Test model architecture
python model.py
```

## 🎯 Next Steps After Training

1. ✅ Model saved to `../src/api/model.pth`
2. ✅ Frontend already updated (8 emotions)
3. 🚀 Start backend and test with web app!

## 💡 Advanced Usage

### Custom Dataset
Modify `dataset.py` to load your own data:
```python
class CustomDataset(RAVDESSDataset):
    def _collect_files(self):
        # Your custom file collection logic
        pass
```

### Different Model Architecture
Edit `model.py`:
```python
# Try bidirectional LSTM
self.lstm = nn.LSTM(..., bidirectional=True)

# Add attention layer
# etc.
```

### Transfer Learning
```python
# In train.py, load pre-trained weights
checkpoint = torch.load('pretrained.pth')
model.load_state_dict(checkpoint, strict=False)
```

## 📚 References

- RAVDESS Dataset: https://zenodo.org/record/1188976
- PyTorch LSTM Docs: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Librosa MFCC: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html

---

**Happy Training! 🎉**

If you encounter any issues, check the main `QUICKSTART.md` in the root directory.
