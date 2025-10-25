# 🚀 Quick Start Guide - Training Your LSTM Model

## ✅ Prerequisites
- Python 3.8+
- CUDA-capable GPU (you have this!)
- ~10GB disk space for dataset

## 📥 Step 1: Download RAVDESS Dataset

1. Go to: https://zenodo.org/record/1188976
2. Download **Audio_Speech_Actors_01-24.zip** (208.5 MB)
   - This contains audio-only speech recordings from all 24 actors
3. Extract to `data/` folder in your project

**Commands:**
```bash
# From project root
mkdir -p data
cd data

# Extract the downloaded zip file (adjust path if needed)
unzip ~/Downloads/Audio_Speech_Actors_01-24.zip

# Verify extraction
ls
```

Your directory structure should look like:
```
data/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-01-01-01-02-01.wav
│   └── ...
├── Actor_02/
│   └── ...
...
├── Actor_24/
    └── ...
```

**Expected:** 24 Actor folders with ~60 .wav files each (1440 total files)

## 🔧 Step 2: Install Training Dependencies

```bash
cd training
pip install -r requirements.txt
```

## 🏋️ Step 3: Train the Model

```bash
cd training
python train.py
```

**What happens:**
- Loads RAVDESS dataset (1440 audio files from 24 actors)
- Splits into train (70%), validation (15%), test (15%)
- Extracts MFCC features (matching backend preprocessing EXACTLY)
- Trains LSTM model with:
  - 256 hidden units
  - 2 LSTM layers
  - Dropout 0.3
  - Data augmentation (noise, pitch shift, time stretch)
- Saves best model based on validation accuracy
- Creates training plots and checkpoints

**Expected training time:** ~30-60 minutes on GPU

**Output:**
- `checkpoints/best_model.pth` - Best model checkpoint
- `checkpoints/training_history.png` - Training curves
- `src/api/model.pth` - Production model (auto-copied)

## 🧪 Step 4: Test the Model

```bash
python test.py
```

**Output:**
- Test accuracy
- Confusion matrix
- Per-class accuracy
- Classification report

**Expected accuracy:** 60-80% (8-class emotion is challenging!)

## ✅ Step 5: Verify Backend Compatibility

```bash
python test_inference.py
```

This script:
- Simulates EXACT backend preprocessing
- Tests frame-by-frame inference with hidden state
- Verifies output format matches frontend expectations

**If this passes, your model is ready!**

## 🎯 Step 6: Deploy to Backend

The model is already saved to `src/api/model.pth`!

Just:
1. **Frontend is already updated** ✅ (8 emotions in `ModelInference.jsx`)
2. Start your backend: `python src/api/app.py`
3. Start your frontend: `cd src/site && npm run dev`
4. Test with audio files!

## 📊 Expected Results

### Model Performance
- **Training Accuracy:** 85-95%
- **Validation Accuracy:** 65-80%
- **Test Accuracy:** 60-75%

Note: Emotion recognition from audio alone is inherently challenging. These results are competitive with research papers!

### Most Confusing Emotions
- Calm vs Neutral
- Angry vs Disgust
- Happy vs Surprised

## 🐛 Troubleshooting

### "CUDA out of memory"
Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # or 8
```

### "Dataset not found"
Check your data directory structure matches Step 1 exactly.

### "Model accuracy too low" (<50%)
- Train for more epochs (increase `MAX_EPOCHS`)
- Try different learning rate
- Check if data augmentation is too aggressive

### Backend predictions always wrong
Run `test_inference.py` - it will show you exact preprocessing differences.

## 🎓 Advanced: Hyperparameter Tuning

Edit `training/config.py` to experiment:

```python
# Try larger model
HIDDEN_SIZE = 512
NUM_LAYERS = 3

# Try different learning rate
LEARNING_RATE = 0.0001

# More aggressive augmentation
AUGMENT_PROB = 0.5
```

Then re-run `python train.py`

## 📝 Model Details

### Architecture
```
Input (batch, seq_len, 13) 
  ↓
LSTM Layer 1 (256 hidden)
  ↓
LSTM Layer 2 (256 hidden)
  ↓
Dropout (0.3)
  ↓
Fully Connected (256 → 8)
  ↓
Output (batch, 8)
```

### Total Parameters: ~988K

### Preprocessing (matches backend EXACTLY)
1. Load audio at 16kHz
2. Normalize to [-1, 1]
3. Frame into 25ms windows (400 samples)
4. 10ms hop length (160 samples)
5. Extract 13 MFCCs per frame
6. Mean pooling across time dimension

## 🎉 Next Steps

Once training is complete and tests pass:
1. Upload audio files to your web app
2. Watch the real-time pipeline in action!
3. Celebrate with your emotion predictions 🎊

## 💡 Tips for Better Accuracy

1. **Collect more data:** RAVDESS is small. Consider:
   - CREMA-D dataset
   - TESS dataset
   - Combine multiple datasets

2. **Feature engineering:**
   - Try 40 MFCCs instead of 13 (requires backend change)
   - Add delta and delta-delta features
   - Try mel spectrograms

3. **Model improvements:**
   - Try bidirectional LSTM
   - Add attention mechanism
   - Use pre-trained models (Wav2Vec2, HuBERT)

4. **Ensemble methods:**
   - Train multiple models with different seeds
   - Average predictions

Good luck! 🚀
