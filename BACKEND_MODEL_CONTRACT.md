# 🎯 Backend-Model Contract Summary

## Critical Requirements ✅

### 1️⃣ Audio Preprocessing (MUST MATCH EXACTLY)

**Backend Pipeline:**
```
Raw Audio (.wav)
    ↓
Load at 16kHz (librosa.load)
    ↓
Normalize to [-1, 1]
    ↓
Frame: 25ms windows (400 samples)
        10ms hop (160 samples)
    ↓
Extract MFCCs per frame:
  - 13 coefficients
  - Mean across time dimension
  - Skip silent frames
    ↓
Output: (num_frames, 13)
```

**Training MUST USE SAME PIPELINE:**
✅ Implemented in `training/dataset.py` → `_extract_mfcc_features()`

---

### 2️⃣ Model Interface (MUST BE COMPATIBLE)

**Backend Inference Pattern:**
```python
# Load full model
model = torch.load(model_path)
model.eval()

# Frame-by-frame inference
hidden = None
for i in range(num_frames):
    frame = input[:, i:i+1, :]  # (1, 1, 13)
    output, hidden = model(frame, hidden)
    probs = torch.softmax(output, dim=1)
```

**Model MUST Support:**
- ✅ Input: `(batch, 1, 13)` - single timestep
- ✅ Hidden state: `(h_n, c_n)` tuple or None
- ✅ Output: `(batch, 8)` - logits for 8 classes
- ✅ Return: `(output, new_hidden)`
- ✅ Saveable with `torch.load()` / `torch.save()`

✅ Implemented in `training/model.py` → `EmotionLSTM.forward()`

---

### 3️⃣ Output Format (FRONTEND EXPECTS)

**Backend sends to frontend:**
```json
{
  "final_prediction": {
    "class": 3,           // Integer 0-7
    "confidence": [       // Array of 8 probabilities
      0.05, 0.10, 0.08, 0.62, 0.07, 0.04, 0.02, 0.02
    ]
  }
}
```

**Frontend displays:**
- Emotion name from `EMOTIONS[class]`
- Emoji from `EMOTIONS[class].emoji`
- Confidence bars for all 8 classes

✅ Frontend updated in `src/site/src/components/ModelInference.jsx`

---

## 🎭 Emotion Mapping

| Class | RAVDESS Label | Frontend Emoji | Color |
|-------|---------------|----------------|-------|
| 0 | Neutral | 😐 | #95A5A6 |
| 1 | Calm | 😌 | #A8D5BA |
| 2 | Happy | 😊 | #FFD700 |
| 3 | Sad | 😢 | #4A90E2 |
| 4 | Angry | 😠 | #E74C3C |
| 5 | Fearful | 😨 | #9B59B6 |
| 6 | Disgust | 🤢 | #7D8C3C |
| 7 | Surprised | 😲 | #F39C12 |

---

## 📊 Data Flow

```
User Upload (.wav)
    ↓
Backend: audio_processing.py
  - Load, normalize, frame
    ↓
Backend: mfcc_extraction.py
  - Compute 13 MFCCs per frame
  - Shape: (variable_frames, 13)
    ↓
Backend: model_inference.py
  - Load PyTorch model
  - Frame-by-frame inference
  - Aggregate predictions
    ↓
Frontend: PipelineContext
  - Receive via WebSocket
  - Update state
    ↓
Frontend: ModelInference.jsx
  - Display emotion + emoji
  - Show confidence bars
  - Celebration animation 🎉
```

---

## ✅ Compatibility Checklist

Before deploying your trained model, verify:

### Preprocessing Match
- [ ] Sample rate: 16000 Hz
- [ ] Frame length: 400 samples (25ms)
- [ ] Hop length: 160 samples (10ms)
- [ ] MFCC coefficients: 13
- [ ] Silent frame handling: Skip if all zeros
- [ ] Normalization: Divide by max absolute value

### Model Interface
- [ ] Accepts: `(batch, 1, input_size)` input
- [ ] Accepts: `hidden` state or None
- [ ] Returns: `(output, new_hidden)` tuple
- [ ] Output shape: `(batch, num_classes)`
- [ ] Loadable with `torch.load(path)`

### Output Format
- [ ] Class: Integer index (0-7)
- [ ] Confidence: List of 8 floats
- [ ] Sum of confidence ≈ 1.0 (probabilities)

### Frontend
- [ ] EMOTIONS mapping has 8 entries (0-7)
- [ ] Each emotion has: name, emoji, color
- [ ] Confidence bars handle 8 classes

---

## 🧪 Verification Script

Run `training/test_inference.py` to verify ALL requirements:

```bash
cd training
python test_inference.py
```

**This script:**
1. ✅ Tests preprocessing matches backend
2. ✅ Tests model interface compatibility  
3. ✅ Tests frame-by-frame inference
4. ✅ Tests output format
5. ✅ Runs on real RAVDESS audio file

**If all tests pass → Model is ready for production! 🎉**

---

## 🚨 Common Issues & Fixes

### Issue: Model predictions stuck on one class
**Cause:** Model not properly trained or weights not loaded  
**Fix:** Check training accuracy, ensure `torch.load()` worked

### Issue: All confidence values ≈ 0.125 (1/8)
**Cause:** Model outputting uniform distribution (untrained)  
**Fix:** Load correct model checkpoint with trained weights

### Issue: Frontend shows wrong emotion
**Cause:** Class mapping mismatch  
**Fix:** Verify EMOTIONS dict in ModelInference.jsx matches training classes

### Issue: Backend error during inference
**Cause:** Input shape mismatch  
**Fix:** Run `test_inference.py` to see exact error

### Issue: Predictions different from training
**Cause:** Preprocessing doesn't match  
**Fix:** Compare training `dataset.py` with backend `mfcc_extraction.py`

---

## 📝 Files Modified/Created

### Created (Training):
- ✅ `training/config.py`
- ✅ `training/dataset.py`
- ✅ `training/model.py`
- ✅ `training/train.py`
- ✅ `training/test.py`
- ✅ `training/test_inference.py`
- ✅ `training/requirements.txt`

### Modified (Frontend):
- ✅ `src/site/src/components/ModelInference.jsx` (8 emotions)

### Not Modified (Backend):
- ✅ `src/api/model_inference.py` (already compatible!)
- ✅ `src/api/mfcc_extraction.py` (num_mfcc=13 ✓)
- ✅ `src/api/audio_processing.py` (preprocessing ✓)
- ✅ `src/api/websocket_handler.py` (loads model correctly ✓)

---

## 🎓 Key Insights

### Why This Works Perfectly

1. **Training preprocessing = Backend preprocessing**  
   No "works in training but fails in production" issues!

2. **Model supports both training and inference modes**  
   - Training: Process full sequences efficiently
   - Inference: Frame-by-frame with hidden state (stateful RNN)

3. **Clear contract between components**  
   - Backend: Expects PyTorch model with specific interface
   - Model: Provides that exact interface
   - Frontend: Expects specific JSON format
   - Backend: Sends that exact format

4. **Comprehensive testing**  
   - `test_inference.py` catches issues BEFORE deployment
   - Simulates exact backend behavior
   - Validates every step of the pipeline

### Design Decisions Explained

**Q: Why 13 MFCCs instead of 40?**  
A: You specified this, and it's computationally efficient. Can be changed if needed (requires backend + training update).

**Q: Why unidirectional LSTM?**  
A: Bidirectional requires seeing full sequence, incompatible with frame-by-frame inference. Unidirectional processes one frame at a time (online/streaming).

**Q: Why frame-by-frame instead of full sequence?**  
A: Your backend design allows real-time intermediate predictions. Useful for long audio files and progress updates.

**Q: Why save full model instead of just weights?**  
A: Backend uses `torch.load(model_path)` expecting full model. Simpler deployment, no need to reconstruct architecture.

---

**Everything is designed to fit perfectly! 🎯**

Follow the QUICKSTART.md to train and deploy! 🚀
