# 🎯 LSTM Emotion Recognition Model Training Plan

## 📋 Backend Requirements Analysis

### 1. **Audio Processing Pipeline** (`audio_processing.py`)
- **Sample Rate**: 16000 Hz (fixed)
- **Frame Size**: 25ms (0.025s)
- **Hop Length**: 10ms (0.010s)
- **Frame Length**: `int(0.025 * 16000) = 400 samples`
- **Hop Samples**: `int(0.010 * 16000) = 160 samples`
- **Normalization**: Audio scaled to [-1, 1] range
- **Output**: `frames` shape = `(400, num_frames)` where num_frames varies by audio length

### 2. **MFCC Extraction** (`mfcc_extraction.py`)
- **Input**: Framed audio from step 1
- **MFCC Coefficients**: `num_mfcc=13` (YOU SPECIFIED THIS)
- **Processing**: Per-frame MFCC computation
  - Computes MFCCs using `librosa.feature.mfcc(y=frame, sr=16000, n_mfcc=13)`
  - Takes **mean across time** for each frame: `np.mean(mfccs, axis=1)` → shape `(13,)`
  - Stacks all frame MFCCs: `np.vstack(mfcc_features)` → shape `(num_frames, 13)`
- **Silent Frame Handling**: Skips frames that are all zeros (reduces sequence length)
- **Output**: `mfcc_features` shape = `(variable_num_frames, 13)`
  - **CRITICAL**: Length varies per audio file!

### 3. **Model Inference** (`model_inference.py`)
- **Framework**: **PyTorch** (as you requested)
- **Model Loading**: `torch.load(model_path)` - expects full model object
- **Input Format**: 
  ```python
  input_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)
  # Shape: (1, seq_len, 13) where seq_len varies per audio
  ```
- **Inference Mode**: **Frame-by-frame with hidden state**
  ```python
  for i in range(total_frames):
      frame = input_tensor[:, i:i+1, :]  # (1, 1, 13)
      output, hidden = model(frame, hidden)  # output: (1, num_classes)
      probabilities = torch.softmax(output, dim=1)
  ```
- **Expected Model Interface**:
  ```python
  model(input, hidden) → (output, new_hidden)
  # input shape: (batch=1, seq_len=1, num_mfcc=13)
  # hidden: tuple of (h_0, c_0) for LSTM
  # output shape: (batch=1, num_classes)
  ```
- **Final Prediction**: Uses last frame's probabilities OR average of all (commented option)
- **Output Format**:
  ```json
  {
    "class": int,  // argmax of probabilities
    "confidence": [float, ...]  // list of probabilities for all classes
  }
  ```

### 4. **Frontend Expectations** (`ModelInference.jsx`)
- **Emotion Classes**: Currently expects 6 classes (0-5)
  - 0: Happy 😊
  - 1: Sad 😢
  - 2: Angry 😠
  - 3: Neutral 😐
  - 4: Fearful 😨
  - 5: Surprised 😲
- **Can be updated** to match RAVDESS emotions

---

## 🎭 RAVDESS Dataset Information

### Emotion Labels (8 classes):
1. **Neutral** (01)
2. **Calm** (02)
3. **Happy** (03)
4. **Sad** (04)
5. **Angry** (05)
6. **Fearful** (06)
7. **Disgust** (07)
8. **Surprised** (08)

### Filename Format:
`03-01-06-01-02-01-12.wav`
- Modality (03 = full-AV, 01 = audio-only)
- Vocal channel (01 = speech, 02 = song)
- **Emotion (01-08)** ← We need this!
- Emotional intensity (01 = normal, 02 = strong)
- Statement (01-02)
- Repetition (01-02)
- Actor (01-24, odd=male, even=female)

---

## 🏗️ Model Architecture Requirements

### **CRITICAL CONSTRAINTS** from backend:

1. **Input Shape**: `(batch, seq_len=1, input_size=13)`
   - Model must accept **single timesteps** with hidden state
   - NOT full sequences at once!

2. **Forward Pass Signature**:
   ```python
   def forward(self, x, hidden):
       # x: (batch, 1, 13)
       # hidden: (h_0, c_0) or None
       # Returns: (output, new_hidden)
       # output: (batch, num_classes)
   ```

3. **Output**: Raw logits or probabilities, shape `(batch, num_classes)`

4. **Model Format**: Must be saveable/loadable with `torch.load()`

---

## 📝 **TRAINING PLAN**

### **Phase 1: Setup & Data Preparation** (Est. 2-3 hours)

#### Step 1.1: Environment Setup
```bash
# Install dependencies
pip install torch torchaudio librosa numpy pandas scikit-learn matplotlib tqdm
```

#### Step 1.2: Download RAVDESS Dataset
- Download from: https://zenodo.org/record/1188976
- Extract to: `data/RAVDESS/`
- Verify structure: `data/RAVDESS/Actor_01/03-01-06-01-02-01-12.wav`

#### Step 1.3: Create Dataset Class
**File**: `training/dataset.py`
```python
import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, sample_rate=16000):
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.files = []
        self.labels = []
        
        # Collect all audio files
        for actor_folder in os.listdir(data_dir):
            actor_path = os.path.join(data_dir, actor_folder)
            if not os.path.isdir(actor_path):
                continue
            
            for filename in os.listdir(actor_path):
                if filename.endswith('.wav'):
                    filepath = os.path.join(actor_path, filename)
                    # Extract emotion from filename
                    emotion = int(filename.split('-')[2]) - 1  # Convert to 0-7
                    self.files.append(filepath)
                    self.labels.append(emotion)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load audio (matching backend preprocessing)
        audio, sr = librosa.load(self.files[idx], sr=self.sample_rate)
        
        # Normalize (matching backend)
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        # Frame audio (matching backend)
        frame_length = int(0.025 * self.sample_rate)  # 400
        hop_length = int(0.010 * self.sample_rate)    # 160
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        
        # Compute MFCCs per frame (matching backend)
        mfcc_features = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            if np.all(frame == 0):  # Skip silent frames
                continue
            
            mfccs = librosa.feature.mfcc(y=frame, sr=self.sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)  # (13,)
            mfcc_features.append(mfcc_mean)
        
        if len(mfcc_features) == 0:
            # Fallback for fully silent audio
            mfcc_features = [np.zeros(13)]
        
        mfcc_features = np.array(mfcc_features)  # (seq_len, 13)
        
        return torch.tensor(mfcc_features, dtype=torch.float32), self.labels[idx]
```

**QUESTION 1**: Do you want to use **all 8 RAVDESS emotions** or reduce to 6 to match current frontend?
- Option A: Use all 8 (requires frontend update)
- Option B: Merge some emotions (e.g., Calm→Neutral, Disgust→Angry)

#### Step 1.4: Data Splitting
**File**: `training/prepare_data.py`
```python
from sklearn.model_selection import train_test_split

# 80/10/10 split
# Use stratified split to ensure balanced emotion distribution
```

**QUESTION 2**: What train/validation/test split do you prefer?
- Recommended: 70% train, 15% validation, 15% test

---

### **Phase 2: Model Architecture** (Est. 1-2 hours)

#### Step 2.1: Define LSTM Model
**File**: `training/model.py`

```python
import torch
import torch.nn as nn

class EmotionLSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=8, dropout=0.3):
        super(EmotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_size) - for training: full sequences
                                              - for inference: (batch, 1, input_size)
            hidden: (h_0, c_0) tuple or None
        
        Returns:
            output: (batch, num_classes) - logits for current timestep
            hidden: (h_n, c_n) - updated hidden state
        """
        # If hidden is None, LSTM will initialize with zeros
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (batch, seq_len, hidden_size)
        
        # Take last timestep
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Apply dropout and FC
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)  # (batch, num_classes)
        
        return logits, hidden
```

**QUESTION 3**: Model hyperparameters:
- **Hidden size**: 128, 256, or 512?
- **Number of LSTM layers**: 1, 2, or 3?
- **Dropout rate**: 0.3, 0.5?

---

### **Phase 3: Training Pipeline** (Est. 2-3 hours)

#### Step 3.1: Collate Function (Handle Variable Lengths)
```python
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Pads sequences to same length in batch
    """
    sequences, labels = zip(*batch)
    
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_sequences, labels
```

#### Step 3.2: Training Loop
**File**: `training/train.py`

Key features:
- Cross-entropy loss
- Adam optimizer with learning rate scheduling
- Early stopping based on validation accuracy
- Model checkpointing (save best model)
- Training metrics: accuracy, loss, confusion matrix

**QUESTION 4**: Training hyperparameters:
- **Batch size**: 16, 32, 64?
- **Learning rate**: 0.001, 0.0001?
- **Epochs**: 50, 100, 200?
- **Optimizer**: Adam, AdamW, SGD with momentum?

---

### **Phase 4: Validation & Export** (Est. 1 hour)

#### Step 4.1: Test Inference Compatibility
Create a test script that mimics backend inference:
```python
# Load model
model = torch.load('model.pth')
model.eval()

# Test frame-by-frame inference
hidden = None
for i in range(seq_len):
    frame = mfcc_features[i:i+1].unsqueeze(0)  # (1, 1, 13)
    output, hidden = model(frame, hidden)
    probs = torch.softmax(output, dim=1)
```

#### Step 4.2: Save Model
```python
# Save entire model (as backend expects)
torch.save(model, 'model.pth')

# Also save just weights as backup
torch.save(model.state_dict(), 'model_weights.pth')
```

#### Step 4.3: Update Frontend Emotion Mapping
Update `ModelInference.jsx` to match your emotion classes.

---

## 🎯 **CRITICAL CHECKLIST**

Before starting training, confirm:

- [ ] **MFCC extraction** in training **EXACTLY matches** backend:
  - `librosa.feature.mfcc(y=frame, sr=16000, n_mfcc=13)`
  - Mean across time dimension per frame
  - Silent frame skipping logic

- [ ] **Model forward pass** accepts:
  - Input: `(batch, 1, 13)` for single timesteps
  - Hidden state: `(h_0, c_0)` tuple or None
  - Returns: `(output, new_hidden)`

- [ ] **Model saved** with `torch.save(model, path)` (full model, not just weights)

- [ ] **Emotion labels** mapped correctly (RAVDESS → your classes)

- [ ] **Frontend** updated with correct emotion names/emojis

---

## ❓ **QUESTIONS BEFORE WE START**

Please answer these to finalize the plan:

1. **Number of classes**: 8 emotions (RAVDESS all) or 6 emotions (merge some)?

2. **Data split**: 70/15/15 train/val/test or different ratio?

3. **Model architecture**:
   - Hidden size: 128, 256, or 512?
   - LSTM layers: 1, 2, or 3?
   - Dropout: 0.3 or 0.5?

4. **Training hyperparameters**:
   - Batch size: 16, 32, or 64?
   - Learning rate: 0.001 or 0.0001?
   - Max epochs: 50, 100, or 200?

5. **Data augmentation**: Do you want to add noise/pitch shift/time stretch for better generalization?

6. **Compute resources**: Do you have a GPU available? (Speeds up training significantly)

---

## 📁 **Proposed Directory Structure**

```
SpeechEmotion-LSTM/
├── data/
│   └── RAVDESS/
│       ├── Actor_01/
│       ├── Actor_02/
│       └── ...
├── training/
│   ├── dataset.py          # Dataset class
│   ├── model.py            # LSTM model definition
│   ├── train.py            # Training loop
│   ├── test.py             # Testing/validation
│   ├── utils.py            # Helper functions
│   └── config.py           # Hyperparameters
├── checkpoints/            # Saved models during training
│   ├── best_model.pth
│   └── model_epoch_50.pth
├── src/api/
│   └── model.pth           # Final production model
└── TRAINING_PLAN.md        # This file
```

---

**Ready to proceed?** Answer the questions above and I'll generate the complete training code! 🚀
