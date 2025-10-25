"""
Training configuration
All hyperparameters in one place
"""

# Dataset
SAMPLE_RATE = 16000
FRAME_SIZE = 0.025  # 25ms (matching backend)
HOP_LENGTH = 0.010  # 10ms (matching backend)
NUM_MFCC = 13       # MUST match backend

# Model Architecture
INPUT_SIZE = 13      # MFCC coefficients
HIDDEN_SIZE = 256    # LSTM hidden dimension
NUM_LAYERS = 2       # Number of LSTM layers
NUM_CLASSES = 8      # RAVDESS emotions (0-7)
DROPOUT = 0.3        # Dropout rate

# Training
BATCH_SIZE = 256         # Optimized for RTX 2070 Super (8GB VRAM, uses only 1.38GB)
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs
WEIGHT_DECAY = 1e-5           # L2 regularization

# Data Split
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Paths
DATA_DIR = '../data'  # Points to data/ folder containing Actor_01, Actor_02, etc.
CHECKPOINT_DIR = 'checkpoints'
MODEL_SAVE_PATH = '../src/api/model.pth'

# Device
DEVICE = 'cuda'  # You have GPU

# Data Augmentation
USE_AUGMENTATION = False  # Disabled - was preventing learning
AUGMENT_PROB = 0.3  # Probability of applying augmentation

# RAVDESS Emotion Mapping
EMOTION_LABELS = {
    0: 'Neutral',
    1: 'Calm',
    2: 'Happy',
    3: 'Sad',
    4: 'Angry',
    5: 'Fearful',
    6: 'Disgust',
    7: 'Surprised'
}

# Frontend Emoji Mapping (to be copied to ModelInference.jsx)
EMOTION_EMOJIS = {
    0: {'name': 'Neutral', 'emoji': '😐', 'color': '#95A5A6'},
    1: {'name': 'Calm', 'emoji': '😌', 'color': '#A8D5BA'},
    2: {'name': 'Happy', 'emoji': '😊', 'color': '#FFD700'},
    3: {'name': 'Sad', 'emoji': '😢', 'color': '#4A90E2'},
    4: {'name': 'Angry', 'emoji': '😠', 'color': '#E74C3C'},
    5: {'name': 'Fearful', 'emoji': '😨', 'color': '#9B59B6'},
    6: {'name': 'Disgust', 'emoji': '🤢', 'color': '#7D8C3C'},
    7: {'name': 'Surprised', 'emoji': '😲', 'color': '#F39C12'}
}
