"""
RAVDESS Dataset Loader
Preprocessing EXACTLY matches backend pipeline
"""
import os
import librosa
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset
from config import *

# Suppress the n_fft warning (expected behavior for 25ms frames)
warnings.filterwarnings('ignore', message='n_fft=2048 is too large for input signal of length=400')


class RAVDESSDataset(Dataset):
    def __init__(self, data_dir, file_list=None, augment=False):
        """
        Args:
            data_dir: Path to RAVDESS dataset
            file_list: List of (filepath, label) tuples for train/val/test split
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.augment = augment
        
        if file_list is not None:
            self.files = [f[0] for f in file_list]
            self.labels = [f[1] for f in file_list]
        else:
            self.files, self.labels = self._collect_files()
    
    def _collect_files(self):
        """Collect all WAV files and extract emotion labels"""
        files = []
        labels = []
        
        for actor_folder in sorted(os.listdir(self.data_dir)):
            actor_path = os.path.join(self.data_dir, actor_folder)
            if not os.path.isdir(actor_path):
                continue
            
            for filename in sorted(os.listdir(actor_path)):
                if filename.endswith('.wav'):
                    filepath = os.path.join(actor_path, filename)
                    
                    # Parse RAVDESS filename: 03-01-06-01-02-01-12.wav
                    # Emotion is 3rd field (index 2)
                    parts = filename.split('-')
                    emotion = int(parts[2]) - 1  # Convert to 0-indexed (0-7)
                    
                    files.append(filepath)
                    labels.append(emotion)
        
        return files, labels
    
    def __len__(self):
        return len(self.files)
    
    def _augment_audio(self, audio):
        """
        Apply random augmentation
        """
        if not self.augment or np.random.rand() > AUGMENT_PROB:
            return audio
        
        # Choose random augmentation
        aug_type = np.random.choice(['noise', 'pitch', 'speed'])
        
        if aug_type == 'noise':
            # Add white noise
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        
        elif aug_type == 'pitch':
            # Pitch shift (±2 semitones)
            n_steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)
        
        elif aug_type == 'speed':
            # Time stretch (0.9x to 1.1x)
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        return audio
    
    def _extract_mfcc_features(self, audio):
        """
        Extract MFCC features EXACTLY matching backend preprocessing
        
        This is CRITICAL - must match:
        - src/api/audio_processing.py: frame_audio()
        - src/api/mfcc_extraction.py: compute_mfcc()
        """
        # Step 1: Normalize (matching backend)
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        
        # Step 2: Frame audio (matching backend)
        frame_length = int(FRAME_SIZE * SAMPLE_RATE)  # 400 samples
        hop_length_samples = int(HOP_LENGTH * SAMPLE_RATE)  # 160 samples
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length_samples)
        # frames shape: (400, num_frames)
        
        # Step 3: Compute MFCCs per frame (matching backend)
        mfcc_features = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            
            # Skip silent frames (matching backend)
            if np.all(frame == 0):
                continue
            
            # Compute MFCCs for this frame
            try:
                mfccs = librosa.feature.mfcc(y=frame, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
                # mfccs shape: (13, time_steps_in_frame)
                
                # Take mean across time (matching backend)
                mfcc_mean = np.mean(mfccs, axis=1)  # shape: (13,)
                mfcc_features.append(mfcc_mean)
            except:
                # Skip problematic frames
                continue
        
        # Handle case where all frames are silent
        if len(mfcc_features) == 0:
            mfcc_features = [np.zeros(NUM_MFCC)]
        
        # Stack into array: (num_frames, 13)
        mfcc_features = np.array(mfcc_features, dtype=np.float32)
        
        return mfcc_features
    
    def __getitem__(self, idx):
        """
        Returns:
            mfcc_features: torch.Tensor of shape (seq_len, 13)
            label: int (0-7)
        """
        # Load audio
        audio, sr = librosa.load(self.files[idx], sr=SAMPLE_RATE)
        
        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment_audio(audio)
        
        # Extract MFCC features (matching backend)
        mfcc_features = self._extract_mfcc_features(audio)
        
        # Convert to tensor
        mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32)
        label = self.labels[idx]
        
        return mfcc_tensor, label


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    Pads sequences to max length in batch
    
    Args:
        batch: List of (sequence, label) tuples
    
    Returns:
        padded_sequences: (batch_size, max_seq_len, 13)
        labels: (batch_size,)
        lengths: (batch_size,) - original lengths before padding
    """
    sequences, labels = zip(*batch)
    
    # Get lengths
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # Pad sequences to max length in batch
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, 
        batch_first=True, 
        padding_value=0.0
    )
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_sequences, labels, lengths


def get_data_loaders(data_dir, batch_size=BATCH_SIZE):
    """
    Create train/val/test data loaders with stratified split
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split
    
    # Collect all files
    dataset = RAVDESSDataset(data_dir)
    all_files = list(zip(dataset.files, dataset.labels))
    
    # Stratified split: first split train vs (val+test)
    train_files, temp_files = train_test_split(
        all_files,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=[label for _, label in all_files],
        random_state=42
    )
    
    # Split (val+test) into val and test
    val_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=(1 - val_size),
        stratify=[label for _, label in temp_files],
        random_state=42
    )
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val:   {len(val_files)} samples")
    print(f"  Test:  {len(test_files)} samples")
    
    # Create datasets
    train_dataset = RAVDESSDataset(data_dir, train_files, augment=USE_AUGMENTATION)
    val_dataset = RAVDESSDataset(data_dir, val_files, augment=False)
    test_dataset = RAVDESSDataset(data_dir, test_files, augment=False)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=10,       # Optimized for 12-core CPU (leave 2 for system)
        pin_memory=True,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=10,       # Optimized for 12-core CPU (leave 2 for system)
        pin_memory=True,
        persistent_workers=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=10,       # Optimized for 12-core CPU (leave 2 for system)
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader
