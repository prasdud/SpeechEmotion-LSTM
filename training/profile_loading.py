"""
Profile data loading bottlenecks
"""
import time
import numpy as np
import librosa
from dataset import RAVDESSDataset
from config import *

print("🔍 Profiling Data Loading Pipeline\n")

# Create dataset
dataset = RAVDESSDataset(DATA_DIR, augment=False)
print(f"Dataset size: {len(dataset)} samples\n")

# Profile single sample loading
print("="*60)
print("SINGLE SAMPLE PROFILING")
print("="*60)

idx = 0
filepath = dataset.files[idx]
print(f"File: {filepath}\n")

# 1. Audio loading
start = time.time()
audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
load_time = time.time() - start
print(f"1. librosa.load():        {load_time*1000:.1f} ms")

# 2. MFCC extraction (current method)
start = time.time()
mfcc_features = dataset._extract_mfcc_features(audio)
mfcc_time = time.time() - start
print(f"2. MFCC extraction:       {mfcc_time*1000:.1f} ms")
print(f"   Total per sample:      {(load_time + mfcc_time)*1000:.1f} ms")
print()

# 3. Estimate batch loading time
print("="*60)
print("BATCH LOADING ESTIMATION")
print("="*60)
batch_size = 256
workers = 4

per_sample_time = load_time + mfcc_time
sequential_time = per_sample_time * batch_size
parallel_time = sequential_time / workers

print(f"Batch size: {batch_size}")
print(f"Workers: {workers}")
print(f"Per-sample time: {per_sample_time*1000:.1f} ms")
print(f"Sequential (1 worker): {sequential_time:.1f} seconds")
print(f"Parallel ({workers} workers): {parallel_time:.1f} seconds")
print()

# 4. Test faster MFCC extraction
print("="*60)
print("OPTIMIZATION: Direct librosa.feature.mfcc()")
print("="*60)

start = time.time()
# Direct MFCC computation (faster)
mfcc_fast = librosa.feature.mfcc(
    y=audio,
    sr=SAMPLE_RATE,
    n_mfcc=NUM_MFCC,
    n_fft=int(FRAME_SIZE * SAMPLE_RATE),
    hop_length=int(HOP_LENGTH * SAMPLE_RATE)
)
# Transpose to (time, features)
mfcc_fast = mfcc_fast.T
mfcc_fast_time = time.time() - start

print(f"Direct MFCC:              {mfcc_fast_time*1000:.1f} ms")
print(f"Speedup:                  {mfcc_time/mfcc_fast_time:.1f}x faster!")
print(f"Shape: {mfcc_fast.shape}")
print()

# 5. Verify they're similar
print("="*60)
print("VERIFICATION")
print("="*60)
print(f"Current method shape: {mfcc_features.shape}")
print(f"Fast method shape:    {mfcc_fast.shape}")
print(f"Shapes match:         {mfcc_features.shape == mfcc_fast.shape}")

if mfcc_features.shape == mfcc_fast.shape:
    diff = np.abs(mfcc_features - mfcc_fast).mean()
    print(f"Mean difference:      {diff:.6f}")
    print(f"Methods equivalent:   {diff < 0.01}")
print()

# 6. Overall impact
print("="*60)
print("IMPACT ON TRAINING")
print("="*60)
new_parallel_time = (load_time + mfcc_fast_time) * batch_size / workers
speedup = parallel_time / new_parallel_time

print(f"OLD first batch time: {parallel_time:.1f} seconds")
print(f"NEW first batch time: {new_parallel_time:.1f} seconds")
print(f"Speedup:              {speedup:.1f}x faster!")
print()
print(f"💡 Recommendation: Use librosa.feature.mfcc() directly")
