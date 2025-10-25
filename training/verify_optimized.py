"""
Verify optimized MFCC extraction is correct and fast
"""
import time
import numpy as np
from dataset import RAVDESSDataset
from config import *

print("🔍 Verifying Optimized MFCC Extraction\n")

# Create dataset
dataset = RAVDESSDataset(DATA_DIR, augment=False)

print("="*60)
print("SPEED TEST")
print("="*60)

# Test 10 samples
num_samples = 10
total_time = 0

for i in range(num_samples):
    start = time.time()
    mfcc, label = dataset[i]
    elapsed = time.time() - start
    total_time += elapsed
    print(f"Sample {i}: {elapsed*1000:.1f} ms, shape: {mfcc.shape}")

avg_time = total_time / num_samples
print(f"\nAverage: {avg_time*1000:.1f} ms per sample")
print()

# Estimate batch loading time
print("="*60)
print("BATCH LOADING ESTIMATION")
print("="*60)
batch_size = 256
workers = 4

sequential_time = avg_time * batch_size
parallel_time = sequential_time / workers

print(f"Batch size: {batch_size}")
print(f"Workers: {workers}")
print(f"Per-sample time: {avg_time*1000:.1f} ms")
print(f"Parallel ({workers} workers): {parallel_time:.1f} seconds")
print()

if parallel_time < 30:
    print("✅ EXCELLENT! First batch should load in < 30 seconds")
elif parallel_time < 60:
    print("✅ GOOD! First batch should load in < 1 minute")
else:
    print("⚠️  Still slow - need more optimization")
