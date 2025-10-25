"""
Test different worker counts to find optimal setting
"""
import time
import torch
from dataset import get_data_loaders
from config import *

print("🔍 Testing Worker Scaling\n")
print(f"Batch size: {BATCH_SIZE}")
print(f"Your CPU cores: 12\n")

worker_counts = [0, 2, 4, 6, 8, 10]
results = {}

for num_workers in worker_counts:
    print("="*60)
    print(f"Testing num_workers = {num_workers}")
    print("="*60)
    
    try:
        # Create dataloader with this worker count
        import torch.utils.data
        from dataset import RAVDESSDataset, collate_fn
        
        dataset = RAVDESSDataset(DATA_DIR, augment=False)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=False
        )
        
        # Time loading first 3 batches
        times = []
        start_total = time.time()
        
        for i, batch in enumerate(loader):
            if i == 0:
                first_batch_time = time.time() - start_total
                print(f"  First batch:  {first_batch_time:.2f} seconds")
                start_total = time.time()
            elif i < 3:
                batch_time = time.time() - start_total
                times.append(batch_time)
                start_total = time.time()
            else:
                break
        
        avg_time = sum(times) / len(times) if times else 0
        print(f"  Avg 2nd-3rd:  {avg_time:.2f} seconds")
        print(f"  Status:       ✅ Working")
        
        results[num_workers] = {
            'first_batch': first_batch_time,
            'avg_batch': avg_time,
            'status': 'OK'
        }
        
    except Exception as e:
        print(f"  Status:       ❌ Error: {e}")
        results[num_workers] = {'status': 'FAILED'}
    
    print()

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
print(f"{'Workers':<10} {'First Batch':<15} {'Avg Batch':<15} {'Status'}")
print("-"*60)

for workers, result in results.items():
    if result['status'] == 'OK':
        print(f"{workers:<10} {result['first_batch']:<15.2f} {result['avg_batch']:<15.2f} ✅")
    else:
        print(f"{workers:<10} {'N/A':<15} {'N/A':<15} ❌")

# Find best
if any(r['status'] == 'OK' for r in results.values()):
    best_workers = min(
        [w for w, r in results.items() if r['status'] == 'OK'],
        key=lambda w: results[w]['avg_batch']
    )
    print()
    print(f"🎯 Recommendation: num_workers = {best_workers}")
    print(f"   First batch: {results[best_workers]['first_batch']:.2f}s")
    print(f"   Avg batch:   {results[best_workers]['avg_batch']:.2f}s")
