"""
Test maximum batch size for your GPU
"""
import torch
from model import create_model

device = 'cuda'
model = create_model().to(device)

print("Testing batch sizes on RTX 2070 Super...")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

for batch_size in batch_sizes:
    try:
        # Clear cache
        torch.cuda.empty_cache()
        
        # Test forward + backward pass
        dummy_input = torch.randn(batch_size, 250, 13).to(device)  # Average seq length
        dummy_labels = torch.randint(0, 8, (batch_size,)).to(device)
        
        # Forward
        output, _ = model(dummy_input)
        loss = torch.nn.functional.cross_entropy(output, dummy_labels)
        
        # Backward
        loss.backward()
        
        # Check memory
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"✅ Batch {batch_size:3d}: {memory_used:.2f} GB used")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ Batch {batch_size:3d}: OUT OF MEMORY")
            break
        else:
            raise e
    finally:
        torch.cuda.empty_cache()

print("\n💡 Recommendation:")
print("   Use largest batch size that fits with ~20% headroom")
