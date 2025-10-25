"""
Pre-Training Verification Script
Run this BEFORE training to catch issues early
"""
import os
import sys
from config import *



def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"   ❌ Python {version.major}.{version.minor} detected")
        print(f"   ⚠️  Python 3.8+ required")
        return False
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'librosa': 'Librosa',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} not installed")
            missing.append(name)
    
    if missing:
        print(f"\n   ⚠️  Install missing packages:")
        print(f"   pip install -r requirements.txt")
        return False
    
    return True


def check_cuda():
    """Check if CUDA is available"""
    print("\n🎮 Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print(f"   ⚠️  CUDA not available (will use CPU)")
            print(f"   Training will be MUCH slower on CPU")
            return True  # Not a failure, just slower
    except:
        print(f"   ❌ Could not check CUDA")
        return False


def check_data_directory():
    """Check if RAVDESS dataset exists"""
    print("\n📂 Checking dataset...")
    
    data_dir = '../data'
    if not os.path.exists(data_dir):
        print(f"   ❌ Dataset directory not found: {data_dir}")
        print(f"\n   📥 Download RAVDESS dataset:")
        print(f"   1. Go to: https://zenodo.org/record/1188976")
        print(f"   2. Download 'Audio_Speech_Actors_01-24.zip' (208.5 MB)")
        print(f"   3. Extract to: data/")
        print(f"\n   Commands:")
        print(f"   mkdir -p data")
        print(f"   unzip Audio_Speech_Actors_01-24.zip -d data/")
        print(f"\n   Expected structure:")
        print(f"   data/")
        print(f"   ├── Actor_01/")
        print(f"   │   ├── 03-01-01-01-01-01-01.wav")
        print(f"   │   └── ...")
        print(f"   ├── Actor_02/")
        print(f"   └── ...")
        return False
    
    # Count actors and files
    actors = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    total_files = 0
    for actor in actors:
        actor_path = os.path.join(data_dir, actor)
        files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
        total_files += len(files)
    
    print(f"   ✅ Dataset found")
    print(f"   Actors: {len(actors)}")
    print(f"   Audio files: {total_files}")
    
    if total_files < 100:
        print(f"   ⚠️  Very few audio files detected")
        print(f"   Expected: ~1440 files (60 per actor × 24 actors)")
        return False
    
    return True


def check_config():
    """Check if config.py is valid"""
    print("\n⚙️  Checking configuration...")
    try:
        
        # Check critical parameters
        checks = [
            (NUM_MFCC == 13, f"NUM_MFCC = {NUM_MFCC} (should be 13 for backend compatibility)"),
            (SAMPLE_RATE == 16000, f"SAMPLE_RATE = {SAMPLE_RATE} (should be 16000)"),
            (NUM_CLASSES == 8, f"NUM_CLASSES = {NUM_CLASSES} (should be 8 for RAVDESS)"),
            (INPUT_SIZE == 13, f"INPUT_SIZE = {INPUT_SIZE} (should be 13)"),
        ]
        
        all_good = True
        for check, msg in checks:
            if check:
                print(f"   ✅ {msg}")
            else:
                print(f"   ❌ {msg}")
                all_good = False
        
        return all_good
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")
        return False


def test_model_creation():
    """Test if model can be created"""
    print("\n🏗️  Testing model creation...")
    try:
        from model import create_model
        import torch
        
        model = create_model()
        print(f"   ✅ Model created successfully")
        print(f"   Parameters: {model.get_num_params():,}")
        
        # Test forward pass
        test_input = torch.randn(1, 1, 13)
        output, hidden = model(test_input, None)
        
        assert output.shape == (1, 8), f"Output shape should be (1, 8), got {output.shape}"
        assert hidden is not None, "Hidden state should not be None"
        assert len(hidden) == 2, "Hidden should be (h, c) tuple"
        
        print(f"   ✅ Forward pass works")
        print(f"   Input: {test_input.shape} → Output: {output.shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test if dataset can be loaded"""
    print("\n📊 Testing dataset loader...")
    try:
        from dataset import RAVDESSDataset
        
        data_dir = '../data/'
        if not os.path.exists(data_dir):
            print(f"   ⚠️  Skipping (dataset not found)")
            return True  # Not a failure if we already warned
        
        dataset = RAVDESSDataset(data_dir)
        print(f"   ✅ Dataset loaded")
        print(f"   Total samples: {len(dataset)}")
        
        # Try to load one sample
        mfcc, label = dataset[0]
        print(f"   ✅ Sample loaded: MFCC shape {mfcc.shape}, label {label}")
        
        assert mfcc.shape[1] == 13, f"MFCC should have 13 features, got {mfcc.shape[1]}"
        assert 0 <= label < 8, f"Label should be 0-7, got {label}"
        
        print(f"   ✅ Dataset format correct")
        
        return True
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks"""
    print("="*60)
    print("🔍 Pre-Training Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("GPU/CUDA", check_cuda),
        ("Dataset", check_data_directory),
        ("Configuration", check_config),
        ("Model Creation", test_model_creation),
        ("Dataset Loader", test_dataset),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All checks passed! Ready to train!")
        print("\nNext steps:")
        print("  python train.py")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Download RAVDESS dataset")
        print("  - Check config.py settings")
    
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
