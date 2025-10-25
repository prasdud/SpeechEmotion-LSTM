"""
Test Backend Inference Compatibility
Simulates EXACTLY how the backend will use the model
"""
import os
import torch
import numpy as np
import librosa
from config import *


def backend_preprocessing(audio_path):
    """
    Simulate EXACT backend preprocessing pipeline
    
    Matches:
    - src/api/audio_processing.py
    - src/api/mfcc_extraction.py
    """
    print(f"\n🔊 Preprocessing audio (matching backend)...")
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    print(f"  Loaded: {len(audio)} samples at {sr} Hz")
    
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    print(f"  Normalized")
    
    # Frame audio
    frame_length = int(FRAME_SIZE * SAMPLE_RATE)  # 400
    hop_length = int(HOP_LENGTH * SAMPLE_RATE)    # 160
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    print(f"  Framed: {frames.shape[1]} frames of length {frame_length}")
    
    # Compute MFCCs per frame
    mfcc_features = []
    silent_count = 0
    for i in range(frames.shape[1]):
        frame = frames[:, i]
        
        # Skip silent frames
        if np.all(frame == 0):
            silent_count += 1
            continue
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=frame, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_features.append(mfcc_mean)
    
    mfcc_features = np.array(mfcc_features)
    print(f"  MFCCs extracted: {mfcc_features.shape} (skipped {silent_count} silent frames)")
    
    return mfcc_features


def backend_inference(model, mfcc_features):
    """
    Simulate EXACT backend inference (frame-by-frame with hidden state)
    
    Matches: src/api/model_inference.py
    """
    print(f"\n🧠 Running inference (matching backend)...")
    
    # Convert to tensor with batch dimension
    input_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0)
    print(f"  Input tensor: {input_tensor.shape}")
    
    # Frame-by-frame inference with hidden state
    hidden = None
    total_frames = input_tensor.shape[1]
    intermediate_predictions = []
    
    print(f"  Processing {total_frames} frames...")
    for i in range(total_frames):
        # Extract single frame
        frame = input_tensor[:, i:i+1, :]  # shape (1, 1, 13)
        
        # Forward pass
        output, hidden = model(frame, hidden)
        
        # Get probabilities
        probabilities = torch.softmax(output, dim=1).detach().numpy()[0]
        intermediate_predictions.append(probabilities)
        
        # Log every 20 frames
        if i % 20 == 0 or i == total_frames - 1:
            pred_class = np.argmax(probabilities)
            print(f"    Frame {i}/{total_frames}: pred={EMOTION_LABELS[pred_class]} ({probabilities[pred_class]:.3f})")
    
    # Final prediction (from last frame)
    final_probabilities = intermediate_predictions[-1]
    final_class = int(np.argmax(final_probabilities))
    
    print(f"\n✅ Final Prediction:")
    print(f"  Emotion: {EMOTION_LABELS[final_class]}")
    print(f"  Confidence: {final_probabilities[final_class]:.3f}")
    print(f"\n  All probabilities:")
    for i, prob in enumerate(final_probabilities):
        print(f"    {EMOTION_LABELS[i]:12s}: {prob:.4f}")
    
    return final_class, final_probabilities


def test_model_compatibility(model_path, audio_path):
    """
    Test that model works exactly like backend expects
    """
    print("="*60)
    print("🧪 Testing Backend Inference Compatibility")
    print("="*60)
    
    # Load model (full model as backend does)
    print(f"\n📦 Loading model: {model_path}")
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    print(f"  ✅ Model loaded successfully")
    
    # Test model interface
    print(f"\n🔍 Testing model interface...")
    test_input = torch.randn(1, 1, NUM_MFCC)
    try:
        output, hidden = model(test_input, None)
        assert output.shape == (1, NUM_CLASSES), f"Expected output shape (1, {NUM_CLASSES}), got {output.shape}"
        assert hidden is not None, "Hidden state should not be None"
        assert len(hidden) == 2, "Hidden state should be tuple of (h, c)"
        print(f"  ✅ Model interface correct")
        print(f"     Input: {test_input.shape} → Output: {output.shape}")
    except Exception as e:
        print(f"  ❌ Model interface test FAILED: {e}")
        return
    
    # Preprocess audio
    mfcc_features = backend_preprocessing(audio_path)
    
    # Run inference
    pred_class, probs = backend_inference(model, mfcc_features)
    
    print("\n" + "="*60)
    print("✅ Backend compatibility test PASSED!")
    print("="*60)
    print("\nThe model is compatible with your backend!")
    print("You can now:")
    print(f"  1. Copy {model_path} to src/api/model.pth")
    print(f"  2. Update frontend emotion mapping in ModelInference.jsx")
    print(f"  3. Test with your web app")


def main():
    """Main test function"""
    # Check if model exists
    model_path = MODEL_SAVE_PATH
    if not os.path.exists(model_path):
        model_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"❌ ERROR: Model not found!")
            print(f"   Run train.py first to create the model")
            return
        else:
            # Load from checkpoint and save as full model
            print(f"📦 Loading from checkpoint: {model_path}")
            checkpoint = torch.load(model_path)
            from model import create_model
            model = create_model()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Save as full model
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model, MODEL_SAVE_PATH)
            print(f"💾 Saved full model to: {MODEL_SAVE_PATH}")
            model_path = MODEL_SAVE_PATH
    
    # Get a test audio file
    test_audio = None
    if os.path.exists(DATA_DIR):
        # Find first audio file in RAVDESS
        for actor_folder in os.listdir(DATA_DIR):
            actor_path = os.path.join(DATA_DIR, actor_folder)
            if os.path.isdir(actor_path):
                for filename in os.listdir(actor_path):
                    if filename.endswith('.wav'):
                        test_audio = os.path.join(actor_path, filename)
                        break
                if test_audio:
                    break
    
    if test_audio is None:
        print("❌ ERROR: No test audio file found!")
        print("   Please provide path to a WAV file")
        return
    
    print(f"Using test audio: {test_audio}")
    
    # Run compatibility test
    test_model_compatibility(model_path, test_audio)


if __name__ == '__main__':
    main()
