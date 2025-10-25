"""
Adaptive Inference Script for Speech Emotion Recognition
---------------------------------------------------------
- Automatically matches MFCC dimensions to model input shape
- Supports both .wav file and live microphone input
"""

import argparse
import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from tensorflow.keras.models import load_model

# Emotion labels (adjust to your dataset)
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


# =========================
# 🎙️ Audio Recording Helper
# =========================
def record_audio(duration=3, sr=22050):
    """Records audio from the microphone."""
    print(f"🎤 Recording {duration}s of audio...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    sf.write("temp_record.wav", audio, sr)
    print("✅ Saved temp_record.wav")
    return "temp_record.wav"


# =========================
# 🎛 MFCC Extraction Helper
# =========================
def extract_mfcc(path, target_shape, sr=22050, n_mfcc=40):
    """Extracts MFCCs and adjusts them to match the model's input shape."""
    y, _ = librosa.load(path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # target_shape is (time_steps, n_mfcc)
    target_frames, target_coeffs = target_shape
    assert target_coeffs == n_mfcc, (
        f"Model expects {target_coeffs} MFCCs, but extractor used {n_mfcc}"
    )

    # Pad or truncate to match model shape
    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :target_frames]

    return np.expand_dims(mfcc.T, axis=0)  # (1, time, n_mfcc)


# =========================
# 🔍 Emotion Prediction
# =========================
def predict_emotion(model_path, ckpt_path=None, wav=None):
    """Loads model, extracts features, predicts emotion."""
    # Load model
    model = load_model(model_path, compile=False)
    if ckpt_path and Path(ckpt_path).exists():
        model.load_weights(ckpt_path)
        print(f"✅ Loaded weights from: {ckpt_path}")

    # Determine expected MFCC shape from model input
    input_shape = model.input_shape[1:]  # e.g. (120, 40)
    print(f"🧩 Model expects input shape: {input_shape}")

    # Use provided wav or record new
    if wav is None:
        try:
            wav = record_audio()
        except Exception as e:
            print(f"⚠️ Mic not available. Use --wav path instead.\n{e}")
            return

    print(f"🔍 Processing {wav} ...")
    mfcc = extract_mfcc(wav, target_shape=input_shape)

    # Predict
    pred = model.predict(mfcc)
    idx = np.argmax(pred)
    conf = np.max(pred)

    print(f"\n🎯 Predicted Emotion: {EMOTIONS[idx].upper()}  (confidence: {conf:.3f})")


# =========================
# 🚀 CLI Entry Point
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default=None, help="Path to .wav file (optional)")
    parser.add_argument("--model", type=str, default="models/final_hybrid_model.keras")
    parser.add_argument("--ckpt", type=str, default="models/best_hybrid.weights.h5")
    args = parser.parse_args()

    predict_emotion(args.model, args.ckpt, args.wav)
