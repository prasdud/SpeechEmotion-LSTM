# inference.py
import argparse
import numpy as np
import librosa
from models import build_bilstm
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import os

LABELS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]

def extract_for_inference(wav_path, sr=16000, n_mfcc=40, n_fft=1024, hop_length=512, duration=3.0):
    y, _ = librosa.load(wav_path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=25)
    max_len = int(sr*duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, delta, delta2]).T  # (T, F)
    return feats.astype(np.float32)

def predict_from_file(model, wav_path, args):
    feats = extract_for_inference(wav_path, sr=args.sr, n_mfcc=args.n_mfcc, n_fft=args.n_fft, hop_length=args.hop_length, duration=args.duration)
    # pad/truncate to model input T
    T = model.input.shape[1]
    feats = feats[:T, :]
    if feats.shape[0] < T:
        pad_len = T - feats.shape[0]
        feats = np.pad(feats, ((0,pad_len),(0,0)))
    preds = model.predict(np.expand_dims(feats, 0))
    idx = int(np.argmax(preds, axis=1)[0])
    return LABELS[idx], float(np.max(preds))

def record_and_save(out_path, duration=3, sr=16000):
    print("Recording for", duration, "seconds...")
    rec = sd.rec(int(duration*sr), samplerate=sr, channels=1)
    sd.wait()
    sf.write(out_path, rec, sr)
    print("Saved recording to", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wav", default=None, help="Path to WAV file. If omitted, script records from microphone.")
    p.add_argument("--ckpt", default="models/best_weights.h5")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n_mfcc", type=int, default=40)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=512)
    p.add_argument("--duration", type=float, default=3.0)
    args = p.parse_args()

    # build model (we need to know input shape - load sample .npy)
    # this uses a sample file to derive T and F
    import glob
    sample_feat_files = glob.glob("data/features/*.npy")
    if not sample_feat_files:
        raise RuntimeError("No features found. Run preprocess.py first to extract features.")
    sample = np.load(sample_feat_files[0])
    T, F = sample.shape
    model = build_bilstm(input_shape=(T,F), n_classes=len(LABELS))
    model.load_weights(args.ckpt)
    print("Loaded model weights.")

    if args.wav:
        label, conf = predict_from_file(model, args.wav, args)
        print(f"Predicted: {label} (confidence {conf:.3f})")
    else:
        tmp = "record_temp.wav"
        record_and_save(tmp, duration=int(args.duration), sr=args.sr)
        label, conf = predict_from_file(model, tmp, args)
        print(f"Predicted: {label} (confidence {conf:.3f})")
        os.remove(tmp)
