# preprocess.py
import argparse
from pathlib import Path
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

# RAVDESS emotion mapping (filename field 3)
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def parse_filename(fname: str):
    # example filename: 03-01-01-01-01-01-01.wav
    parts = Path(fname).stem.split("-")
    # parts[2] is emotion code in standard RAVDESS
    code = parts[2]
    emotion = EMOTION_MAP.get(code, "unknown")
    actor = parts[-1]  # last part is actor id (e.g., '01')
    return emotion, actor.zfill(2)

def build_manifest(root: Path, out_csv: Path):
    rows = []
    for wav in root.rglob("*.wav"):
        emotion, actor = parse_filename(wav.name)
        rows.append({"path": str(wav), "emotion": emotion, "actor": actor})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Manifest saved to {out_csv}, {len(df)} files found.")
    return df

def extract_mfcc(path, sr=16000, n_mfcc=40, n_fft=1024, hop_length=512,
                 duration=3.0):
    y, orig_sr = librosa.load(path, sr=sr)
    # trim silence
    y, _ = librosa.effects.trim(y, top_db=25)
    # ensure fixed duration (pad or truncate)
    max_len = int(sr * duration)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats = np.vstack([mfcc, delta, delta2])  # shape (n_feats, T)
    return feats.astype(np.float32)  # (F, T)

def main(args):
    root = Path(args.datadir)
    manifest_path = Path(args.outdir) / "manifest.csv"
    df = build_manifest(root, manifest_path)
    feat_dir = Path(args.outdir) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCCs"):
        wav_path = row["path"]
        feats = extract_mfcc(wav_path,
                             sr=args.sr,
                             n_mfcc=args.n_mfcc,
                             n_fft=args.n_fft,
                             hop_length=args.hop_length,
                             duration=args.duration)
        # save as (T, F) for easier tf usage
        feats_t = feats.T  # shape (T, F)
        out_file = feat_dir / (Path(wav_path).stem + ".npy")
        np.save(out_file, feats_t)

    print("Feature extraction complete. Features saved under:", feat_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", type=str, default=r"C:\Speech\ravdess\Audio_Speech_Actors_01-24")
    p.add_argument("--outdir", type=str, default="data")
    p.add_argument("--sr", type=int, default=16000)
    p.add_argument("--n_mfcc", type=int, default=40)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=512)
    p.add_argument("--duration", type=float, default=3.0)
    args = p.parse_args()
    main(args)
