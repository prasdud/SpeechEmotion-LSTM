import os
import argparse
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def extract_features(file_path, n_mfcc=40, sr=22050):
    y, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = librosa.util.fix_length(mfcc, size=120, axis=1)
    return mfcc.T  # (time, n_mfcc)

def build_manifest(root, manifest_csv):
    audio_paths, labels = [], []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".wav"):
                full_path = os.path.join(dirpath, f)
                parts = Path(f).stem.split("-")
                if len(parts) > 2:
                    emotion = int(parts[2])  # 01-08
                    labels.append(emotion - 1)
                    audio_paths.append(full_path)
    df = pd.DataFrame({"path": audio_paths, "label": labels})
    df.to_csv(manifest_csv, index=False)
    print(f"Manifest saved to {manifest_csv}, {len(df)} files found.")
    return df

def main(args):
    os.makedirs(args.outdir, exist_ok=True)
    featdir = os.path.join(args.outdir, "features")
    os.makedirs(featdir, exist_ok=True)
    manifest_csv = os.path.join(args.outdir, "manifest.csv")

    df = build_manifest(args.datadir, manifest_csv)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCCs"):
        mfcc = extract_features(row["path"])
        np.save(os.path.join(featdir, f"{Path(row['path']).stem}.npy"), mfcc)

    print(f"Feature extraction complete. Features saved under: {featdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="data")
    args = parser.parse_args()
    main(args)
