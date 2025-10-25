import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

def load_features(df, featdir):
    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading features"):
        feat_path = Path(featdir) / f"{Path(row['path']).stem}.npy"
        if feat_path.exists():
            X.append(np.load(feat_path))
            y.append(row["label"])
    X = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post", dtype="float32")
    y = np.array(y)
    return X, y

def main(args):
    df = pd.read_csv(args.manifest)
    X, y = load_features(df, args.featdir)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, stratify=y_temp, random_state=42)

    os.makedirs("data", exist_ok=True)
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_val.npy", X_val)
    np.save("data/y_val.npy", y_val)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)

    print(f"Train/test/val sizes: {len(X_train)} {len(X_test)} {len(X_val)}")
    print(f"X {X_train.shape} y {y_train.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--featdir", type=str, required=True)
    args = parser.parse_args()
    main(args)
