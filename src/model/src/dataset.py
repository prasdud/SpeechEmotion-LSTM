# dataset.py
import argparse
import pandas as pd
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

LABELS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
LABEL_TO_IDX = {l:i for i,l in enumerate(LABELS)}

def load_manifest(manifest_csv):
    df = pd.read_csv(manifest_csv)
    # drop unknown emotions if any
    df = df[df['emotion'].isin(LABELS)].reset_index(drop=True)
    return df

def train_val_test_split(df, test_size=0.15, val_size=0.15, random_state=42):
    # speaker-independent split by actor using GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size+val_size, random_state=random_state)
    train_idx, hold_idx = next(gss.split(df, groups=df['actor']))
    hold = df.iloc[hold_idx]
    # split hold into val and test
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(test_size+val_size), random_state=random_state)
    val_idx, test_idx = next(gss2.split(hold, groups=hold['actor']))
    val = hold.iloc[val_idx]
    test = hold.iloc[test_idx]
    train = df.iloc[train_idx]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def npy_loader(path):
    arr = np.load(path.decode('utf-8'))
    return arr.astype(np.float32)

def make_dataset(df, feat_dir, batch_size=32, shuffle=True, max_len=94):
    feat_dir = Path(feat_dir)
    paths = [str(feat_dir / (Path(p).stem + ".npy")) for p in df['path'].values]
    labels = [LABEL_TO_IDX[e] for e in df['emotion'].values]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths))

    def _read(path, label):
        # use numpy loader to load variable-length arrays
        feat = tf.numpy_function(lambda x: np.load(x.decode()), [path], tf.float32)
        feat.set_shape([None, None])  # (T, F)
        # pad / truncate to max_len
        T = tf.shape(feat)[0]
        F = tf.shape(feat)[1]
        feat = feat[:max_len, :]
        pad_len = max_len - tf.shape(feat)[0]
        feat = tf.cond(pad_len>0, lambda: tf.pad(feat, [[0,pad_len],[0,0]]), lambda: feat)
        # return (max_len, F), label
        return feat, label

    ds = ds.map(_read, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--featdir", default="data/features")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    df = load_manifest(args.manifest)
    tr, val, te = train_val_test_split(df)
    print("Train/test/val sizes:", len(tr), len(val), len(te))
    ds = make_dataset(tr, args.featdir, batch_size=args.batch_size)
    for X,y in ds.take(1):
        print("X", X.shape, "y", y.shape)
