# evaluate.py
import argparse
import numpy as np
from dataset import load_manifest, train_val_test_split, make_dataset
from models import build_bilstm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

LABELS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]

def main(args):
    df = load_manifest(args.manifest)
    tr, val, te = train_val_test_split(df, test_size=args.test_size, val_size=args.val_size)
    sample_feat = np.load(os.path.join(args.featdir, os.path.splitext(os.path.basename(tr['path'].iloc[0]))[0] + ".npy"))
    T, F = sample_feat.shape
    input_shape = (T, F)
    n_classes = len(LABELS)

    model = build_bilstm(input_shape, n_classes, units=args.units, dropout=args.dropout)
    model.load_weights(args.ckpt)
    print("Loaded weights from", args.ckpt)

    test_ds = make_dataset(te, args.featdir, batch_size=args.batch_size, shuffle=False, max_len=T)
    y_true = []
    y_pred = []
    for Xb, yb in test_ds:
        preds = model.predict(Xb)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(preds.argmax(axis=1).tolist())

    print(classification_report(y_true, y_pred, target_names=LABELS))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    cmn = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(9,7))
    sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=200)
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--featdir", default="data/features")
    p.add_argument("--ckpt", default="models/best_weights.h5")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--units", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    args = p.parse_args()
    main(args)
