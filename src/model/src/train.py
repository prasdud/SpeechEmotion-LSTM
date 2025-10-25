# train.py
import argparse
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from dataset import load_manifest, train_val_test_split, make_dataset
from models import build_bilstm
import os
import numpy as np

def main(args):
    # mixed precision
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled.")
    except Exception as e:
        print("Mixed precision not enabled:", e)

    df = load_manifest(args.manifest)
    train_df, val_df, test_df = train_val_test_split(df, test_size=args.test_size, val_size=args.val_size)
    print("Split sizes:", len(train_df), len(val_df), len(test_df))

    sample_feat = np.load(str(args.featdir) + "/" + os.path.splitext(os.path.basename(train_df['path'].iloc[0]))[0] + ".npy")
    T = sample_feat.shape[0]
    F = sample_feat.shape[1]
    input_shape = (T, F)
    n_classes = 8

    train_ds = make_dataset(train_df, args.featdir, batch_size=args.batch_size, shuffle=True, max_len=T)
    val_ds = make_dataset(val_df, args.featdir, batch_size=args.batch_size, shuffle=False, max_len=T)

    model = build_bilstm(input_shape=input_shape, n_classes=n_classes, units=args.units, dropout=args.dropout)
    opt = optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    ckpt = callbacks.ModelCheckpoint(args.ckpt, monitor='val_accuracy', save_best_only=True, save_weights_only=True)
    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[ckpt, rlr, es])
    print("Training finished. Best weights saved to", args.ckpt)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.csv")
    p.add_argument("--featdir", default="data/features")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--units", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--ckpt", default="models/best_weights.h5")
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--val_size", type=float, default=0.15)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    main(args)
