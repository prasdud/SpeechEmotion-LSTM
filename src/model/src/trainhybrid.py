"""
Speech Emotion Recognition - CNN + BiLSTM Hybrid (train_v3_hybrid.py)

Enhancements over v2:
  - CNN layers for spatial MFCC feature extraction
  - BiLSTM for temporal dependencies
  - Data augmentation (noise, pitch, time stretch)
  - L2 + Dropout regularization
  - ReduceLROnPlateau + EarlyStopping
  - Mixed precision for RTX GPU
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import librosa
import random
from pathlib import Path
import joblib
import pandas as pd

# ====================================
# 🎛 Audio Augmentation Functions
# ====================================
def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def stretch_audio(data, rate=1.0):
    return librosa.effects.time_stretch(data, rate)

def pitch_shift(data, sr, n_steps=2):
    return librosa.effects.pitch_shift(data, sr, n_steps=n_steps)

def augment_audio(path, sr=22050):
    """Applies random augmentations with 50% probability each."""
    y, _ = librosa.load(path, sr=sr)
    if random.random() < 0.5:
        y = add_noise(y)
    if random.random() < 0.5:
        y = stretch_audio(y, rate=random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        y = pitch_shift(y, sr, n_steps=random.choice([-2, -1, 1, 2]))
    return y


# ====================================
# 🧩 Dataset Loader
# ====================================
def load_dataset(manifest, featdir):
    df = pd.read_csv(manifest)
    X, y = [], []
    for _, row in df.iterrows():
        feat_path = Path(featdir) / f"{Path(row['path']).stem}.npy"
        if feat_path.exists():
            X.append(np.load(feat_path))
            y.append(row['label'])
    X = np.array(X)
    y = np.array(y)
    return X, y


# ====================================
# 🧠 CNN + BiLSTM Model
# ====================================
def build_cnn_bilstm_model(input_shape, num_classes):
    reg = regularizers.l2(1e-4)
    inp = layers.Input(shape=input_shape, name="mfcc_input")

    # CNN feature extractor
    x = layers.Reshape((input_shape[0], input_shape[1], 1))(inp)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Flatten CNN output for LSTM input
    x = layers.Reshape((x.shape[1], -1))(x)

    # BiLSTM temporal modeling
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, kernel_regularizer=reg)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, dropout=0.3, kernel_regularizer=reg)
    )(x)

    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ====================================
# 🏋️ Training Loop
# ====================================
def main(args):
    # Load split data
    X_train, y_train = np.load('data/X_train.npy'), np.load('data/y_train.npy')
    X_val, y_val = np.load('data/X_val.npy'), np.load('data/y_val.npy')
    X_test, y_test = np.load('data/X_test.npy'), np.load('data/y_test.npy')

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    model = build_cnn_bilstm_model(input_shape, num_classes)
    model.summary()

    # Callbacks
    cb = [
        callbacks.ModelCheckpoint(
            args.ckpt,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
        )
    ]

    # Mixed precision for NVIDIA GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled.")

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        verbose=1
    )

    # Evaluate final test accuracy
    test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"✅ Final Test Accuracy: {test_acc:.4f}")

    # Save model + history
    os.makedirs("models", exist_ok=True)
    model.save("models/final_hybrid_model.keras")
    joblib.dump(history.history, "models/hybrid_training_history.pkl")
    print("✅ Model + training history saved.")


# ====================================
# 🚀 CLI Entry
# ====================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default="models/best_hybrid.weights.h5")
    args = parser.parse_args()
    main(args)
