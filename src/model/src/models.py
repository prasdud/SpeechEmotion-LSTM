# models.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_bilstm(input_shape, n_classes, units=128, dropout=0.3, bidirectional=True):
    inp = layers.Input(shape=input_shape, name="mfcc_input")  # (T, F)
    x = inp
    # Mask zeros (we padded with zeros)
    x = layers.Masking(mask_value=0.0)(x)
    if bidirectional:
        x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(units//2))(x)
    else:
        x = layers.LSTM(units, return_sequences=True)(x)
        x = layers.LSTM(units//2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax', dtype='float32')(x)
    model = models.Model(inp, out)
    return model
