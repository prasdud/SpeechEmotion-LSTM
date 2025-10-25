import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

EMOTIONS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]

def evaluate_model(model_path, ckpt_path=None):
    X_test, y_test = np.load("data/X_test.npy"), np.load("data/y_test.npy")

    model = load_model(model_path, compile=False)
    if ckpt_path and tf.io.gfile.exists(ckpt_path):
        model.load_weights(ckpt_path)

    preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"✅ Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")

    unique_labels = sorted(np.unique(y_test))
    emotion_subset = [EMOTIONS[i] for i in unique_labels]
    report = classification_report(y_test, y_pred, target_names=emotion_subset, output_dict=True)

    pd.DataFrame(report).transpose().to_csv("models/evaluation_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/final_hybrid_model.keras")
    parser.add_argument("--ckpt", type=str, default="models/best_hybrid.weights.h5")
    args = parser.parse_args()
    evaluate_model(args.model, args.ckpt)
