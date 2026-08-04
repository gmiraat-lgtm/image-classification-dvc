"""
Stage 5 - Model Evaluation
--------------------------
Loads the trained model and the held-out test set, computes accuracy and a
full per-class classification report, and saves both a metrics JSON and a
confusion-matrix image.

Input : models/model.joblib (or model.keras), data/features/X_test.npy,
        data/processed/y_test.npy
Output: reports/metrics.json, reports/confusion_matrix.png
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless backend for pipeline runs
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from utils import load_params, ensure_dir, log


def load_model(cfg):
    if cfg["framework"].lower() == "keras":
        from tensorflow.keras import models
        return models.load_model("models/model.keras"), "keras"
    import joblib
    return joblib.load(cfg["model_path"]), "sklearn"


def main():
    params = load_params()
    cfg = params["model_training"]
    ev = params["model_evaluation"]

    model, framework = load_model(cfg)
    X_test = np.load("data/features/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    log("model_evaluation", f"Evaluating on test set: {X_test.shape}")

    if framework == "keras":
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "framework": framework,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "n_test_samples": int(len(y_test)),
        "per_class_f1": {str(k): float(v["f1-score"])
                         for k, v in report.items() if k.isdigit()},
    }
    ensure_dir(ev["metrics_path"])
    with open(ev["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix figure
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix (accuracy={acc:.3f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.colorbar(im, ax=ax)
    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    ensure_dir(ev["confusion_matrix_path"])
    fig.savefig(ev["confusion_matrix_path"], dpi=120)
    plt.close(fig)

    log("model_evaluation",
        f"Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f} | "
        f"metrics -> {ev['metrics_path']}, figure -> {ev['confusion_matrix_path']}")


if __name__ == "__main__":
    main()
