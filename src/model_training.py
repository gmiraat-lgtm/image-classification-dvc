"""
Stage 4 - Model Training
------------------------
Trains a neural-network classifier on the selected features. Two frameworks
are supported (chosen in params.yaml -> model_training.framework):

  * "sklearn" (default) - MLPClassifier. Runs anywhere, no GPU, no heavy
                          dependencies. Uses learning_rate, batch_size,
                          epochs (max_iter), optimizer (solver),
                          hidden_units and early_stopping.
  * "keras"             - a Dense network with Dropout, using the same
                          parameters natively (learning_rate, optimizer,
                          dropout_rate, batch_size, epochs).

Also writes reports/training_curve.csv (per-iteration loss) for DVC plots.

Input : data/features/{X_train,X_val}.npy, data/processed/{y_train,y_val}.npy
Output: models/model.joblib   (+ models/history.json, reports/training_curve.csv)
"""
import csv
import json

import joblib
import numpy as np

from utils import load_params, ensure_dir, log


def save_training_curve(losses):
    """Write per-iteration loss to a CSV that DVC can render as a plot."""
    ensure_dir("reports/training_curve.csv")
    with open("reports/training_curve.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "loss"])
        for i, loss in enumerate(losses):
            writer.writerow([i, float(loss)])


def train_sklearn(X_train, y_train, X_val, y_val, cfg, seed):
    from sklearn.neural_network import MLPClassifier

    solver = "adam" if cfg["optimizer"].lower() == "adam" else "sgd"
    model = MLPClassifier(
        hidden_layer_sizes=(cfg["hidden_units"], cfg["hidden_units"] // 2),
        solver=solver,
        learning_rate_init=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        max_iter=cfg["epochs"],
        alpha=cfg["dropout_rate"],            # L2 regularisation ~ dropout analogue
        early_stopping=cfg["early_stopping"],
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=seed,
        verbose=False,
    )
    model.fit(X_train, y_train)

    # Per-iteration loss curve for DVC plots
    save_training_curve(model.loss_curve_)

    history = {
        "framework": "sklearn",
        "train_accuracy": float(model.score(X_train, y_train)),
        "val_accuracy": float(model.score(X_val, y_val)),
        "n_iterations": int(model.n_iter_),
    }
    return model, history


def train_keras(X_train, y_train, X_val, y_val, cfg, seed, n_classes):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    tf.random.set_seed(seed)
    opt = (optimizers.Adam(cfg["learning_rate"]) if cfg["optimizer"].lower() == "adam"
           else optimizers.SGD(cfg["learning_rate"]))

    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(cfg["hidden_units"], activation="relu"),
        layers.Dropout(cfg["dropout_rate"]),
        layers.Dense(cfg["hidden_units"] // 2, activation="relu"),
        layers.Dropout(cfg["dropout_rate"]),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = []
    if cfg["early_stopping"]:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True))
    hist = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        callbacks=callbacks, verbose=0,
    )

    # Per-epoch loss curve for DVC plots
    save_training_curve(hist.history["loss"])

    history = {
        "framework": "keras",
        "train_accuracy": float(hist.history["accuracy"][-1]),
        "val_accuracy": float(hist.history["val_accuracy"][-1]),
        "epochs_run": len(hist.history["accuracy"]),
    }
    return model, history


def main():
    params = load_params()
    base = params["base"]
    cfg = params["model_training"]
    seed = base["random_state"]

    X_train = np.load("data/features/X_train.npy")
    X_val = np.load("data/features/X_val.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_val = np.load("data/processed/y_val.npy")
    log("model_training", f"Training features: {X_train.shape}, framework={cfg['framework']}")

    if cfg["framework"].lower() == "keras":
        model, history = train_keras(X_train, y_train, X_val, y_val, cfg, seed, base["n_classes"])
        model_path = "models/model.keras"
        ensure_dir(model_path)
        model.save(model_path)
    else:
        model, history = train_sklearn(X_train, y_train, X_val, y_val, cfg, seed)
        model_path = cfg["model_path"]
        ensure_dir(model_path)
        joblib.dump(model, model_path)

    ensure_dir("models/history.json")
    with open("models/history.json", "w") as f:
        json.dump(history, f, indent=2)

    log("model_training",
        f"Saved model -> {model_path} | "
        f"train_acc={history['train_accuracy']:.4f}, val_acc={history['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()