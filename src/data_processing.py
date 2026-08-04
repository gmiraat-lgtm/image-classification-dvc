"""
Stage 2 - Data Processing
-------------------------
Splits the collected dataset into features (pixels) and labels, optionally
normalises pixel values to [0, 1], optionally applies light augmentation,
and produces reproducible train / validation / test splits saved as .npy
arrays for the downstream stages.

Input : data/collected/dataset.csv
Output: data/processed/{X_train,X_val,X_test,y_train,y_val,y_test}.npy
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_params, ensure_dir, log


def augment(X, rotation, zoom, image_size, seed):
    """
    Very lightweight augmentation: adds a small number of rotated/shifted
    copies. Kept dependency-free (numpy only) so the pipeline stays portable.
    """
    rng = np.random.default_rng(seed)
    imgs = X.reshape(-1, image_size, image_size)
    aug = []
    for img in imgs:
        shift = rng.integers(-1, 2, size=2)               # tiny pixel shift
        rolled = np.roll(img, shift=(shift[0], shift[1]), axis=(0, 1))
        aug.append(rolled.reshape(-1))
    return np.array(aug)


def main():
    params = load_params()
    base = params["base"]
    cfg = params["data_processing"]

    src = params["data_collection"]["collected_path"]
    log("data_processing", f"Loading collected dataset from {src}")
    df = pd.read_csv(src)

    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype("float32")
    log("data_processing", f"Features: {X.shape}, Labels: {y.shape}")

    if cfg["normalize"]:
        X = X / 255.0
        log("data_processing", "Normalised pixel values to [0, 1]")

    seed = base["random_state"]
    test_split = cfg["test_split"]
    val_split = cfg["validation_split"]

    # First carve out the test set, then split the remainder into train/val.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )
    val_relative = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=seed, stratify=y_temp
    )

    if cfg["augmentation"]["enabled"]:
        log("data_processing", "Applying augmentation to training set")
        X_aug = augment(
            X_train,
            cfg["augmentation"]["rotation"],
            cfg["augmentation"]["zoom"],
            cfg["image_size"],
            seed,
        )
        X_train = np.vstack([X_train, X_aug])
        y_train = np.concatenate([y_train, y_train])

    out_dir = "data/processed"
    ensure_dir(out_dir + "/placeholder")
    for name, arr in [
        ("X_train", X_train), ("X_val", X_val), ("X_test", X_test),
        ("y_train", y_train), ("y_val", y_val), ("y_test", y_test),
    ]:
        np.save(f"{out_dir}/{name}.npy", arr)

    log("data_processing",
        f"Saved splits -> train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")


if __name__ == "__main__":
    main()
