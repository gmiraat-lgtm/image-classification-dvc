"""
Stage 3 - Feature Selection
---------------------------
Reduces the 784-pixel feature space to a smaller, more informative set.
Two selectable methods (via params.yaml):

  * "variance" - drops near-constant pixels (e.g. always-black borders)
                 using a VarianceThreshold.
  * "pca"      - projects pixels onto the top-N principal components.

The fitted selector is learnt on the TRAINING set only and then applied to
validation and test sets, so there is no information leakage.

Input : data/processed/{X_train,X_val,X_test}.npy
Output: data/features/{X_train,X_val,X_test}.npy
        data/features/selector.joblib
        data/features/feature_report.json
"""
import json

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from utils import load_params, ensure_dir, log


def main():
    params = load_params()
    base = params["base"]
    cfg = params["feature_selection"]

    proc = "data/processed"
    X_train = np.load(f"{proc}/X_train.npy")
    X_val = np.load(f"{proc}/X_val.npy")
    X_test = np.load(f"{proc}/X_test.npy")
    log("feature_selection", f"Input feature dimension: {X_train.shape[1]}")

    method = cfg["method"].lower()
    if method == "pca":
        selector = PCA(n_components=cfg["n_components"], random_state=base["random_state"])
        log("feature_selection", f"Method=PCA, components={cfg['n_components']}")
    elif method == "variance":
        selector = VarianceThreshold(threshold=cfg["variance_threshold"])
        log("feature_selection", f"Method=VarianceThreshold, threshold={cfg['variance_threshold']}")
    else:
        raise ValueError(f"Unknown feature_selection.method: {method}")

    X_train_fs = selector.fit_transform(X_train)
    X_val_fs = selector.transform(X_val)
    X_test_fs = selector.transform(X_test)

    out_dir = "data/features"
    ensure_dir(out_dir + "/placeholder")
    np.save(f"{out_dir}/X_train.npy", X_train_fs)
    np.save(f"{out_dir}/X_val.npy", X_val_fs)
    np.save(f"{out_dir}/X_test.npy", X_test_fs)
    joblib.dump(selector, f"{out_dir}/selector.joblib")

    report = {
        "method": method,
        "input_features": int(X_train.shape[1]),
        "output_features": int(X_train_fs.shape[1]),
    }
    if method == "pca":
        report["explained_variance_ratio_sum"] = float(np.sum(selector.explained_variance_ratio_))

    with open(f"{out_dir}/feature_report.json", "w") as f:
        json.dump(report, f, indent=2)

    log("feature_selection",
        f"Reduced {report['input_features']} -> {report['output_features']} features")


if __name__ == "__main__":
    main()
