"""
Stage 1 - Data Collection
-------------------------
Reads the raw Fashion-MNIST CSV, optionally shuffles and samples a subset
(for faster experimentation), and writes a clean 'collected' dataset that
the rest of the pipeline consumes.

Input : data/raw/fashion-mnist_train.csv   (label + 784 pixel columns)
Output: data/collected/dataset.csv
"""
import pandas as pd

from utils import load_params, ensure_dir, log


def main():
    params = load_params()
    base = params["base"]
    cfg = params["data_collection"]

    raw_path = base["raw_data"]
    log("data_collection", f"Reading raw data from {raw_path}")

    df = pd.read_csv(raw_path)
    log("data_collection", f"Raw dataset shape: {df.shape}")

    if cfg["shuffle"]:
        df = df.sample(frac=1.0, random_state=base["random_state"]).reset_index(drop=True)
        log("data_collection", "Shuffled dataset")

    sample_size = cfg["sample_size"]
    if sample_size and sample_size < len(df):
        df = df.iloc[:sample_size].reset_index(drop=True)
        log("data_collection", f"Sampled {sample_size} rows")

    out_path = cfg["collected_path"]
    ensure_dir(out_path)
    df.to_csv(out_path, index=False)
    log("data_collection", f"Saved collected dataset -> {out_path}  (shape {df.shape})")


if __name__ == "__main__":
    main()
