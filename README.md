# Modular Image Classification with a DVC Pipeline

An end-to-end, modular image-classification project built on the
**Fashion-MNIST** dataset (10 clothing classes, 28x28 grayscale images).
Every stage is a separate Python module, all parameters live in
`params.yaml`, and the whole workflow is automated with **DVC**
(`dvc repro`).

## Pipeline stages

| Stage | Module | Purpose |
|-------|--------|---------|
| 1. Data Collection | `src/data_collection.py` | Read raw CSV, shuffle, sample a subset |
| 2. Data Processing | `src/data_processing.py` | Normalise pixels, split into train/val/test |
| 3. Feature Selection | `src/feature_selection.py` | PCA / variance-threshold dimensionality reduction |
| 4. Model Training | `src/model_training.py` | Train an MLP classifier (sklearn or keras) |
| 5. Model Evaluation | `src/model_evaluation.py` | Accuracy, macro-F1, confusion matrix |

## Project structure

```
image-classification-dvc/
├── data/
│   ├── raw/            <-- place fashion-mnist_train.csv here
│   ├── collected/      (generated) sampled dataset
│   ├── processed/      (generated) train/val/test .npy arrays
│   └── features/       (generated) reduced features + selector
├── models/             (generated) trained model + history
├── reports/            (generated) metrics.json + confusion_matrix.png
├── src/                5 stage modules + utils.py
├── params.yaml         all 18 tunable parameters
├── dvc.yaml            pipeline definition (5 stages)
├── requirements.txt
└── README.md
```

## Setup

1. **Get the dataset** from Kaggle:
   https://www.kaggle.com/datasets/zalando-research/fashionmnist
   Download `fashion-mnist_train.csv` and place it in `data/raw/`.

2. **Create a virtual environment and install dependencies:**
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1        # Windows PowerShell
   pip install -r requirements.txt
   ```

3. **Initialise git and DVC:**
   ```
   git init
   dvc init
   git add .
   git commit -m "Initial project scaffold"
   ```

## Run the pipeline

```
dvc repro
```

DVC runs the five stages in order. Run it again and unchanged stages are
skipped — that is the reproducibility guarantee. Outputs land in
`data/`, `models/` and `reports/`.

## Tuning

Edit any value in `params.yaml` (for example `feature_selection.n_components`,
`model_training.learning_rate`, or `model_training.epochs`) and re-run
`dvc repro`. DVC detects the parameter change and re-executes only the
affected stages.

Switch the model backend with `model_training.framework`:
`sklearn` (default, no GPU) or `keras` (requires `tensorflow`).

## Results

After a run, see:
- `reports/metrics.json` — accuracy, macro-F1, per-class F1
- `reports/confusion_matrix.png` — confusion matrix figure
- `dvc metrics show` — quick metrics summary from the command line
