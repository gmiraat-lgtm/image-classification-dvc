# =====================================================================
# run_experiments.ps1
# Hyperparameter Optimization for Image Classification - DVC Experiments
# Queues 20 experiments (each derived independently from the committed
# baseline), then shows the queue. Execute them afterwards with:
#     dvc exp run --run-all
#
# Baseline (params.yaml): learning_rate=0.01, hidden_units=128,
# batch_size=64, epochs=30, optimizer=adam, dropout_rate=0.20,
# n_components=100 (PCA)
# =====================================================================

# --- Sweep 1: learning rate (5 experiments) --------------------------
dvc exp run --queue --name exp01-lr-0.0005 -S model_training.learning_rate=0.0005
dvc exp run --queue --name exp02-lr-0.001  -S model_training.learning_rate=0.001
dvc exp run --queue --name exp03-lr-0.005  -S model_training.learning_rate=0.005
dvc exp run --queue --name exp04-lr-0.02   -S model_training.learning_rate=0.02
dvc exp run --queue --name exp05-lr-0.05   -S model_training.learning_rate=0.05

# --- Sweep 2: hidden units / network width (3 experiments) -----------
dvc exp run --queue --name exp06-hu-64  -S model_training.hidden_units=64
dvc exp run --queue --name exp07-hu-256 -S model_training.hidden_units=256
dvc exp run --queue --name exp08-hu-512 -S model_training.hidden_units=512

# --- Sweep 3: PCA components / feature dimensionality (4 experiments) -
dvc exp run --queue --name exp09-pca-50  -S feature_selection.n_components=50
dvc exp run --queue --name exp10-pca-150 -S feature_selection.n_components=150
dvc exp run --queue --name exp11-pca-200 -S feature_selection.n_components=200
dvc exp run --queue --name exp12-pca-300 -S feature_selection.n_components=300

# --- Sweep 4: batch size (3 experiments) -----------------------------
dvc exp run --queue --name exp13-bs-32  -S model_training.batch_size=32
dvc exp run --queue --name exp14-bs-128 -S model_training.batch_size=128
dvc exp run --queue --name exp15-bs-256 -S model_training.batch_size=256

# --- Sweep 5: optimizer (1 experiment) -------------------------------
dvc exp run --queue --name exp16-opt-sgd -S model_training.optimizer=sgd

# --- Sweep 6: regularisation strength (2 experiments) ----------------
# (dropout_rate maps to the MLPClassifier L2 penalty "alpha")
dvc exp run --queue --name exp17-alpha-0.0001 -S model_training.dropout_rate=0.0001
dvc exp run --queue --name exp18-alpha-0.01   -S model_training.dropout_rate=0.01

# --- Sweep 7: training length (1 experiment) -------------------------
dvc exp run --queue --name exp19-epochs-60 -S model_training.epochs=60

# --- Sweep 8: combined best-guess configuration (1 experiment) -------
dvc exp run --queue --name exp20-combo -S model_training.learning_rate=0.005 -S model_training.hidden_units=256 -S feature_selection.n_components=200

# --- Show the queue ---------------------------------------------------
dvc queue status
Write-Host ""
Write-Host "20 experiments queued. Now run them all with:  dvc exp run --run-all"
