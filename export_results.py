"""
export_results.py
-----------------
Collects the results of all DVC experiments (`dvc exp show --csv`) and
exports a clean CSV containing the tuned hyperparameter values and the
corresponding performance metrics, ranked from best to worst accuracy.

Usage:  python export_results.py
Output: experiment_results.csv
"""
import csv
import io
import subprocess

# Hyperparameters varied during the optimization (params.yaml dotted paths)
PARAM_COLS = [
    "model_training.learning_rate",
    "model_training.hidden_units",
    "model_training.batch_size",
    "model_training.epochs",
    "model_training.optimizer",
    "model_training.dropout_rate",
    "feature_selection.n_components",
]
METRIC_COLS = ["accuracy", "macro_f1"]
OUT_FILE = "experiment_results.csv"


def main():
    raw = subprocess.run(
        ["dvc", "exp", "show", "--csv"],
        capture_output=True, text=True, check=True,
    ).stdout

    reader = csv.DictReader(io.StringIO(raw))
    rows, baseline_taken = [], False
    for r in reader:
        name = (r.get("Experiment") or "").strip()
        typ = (r.get("typ") or "").strip()
        try:
            metrics = {m: round(float(r[m]), 4) for m in METRIC_COLS}
        except (TypeError, ValueError, KeyError):
            continue  # row without valid metrics (e.g. uncommitted baseline)
        if typ == "baseline" and not baseline_taken:
            label, baseline_taken = "baseline", True
        elif name.startswith("exp"):
            label = name
        else:
            continue
        row = {"Experiment": label}
        for c in PARAM_COLS:
            row[c] = r.get(c, "")
        row.update(metrics)
        rows.append(row)

    rows.sort(key=lambda x: x["accuracy"], reverse=True)

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Experiment"] + PARAM_COLS + METRIC_COLS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} rows -> {OUT_FILE}\n")
    print(f"{'Experiment':<28}{'accuracy':>10}{'macro_f1':>10}")
    print("-" * 48)
    for r in rows:
        print(f"{r['Experiment']:<28}{r['accuracy']:>10}{r['macro_f1']:>10}")
    best = rows[0]
    print(f"\nBEST EXPERIMENT: {best['Experiment']}  (accuracy={best['accuracy']}, macro_f1={best['macro_f1']})")
    print(f"Apply it with:   dvc exp apply {best['Experiment'].replace(' (baseline)','')}")


if __name__ == "__main__":
    main()
