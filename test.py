"""
test.py - Automated tests for the Image Classification DVC project.
Author: Miraat Gupta | PRN 23070126073 | AIML - A3

Executed by the GitHub Actions CI pipeline (.github/workflows/ci.yaml)
on every push to the main/master branch, and runnable locally with:

    pytest test.py -v

The suite tests the application at four levels, and every functional test
uses a small synthetic dataset generated in-memory, because the raw
Fashion-MNIST CSV is DVC-tracked (not in git) and hence is not present in
the CI runner's checkout:

  1. Structure     - all pipeline files and stage scripts exist
  2. Compilation   - every src script compiles (catches syntax errors on push)
  3. Configuration - params.yaml and dvc.yaml are valid and consistent
  4. Functionality - the project's own utils and train_sklearn() work, and
                     the processing / PCA / evaluation logic is correct
"""

import importlib.util
import json
import os
import py_compile
import sys

import joblib
import numpy as np
import pytest
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)  # stage scripts import each other as "from utils import ..."

PIPELINE_STAGES = [
    "data_collection",
    "data_processing",
    "feature_selection",
    "model_training",
    "model_evaluation",
]

REQUIRED_PARAM_SECTIONS = ["base"] + PIPELINE_STAGES


def _load_yaml(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return yaml.safe_load(f)


def _make_synthetic_dataset(n_samples=400, n_features=64, n_classes=4, seed=42):
    """Deterministic dataset shaped like flattened grayscale images
    (pixel values 0-255) so it exercises the same code paths as the real
    Fashion-MNIST data, without needing the 126 MB DVC-tracked CSV in CI."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n_samples)
    centers = rng.uniform(60, 200, size=(n_classes, n_features))
    X = centers[y] + rng.normal(0, 12, size=(n_samples, n_features))
    return np.clip(X, 0, 255), y


# ===========================================================================
# 1. PROJECT STRUCTURE
# ===========================================================================

class TestProjectStructure:

    def test_params_yaml_exists(self):
        assert os.path.isfile(os.path.join(PROJECT_ROOT, "params.yaml"))

    def test_dvc_yaml_exists(self):
        assert os.path.isfile(os.path.join(PROJECT_ROOT, "dvc.yaml"))

    def test_all_stage_scripts_exist(self):
        for stage in PIPELINE_STAGES:
            assert os.path.isfile(os.path.join(SRC_DIR, f"{stage}.py")), \
                f"Missing stage script: src/{stage}.py"

    def test_utils_module_exists(self):
        assert os.path.isfile(os.path.join(SRC_DIR, "utils.py"))


# ===========================================================================
# 2. COMPILATION - every source file must at least be valid Python
# ===========================================================================

class TestCompilation:

    @pytest.mark.parametrize("script", PIPELINE_STAGES + ["utils"])
    def test_script_compiles(self, script):
        py_compile.compile(os.path.join(SRC_DIR, f"{script}.py"), doraise=True)


# ===========================================================================
# 3. CONFIGURATION
# ===========================================================================

class TestConfiguration:

    def test_params_yaml_has_all_sections(self):
        params = _load_yaml("params.yaml")
        for section in REQUIRED_PARAM_SECTIONS:
            assert section in params, f"params.yaml missing section: {section}"

    def test_hyperparameters_are_sane(self):
        params = _load_yaml("params.yaml")
        train = params["model_training"]
        assert 0 < train["learning_rate"] < 1
        assert train["batch_size"] > 0
        assert train["epochs"] > 0
        assert train["hidden_units"] > 0
        assert params["feature_selection"]["n_components"] > 0
        dp = params["data_processing"]
        assert 0 < dp["validation_split"] < 0.5
        assert 0 < dp["test_split"] < 0.5

    def test_dvc_yaml_defines_all_five_stages(self):
        dvc = _load_yaml("dvc.yaml")
        assert "stages" in dvc
        for stage in PIPELINE_STAGES:
            assert stage in dvc["stages"], f"dvc.yaml missing stage: {stage}"

    def test_every_stage_has_cmd_deps_and_existing_script(self):
        dvc = _load_yaml("dvc.yaml")
        for stage, spec in dvc["stages"].items():
            assert "cmd" in spec and "deps" in spec, f"Stage '{stage}' incomplete"
            script = spec["cmd"].split()[-1]  # "python src/<stage>.py"
            assert os.path.isfile(os.path.join(PROJECT_ROOT, script)), \
                f"Stage '{stage}' cmd points to missing script {script}"


# ===========================================================================
# 4. FUNCTIONALITY - the project's own code, on synthetic data
# ===========================================================================

class TestUtils:

    def test_load_params_returns_full_config(self):
        import utils
        params = utils.load_params(os.path.join(PROJECT_ROOT, "params.yaml"))
        assert isinstance(params, dict) and "model_training" in params

    def test_ensure_dir_creates_nested_directories(self, tmp_path):
        import utils
        target = tmp_path / "a" / "b" / "file.json"
        utils.ensure_dir(str(target))
        assert (tmp_path / "a" / "b").is_dir()


class TestDataProcessingLogic:
    """Mirrors src/data_processing.py: normalize, then stratified
    test split followed by relative validation split."""

    def test_normalization_bounds_and_no_nan(self):
        X, _ = _make_synthetic_dataset()
        X_norm = X.astype("float32") / 255.0
        assert X_norm.min() >= 0.0 and X_norm.max() <= 1.0
        assert not np.isnan(X_norm).any()

    def test_split_proportions_match_params(self):
        X, y = _make_synthetic_dataset()
        cfg = _load_yaml("params.yaml")["data_processing"]
        test, val = cfg["test_split"], cfg["validation_split"]
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=test, random_state=42, stratify=y)
        val_rel = val / (1.0 - test)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=val_rel, random_state=42, stratify=y_tmp)
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert abs(len(X_test) / len(X) - test) < 0.03
        assert abs(len(X_val) / len(X) - val) < 0.03


class TestFeatureSelectionLogic:
    """Mirrors src/feature_selection.py: PCA reduction and its report."""

    def test_pca_reduces_dimensionality(self):
        X, _ = _make_synthetic_dataset(n_features=64)
        X_red = PCA(n_components=10, random_state=42).fit_transform(X / 255.0)
        assert X_red.shape == (X.shape[0], 10)

    def test_pca_explained_variance_is_descending(self):
        X, _ = _make_synthetic_dataset(n_features=64)
        pca = PCA(n_components=10, random_state=42).fit(X / 255.0)
        r = pca.explained_variance_ratio_
        assert all(r[i] >= r[i + 1] for i in range(len(r) - 1))


class TestModelTraining:
    """Calls the project's real train_sklearn() from src/model_training.py."""

    def _train(self):
        from model_training import train_sklearn
        X, y = _make_synthetic_dataset()
        X = X / 255.0
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
        cfg = dict(_load_yaml("params.yaml")["model_training"])
        cfg.update(hidden_units=32, epochs=120, early_stopping=False)  # fast CI config
        return train_sklearn(X_tr, y_tr, X_val, y_val, cfg, seed=42)

    def test_train_sklearn_returns_model_and_history(self):
        model, history = self._train()
        assert hasattr(model, "predict")
        assert {"framework", "train_accuracy", "val_accuracy",
                "n_iterations"} <= set(history)

    def test_trained_model_beats_random_chance(self):
        _, history = self._train()
        assert history["val_accuracy"] > 0.5, \
            f"val_accuracy={history['val_accuracy']:.3f} (chance = 0.25 for 4 classes)"


class TestModelEvaluationLogic:
    """Mirrors src/model_evaluation.py: metrics computation + JSON report."""

    def test_metrics_json_structure_and_ranges(self, tmp_path):
        from model_training import train_sklearn
        X, y = _make_synthetic_dataset()
        X = X / 255.0
        X_tr, X_test, y_tr, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
        cfg = dict(_load_yaml("params.yaml")["model_training"])
        cfg.update(hidden_units=32, epochs=120, early_stopping=False)
        model, _ = train_sklearn(X_tr, y_tr, X_test, y_test, cfg, seed=42)
        y_pred = model.predict(X_test)
        metrics = {"accuracy": float(accuracy_score(y_test, y_pred)),
                   "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                   "n_test_samples": int(len(y_test))}
        report = tmp_path / "metrics.json"
        report.write_text(json.dumps(metrics, indent=2))
        loaded = json.loads(report.read_text())
        assert 0.0 <= loaded["accuracy"] <= 1.0
        assert 0.0 <= loaded["macro_f1"] <= 1.0
        assert loaded["n_test_samples"] == len(y_test)


class TestModelPersistence:
    """Mirrors the joblib save/load used by training and evaluation stages."""

    def test_joblib_roundtrip_predictions_identical(self, tmp_path):
        from model_training import train_sklearn
        X, y = _make_synthetic_dataset(n_samples=200)
        X = X / 255.0
        cfg = dict(_load_yaml("params.yaml")["model_training"])
        cfg.update(hidden_units=16, epochs=80, early_stopping=False)
        model, _ = train_sklearn(X, y, X, y, cfg, seed=42)
        path = tmp_path / "model.joblib"
        joblib.dump(model, path)
        restored = joblib.load(path)
        assert np.array_equal(model.predict(X[:30]), restored.predict(X[:30]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
