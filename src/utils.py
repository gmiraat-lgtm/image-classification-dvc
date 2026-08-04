"""
utils.py
Shared helpers used by every pipeline stage: loading params.yaml,
creating output directories, and consistent logging.
"""
import os
import sys
import yaml


def load_params(path="params.yaml"):
    """Read the full params.yaml into a nested dictionary."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    """Create the parent directory of a file path if it does not exist."""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def log(stage, message):
    """Uniform stage logging so pipeline output is easy to read."""
    print(f"[{stage}] {message}")
    sys.stdout.flush()
