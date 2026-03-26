"""I/O utilities for data loading, saving, and model artifact management."""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import joblib
import yaml

from utils.logger import logger


# ── Base paths ──
ML_ROOT = Path(__file__).parent.parent
DATA_DIR = ML_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
FEATURES_DIR = DATA_DIR / "features"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = ML_ROOT / "models"


def ensure_dirs() -> None:
    """Create all data directories if they don't exist."""
    for d in [RAW_DIR, CLEAN_DIR, FEATURES_DIR, SYNTHETIC_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    logger.info("loading_csv", path=str(path))
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("saved_csv", path=str(path), rows=len(df))


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame."""
    logger.info("loading_parquet", path=str(path))
    return pd.read_parquet(path)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to Parquet format (compact, fast, typed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("saved_parquet", path=str(path), rows=len(df))


def load_yaml(path: Path) -> dict:
    """Load a YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_model(model: Any, model_name: str, version: int, metrics: dict,
               hyperparams: dict, dataset_path: Path | None = None) -> Path:
    """
    Save a trained model artifact with full metadata for reproducibility.

    Creates:
      models/{model_name}/v{version}/
        ├── model.pkl          # serialised model
        └── metadata.json      # training context
    """
    model_dir = MODELS_DIR / model_name / f"v{version}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)

    # Compute dataset hash for reproducibility tracking
    dataset_hash = None
    if dataset_path and dataset_path.exists():
        dataset_hash = hashlib.md5(dataset_path.read_bytes()).hexdigest()[:12]

    # Save metadata
    metadata = {
        "model_name": model_name,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "hyperparams": hyperparams,
        "dataset_hash": dataset_hash,
        "model_file": "model.pkl",
    }
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("saved_model",
                model=model_name,
                version=version,
                metrics=metrics,
                path=str(model_dir))
    return model_dir


def load_model(model_name: str, version: int | None = None) -> tuple[Any, dict]:
    """
    Load a trained model and its metadata.
    If version is None, loads the latest version.
    """
    model_base = MODELS_DIR / model_name

    if version is None:
        # Find latest version
        versions = sorted(
            [d for d in model_base.iterdir() if d.is_dir() and d.name.startswith("v")],
            key=lambda d: int(d.name[1:])
        )
        if not versions:
            raise FileNotFoundError(f"No trained models found for {model_name}")
        model_dir = versions[-1]
    else:
        model_dir = model_base / f"v{version}"

    model = joblib.load(model_dir / "model.pkl")
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    logger.info("loaded_model",
                model=model_name,
                version=metadata["version"],
                metrics=metadata["metrics"])
    return model, metadata
