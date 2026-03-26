"""Evaluation metrics and reporting utilities for NexusIQ ML models."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_absolute_percentage_error,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from utils.logger import logger


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           y_prob: np.ndarray | None = None) -> dict:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob: np.ndarray | None = None) -> dict:
    """Compute binary classification metrics with false positive rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = {
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }
    if y_prob is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "median_ae": float(np.median(np.abs(y_true - y_pred))),
    }


def calibration_error(y_true: np.ndarray, y_pred_quantiles: dict[str, np.ndarray]) -> dict:
    """
    Compute calibration error for quantile predictions (ETA model).
    Checks if actual fraction within each quantile band matches the nominal coverage.
    """
    errors = {}
    for quantile_name, threshold in y_pred_quantiles.items():
        nominal = float(quantile_name.replace("p", "")) / 100.0
        actual_coverage = float(np.mean(y_true <= threshold))
        errors[quantile_name] = {
            "nominal": nominal,
            "actual": actual_coverage,
            "error": abs(nominal - actual_coverage),
        }
    avg_error = float(np.mean([v["error"] for v in errors.values()]))
    return {"per_quantile": errors, "mean_calibration_error": avg_error}


def save_eval_report(metrics: dict, model_name: str, version: int,
                     output_dir: Path | None = None) -> Path:
    """Save evaluation report as JSON."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "models" / model_name / f"v{version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("saved_eval_report", path=str(report_path))
    return report_path


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: list[str], save_path: Path) -> None:
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved_confusion_matrix", path=str(save_path))


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray,
                          save_path: Path) -> None:
    """Generate and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved_pr_curve", path=str(save_path))
