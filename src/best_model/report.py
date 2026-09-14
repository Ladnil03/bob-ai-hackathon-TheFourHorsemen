"""Reporting: console tables, JSON metrics, SHAP summary and plots."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("best_model.report")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    log.info("wrote %s", path)


def print_metrics_table(rows: list[dict], title: str = "MODEL COMPARISON") -> None:
    headers = ["model", "auc_roc", "pr_auc", "f1", "precision", "recall", "accuracy", "ber"]
    print("\n" + "=" * 92)
    print(f"{title}")
    print("=" * 92)
    print(f"{headers[0]:<16}" + "".join(f"{h:>10}" for h in headers[1:]))
    for row in rows:
        line = f"{str(row.get('model', '')):<16}"
        for h in headers[1:]:
            line += f"{float(row.get(h, 0.0)):>10.4f}"
        print(line)
    print("=" * 92)


def shap_top_features(model, X_explain: np.ndarray, feature_names: list[str], top_k: int = 20):
    """Mean-|SHAP| ranking using TreeExplainer when available."""
    try:
        import shap
    except Exception:
        return []
    if not hasattr(model, "predict"):
        return []
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_explain)
        if isinstance(values, list):
            values = values[1] if len(values) == 2 else values[-1]
        if values.ndim == 3:
            values = values[:, :, 1]
        mean_abs = np.abs(values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_k]
        return [
            {"feature": feature_names[i], "mean_abs_shap": round(float(mean_abs[i]), 6)}
            for i in order
        ]
    except Exception as exc:  # pragma: no cover
        log.warning("SHAP explanation failed: %s", exc)
        return []


def save_curves(y: np.ndarray, prob: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    prec, rec, _ = precision_recall_curve(y, prob)
    axes[0].plot(rec, prec); axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall")
    fpr, tpr, _ = roc_curve(y, prob)
    axes[1].plot(fpr, tpr); axes[1].plot([0, 1], [0, 1], "--", alpha=0.4)
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log.info("wrote %s", path)