"""Data loading and basic cleaning for the SECOM dataset.

The tree-based models consume the raw feature matrix (NaN left intact —
XGBoost, LightGBM and CatBoost all support missing values natively).
Imputation + scaling is applied only for TabPFN and the stacking meta-model,
and always fitted inside the training fold to avoid leakage.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .config import SECOM_DIR, MAX_NAN_FRAC


class SecomData:
    """Container for raw SECOM features, target and chronologically-sorted times."""

    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: list[str], times: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.times = times

    @property
    def shape(self) -> tuple:
        return self.X.shape


def load_secom() -> SecomData:
    """Load secom.data + secom_labels.data, sort rows chronologically.

    Returns
    -------
    SecomData
        X (float32, n x p), y (int, 0/1), feature_names, times (datetime64).
    """
    features = pd.read_csv(
        SECOM_DIR / "secom.data",
        sep=r"\s+",
        header=None,
        na_values=["NaN", "nan"],
        dtype=np.float32,
    )
    labels = pd.read_csv(
        SECOM_DIR / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["label", "timestamp"],
        dtype={"label": np.int8},
    )
    times = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")
    y = (labels["label"] == 1).astype(np.int8).to_numpy()

    order = np.argsort(times.to_numpy(), stable=True)
    features = features.iloc[order].reset_index(drop=True)
    times = times.iloc[order].reset_index(drop=True)
    y = y[order]
    times_ordered = times.to_numpy()

    feature_names = [f"s{i}" for i in range(features.shape[1])]
    return SecomData(
        X=features.to_numpy(dtype=np.float32),
        y=y,
        feature_names=feature_names,
        times=times_ordered,
    )


def basic_mask(data: SecomData, max_nan_frac: float = MAX_NAN_FRAC) -> np.ndarray:
    """Mask of columns kept: not >max_nan_frac missing and not (near-)constant."""
    X = data.X
    n = X.shape[0]
    nan_frac = np.isnan(X).sum(axis=0) / n
    keep_nan = nan_frac <= max_nan_frac

    with np.errstate(all="ignore"):
        std = np.nanstd(X, axis=0)
        mean = np.nanmean(X, axis=0)
        finite_mean_std = np.isfinite(std) & np.isfinite(mean)
        non_constant = std > 1e-8
    mask = keep_nan & finite_mean_std & non_constant
    return mask


def prepare_matrix(data: SecomData) -> tuple[np.ndarray, list[str]]:
    """Return the cleaned (m) x (p) float array and its column names after basic_mask."""
    mask = basic_mask(data)
    names = [data.feature_names[i] for i in range(len(mask)) if mask[i]]
    return data.X[:, mask].astype(np.float32), names