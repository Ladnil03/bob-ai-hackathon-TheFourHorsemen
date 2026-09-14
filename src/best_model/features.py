"""Feature engineering for SECOM.

Feature groups (each toggleable, evaluated via a forward ablation before use):

  A  missingness        - per-row NaN count / fraction
  B  wafer signature    - row-wise distribution statistics across sensors
  C  time features      - clock + chronological signals (hour, dow, gap, age)
     time_history       - fold-aware expanding-window past failure rate
  D  unsupervised       - PCA components, KMeans clusters/distances,
                          IsolationForest anomaly score (fit inside folds)
  E  robust z-scores    - MAD-based z-scores on the top-MI raw columns
  F  similarity         - nearest-neighbour distance in PCA space

Unsupervised transforms (D, F) and the mutual-information selector are fitted
on training folds only; everything else is a deterministic per-row transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from scipy.stats import skew, kurtosis


# ---------------------------------------------------------------------------
# Deterministic per-row transforms (no fitted parameters)
# ---------------------------------------------------------------------------

def missingness_features(X: np.ndarray) -> np.ndarray:
    nan_counts = np.isnan(X).sum(axis=1)
    return np.column_stack([nan_counts, nan_counts / max(X.shape[1], 1)])


def rowstats_features(X: np.ndarray) -> np.ndarray:
    n_out = 14  # mean, median, std, min, max, range, q05, q25, q50, q75, q95, mad, skew, kurtosis
    f = np.full((X.shape[0], n_out), np.nan, dtype=np.float32)
    with np.errstate(all="ignore"):
        quantiles = np.nanquantile(X, [0.05, 0.25, 0.50, 0.75, 0.95], axis=1)
        f[:, 0] = np.nanmean(X, axis=1)
        f[:, 1] = np.nanmedian(X, axis=1)
        f[:, 2] = np.nanstd(X, axis=1)
        f[:, 3] = np.nanmin(X, axis=1)
        f[:, 4] = np.nanmax(X, axis=1)
        f[:, 5] = np.nanmax(X, axis=1) - np.nanmin(X, axis=1)
        f[:, 6:11] = quantiles.T
        f[:, 11] = np.nanmean(abs(X - np.nanmedian(X, axis=1)[:, None]), axis=1)
        # Skewness and kurtosis per row (robust for tiny datasets)
        for i in range(X.shape[0]):
            row = X[i, ~np.isnan(X[i, :])]
            if len(row) > 2:
                f[i, 12] = skew(row, nan_policy="omit")
                f[i, 13] = kurtosis(row, nan_policy="omit")
    f = np.where(np.isfinite(f), f, 0.0)
    return f.astype(np.float32)


def time_features(times: np.ndarray) -> np.ndarray:
    dt = pd.to_datetime(times)
    t0 = dt.min()
    seconds = (dt - t0).total_seconds().to_numpy(dtype=np.float64)
    gaps = np.zeros_like(seconds)
    gaps[1:] = np.diff(seconds)
    hours = dt.hour.to_numpy()
    dow = dt.dayofweek.to_numpy()
    f = np.column_stack([
        seconds,
        seconds / max(seconds.max(), 1.0),
        np.sin(2 * np.pi * hours / 24.0),
        np.cos(2 * np.pi * hours / 24.0),
        dow,
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        gaps,
    ]).astype(np.float32)
    return f


def rolling_prior_failures(time_query: np.ndarray, time_hist: np.ndarray,
                           y_hist: np.ndarray, k: int = 30) -> np.ndarray:
    """Fraction of failures among the last k historical samples strictly before each query time.

    Uses only past information (no look-ahead), fitting the same logic used
    inside CV folds for both train and test partitions.
    """
    order = np.argsort(time_hist, kind="mergesort")
    time_hist_s = time_hist[order]
    y_hist_s = y_hist[order].astype(np.float64)
    prefix = np.concatenate([[0.0], np.cumsum(y_hist_s)])
    j = np.searchsorted(time_hist_s, time_query, side="left")

    j_c = np.clip(j, 0, len(time_hist_s))
    jk = np.clip(j - k, 0, len(time_hist_s))
    total = prefix[j_c] - prefix[jk]
    count = np.minimum(j, k).astype(np.float64)
    frac = np.divide(total, count, out=np.zeros_like(total, dtype=np.float64), where=count > 0)

    gap = np.zeros_like(j, dtype=np.float64)
    prev_mask = j > 0
    if prev_mask.any():
        inds = np.clip(j[prev_mask] - 1, 0, len(time_hist_s) - 1)
        query_s = time_query[prev_mask].astype("datetime64[s]").astype(np.float64)
        hist_s = time_hist_s[inds].astype("datetime64[s]").astype(np.float64)
        gap[prev_mask] = query_s - hist_s
    return np.column_stack([frac, gap]).astype(np.float32)


def mad_zscore_features(X: np.ndarray, top_idx: np.ndarray) -> np.ndarray:
    if top_idx.size == 0:
        return np.empty((X.shape[0], 0), dtype=np.float32)
    sub = X[:, top_idx]
    med = np.nanmedian(sub, axis=0)
    with np.errstate(all="ignore"):
        mad = np.nanmedian(np.abs(sub - med), axis=0)
    mad = np.where(mad == 0, 1.0, mad)
    z = (sub - med) / mad
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def interaction_features(X: np.ndarray, top_idx: np.ndarray, n_pairs: int = 8) -> np.ndarray:
    if top_idx.size < 2:
        return np.empty((X.shape[0], 0), dtype=np.float32)
    cols = X[:, top_idx[: min(len(top_idx), n_pairs)]]
    med = np.nanmedian(cols, axis=0)
    cols = np.where(np.isfinite(cols), cols, med)
    out = []
    for i in range(cols.shape[1] - 1):
        for j in range(i + 1, cols.shape[1]):
            out.append(cols[:, i] * cols[:, j])
            with np.errstate(all="ignore"):
                ratio = cols[:, i] / (cols[:, j] + 1e-9)
            out.append(np.where(np.isfinite(ratio), ratio, 0.0))
    return np.column_stack(out).astype(np.float32) if out else np.empty((X.shape[0], 0), dtype=np.float32)


# ---------------------------------------------------------------------------
# Fitted unsupervised transforms (fold-aware; never fit on test data)
# ---------------------------------------------------------------------------

class UnsupervisedComposites(BaseEstimator, TransformerMixin):
    """PCA + KMeans + IsolationForest + nearest-neighbour features."""

    def __init__(self, n_components: int = 20, n_clusters: int = 6,
                 contamination: float = 0.1, n_neighbors: int = 10, seed: int = 42) -> None:
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.seed = seed
        self._pca = None
        self._kmeans = None
        self._iso = None
        self._nn = None
        self._median = None

    def _fill(self, X: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(X), self._median, X)
        return filled.astype(np.float32)

    def fit(self, X: np.ndarray, y=None):
        self._median = np.nanmedian(X, axis=0)
        filled = self._fill(X)
        n_comp = min(self.n_components, X.shape[1], X.shape[0] - 1)
        n_comp = max(n_comp, 1)
        self._pca = PCA(n_components=n_comp,
                        random_state=self.seed).fit(filled)
        comps = self._pca.transform(filled)
        n_clust = min(self.n_clusters, X.shape[0])
        self._kmeans = MiniBatchKMeans(n_clusters=n_clust, random_state=self.seed,
                                       batch_size=min(512, X.shape[0]), n_init="auto").fit(comps)
        self._iso = IsolationForest(contamination=self.contamination, random_state=self.seed,
                                    n_estimators=200).fit(filled)
        n_nn = min(self.n_neighbors, X.shape[0] - 1)
        n_nn = max(n_nn, 1)
        self._nn = NearestNeighbors(n_neighbors=n_nn).fit(comps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        filled = self._fill(X)
        comps = self._pca.transform(filled)
        dist = self._kmeans.transform(comps)
        cdist = np.min(dist, axis=1, keepdims=True)
        iso = self._iso.score_samples(filled).reshape(-1, 1)
        knn_dist, _ = self._nn.kneighbors(comps)
        knn = np.mean(knn_dist, axis=1, keepdims=True)
        return np.column_stack([comps, dist, cdist, iso, knn]).astype(np.float32)


# ---------------------------------------------------------------------------
# Mutual-information-based selector (fitted on the train fold)
# ---------------------------------------------------------------------------

class MIRanker:
    """Top-K features by mutual information with the binary target."""

    def __init__(self, top_k: int) -> None:
        self.top_k = top_k
        self.selected_idx: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MIRanker":
        with np.errstate(all="ignore"):
            median = np.nanmedian(X, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        Xf = np.where(np.isnan(X), median, X)
        Xf = np.clip(Xf, -1e6, 1e6)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = mutual_info_classif(Xf, y, random_state=0, n_neighbors=min(6, max(1, len(y) - 1)))
        idx = np.argsort(scores)[::-1]
        self.selected_idx = idx[: min(self.top_k, len(idx))]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_idx]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


# ---------------------------------------------------------------------------
# Full feature-column builder (applies all enabled groups deterministically)
# ---------------------------------------------------------------------------

_GROUP_NAMES = ["A_missing", "B_rowstats", "C_time", "E_zscores", "E_interactions"]


def build_derived(X: np.ndarray, times: np.ndarray,
                  top_mi_idx: np.ndarray, enabled: set[str]) -> np.ndarray:
    """Stack the deterministic feature groups into one float32 matrix.

    ``top_mi_idx`` are raw-column indices used by groups E (computed by the
    caller on the training fold for leakage safety).
    """
    cols = []
    if "A_missing" in enabled:
        cols.append(missingness_features(X))
    if "B_rowstats" in enabled:
        cols.append(rowstats_features(X))
    if "C_time" in enabled:
        cols.append(time_features(times))
    if "E_zscores" in enabled:
        cols.append(mad_zscore_features(X, top_mi_idx))
    if "E_interactions" in enabled:
        cols.append(interaction_features(X, top_mi_idx))
    if not cols:
        return np.empty((X.shape[0], 0), dtype=np.float32)
    return np.column_stack(cols).astype(np.float32)


def derived_group_slices(enabled: set[str]) -> list[tuple[str, int, int]]:
    """Map each enabled deterministic group to its column slice in build_derived output."""
    slices = []
    start = 0
    spec = {
        "A_missing": 2,
        "B_rowstats": 14,  # updated: now includes skew + kurtosis
        "C_time": 8,
        "E_zscores": None,
        "E_interactions": None,
    }
    for name in _GROUP_NAMES:
        if name not in enabled:
            continue
        slices.append((name, start, start + (spec[name] or 0)))
        start += spec[name] or 0
    return slices