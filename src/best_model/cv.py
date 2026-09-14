"""Cross-validation utilities: fold-aware feature pipeline + OOF evaluation.

The main trick for the SECOM dataset: tree models receive raw NaN values,
so the only fitted transforms are (a) the unsupervised composites, (b) the
mutual-information selector, and (c) imputer/scaler for TabPFN and the
stacking meta-learner.  All are fitted on the training fold only.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from .config import (
    SEED,
    N_FOLDS_TUNE,
    N_FOLDS_FINAL,
    N_REPEATS_FINAL,
    TOP_K_FEATURES,
    EARLY_STOPPING_ROUNDS,
    VALIDATION_FRACTION,
)
from .features import (
    UnsupervisedComposites,
    MIRanker,
    build_derived,
    rolling_prior_failures,
)


def make_folds(y: np.ndarray, n_splits: int, n_repeats: int,
               seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified fold indices.

    Parameters
    ----------
    y : 1-d array of labels (used for stratification).
    n_splits, n_repeats : CV geometry.
    seed : random state.
    """
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                  random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]


class FoldFeaturePipeline:
    """Applies derived features, fold-aware history, MI selection and composites."""

    def __init__(self, raw: np.ndarray, times: np.ndarray, enabled: set[str],
                 top_k: int = TOP_K_FEATURES, n_unsup: int = 12, seed: int = SEED) -> None:
        self.raw = raw.astype(np.float32)
        self.times = times
        self.enabled = enabled
        self.top_k = top_k
        self.seed = seed
        self.n_unsup = n_unsup

    def _mi_idx_train(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        import warnings

        selector = MIRanker(self.top_k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selector.fit(X_train, y_train)
        return selector.selected_idx

    def fit_transform_fold(self, tr: np.ndarray, te: np.ndarray, y: np.ndarray,
                           with_history: bool = True) -> tuple[np.ndarray, np.ndarray]:
        X_tr, X_te = self.raw[tr], self.raw[te]
        mi_idx = self._mi_idx_train(X_tr, y[tr])

        pieces_tr = [X_tr[:, mi_idx]]
        pieces_te = [X_te[:, mi_idx]]

        derived_enabled = {g for g in ("A_missing", "B_rowstats", "C_time", "E_zscores", "E_interactions") if g in self.enabled}
        d_tr = build_derived(X_tr, self.times[tr], mi_idx, derived_enabled)
        d_te = build_derived(X_te, self.times[te], mi_idx, derived_enabled)
        if d_tr.shape[1] > 0:
            pieces_tr.append(d_tr)
            pieces_te.append(d_te)

        if "C_time" in self.enabled:
            pieces_tr.append(rolling_prior_failures(self.times[tr], self.times[tr], y[tr]) if with_history else np.empty((len(tr), 0), np.float32))
            pieces_te.append(rolling_prior_failures(self.times[te], self.times[tr], y[tr]) if with_history else np.empty((len(te), 0), np.float32))

        if "D_unsupervised" in self.enabled:
            unsup = UnsupervisedComposites(n_components=self.n_unsup, seed=self.seed).fit(X_tr)
            pieces_tr.append(unsup.transform(X_tr))
            pieces_te.append(unsup.transform(X_te))

        X_tr = np.hstack(pieces_tr).astype(np.float32)
        X_te = np.hstack(pieces_te).astype(np.float32)
        return X_tr, X_te


class ImputeScale:
    """Median imputation + standardisation, fitted on the training fold."""

    def __init__(self, seed: int = SEED) -> None:
        self._imp = SimpleImputer(strategy="median", add_indicator=False)
        self._scaler = StandardScaler()

    def fit(self, X: np.ndarray) -> "ImputeScale":
        self._imp.fit(X)
        self._scaler.fit(self._imp.transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._scaler.transform(self._imp.transform(X)).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def _fit_predict_one(model, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
                     use_early_stop: bool = True, es_supported: bool = True) -> tuple[np.ndarray, object]:
    """Fit ``model`` with inner validation for early stopping and return (prob, model)."""
    if not use_early_stop or not es_supported:
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        return p, model

    n_val = max(int(len(y_tr) * VALIDATION_FRACTION), 1)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(y_tr))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xv, yv = X_tr[tr_idx], y_tr[tr_idx]
    Xe, ye = X_tr[val_idx], y_tr[val_idx]
    eval_set = [(Xe, ye)]

    is_lgbm = type(model).__module__.startswith("lightgbm")
    if is_lgbm:
        import lightgbm as lgb

        model.fit(Xv, yv, eval_set=eval_set,
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    else:
        model.fit(Xv, yv, eval_set=eval_set, verbose=False)
    p = model.predict_proba(X_te)[:, 1]
    return p, model


def compute_metrics(y: np.ndarray, prob: np.ndarray, threshold: float | None = None):
    """Full metric dictionary. Threshold defaults to argmax F1 on the given sample."""
    if threshold is None:
        prec, rec, thr = precision_recall_curve(y, prob)
        f1s = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
        threshold = float(thr[np.argmax(f1s[:-1])]) if thr.size else 0.5
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    ber = 0.5 * (tp / max(1, fn + tp) + tn / max(1, tn + fp))
    return {
        "threshold": round(float(threshold), 6),
        "accuracy": round(float(accuracy_score(y, pred)), 6),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 6),
        "auc_roc": round(float(roc_auc_score(y, prob)), 6),
        "pr_auc": round(float(average_precision_score(y, prob)), 6),
        "ber": round(float(ber), 6),
        "confusion_matrix": cm,
    }


def tune_cv_metric(model_factory, X: np.ndarray, y: np.ndarray, pair: tuple[np.ndarray, np.ndarray]):
    """Fit on a single fold pair, early-stop, and return the OOF PR-AUC."""
    tr, te = pair
    model = model_factory()
    p, _ = _fit_predict_one(model, X[tr], y[tr], X[te])
    return float(average_precision_score(y[te], p))


def oof_proba(model_factory, X: np.ndarray, y: np.ndarray,
              folds: list[tuple[np.ndarray, np.ndarray]], use_early_stop: bool = True):
    """Compute out-of-fold probabilities across folds; returns proba + fitted-on-full list."""
    oof = np.zeros(len(y), dtype=np.float64)
    per_fold = np.full(len(y), np.nan)
    for fi, (tr, te) in enumerate(folds):
        p, _ = _fit_predict_one(model_factory(), X[tr], y[tr], X[te], use_early_stop=use_early_stop)
        oof[te] = p
        per_fold[te] = fi
    return oof


def build_fold_matrices(pipe: FoldFeaturePipeline, folds: list,
                        y: np.ndarray) -> list[dict]:
    """Precompute the transformed train/test matrices for every fold.

    The expensive fitted transforms (MI ranking, PCA/KMeans/IsolationForest,
    history features) are computed exactly once per fold and reused across all
    Optuna trials and the final evaluation.
    """
    out = []
    for tr, te in folds:
        X_tr, X_te = pipe.fit_transform_fold(tr, te, y)
        out.append({
            "tr": tr, "te": te,
            "X_tr": X_tr.astype(np.float32), "X_te": X_te.astype(np.float32),
            "y_tr": y[tr].copy(),
        })
    return out