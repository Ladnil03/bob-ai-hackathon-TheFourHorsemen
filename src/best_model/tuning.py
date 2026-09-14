"""Optuna tuning of the boosting models under fold-consistent CV.

All fitted feature transforms are computed once per fold (see
``cv.build_fold_matrices``), so each Optuna trial only re-fits the model.
"""

from __future__ import annotations

import logging
import numpy as np

from .config import (
    N_FOLDS_TUNE,
    N_TRIALS_PER_MODEL,
    FEATURE_ABLATION_DELTA,
    HAS_ES,
    NEEDS_IMPUTE,
)
from .models import FACTORIES, suggest_params
from .cv import FoldFeaturePipeline, make_folds, build_fold_matrices, _fit_predict_one, ImputeScale
from sklearn.metrics import average_precision_score

log = logging.getLogger("best_model.tuning")


def tune_model(name: str, X: np.ndarray, y: np.ndarray, times: np.ndarray,
               enabled: set[str], n_trials: int = N_TRIALS_PER_MODEL,
               class_ratio: float = 14.0, seed: int = 42) -> tuple[dict, float]:
    """Run an Optuna study for model ``name``; returns (best_params, best_pr_auc)."""
    import optuna

    folds = make_folds(len(y), N_FOLDS_TUNE, 1, seed)
    pipe = FoldFeaturePipeline(X, times, enabled, seed=seed)
    fold_sets = build_fold_matrices(pipe, folds, y)
    es_supported = HAS_ES.get(name, True)
    needs_impute = name in NEEDS_IMPUTE

    def objective(trial) -> float:
        params = suggest_params(name, trial, class_ratio)
        scores = []
        for fs in fold_sets:
            Xtr, Xte = fs["X_tr"], fs["X_te"]
            if needs_impute:
                is_ = ImputeScale().fit(Xtr)
                Xtr, Xte = is_.transform(Xtr), is_.transform(Xte)
            model = FACTORIES[name](params, seed=seed)
            p, _ = _fit_predict_one(model, Xtr, fs["y_tr"], Xte, es_supported=es_supported)
            scores.append(average_precision_score(y[fs["te"]], p))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    score = study.best_value
    log.info("Tuned %s -> OOF PR-AUC %.4f (trial %d)", name, score, study.best_trial.number)
    return best, float(score)


def feature_group_ablation(X: np.ndarray, y: np.ndarray, times: np.ndarray,
                           groups: list[frozenset], class_ratio: float,
                           seed: int = 42) -> set[str]:
    """Forward ablation with default LightGBM.

    A candidate group is kept only when OOF PR-AUC improves by at least
    ``FEATURE_ABLATION_DELTA`` over the current best configuration.
    """
    from .models import new_lgbm

    folds = make_folds(len(y), N_FOLDS_TUNE, 1, seed)
    chosen: set[str] = set()

    def evaluate(group_set: set[str]) -> float:
        pipe = FoldFeaturePipeline(X, times, group_set, seed=seed)
        fold_sets = build_fold_matrices(pipe, folds, y)
        fixed = {"learning_rate": 0.05, "num_leaves": 31, "scale_pos_weight": class_ratio}
        scores = []
        for fs in fold_sets:
            model = new_lgbm(fixed, seed=seed)
            p, _ = _fit_predict_one(model, fs["X_tr"], fs["y_tr"], fs["X_te"])
            scores.append(average_precision_score(y[fs["te"]], p))
        return float(np.mean(scores))

    base_score = evaluate(chosen)
    log.info("FE ablation | baseline PR-AUC %.4f", base_score)
    for group in groups:
        candidate = chosen | set(group)
        score = evaluate(candidate)
        delta = score - base_score
        keep = delta >= FEATURE_ABLATION_DELTA
        log.info("  candidate %-18s PR-AUC %.4f (delta %+.4f) %s",
                 sorted(group), score, delta, "KEEP" if keep else "skip")
        if keep:
            chosen = candidate
            base_score = score
    return chosen