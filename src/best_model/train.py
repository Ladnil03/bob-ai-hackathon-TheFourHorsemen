"""End-to-end best-model pipeline: tune, evaluate OOF, blend/stack, persist."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import joblib
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .config import (
    ARTIFACT_DIR,
    SEED,
    N_FOLDS_FINAL,
    N_REPEATS_FINAL,
    N_TRIALS_PER_MODEL,
    N_TRIALS_EXTRA_MODEL,
    TOP_K_FEATURES,
    BOOSTERS,
    EXTRA_MODELS,
    HAS_ES,
    NEEDS_IMPUTE,
    USE_TABPFN,
)
from .data import load_secom, prepare_matrix
from .features import build_derived, rolling_prior_failures, UnsupervisedComposites, MIRanker
from .cv import (
    FoldFeaturePipeline,
    ImputeScale,
    make_folds,
    build_fold_matrices,
    _fit_predict_one,
    compute_metrics,
)
from .models import FACTORIES, new_tabpfn, HAS_TABPFN
from .tuning import tune_model, feature_group_ablation
from .report import save_json, print_metrics_table, shap_top_features, save_curves

log = logging.getLogger("best_model")

FEATURE_GROUPS = [
    frozenset({"A_missing"}),
    frozenset({"B_rowstats"}),
    frozenset({"C_time"}),
    frozenset({"D_unsupervised"}),
    frozenset({"E_zscores"}),
    frozenset({"E_interactions"}),
]


def _softmax_free(wfree: np.ndarray, m: int) -> np.ndarray:
    w = np.concatenate([wfree, np.array([1.0 - wfree.sum()])])
    w = np.clip(w, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(m) / m


class BestEnsemble:
    """Pickle-able artifact: base models + either blend weights or a logistic stacker."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.models: dict[str, object] = {}
        self.weights: dict[str, float] = {}
        self.stacker: object | None = None
        self.enabled: set[str] = set()
        self.top_k = TOP_K_FEATURES
        self.seed = SEED
        self.mi_idx: np.ndarray | None = None
        self.unsup: UnsupervisedComposites | None = None
        self.train_times: np.ndarray | None = None
        self.train_y_hist: np.ndarray | None = None
        self.imputer_scale: dict[str, ImputeScale] = {}
        self.threshold: float = 0.5
        self.feature_names: list[str] = []

    def _derived(self, X: np.ndarray, times: np.ndarray, with_history: bool) -> np.ndarray:
        der = build_derived(
            X, times, self.mi_idx,
            {g for g in ("A_missing", "B_rowstats", "C_time", "E_zscores", "E_interactions") if g in self.enabled},
        )
        pieces = [X[:, self.mi_idx]]
        if der.shape[1] > 0:
            pieces.append(der)
        if "C_time" in self.enabled and with_history:
            pieces.append(rolling_prior_failures(times, self.train_times, self.train_y_hist))
        if self.unsup is not None:
            pieces.append(self.unsup.transform(X))
        return np.hstack(pieces) if len(pieces) > 1 else pieces[0]

    def fit_final(self, X: np.ndarray, y: np.ndarray, times: np.ndarray,
                  oof_models: dict[str, np.ndarray], z_oof: np.ndarray) -> "BestEnsemble":
        self.train_times = times.copy()
        self.train_y_hist = y.copy()
        self.mi_idx = MIRanker(self.top_k).fit(X, y).selected_idx
        if "D_unsupervised" in self.enabled:
            self.unsup = UnsupervisedComposites(seed=self.seed).fit(X)
        F = self._derived(X, times, with_history=True)

        for name in oof_models:
            if name == "tabpfn":
                self.imputer_scale["tabpfn"] = ImputeScale().fit(F)
                tp = new_tabpfn(self.seed, n_ens=2)
                tp.fit(self.imputer_scale["tabpfn"].transform(F), y, overwrite_warning=True)
                self.models[name] = tp
            elif name in NEEDS_IMPUTE:
                self.imputer_scale[name] = ImputeScale().fit(F)
                model = FACTORIES[name]({}, seed=self.seed)
                model.fit(self.imputer_scale[name].transform(F), y)
                self.models[name] = model
            else:
                model = FACTORIES[name]({}, seed=self.seed)
                _, model = _fit_predict_one(model, F, y, F)
                self.models[name] = model

        if self.mode == "blend":
            keys = list(oof_models)
            if len(keys) == 1:
                w = np.array([1.0])
            else:
                res = differential_evolution(
                    lambda w: -_blend_pr_auc(w, y, oof_models),
                    [(0.0, 1.0)] * (len(keys) - 1), seed=self.seed, maxiter=60,
                    polish=True, tol=1e-6,
                )
                w = _softmax_free(res.x, len(keys))
            self.weights = {keys[i]: float(w[i]) for i in range(len(keys))}
            log.info("blend weights %s", self.weights)

        if self.mode == "stack":
            self.stacker = LogisticRegression(C=1.0, max_iter=2000).fit(z_oof, y)

        self.threshold = float(compute_metrics(y, _ensemble_proba(self, F, z_oof, oof_models), None)["threshold"])
        return self

    def predict_proba(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        F = self._derived(X, times, with_history=True)
        probe_map = {}
        for name, model in self.models.items():
            if name == "tabpfn":
                probe_map[name] = model.predict_proba(self.imputer_scale["tabpfn"].transform(F))[:, 1]
            elif name in NEEDS_IMPUTE:
                probe_map[name] = model.predict_proba(self.imputer_scale[name].transform(F))[:, 1]
            else:
                probe_map[name] = model.predict_proba(F)[:, 1]
        if self.mode == "blend":
            P = np.column_stack([probe_map[k] for k in self.weights if k in probe_map])
            w = np.array([self.weights[k] for k in self.weights if k in probe_map])
            return P @ w
        names = [k for k in self.models]
        Z = np.column_stack([np.log(np.clip(probe_map[k], 1e-9, 1 - 1e-9)) for k in names])
        Z = self.imputer_scale["stack"].transform(Z)
        return self.stacker.predict_proba(Z)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        type(self).__module__ = "best_model.train"
        joblib.dump(self, path, compress=3)
        log.info("saved artifact %s", path)


def _blend_pr_auc(wfree: np.ndarray, y: np.ndarray, proba_map: dict[str, np.ndarray]) -> float:
    w = _softmax_free(wfree, len(wfree) + 1)
    P = np.column_stack([proba_map[k] for k in proba_map])
    return average_precision_score(y, P @ w)


def _ensemble_proba(ens: BestEnsemble, F: np.ndarray, z_oof: np.ndarray,
                    oof_models: dict[str, np.ndarray]) -> np.ndarray:
    if ens.mode == "stack":
        return ens.stacker.predict_proba(z_oof)[:, 1]
    w = np.array([ens.weights[k] for k in oof_models if k in ens.weights])
    names = [k for k in oof_models if k in ens.weights]
    P = np.column_stack([oof_models[k] for k in names])
    return P @ w


def _fold_level_pipeline(X: np.ndarray, y: np.ndarray, times: np.ndarray, enabled: set[str],
                         folds, params: dict[str, dict], seed: int = SEED) -> dict[str, np.ndarray]:
    pipe = FoldFeaturePipeline(X, times, enabled, seed=seed)
    fold_sets = build_fold_matrices(pipe, folds, y)
    oof: dict[str, np.ndarray] = {}
    for name in params:
        es = HAS_ES.get(name, True)
        impute = name in NEEDS_IMPUTE
        out = np.zeros(len(y), dtype=np.float64)
        for fs in fold_sets:
            Xtr, Xte = fs["X_tr"], fs["X_te"]
            if impute:
                is_ = ImputeScale(seed).fit(Xtr)
                Xtr, Xte = is_.transform(Xtr), is_.transform(Xte)
            model = FACTORIES[name](params[name], seed=seed)
            p, _ = _fit_predict_one(model, Xtr, fs["y_tr"], Xte, es_supported=es)
            out[fs["te"]] = p
        oof[name] = out
        log.info("OOF %-5s PR-AUC %.4f  AUC %.4f", name,
                 average_precision_score(y, out),
                 compute_metrics(y, out, None)["auc_roc"])

    if HAS_TABPFN:
        try:
            out = np.zeros(len(y), dtype=np.float64)
            for fs in fold_sets:
                is_ = ImputeScale(seed).fit(fs["X_tr"])
                model = new_tabpfn(seed, n_ens=2)
                model.fit(is_.transform(fs["X_tr"]), fs["y_tr"], overwrite_warning=True)
                out[fs["te"]] = model.predict_proba(is_.transform(fs["X_te"]))[:, 1]
            oof["tabpfn"] = out
            log.info("OOF tabpfn PR-AUC %.4f  AUC %.4f",
                     average_precision_score(y, out), compute_metrics(y, out, None)["auc_roc"])
        except Exception as exc:
            log.warning("TabPFN OOF failed, dropping it: %s", exc)
    return oof


def run(smoke: bool = False, artifact_dir: Path | None = None,
        n_trials: int | None = None, fe_all: bool = False) -> dict:
    """Execute the complete pipeline and persist artifacts under ``artifact_dir``."""
    import warnings

    warnings.filterwarnings("ignore")
    t_start = time.time()
    artifact_dir = Path(artifact_dir or ARTIFACT_DIR)
    n_trials = n_trials or N_TRIALS_PER_MODEL

    data = load_secom()
    if smoke:
        rng = np.random.RandomState(SEED)
        pos_idx = np.where(data.y == 1)[0]
        keep = np.sort(np.concatenate([
            rng.choice(pos_idx, min(10, len(pos_idx)), replace=False),
            rng.choice(np.where(data.y == 0)[0], 110, replace=False),
        ]))
        data.X, data.y = data.X[keep], data.y[keep]
        data.times = data.times[keep]

    X_raw, names = prepare_matrix(data)
    y, times = data.y, data.times
    n_pos = int(y.sum())
    class_ratio = float((len(y) - n_pos) / max(n_pos, 1))
    log.info("data %s, failures=%d (%.2f%%)", X_raw.shape, n_pos, 100 * n_pos / len(y))

    if fe_all:
        enabled = {g for group in FEATURE_GROUPS for g in group}
        log.info("all FE groups enabled: %s", sorted(enabled))
    else:
        enabled = feature_group_ablation(X_raw, y, times, FEATURE_GROUPS, class_ratio)
        log.info("chosen FE groups: %s", sorted(enabled))

    model_names = list(BOOSTERS) + list(EXTRA_MODELS)
    best_params: dict[str, dict] = {}
    for name in model_names:
        trials = n_trials if (name in BOOSTERS or smoke) else max(int(n_trials * 0.5), N_TRIALS_EXTRA_MODEL)
        best_params[name], _ = tune_model(name, X_raw, y, times, enabled, trials, class_ratio)

    folds_final = make_folds(len(y), N_FOLDS_FINAL, N_REPEATS_FINAL)
    oof = _fold_level_pipeline(X_raw, y, times, enabled, folds_final, best_params)
    rows = [{"model": n, **compute_metrics(y, p, None)} for n, p in oof.items()]
    print_metrics_table(rows, "OOF METRICS (5x2 repeated stratified CV)")

    z_oof = np.column_stack([np.log(np.clip(p, 1e-9, 1 - 1e-9)) for p in oof.values()])
    stack_oof = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), z_oof, y, cv=5,
                                  method="predict_proba")[:, 1]
    stack_metrics = compute_metrics(y, stack_oof, None)
    stack_metrics = {"model": "stack_all", **stack_metrics}

    blend, blend_metrics = None, None
    if len(oof) > 1:
        keys = list(oof)
        res = differential_evolution(
            lambda w: -_blend_pr_auc(w, y, oof), [(0.0, 1.0)] * (len(keys) - 1),
            seed=SEED, maxiter=60, polish=True, tol=1e-6,
        )
        w = _softmax_free(res.x, len(keys))
        blend = np.column_stack([oof[k] for k in keys]) @ w
        blend_metrics = {"model": "blend", **compute_metrics(y, blend, None),
                         "weights": {k: round(float(v), 4) for k, v in zip(keys, w)}}
        print_metrics_table(rows + [blend_metrics, stack_metrics], "FINAL OOF METRICS")

    blend_pr = float(blend_metrics["pr_auc"]) if blend_metrics else -1.0
    mode = "blend" if blend_pr >= float(stack_metrics["pr_auc"]) else "stack"
    log.info("ensemble mode chosen: %s", mode)

    ens = BestEnsemble(mode)
    ens.enabled = enabled
    ens.feature_names = names
    if mode == "stack":
        ens.imputer_scale["stack"] = ImputeScale().fit(z_oof)
    ens.fit_final(X_raw, y, times, oof, z_oof)

    chosen_oof = blend if mode == "blend" else stack_oof
    final_metrics = compute_metrics(y, chosen_oof, None)

    F_full = ens._derived(X_raw, times, True)
    F_explain = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(F_full))
    shap_list = []
    base_candidates = [m for m in ens.models if m != "tabpfn"]
    if base_candidates:
        shap_list = shap_top_features(ens.models[base_candidates[0]], F_explain,
                                      [f"f{i}" for i in range(F_explain.shape[1])])

    metrics = {
        "seed": SEED,
        "data_shape": [int(v) for v in X_raw.shape],
        "n_failures": int(n_pos),
        "failure_rate": round(float(100 * n_pos / len(y)), 2),
        "fe_groups_chosen": sorted(enabled),
        "best_params": {k: {kk: (float(vv) if isinstance(vv, (int, float)) else vv) for kk, vv in v.items()} for k, v in best_params.items()},
        "oof_models": rows,
        "ensemble_mode": mode,
        "blend": blend_metrics,
        "stack": {"pr_auc": float(stack_metrics["pr_auc"])} if mode == "stack" else None,
        "final_metrics": final_metrics,
        "threshold": float(ens.threshold),
        "shap_top": shap_list,
        "runtime_sec": round(time.time() - t_start, 1),
    }
    save_json(metrics, artifact_dir / "metrics.json")
    ens.save(artifact_dir / "best_model.pkl")
    save_curves(y, chosen_oof, artifact_dir / "curves.png")

    print("\n" + "=" * 60)
    print(f"FINAL ENSEMBLE OOF METRICS  (mode = {mode})")
    print("=" * 60)
    for k, v in final_metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:>10}: {v}")
    print(f"  runtime   : {metrics['runtime_sec']}s")
    print(f"Artifacts  : {artifact_dir}")
    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="fast smoke run on 120 samples")
    ap.add_argument("--fe-all", action="store_true", help="enable every feature group (skip ablation)")
    ap.add_argument("--artifacts", default=str(ARTIFACT_DIR))
    ap.add_argument("--trials", type=int, default=N_TRIALS_PER_MODEL)
    args = ap.parse_args()
    run(smoke=args.smoke, artifact_dir=Path(args.artifacts), n_trials=args.trials, fe_all=args.fe_all)