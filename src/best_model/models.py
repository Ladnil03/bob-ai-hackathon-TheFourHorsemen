"""Model factories and Optuna hyper-parameter spaces for the boosters."""

from __future__ import annotations

import numpy as np

HAS_TABPFN = False
try:
    from tabpfn import TabPFNClassifier

    HAS_TABPFN = True
except Exception:
    TabPFNClassifier = None


def new_xgb(params: dict, seed: int = 42):
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", 6),
        min_child_weight=params.get("min_child_weight", 1.0),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        gamma=params.get("gamma", 0.0),
        reg_alpha=params.get("reg_alpha", 1e-3),
        reg_lambda=params.get("reg_lambda", 1.0),
        scale_pos_weight=params.get("scale_pos_weight", 10.0),
        max_bin=params.get("max_bin", 256),
        grow_policy=params.get("grow_policy", "depthwise"),
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=60,
        device="cpu",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )


def new_lgbm(params: dict, seed: int = 42):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=3000,
        learning_rate=params.get("learning_rate", 0.05),
        num_leaves=params.get("num_leaves", 31),
        max_depth=params.get("max_depth", -1),
        min_child_samples=params.get("min_child_samples", 20),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 1e-3),
        reg_lambda=params.get("reg_lambda", 1.0),
        min_sum_hessian_in_leaf=params.get("min_sum_hessian_in_leaf", 10.0),
        scale_pos_weight=params.get("scale_pos_weight", 10.0),
        max_bin=params.get("max_bin", 255),
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )


def new_cat(params: dict, seed: int = 42):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=3000,
        learning_rate=params.get("learning_rate", 0.08),
        depth=params.get("depth", 7),
        l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
        min_data_in_leaf=params.get("min_data_in_leaf", 5),
        subsample=params.get("subsample", 0.9),
        colsample_bylevel=params.get("colsample_bylevel", 1.0),
        auto_class_weights="Balanced",
        loss_function="Logloss",
        eval_metric="PRAUC",
        early_stopping_rounds=60,
        allow_writing_files=False,
        thread_count=-1,
        random_seed=seed,
        verbose=False,
    )


def new_histgb(params: dict, seed: int = 42):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=params.get("learning_rate", 0.05),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        min_samples_leaf=params.get("min_samples_leaf", 20),
        l2_regularization=params.get("l2_regularization", 1.0),
        max_bins=255,
        early_stopping=True,
        validation_fraction=0.15,
        class_weight="balanced",
        random_state=seed,
    )


def new_rf(params: dict, seed: int = 42):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=500,
        max_features=params.get("max_features", 0.3),
        min_samples_leaf=params.get("min_samples_leaf", 5),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def new_et(params: dict, seed: int = 42):
    from sklearn.ensemble import ExtraTreesClassifier

    return ExtraTreesClassifier(
        n_estimators=500,
        max_features=params.get("max_features", 0.3),
        min_samples_leaf=params.get("min_samples_leaf", 5),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def new_mlp(params: dict, seed: int = 42):
    from sklearn.neural_network import MLPClassifier

    return MLPClassifier(
        hidden_layer_sizes=(int(params.get("h1", 64)), int(params.get("h2", 32))),
        alpha=params.get("alpha", 1e-3),
        learning_rate_init=params.get("learning_rate_init", 1e-3),
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


FACTORIES = {
    "xgb": new_xgb,
    "lgbm": new_lgbm,
    "cat": new_cat,
    "histgb": new_histgb,
    "rf": new_rf,
    "et": new_et,
    "mlp": new_mlp,
}


def params_to_numpy_model(name: str, params: dict) -> object:
    """Signature test helper — not used by the pipeline."""
    return FACTORIES[name](params)


def log_range(trial, name: str, low: float, high: float) -> float:
    return trial.suggest_float(name, low, high, log=True)


def suggest_params_boost(name: str, trial: object, class_ratio: float) -> dict:
    """Suggest one hyper-parameter configuration for boosting model ``name``."""
    p: dict = {}
    if name == "xgb":
        p["learning_rate"] = log_range(trial, "learning_rate", 0.005, 0.3)
        p["max_depth"] = trial.suggest_int("max_depth", 3, 9)
        p["min_child_weight"] = log_range(trial, "min_child_weight", 0.5, 30.0)
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.3, 1.0)
        p["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)
        p["reg_alpha"] = log_range(trial, "reg_alpha", 1e-8, 10.0)
        p["reg_lambda"] = log_range(trial, "reg_lambda", 1e-8, 10.0)
        p["max_bin"] = trial.suggest_int("max_bin", 128, 511)
        p["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    elif name == "lgbm":
        p["learning_rate"] = log_range(trial, "learning_rate", 0.005, 0.3)
        p["num_leaves"] = trial.suggest_int("num_leaves", 8, 128)
        p["max_depth"] = trial.suggest_int("max_depth", 3, 20)
        p["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100)
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        p["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.3, 1.0)
        p["reg_alpha"] = log_range(trial, "reg_alpha", 1e-8, 10.0)
        p["reg_lambda"] = log_range(trial, "reg_lambda", 1e-8, 10.0)
        p["min_sum_hessian_in_leaf"] = log_range(trial, "min_sum_hessian_in_leaf", 0.001, 20.0)
        p["max_bin"] = trial.suggest_int("max_bin", 127, 511)
    else:
        p["learning_rate"] = log_range(trial, "learning_rate", 0.005, 0.3)
        p["depth"] = trial.suggest_int("depth", 4, 10)
        p["l2_leaf_reg"] = log_range(trial, "l2_leaf_reg", 0.5, 20.0)
        p["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 1, 50)
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
        p["colsample_bylevel"] = trial.suggest_float("colsample_bylevel", 0.5, 1.0)
    p["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", *((2.0, max(30.0, 2.0 * class_ratio)) if class_ratio > 1 else (1.0, 5.0)))
    return p


def suggest_params_extra(name: str, trial: object, class_ratio: float) -> dict:
    """Hyper-parameter space for the non-boosting sklearn models."""
    p: dict = {}
    if name == "histgb":
        p["learning_rate"] = log_range(trial, "learning_rate", 0.01, 0.3)
        p["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 8, 64)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 60)
        p["l2_regularization"] = log_range(trial, "l2_regularization", 1e-3, 10.0)
    elif name in ("rf", "et"):
        p["max_features"] = trial.suggest_float("max_features", 0.05, 0.7)
        p["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 30)
    else:
        p["h1"] = trial.suggest_categorical("h1", [32, 64, 128])
        p["h2"] = trial.suggest_categorical("h2", [16, 32, 64])
        p["alpha"] = log_range(trial, "alpha", 1e-5, 1.0)
        p["learning_rate_init"] = log_range(trial, "learning_rate_init", 1e-4, 1e-2)
    return p


def suggest_params(name: str, trial: object, class_ratio: float) -> dict:
    if name in ("histgb", "rf", "et", "mlp"):
        return suggest_params_extra(name, trial, class_ratio)
    return suggest_params_boost(name, trial, class_ratio)


def new_tabpfn(seed: int = 42, n_ens: int = 4):
    if not HAS_TABPFN:
        return None
    return TabPFNClassifier(
        device="cpu",
        seed=seed,
        N_ensemble_configurations=n_ens,
        subsample_features=True,
    )