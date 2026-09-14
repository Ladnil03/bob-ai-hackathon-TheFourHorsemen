"""Configuration for the best_model pipeline (paths, seeds, CV, HPO bounds, GPU)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SECOM_DIR = Path(__file__).resolve().parent.parent / "data" / "secom"

ARTIFACT_DIR = Path(os.environ.get("BOB_MODELS_DIR", r"D:\bob_models"))
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = ARTIFACT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── CV layout ────────────────────────────────────────────────────────────
N_FOLDS_TUNE = 5
N_FOLDS_FINAL = 5
N_REPEATS_FINAL = 2

# ── Optuna HPO ───────────────────────────────────────────────────────────
N_TRIALS_PER_MODEL = 80
N_TRIALS_EXTRA_MODEL = 40

# ── Smoke-test overrides ─────────────────────────────────────────────────
SMOKE_N_TRIALS = 3
SMOKE_N_FOLDS = 3
SMOKE_N_REPEATS = 1
SMOKE_N_SAMPLES = 120

# ── Feature engineering ──────────────────────────────────────────────────
TOP_K_FEATURES = 220
MAX_NAN_FRAC = 0.5
FEATURE_ABLATION_DELTA = 0.005

# ── Models ───────────────────────────────────────────────────────────────
BOOSTERS = ("xgb", "lgbm", "cat")
EXTRA_MODELS = ("histgb", "rf", "et", "mlp")
HAS_ES = {
    "xgb": True, "lgbm": True, "cat": True,
    "histgb": False, "rf": False, "et": False, "mlp": False,
}
NEEDS_IMPUTE = {"histgb", "rf", "et", "mlp"}
USE_TABPFN = bool(os.environ.get("BOB_USE_TABPFN", "1") == "1")

MODEL_METRIC = "pr_auc"
EARLY_STOPPING_ROUNDS = 60
VALIDATION_FRACTION = 0.2

CLASS_WEIGHT_RANGE = (2.0, 30.0)

# ── GPU auto-detection ───────────────────────────────────────────────────
# CatBoost GPU works on this machine; XGB/LGB pip wheels lack GPU support.

def _detect_gpu() -> dict[str, str]:
    """Detect per-framework GPU availability and return device settings."""
    gpu_info: dict[str, str] = {
        "xgb": "cpu",
        "lgbm": "cpu",
        "cat": "CPU",
    }

    # CatBoost GPU check
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=2, task_type="GPU",
                               verbose=False, allow_writing_files=False)
        m.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 1, 0, 1])
        gpu_info["cat"] = "GPU"
    except Exception:
        pass

    # XGBoost — try cuda, silently fall back to cpu
    try:
        import xgboost as xgb
        m = xgb.XGBClassifier(device="cuda", tree_method="hist",
                              n_estimators=2, verbosity=0)
        m.fit([[1, 2], [3, 4], [5, 6], [7, 8]], [0, 1, 0, 1])
        # XGB warns and falls back if no real GPU found; check booster config
        cfg = m.get_booster().save_config()
        if '"cuda"' in cfg or '"gpu"' in cfg:
            gpu_info["xgb"] = "cuda"
    except Exception:
        pass

    return gpu_info


# Detect once at import time
GPU_DEVICES = _detect_gpu()


def env_cache_dir() -> Path:
    """Return the shared on-D: cache directory used by all tools."""
    cache = Path(os.environ.get("BOB_CACHE_DIR", r"D:\cache"))
    cache.mkdir(parents=True, exist_ok=True)
    return cache