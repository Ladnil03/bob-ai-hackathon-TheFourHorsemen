"""
best_model — maximum-quality SECOM classifier.

Optimized for a rare-event binary failure prediction on the UCI SECOM
dataset (1,567 wafers x 590 sensors, 6.6% failures).

Pipeline
--------
1. Load + basic cleaning (drop constant / near-constant and >50% NaN columns).
2. Feature engineering (missingness, row statistics, time features,
   unsupervised composites, MAD z-scores, nearest-neighbour distances).
3. Fold-aware feature selection via mutual information (no leakage).
4. Optuna hyper-parameter tuning for XGBoost / LightGBM / CatBoost,
   optimising PR-AUC under repeated stratified K-fold CV.
5. Final out-of-fold evaluation with 5x2 repeated stratified folding,
   threshold tuned to maximise F1, and two ensemble strategies
   (weighted probability blend vs. logistic stacking) — the better one wins.
6. Optional TabPFN (pretrained tabular transformer) as an extra ensemble
   member when it is installed and usable on this machine.
7. Refit on the full dataset, persist artifacts under D:\\bob_models\\ and
   write metrics.json + console/plot reports.
"""

from .config import ARTIFACT_DIR, SEED, N_FOLDS_FINAL, N_REPEATS_FINAL

__all__ = ["ARTIFACT_DIR", "SEED", "N_FOLDS_FINAL", "N_REPEATS_FINAL"]