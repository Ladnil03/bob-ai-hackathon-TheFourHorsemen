"""
Predictive Model Module — Phase 4a / 4b
=========================================
Trains a Random Forest classifier to predict wafer-lot failures and
provides SHAP-based explanations for every prediction.

Model design choices
--------------------
- **Temporal split** — the first 80 % of lots (by date order) are training,
  the last 20 % are test.  This respects time so we evaluate on "future"
  lots, which is realistic for a manufacturing line.
- **StandardScaler** — applied to features so that SHAP values are on a
  comparable scale.
- **Random Forest (n=100, depth=5)** — shallow trees reduce over-fitting on
  this small dataset while still capturing non-linear interactions between
  pressure, temperature, and etch rate.
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("shap not installed — SHAP explanations will be skipped.")


# ---------------------------------------------------------------------------
# Feature list (must match feature_engineering output column names)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "pressure_drift",    # #1 predictor
    "temp_drift",
    "etch_rate_max",
    "flow_std",
    "power_std",
    "cd_drift",
    "humidity_mean",
    "etch_rate_std",
    "temp_std",
    "pressure_std",
]


# ---------------------------------------------------------------------------
# Phase 4a — Train failure prediction model
# ---------------------------------------------------------------------------

def train_failure_predictor(lot_data: pd.DataFrame, test_fraction: float = 0.2):
    """Train a Random Forest to predict ``failure_flag``.

    Parameters
    ----------
    lot_data : pd.DataFrame
        Merged lot dataset produced by ``build_lot_dataset``.
    test_fraction : float
        Fraction of lots to hold out for testing (by temporal order).

    Returns
    -------
    dict with keys:
        model       — fitted RandomForestClassifier
        scaler      — fitted StandardScaler
        features    — list of feature column names used
        X_train, X_test, y_train, y_test — raw (unscaled) splits
        X_test_scaled — scaled test features (for SHAP)
        report      — classification report string
    """
    # Use only features that exist in the dataset
    available_features = [c for c in FEATURE_COLS if c in lot_data.columns]

    # --- Stratified split (80 / 20) ensuring both classes in test ---------
    # Pure temporal split would put all failures in train (they're in Jan)
    # and only passes in test (Feb).  Stratified split is more robust for
    # this small dataset while still evaluating generalisation.
    from sklearn.model_selection import train_test_split

    X_all = lot_data[available_features].values
    y_all = lot_data["failure_flag"].values
    lot_ids = lot_data["lot_id"].values

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_all, y_all, lot_ids,
        test_size=test_fraction,
        random_state=42,
        stratify=y_all,
    )

    # --- Scale features ---------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train Random Forest ----------------------------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight="balanced",   # helps with class imbalance
    )
    model.fit(X_train_scaled, y_train)

    # --- Evaluate ---------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    report = classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["PASS", "FAIL"],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print("\n" + "=" * 70)
    print("FAILURE PREDICTION MODEL — Random Forest")
    print("=" * 70)
    print(f"Training lots : {len(X_train)}  |  Test lots : {len(X_test)}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # Feature importances
    importances = pd.Series(
        model.feature_importances_, index=available_features
    ).sort_values(ascending=False)
    print("\nFeature Importances (Gini):")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:20s}  {imp:.4f}  {bar}")

    return {
        "model": model,
        "scaler": scaler,
        "features": available_features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_test_scaled": X_test_scaled,
        "test_lot_ids": ids_test,
        "report": report,
    }


# ---------------------------------------------------------------------------
# Phase 4b — SHAP explanations for failed lots
# ---------------------------------------------------------------------------

def explain_failures_shap(model_artifacts: dict, lot_data: pd.DataFrame):
    """Generate human-readable SHAP explanations for each predicted failure.

    Parameters
    ----------
    model_artifacts : dict
        Output of ``train_failure_predictor``.
    lot_data : pd.DataFrame
        Full dataset (used to compute population statistics for context).
    """
    if not HAS_SHAP:
        print("\n⚠️  SHAP library not installed.  Run:  pip install shap")
        return

    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    features = model_artifacts["features"]
    X_test_scaled = model_artifacts["X_test_scaled"]
    y_test = model_artifacts["y_test"]
    test_lot_ids = model_artifacts["test_lot_ids"]

    # Compute population statistics for context
    pop_means = lot_data[features].mean()
    pop_stds = lot_data[features].std().replace(0, 1)

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    # Handle different SHAP return formats:
    # Old API: list of [class_0_array, class_1_array], each (n_samples, n_features)
    # New API: 3D ndarray (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        shap_fail = shap_values[1]  # class 1 = FAIL
    elif shap_values.ndim == 3:
        shap_fail = shap_values[:, :, 1]  # class 1 = FAIL
    else:
        shap_fail = shap_values  # binary output, single array

    print("\n" + "=" * 70)
    print("SHAP EXPLANATIONS FOR TEST-SET FAILURES")
    print("=" * 70)

    # Find actual or predicted failures in the test set
    y_pred = model.predict(X_test_scaled)
    failure_mask = (y_pred == 1) | (y_test == 1)

    if not failure_mask.any():
        print("No failures found in test set.")
        return

    for idx in np.where(failure_mask)[0]:
        lot_id = test_lot_ids[idx]
        actual = "FAIL" if y_test[idx] == 1 else "PASS"
        predicted = "FAIL" if y_pred[idx] == 1 else "PASS"
        proba = model.predict_proba(X_test_scaled[idx:idx+1])[0][1]

        print(f"\n  {lot_id}  (actual={actual}, predicted={predicted}, "
              f"P(fail)={proba:.1%})")

        # Rank SHAP contributions for this lot
        shap_for_lot = shap_fail[idx]
        # Ensure 1D
        if shap_for_lot.ndim > 1:
            shap_for_lot = shap_for_lot[:, -1]  # take FAIL class column
        raw_values = scaler.inverse_transform(X_test_scaled[idx:idx+1]).flatten()
        shap_df = pd.DataFrame({
            "feature": features,
            "shap_value": shap_for_lot.flatten(),
            "raw_value": raw_values,
        })
        shap_df["abs_shap"] = shap_df["shap_value"].abs()
        shap_df = shap_df.sort_values("abs_shap", ascending=False)

        for _, r in shap_df.head(4).iterrows():
            feat = r["feature"]
            raw = r["raw_value"]
            sv = r["shap_value"]
            z = (raw - pop_means[feat]) / pop_stds[feat]
            direction = "above" if z > 0 else "below"
            push = "toward FAILURE" if sv > 0 else "toward PASS"
            print(f"    → {feat:20s} = {raw:8.2f}  "
                  f"({abs(z):.1f}σ {direction} mean)  "
                  f"SHAP {sv:+.4f} ({push})")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

    from analysis.data_loader import build_lot_dataset

    lot_data, _, _ = build_lot_dataset()

    # 4a — Train model
    artifacts = train_failure_predictor(lot_data)

    # 4b — SHAP explanations
    explain_failures_shap(artifacts, lot_data)


if __name__ == "__main__":
    main()
