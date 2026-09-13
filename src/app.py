"""
Semiconductor Yield Optimization — Main Application
=====================================================
Entry point that wires together every analysis module:

1. Load & merge data            (Phase 2)
2. Anomaly detection            (Phase 3a)
3. Equipment performance        (Phase 3b)
4. Sensor–failure correlations  (Phase 3c)
5. Predictive model + SHAP      (Phase 4)
6. Root cause analysis          (Phase 5)
7. Batch risk scoring           (Phase 6)

Usage
-----
    # Full pipeline (all phases)
    python src/app.py

    # Score an upcoming batch
    python src/app.py --predict_batch upcoming_recipe.csv
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure src/ is on the path so that `analysis.*` imports resolve
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.data_loader import build_lot_dataset, validate_dataset
from analysis.anomaly_detection import (
    detect_sensor_anomalies,
    print_anomaly_report,
    analyze_equipment_performance,
    sensor_failure_correlation,
)
from analysis.predictive_model import (
    train_failure_predictor,
    explain_failures_shap,
    FEATURE_COLS,
)
from analysis.root_cause_analyzer import (
    rank_root_causes,
    print_root_cause_report,
)


# ---------------------------------------------------------------------------
# Phase 6 — Batch Risk Scoring
# ---------------------------------------------------------------------------

def score_upcoming_batch(
    batch_params: dict,
    model_artifacts: dict,
    lot_data: pd.DataFrame,
) -> dict:
    """Score a planned batch and return a risk assessment.

    Parameters
    ----------
    batch_params : dict
        Planned parameters.  Keys should include recipe targets such as
        ``target_temp_C``, ``target_pressure_pa``, ``target_power_w``,
        ``equipment_id``, ``process_recipe``, etc.
    model_artifacts : dict
        Output of ``train_failure_predictor``.
    lot_data : pd.DataFrame
        Historical lot data (for population statistics).

    Returns
    -------
    dict
        Risk assessment with score, classification, factors, and recs.
    """
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    features = model_artifacts["features"]

    # Map recipe targets → approximate sensor features
    # In production this would come from a physics model; here we use
    # the historical mean of matching recipes as a proxy.
    recipe = batch_params.get("process_recipe", "")
    equipment = batch_params.get("equipment_id", "UNKNOWN")

    # Find historical lots with similar recipe
    similar = lot_data[lot_data["process_recipe"].str.contains(
        recipe.split("_V")[0] if "_V" in recipe else recipe, na=False
    )]
    if similar.empty:
        similar = lot_data  # fallback to all lots

    # Build a feature vector from planned params + historical means
    feature_vector = {}
    for feat in features:
        if feat in batch_params:
            feature_vector[feat] = batch_params[feat]
        elif feat in similar.columns:
            feature_vector[feat] = similar[feat].mean()
        else:
            feature_vector[feat] = 0.0

    # Override with planned parameters where applicable
    if "target_pressure_pa" in batch_params and "pressure_drift" in features:
        # Low drift expected for a well-tuned recipe
        feature_vector["pressure_drift"] = max(0.5, similar["pressure_drift"].quantile(0.25))
    if "target_temp_C" in batch_params and "temp_drift" in features:
        feature_vector["temp_drift"] = max(1.0, similar["temp_drift"].quantile(0.25))

    # Build DataFrame for prediction
    X_batch = pd.DataFrame([feature_vector])[features]
    X_scaled = scaler.transform(X_batch.values)

    # Predict
    proba = model.predict_proba(X_scaled)[0][1]
    risk_pct = round(proba * 100, 1)

    if risk_pct < 25:
        classification = "LOW"
        symbol = "✅"
        status = "APPROVED TO RUN"
        rec = "Standard monitoring (no special precautions)"
    elif risk_pct < 50:
        classification = "MEDIUM"
        symbol = "⚠️"
        status = "CONDITIONAL APPROVAL"
        rec = "Enhanced monitoring — check pressure & temp every 2 min"
    else:
        classification = "HIGH"
        symbol = "🛑"
        status = "HOLD — Review required"
        rec = "Do NOT run until root cause from prior failures is resolved"

    # Equipment history
    eq_lots = lot_data[lot_data["equipment_id"] == equipment]
    eq_yield = eq_lots["yield_percent"].mean() if len(eq_lots) > 0 else None
    eq_failures = eq_lots["failure_flag"].sum() if len(eq_lots) > 0 else None

    # Find similar good lots
    passing = lot_data[lot_data["failure_flag"] == 0]
    best_matches = passing.nlargest(2, "yield_percent")["lot_id"].tolist()

    # Risk factors
    pop_means = lot_data[features].mean()
    pop_stds = lot_data[features].std().replace(0, 1)
    z_scores = ((X_batch.iloc[0] - pop_means) / pop_stds).abs().sort_values(ascending=False)
    risk_factors = []
    for feat, z in z_scores.head(3).items():
        if z > 0.5:
            risk_factors.append(f"{feat} is {z:.1f}σ from historical mean")

    return {
        "risk_score": risk_pct,
        "classification": classification,
        "symbol": symbol,
        "status": status,
        "recommendation": rec,
        "equipment": equipment,
        "recipe": recipe,
        "eq_avg_yield": round(eq_yield, 1) if eq_yield else "N/A",
        "eq_failures": int(eq_failures) if eq_failures is not None else "N/A",
        "risk_factors": risk_factors,
        "similar_good_lots": best_matches,
        "planned_features": {k: round(v, 3) for k, v in feature_vector.items()},
    }


def print_batch_risk_report(assessment: dict, batch_params: dict):
    """Pretty-print a batch risk assessment."""
    print("\n" + "=" * 70)
    print("BATCH RISK ASSESSMENT")
    print("=" * 70)
    print(f"Equipment : {assessment['equipment']}")
    print(f"Recipe    : {assessment['recipe']}")

    planned = batch_params
    for key in ["target_temp_C", "target_pressure_pa", "target_power_w"]:
        if key in planned:
            print(f"  {key}: {planned[key]}")

    print(f"\nRISK SCORE: {assessment['risk_score']}%  "
          f"({assessment['classification']} RISK) {assessment['symbol']}")
    print(f"\nStatus          : {assessment['status']}")
    print(f"Recommendation  : {assessment['recommendation']}")

    if assessment["risk_factors"]:
        print(f"\nRisk factors:")
        for rf in assessment["risk_factors"]:
            print(f"  ⚠️  {rf}")
    else:
        print("\n  ✓ All parameters within normal range")

    print(f"\nEquipment history:")
    print(f"  Avg yield  : {assessment['eq_avg_yield']}%")
    print(f"  Failures   : {assessment['eq_failures']}")

    if assessment["similar_good_lots"]:
        print(f"\nSimilar successful lots: {', '.join(assessment['similar_good_lots'])}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline():
    """Execute every analysis phase end-to-end."""
    print("╔" + "═" * 68 + "╗")
    print("║   SEMICONDUCTOR YIELD OPTIMIZATION PIPELINE                        ║")
    print("║   Phases 2–6: Data → Features → Anomalies → Prediction → Action   ║")
    print("╚" + "═" * 68 + "╝")

    # ── Phase 2: Load & merge ──────────────────────────────────────────
    print("\n▶ PHASE 2: Data Pipeline & Feature Engineering")
    lot_data, sensors, defects = build_lot_dataset()
    validate_dataset(lot_data)

    # ── Phase 3a: Anomaly detection ────────────────────────────────────
    print("\n▶ PHASE 3a: Anomaly Detection (Isolation Forest)")
    lot_data = detect_sensor_anomalies(lot_data)
    print_anomaly_report(lot_data)

    # ── Phase 3b: Equipment performance ────────────────────────────────
    print("\n▶ PHASE 3b: Equipment Performance")
    analyze_equipment_performance(lot_data)

    # ── Phase 3c: Sensor-failure correlation ───────────────────────────
    print("\n▶ PHASE 3c: Sensor–Failure Correlation")
    sensor_failure_correlation(lot_data)

    # ── Phase 4: Predictive model ──────────────────────────────────────
    print("\n▶ PHASE 4: Failure Prediction Model + SHAP")
    model_artifacts = train_failure_predictor(lot_data)
    explain_failures_shap(model_artifacts, lot_data)

    # ── Phase 5: Root cause analysis ───────────────────────────────────
    print("\n▶ PHASE 5: Root Cause Analysis")
    failed_lots = lot_data[lot_data["failure_flag"] == 1]["lot_id"].tolist()
    for lot_id in failed_lots:
        report = rank_root_causes(lot_id, lot_data)
        print_root_cause_report(report)

    # ── Phase 6: Demo batch risk scoring ───────────────────────────────
    print("\n▶ PHASE 6: Batch Risk Scoring (demo)")
    demo_batch_good = {
        "equipment_id": "EQPM_A_AFTER_MAINT",
        "process_recipe": "Recipe_3nm_V2.2",
        "target_temp_C": 301.5,
        "target_pressure_pa": 134.5,
        "target_power_w": 451.5,
        "etch_time_sec": 700,
    }
    assessment = score_upcoming_batch(demo_batch_good, model_artifacts, lot_data)
    print_batch_risk_report(assessment, demo_batch_good)

    demo_batch_risky = {
        "equipment_id": "EQPM_B",
        "process_recipe": "Recipe_3nm_V2.1",
        "target_temp_C": 295.0,
        "target_pressure_pa": 130.0,
        "target_power_w": 445.0,
        "etch_time_sec": 720,
    }
    assessment2 = score_upcoming_batch(demo_batch_risky, model_artifacts, lot_data)
    print_batch_risk_report(assessment2, demo_batch_risky)

    print("\n✅ Pipeline complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semiconductor Yield Optimization Pipeline"
    )
    parser.add_argument(
        "--predict_batch", type=str, default=None,
        help="Path to CSV with upcoming batch parameters to score"
    )
    args = parser.parse_args()

    if args.predict_batch:
        # Score a specific batch from CSV
        lot_data, _, _ = build_lot_dataset()
        model_artifacts = train_failure_predictor(lot_data)

        batch_df = pd.read_csv(args.predict_batch)
        for _, row in batch_df.iterrows():
            batch_params = row.to_dict()
            assessment = score_upcoming_batch(batch_params, model_artifacts, lot_data)
            print_batch_risk_report(assessment, batch_params)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
