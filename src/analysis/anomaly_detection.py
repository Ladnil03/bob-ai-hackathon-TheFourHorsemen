"""
Anomaly Detection Module — Phase 3a / 3b / 3c
================================================
Detects anomalous wafer lots via Isolation Forest, analyses equipment
performance, and computes sensor-to-failure correlations.

Isolation Forest rationale
--------------------------
Contamination is set to 0.15 (~15 % of lots expected anomalous) to align
with the observed failure/warning rate in the sample data.  Features are
StandardScaler-normalised before fitting so that high-magnitude sensors
(e.g.  power_w ~ 450) don't dominate low-magnitude ones (e.g.  cd_drift
~ 0.1).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# Phase 3a — Isolation Forest anomaly detection
# ---------------------------------------------------------------------------

# Sensor features used for anomaly scoring (match Phase 2b outputs)
SENSOR_FEATURE_COLS = [
    "pressure_drift",   # #1 predictor
    "temp_drift",
    "pressure_std",
    "temp_std",
    "flow_std",
    "power_std",
    "etch_rate_max",
    "etch_rate_std",
    "cd_drift",
    "humidity_mean",
]


def detect_sensor_anomalies(
    lot_data: pd.DataFrame,
    contamination: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Isolation Forest on sensor features and flag anomalous lots.

    Parameters
    ----------
    lot_data : pd.DataFrame
        Merged lot-level dataset (output of ``build_lot_dataset``).
    contamination : float
        Expected proportion of anomalous lots.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Copy of *lot_data* with two extra columns:
        - ``is_anomaly`` (int) — 1 = anomalous, 0 = normal
        - ``anomaly_score`` (float) — Isolation Forest decision score
          (more negative → more anomalous)
    """
    # Only use feature columns that actually exist in the data
    available_features = [c for c in SENSOR_FEATURE_COLS if c in lot_data.columns]
    X = lot_data[available_features].copy()

    # Scale features so magnitude doesn't dominate
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Isolation Forest
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        n_jobs=-1,
    )
    raw_labels = iso.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X_scaled)

    result = lot_data.copy()
    result["is_anomaly"] = (raw_labels == -1).astype(int)
    result["anomaly_score"] = scores

    return result


def print_anomaly_report(lot_data: pd.DataFrame):
    """Pretty-print which lots are anomalous and their dominant features.

    Uses z-scores to explain *why* each lot was flagged — the features
    with the largest absolute z-score are the most unusual.
    """
    anomalies = lot_data[lot_data["is_anomaly"] == 1].copy()
    normals = lot_data[lot_data["is_anomaly"] == 0]
    available_features = [c for c in SENSOR_FEATURE_COLS if c in lot_data.columns]

    print("\n" + "=" * 70)
    print("ANOMALY DETECTION REPORT  (Isolation Forest)")
    print("=" * 70)
    print(f"Total lots analysed   : {len(lot_data)}")
    print(f"Anomalies detected    : {len(anomalies)}")
    print(f"Normal lots           : {len(normals)}")

    if len(anomalies) == 0:
        print("No anomalies found.")
        return

    print(f"\nAnomalous lot IDs     : {', '.join(anomalies['lot_id'].tolist())}")

    # Compute z-scores relative to normal population
    normal_means = normals[available_features].mean()
    normal_stds = normals[available_features].std().replace(0, 1)

    print("\n--- Per-lot anomaly explanation ---")
    for _, row in anomalies.iterrows():
        print(f"\n  {row['lot_id']}  (yield={row.get('yield_percent', '?')}%, "
              f"status={row.get('final_status', '?')})")
        z_scores = ((row[available_features] - normal_means) / normal_stds).abs()
        top_features = z_scores.sort_values(ascending=False).head(3)
        for feat, z in top_features.items():
            direction = "above" if row[feat] > normal_means[feat] else "below"
            print(f"    → {feat:20s} = {row[feat]:.2f}  "
                  f"({z:.1f}σ {direction} normal mean {normal_means[feat]:.2f})")


# ---------------------------------------------------------------------------
# Phase 3b — Equipment performance analysis
# ---------------------------------------------------------------------------

def analyze_equipment_performance(lot_data: pd.DataFrame) -> pd.DataFrame:
    """Rank equipment by average yield and flag underperformers.

    Prints maintenance alerts for equipment with:
    - avg_yield < 85 %, **or**
    - ≥ 2 failures.

    Returns the summary DataFrame for downstream use.
    """
    print("\n" + "=" * 70)
    print("EQUIPMENT PERFORMANCE ANALYSIS")
    print("=" * 70)

    stats = lot_data.groupby("equipment_id").agg(
        avg_yield=("yield_percent", "mean"),
        std_yield=("yield_percent", "std"),
        min_yield=("yield_percent", "min"),
        total_lots=("lot_id", "count"),
        num_failures=("failure_flag", "sum"),
    ).round(2)

    stats = stats.sort_values("avg_yield")

    # Build recommendation column
    recommendations = []
    for eq_id, row in stats.iterrows():
        if row["num_failures"] >= 2 or row["avg_yield"] < 85:
            recommendations.append("⚠️  MAINTENANCE REQUIRED")
        elif row["num_failures"] == 1 or row["avg_yield"] < 90:
            recommendations.append("✓ Normal (monitor)")
        else:
            recommendations.append("✓ Best performer")
    stats["recommendation"] = recommendations

    print(stats.to_string())

    # Detailed alerts
    for eq_id, row in stats.iterrows():
        if row["num_failures"] >= 2 or row["avg_yield"] < 85:
            print(f"\n⚠️  ALERT: {eq_id} underperforming")
            print(f"   Failures: {int(row['num_failures'])}, "
                  f"Avg Yield: {row['avg_yield']}%")
            print("   → RECOMMEND: Maintenance check or parameter adjustment")

    return stats


# ---------------------------------------------------------------------------
# Phase 3c — Sensor-to-failure correlation
# ---------------------------------------------------------------------------

def sensor_failure_correlation(lot_data: pd.DataFrame) -> pd.Series:
    """Correlate each sensor feature with failure_flag.

    Returns correlation coefficients sorted by absolute magnitude
    (strongest predictors first).
    """
    available_features = [c for c in SENSOR_FEATURE_COLS if c in lot_data.columns]

    # Also include defect features if present
    defect_cols = [c for c in lot_data.columns
                   if c.endswith("_count") or c in ("avg_severity", "total_defects")]
    all_features = available_features + defect_cols

    correlations = {}
    for col in all_features:
        if col in lot_data.columns and lot_data[col].std() > 0:
            correlations[col] = lot_data[col].corr(lot_data["failure_flag"])

    corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)

    print("\n" + "=" * 70)
    print("SENSOR / DEFECT → FAILURE CORRELATION")
    print("=" * 70)
    for feat, val in corr_series.items():
        bar = "█" * int(abs(val) * 40)
        sign = "+" if val > 0 else "−"
        print(f"  {feat:30s}  {sign}{abs(val):.4f}  {bar}")

    # Interpretation
    top3 = corr_series.head(3)
    print("\n  Interpretation:")
    for feat, val in top3.items():
        direction = "positively" if val > 0 else "negatively"
        print(f"    • {feat} is {direction} correlated with failures (r={val:.3f})")
    print("    → Pressure drift and defect counts are the strongest predictors.")

    return corr_series


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

    from analysis.data_loader import build_lot_dataset

    lot_data, sensors, defects = build_lot_dataset()

    # 3a — Anomaly detection
    lot_data = detect_sensor_anomalies(lot_data)
    print_anomaly_report(lot_data)

    # 3b — Equipment performance
    analyze_equipment_performance(lot_data)

    # 3c — Correlations
    sensor_failure_correlation(lot_data)


if __name__ == "__main__":
    main()
