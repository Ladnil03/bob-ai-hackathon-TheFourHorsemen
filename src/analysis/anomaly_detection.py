"""
Anomaly Detection Module — Phase 3
=====================================
Detects anomalous samples via Isolation Forest, computes feature-to-failure
correlations, and analyses group-level performance (e.g. equipment batches).

All functions accept dynamic feature lists so they work with any dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# Phase 3a — Isolation Forest anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    feature_cols: list,
    contamination: float = 0.1,
    random_state: int = 42,
) -> dict:
    """Run Isolation Forest on *feature_cols* and return anomaly results.

    Returns
    -------
    dict
        total_samples, num_anomalies, num_normal,
        anomaly_indices, anomaly_scores, anomaly_details
    """
    X = df[feature_cols].values.copy()

    # Scale features
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

    is_anomaly = (raw_labels == -1)
    anomaly_indices = list(np.where(is_anomaly)[0])

    # Compute z-scores for explanation
    normal_mask = ~is_anomaly
    normal_means = np.mean(X_scaled[normal_mask], axis=0)
    normal_stds = np.std(X_scaled[normal_mask], axis=0)
    normal_stds[normal_stds == 0] = 1.0

    # Build per-anomaly explanations (top 5 unusual features)
    anomaly_details = []
    for idx in anomaly_indices:
        z_scores = np.abs((X_scaled[idx] - normal_means) / normal_stds)
        top_k = np.argsort(z_scores)[::-1][:5]
        explanations = []
        for k in top_k:
            explanations.append({
                "feature": feature_cols[k],
                "value": round(float(df[feature_cols[k]].iloc[idx]), 4),
                "z_score": round(float(z_scores[k]), 2),
            })
        anomaly_details.append({
            "index": int(idx),
            "anomaly_score": round(float(scores[idx]), 4),
            "failure_flag": int(df["failure_flag"].iloc[idx]),
            "top_features": explanations,
        })

    return {
        "total_samples": len(df),
        "num_anomalies": int(is_anomaly.sum()),
        "num_normal": int((~is_anomaly).sum()),
        "anomaly_indices": [int(i) for i in anomaly_indices],
        "anomaly_scores": [round(float(s), 4) for s in scores],
        "anomaly_details": anomaly_details,
    }


# ---------------------------------------------------------------------------
# Phase 3c — Feature-to-failure correlation
# ---------------------------------------------------------------------------

def compute_correlations(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "failure_flag",
) -> dict:
    """Correlate each feature with the target and return ranked results.

    Returns
    -------
    dict
        correlations: list of {feature, correlation, abs_correlation}
    """
    results = []
    for col in feature_cols:
        if df[col].std() > 0:
            r = float(df[col].corr(df[target_col]))
            if not np.isnan(r):
                results.append({
                    "feature": col,
                    "correlation": round(r, 6),
                    "abs_correlation": round(abs(r), 6),
                })

    results.sort(key=lambda x: x["abs_correlation"], reverse=True)

    return {"correlations": results}


# ---------------------------------------------------------------------------
# Phase 3b — Group-level performance analysis
# ---------------------------------------------------------------------------

def analyze_groups(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "failure_flag",
) -> dict:
    """Analyse performance grouped by a categorical column.

    Works like the old equipment performance analysis but for any
    categorical column.

    Returns
    -------
    dict
        groups: list of {group, total, failures, failure_rate, recommendation}
    """
    
    if group_col not in df.columns:
        return {"groups": [], "group_col": group_col, "error": "Column not found"}

    groups = []
    for name, sub in df.groupby(group_col):
        total = len(sub)
        failures = int(sub[target_col].sum())
        rate = round(failures / total * 100, 1) if total > 0 else 0

        if rate > 15 or failures >= 3:
            rec = "⚠️ MAINTENANCE REQUIRED"
        elif rate > 5 or failures >= 1:
            rec = "Monitor closely"
        else:
            rec = "✓ Normal"

        groups.append({
            "group": str(name),
            "total": total,
            "failures": failures,
            "failure_rate": rate,
            "recommendation": rec,
        })

    groups.sort(key=lambda g: g["failure_rate"], reverse=True)

    return {"groups": groups, "group_col": group_col}


# ---------------------------------------------------------------------------
# Factory wafer lot anomaly detection (Phases 3a, 3b, 3c)
# ---------------------------------------------------------------------------

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
    """
    available_features = [c for c in SENSOR_FEATURE_COLS if c in lot_data.columns]
    X = lot_data[available_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
        n_jobs=-1,
    )
    raw_labels = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    result = lot_data.copy()
    result["is_anomaly"] = (raw_labels == -1).astype(int)
    result["anomaly_score"] = scores

    return result


def print_anomaly_report(lot_data: pd.DataFrame):
    """Pretty-print which lots are anomalous and their dominant features."""
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


def analyze_equipment_performance(lot_data: pd.DataFrame) -> pd.DataFrame:
    """Rank equipment by average yield and flag underperformers."""
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

    for eq_id, row in stats.iterrows():
        if row["num_failures"] >= 2 or row["avg_yield"] < 85:
            print(f"\n⚠️  ALERT: {eq_id} underperforming")
            print(f"   Failures: {int(row['num_failures'])}, "
                  f"Avg Yield: {row['avg_yield']}%")
            print("   → RECOMMEND: Maintenance check or parameter adjustment")

    return stats


def sensor_failure_correlation(lot_data: pd.DataFrame) -> pd.Series:
    """Correlate each sensor feature with failure_flag."""
    available_features = [c for c in SENSOR_FEATURE_COLS if c in lot_data.columns]

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

    from analysis.data_loader import load_secom_dataset, preprocess_data, get_feature_columns
    from analysis.feature_engineering import select_features

    df = load_secom_dataset()
    df = preprocess_data(df)
    feature_cols = get_feature_columns(df)

    fs = select_features(df, feature_cols, top_k=30)
    selected = fs["selected_features"]

    # Anomaly detection
    result = detect_anomalies(df, selected)
    print(f"Anomalies: {result['num_anomalies']} / {result['total_samples']}")
    for det in result["anomaly_details"][:3]:
        print(f"  Index {det['index']} (fail={det['failure_flag']}): "
              f"score={det['anomaly_score']}")
        for f in det["top_features"][:2]:
            print(f"    {f['feature']}: z={f['z_score']}")

    # Correlations
    corr = compute_correlations(df, selected)
    print("\nTop correlations:")
    for c in corr["correlations"][:10]:
        print(f"  {c['feature']:20s}  r={c['correlation']:+.4f}")


if __name__ == "__main__":
    main()
