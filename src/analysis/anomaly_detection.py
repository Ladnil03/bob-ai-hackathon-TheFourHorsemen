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
