"""
Root Cause Analyzer Module — Phase 5
=======================================
Given a dataset with failures, this module:

1. Finds similar *passing* samples for each failed sample (cosine similarity).
2. Compares the failed sample against the best-matching passer.
3. Ranks which features deviated the most (z-score based).
4. Produces an actionable report with ranked root causes.

Works with any dataset — no hardcoded sensor names.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def analyze_root_causes(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "failure_flag",
    n_similar: int = 3,
    top_k_causes: int = 5,
    max_failures: int = 20,
) -> dict:
    """Produce root-cause analysis for all failed samples.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list
    target_col : str
    n_similar : int — number of similar passing samples to find
    top_k_causes : int — number of root causes per failure
    max_failures : int — cap on failures to analyse (for performance)

    Returns
    -------
    dict
        total_failures, analyses (list of per-failure reports)
    """
    failed = df[df[target_col] == 1]
    passing = df[df[target_col] == 0]

    if len(failed) == 0:
        return {"total_failures": 0, "analyses": []}
    if len(passing) == 0:
        return {"total_failures": len(failed), "analyses": [],
                "error": "No passing samples for comparison"}

    # Scale features
    scaler = StandardScaler()
    all_features = df[feature_cols].values
    scaler.fit(all_features)

    passing_scaled = scaler.transform(passing[feature_cols].values)
    pop_means = df[feature_cols].mean()
    pop_stds = df[feature_cols].std().replace(0, 1)

    analyses = []
    for i, (idx, row) in enumerate(failed.iterrows()):
        if i >= max_failures:
            break

        target_vec = scaler.transform(row[feature_cols].values.reshape(1, -1))

        # Find most similar passing samples
        sims = cosine_similarity(target_vec, passing_scaled).flatten()
        top_indices = sims.argsort()[::-1][:n_similar]

        similar_passing = []
        for si in top_indices:
            pass_row = passing.iloc[si]
            diffs = {}
            for feat in feature_cols[:20]:  # Limit for response size
                diffs[feat] = round(float(row[feat] - pass_row[feat]), 4)
            similar_passing.append({
                "index": int(passing.index[si]),
                "similarity": round(float(sims[si]), 4),
                "key_diffs": dict(sorted(diffs.items(),
                                         key=lambda x: abs(x[1]),
                                         reverse=True)[:5]),
            })

        # Compare vs best passer
        best_pass = passing.iloc[top_indices[0]]

        deviations = []
        for feat in feature_cols:
            diff = abs(float(row[feat] - best_pass[feat]))
            z = abs(float(row[feat] - pop_means[feat])) / float(pop_stds[feat])
            deviations.append({
                "feature": feat,
                "failed_value": round(float(row[feat]), 4),
                "best_pass_value": round(float(best_pass[feat]), 4),
                "abs_diff": round(diff, 4),
                "z_score": round(z, 2),
            })

        deviations.sort(key=lambda d: d["z_score"], reverse=True)

        # Root causes — top deviations
        root_causes = []
        for rank, dev in enumerate(deviations[:top_k_causes], 1):
            confidence = min(95, int(50 + dev["z_score"] * 12))
            level = "CRITICAL" if dev["z_score"] > 2.0 else \
                    "HIGH" if dev["z_score"] > 1.5 else \
                    "MODERATE" if dev["z_score"] > 1.0 else "LOW"
            root_causes.append({
                "rank": rank,
                "feature": dev["feature"],
                "confidence": confidence,
                "level": level,
                "failed_value": dev["failed_value"],
                "best_pass_value": dev["best_pass_value"],
                "z_score": dev["z_score"],
            })

        # Anomaly flags
        anomalies = []
        for feat in feature_cols:
            z = (float(row[feat]) - float(pop_means[feat])) / float(pop_stds[feat])
            if abs(z) > 1.0:
                level = "CRITICAL" if abs(z) > 2.0 else "HIGH"
                anomalies.append({
                    "feature": feat,
                    "value": round(float(row[feat]), 4),
                    "z_score": round(z, 2),
                    "level": level,
                })
        anomalies.sort(key=lambda a: abs(a["z_score"]), reverse=True)

        analyses.append({
            "sample_index": int(idx),
            "root_causes": root_causes,
            "anomalies": anomalies[:10],
            "similar_passing": similar_passing,
        })

    return {
        "total_failures": int(len(failed)),
        "analyzed": len(analyses),
        "analyses": analyses,
    }


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

    result = analyze_root_causes(df, selected, max_failures=3)
    print(f"Total failures: {result['total_failures']}")
    for a in result["analyses"]:
        print(f"\n--- Sample {a['sample_index']} ---")
        for rc in a["root_causes"][:3]:
            print(f"  #{rc['rank']}: {rc['feature']} "
                  f"(z={rc['z_score']}, conf={rc['confidence']}%)")


if __name__ == "__main__":
    main()
