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
# Factory wafer lot root cause analysis (Phase 5 / CLI pipeline)
# ---------------------------------------------------------------------------

SENSOR_DEFECT_MAP = {
    "pressure_drift": {
        "defect": "METAL_VOID",
        "mechanism": "Unstable chamber pressure causes incomplete metal fill",
        "action": "Schedule pressure transducer calibration / replacement",
    },
    "pressure_std": {
        "defect": "METAL_VOID",
        "mechanism": "Pressure oscillation disrupts deposition uniformity",
        "action": "Check mass-flow controller and throttle valve",
    },
    "temp_drift": {
        "defect": "CRITICAL_DIMENSION_SHIFT",
        "mechanism": "Temperature ramp causes etch-rate variation → CD loss",
        "action": "Check heater element and PID tuning parameters",
    },
    "temp_std": {
        "defect": "CRITICAL_DIMENSION_SHIFT",
        "mechanism": "Temperature instability leads to non-uniform etching",
        "action": "Inspect thermocouple placement and PID loop gain",
    },
    "etch_rate_max": {
        "defect": "GATE_DIELECTRIC_THIN",
        "mechanism": "Excessive etch rate thins gate dielectric beyond spec",
        "action": "Reduce RF power by 2 W or increase chamber pressure slightly",
    },
    "etch_rate_std": {
        "defect": "GATE_DIELECTRIC_THIN",
        "mechanism": "Etch-rate instability creates thickness non-uniformity",
        "action": "Stabilise gas flow and verify endpoint detection",
    },
    "cd_drift": {
        "defect": "CRITICAL_DIMENSION_SHIFT",
        "mechanism": "CD variation indicates process window excursion",
        "action": "Tighten lithography focus/dose and verify resist profile",
    },
    "flow_std": {
        "defect": "LINE_ROUGHNESS",
        "mechanism": "Flow instability causes resist-profile roughness",
        "action": "Check MFC calibration and gas-line integrity",
    },
    "power_std": {
        "defect": "GATE_DIELECTRIC_THIN",
        "mechanism": "RF power jitter modulates plasma density → etch uniformity",
        "action": "Inspect RF generator / matching network",
    },
    "humidity_mean": {
        "defect": "PARTICLE_CONTAMINATION",
        "mechanism": "Elevated humidity promotes moisture absorption and oxidation",
        "action": "Check fab HVAC system and humidity control setpoints",
    },
}

COMPARISON_FEATURES = [
    "pressure_drift", "temp_drift", "pressure_std", "temp_std",
    "flow_std", "power_std", "etch_rate_max", "etch_rate_std",
    "cd_drift", "humidity_mean",
]


def rank_root_causes(
    lot_id: str,
    lot_data: pd.DataFrame,
    n_similar: int = 3,
    top_k: int = 3,
) -> dict:
    """Produce a ranked root-cause report for a single failed lot."""
    available_features = [c for c in COMPARISON_FEATURES if c in lot_data.columns]

    target = lot_data[lot_data["lot_id"] == lot_id]
    if target.empty:
        raise ValueError(f"lot_id '{lot_id}' not found in dataset.")
    target = target.iloc[0]

    passing = lot_data[lot_data["failure_flag"] == 0].copy()
    if passing.empty:
        raise ValueError("No passing lots available for comparison.")

    scaler = StandardScaler()
    all_features = lot_data[available_features].values
    scaler.fit(all_features)

    target_vec = scaler.transform(target[available_features].values.reshape(1, -1))
    passing_vecs = scaler.transform(passing[available_features].values)

    sims = cosine_similarity(target_vec, passing_vecs).flatten()
    top_indices = sims.argsort()[::-1][:n_similar]

    similar_lots = []
    for idx in top_indices:
        row = passing.iloc[idx]
        diffs = {}
        for feat in available_features:
            diffs[feat] = round(target[feat] - row[feat], 3)
        similar_lots.append({
            "lot_id": row["lot_id"],
            "equipment": row.get("equipment_id", "?"),
            "yield": row.get("yield_percent", "?"),
            "similarity": round(float(sims[idx]), 4),
            "diffs": diffs,
        })

    best_pass = passing.iloc[top_indices[0]]

    pop_means = lot_data[available_features].mean()
    pop_stds = lot_data[available_features].std().replace(0, 1)

    deviations = []
    for feat in available_features:
        diff = abs(target[feat] - best_pass[feat])
        z_score = abs(target[feat] - pop_means[feat]) / pop_stds[feat]
        deviations.append({
            "feature": feat,
            "target_value": round(float(target[feat]), 3),
            "best_pass_value": round(float(best_pass[feat]), 3),
            "abs_diff": round(float(diff), 3),
            "z_score": round(float(z_score), 2),
        })

    deviations.sort(key=lambda d: d["abs_diff"], reverse=True)

    root_causes = []
    for rank, dev in enumerate(deviations[:top_k], 1):
        feat = dev["feature"]
        mapping = SENSOR_DEFECT_MAP.get(feat, {
            "defect": "UNKNOWN",
            "mechanism": "No domain mapping available",
            "action": "Investigate manually",
        })
        confidence = min(95, int(50 + dev["z_score"] * 15))
        root_causes.append({
            "rank": rank,
            "feature": feat,
            "confidence": confidence,
            "defect_type": mapping["defect"],
            "mechanism": mapping["mechanism"],
            "action": mapping["action"],
            "target_value": dev["target_value"],
            "best_pass_value": dev["best_pass_value"],
            "z_score": dev["z_score"],
        })

    anomalies = []
    for feat in available_features:
        z = (target[feat] - pop_means[feat]) / pop_stds[feat]
        if abs(z) > 0.5:
            level = "CRITICAL" if abs(z) > 1.5 else "HIGH" if abs(z) > 1.0 else "MODERATE"
            anomalies.append({
                "feature": feat,
                "value": round(float(target[feat]), 3),
                "z_score": round(float(z), 2),
                "level": level,
            })
    anomalies.sort(key=lambda a: abs(a["z_score"]), reverse=True)

    report = {
        "lot_id": lot_id,
        "equipment": target.get("equipment_id", "?"),
        "yield_percent": target.get("yield_percent", "?"),
        "status": target.get("final_status", "?"),
        "anomalies": anomalies,
        "root_causes": root_causes,
        "similar_lots": similar_lots,
    }

    return report


def print_root_cause_report(report: dict):
    """Pretty-print a root-cause report dict as a terminal report."""
    print("\n" + "=" * 70)
    print(f"ROOT CAUSE ANALYSIS: {report['lot_id']}  "
          f"(Yield: {report['yield_percent']}%)")
    print(f"Equipment: {report['equipment']}")
    print(f"Status: {report['status']}")
    print("=" * 70)

    if report["anomalies"]:
        print("\nAnomalies Detected:")
        for a in report["anomalies"]:
            print(f"  - {a['feature']:20s}: {a['value']:.2f}  "
                  f"({a['level']} — {abs(a['z_score']):.1f}σ from mean)")

    if report["root_causes"]:
        print(f"\nProbable Root Causes (ranked):")
        for rc in report["root_causes"]:
            print(f"\n  #{rc['rank']}: {rc['mechanism']}")
            print(f"      Confidence : {rc['confidence']}%")
            print(f"      Feature    : {rc['feature']} = {rc['target_value']} "
                  f"(best pass: {rc['best_pass_value']})")
            print(f"      Defect type: {rc['defect_type']}")
            print(f"      Action     : {rc['action']}")

    if report["similar_lots"]:
        print(f"\nSimilar PASSING lots for comparison:")
        for sl in report["similar_lots"]:
            print(f"  - {sl['lot_id']}  "
                  f"(Equipment: {sl['equipment']}, Yield: {sl['yield']}%, "
                  f"Similarity: {sl['similarity']:.3f})")
            sorted_diffs = sorted(sl["diffs"].items(),
                                  key=lambda x: abs(x[1]), reverse=True)
            for feat, diff in sorted_diffs[:2]:
                print(f"      Δ {feat}: {diff:+.3f}")


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
