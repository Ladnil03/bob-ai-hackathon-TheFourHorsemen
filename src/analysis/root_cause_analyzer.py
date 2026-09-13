"""
Root Cause Analyzer Module — Phase 5a
=======================================
Given a failed lot, this module:

1. Retrieves the lot's sensor drift profile.
2. Finds the 3 most similar *passing* lots (cosine similarity on scaled
   sensor features).
3. Compares the failed lot against the best-matching passer to rank
   which parameters deviated the most.
4. Maps dominant deviations to likely defect types based on domain
   knowledge (pressure → METAL_VOID, temperature → CD_SHIFT, etc.).
5. Produces an actionable markdown report with ranked root causes and
   recommended corrective actions.

This is the **highest-value** module — it turns raw numbers into
engineering insights that a process engineer can act on immediately.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Domain knowledge: sensor → defect type mapping
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

# Features to compare (must match feature_engineering output)
COMPARISON_FEATURES = [
    "pressure_drift", "temp_drift", "pressure_std", "temp_std",
    "flow_std", "power_std", "etch_rate_max", "etch_rate_std",
    "cd_drift", "humidity_mean",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_root_causes(
    lot_id: str,
    lot_data: pd.DataFrame,
    n_similar: int = 3,
    top_k: int = 3,
) -> dict:
    """Produce a ranked root-cause report for a single failed lot.

    Parameters
    ----------
    lot_id : str
        The lot to analyse (must exist in *lot_data*).
    lot_data : pd.DataFrame
        Full merged dataset.
    n_similar : int
        Number of similar passing lots to retrieve.
    top_k : int
        Number of root causes to report.

    Returns
    -------
    dict
        Structured report with keys: ``lot_id``, ``equipment``, ``yield``,
        ``status``, ``anomalies``, ``root_causes``, ``similar_lots``.
    """
    available_features = [c for c in COMPARISON_FEATURES if c in lot_data.columns]

    # Fetch the target lot
    target = lot_data[lot_data["lot_id"] == lot_id]
    if target.empty:
        raise ValueError(f"lot_id '{lot_id}' not found in dataset.")
    target = target.iloc[0]

    # Separate passing lots
    passing = lot_data[lot_data["failure_flag"] == 0].copy()
    if passing.empty:
        raise ValueError("No passing lots available for comparison.")

    # ------------------------------------------------------------------
    # Step 1 — Find most similar passing lots (cosine similarity)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 2 — Compare target vs. best passing lot
    # ------------------------------------------------------------------
    best_pass = passing.iloc[top_indices[0]]

    # Population stats for z-score context
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

    # ------------------------------------------------------------------
    # Step 3 — Build root causes from top deviations
    # ------------------------------------------------------------------
    root_causes = []
    for rank, dev in enumerate(deviations[:top_k], 1):
        feat = dev["feature"]
        mapping = SENSOR_DEFECT_MAP.get(feat, {
            "defect": "UNKNOWN",
            "mechanism": "No domain mapping available",
            "action": "Investigate manually",
        })
        # Confidence heuristic: higher z-score → higher confidence
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

    # ------------------------------------------------------------------
    # Step 4 — Anomaly summary (features > 1σ from mean)
    # ------------------------------------------------------------------
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

    # Anomalies
    if report["anomalies"]:
        print("\nAnomalies Detected:")
        for a in report["anomalies"]:
            print(f"  - {a['feature']:20s}: {a['value']:.2f}  "
                  f"({a['level']} — {abs(a['z_score']):.1f}σ from mean)")

    # Root causes
    if report["root_causes"]:
        print(f"\nProbable Root Causes (ranked):")
        for rc in report["root_causes"]:
            print(f"\n  #{rc['rank']}: {rc['mechanism']}")
            print(f"      Confidence : {rc['confidence']}%")
            print(f"      Feature    : {rc['feature']} = {rc['target_value']} "
                  f"(best pass: {rc['best_pass_value']})")
            print(f"      Defect type: {rc['defect_type']}")
            print(f"      Action     : {rc['action']}")

    # Similar passing lots
    if report["similar_lots"]:
        print(f"\nSimilar PASSING lots for comparison:")
        for sl in report["similar_lots"]:
            print(f"  - {sl['lot_id']}  "
                  f"(Equipment: {sl['equipment']}, Yield: {sl['yield']}%, "
                  f"Similarity: {sl['similarity']:.3f})")
            # Top 2 biggest differences
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

    from analysis.data_loader import build_lot_dataset

    lot_data, _, _ = build_lot_dataset()

    # Analyse all failed lots
    failed_lots = lot_data[lot_data["failure_flag"] == 1]["lot_id"].tolist()
    print(f"\nFound {len(failed_lots)} failed lots to analyse: {failed_lots}")

    for lot_id in failed_lots:
        report = rank_root_causes(lot_id, lot_data)
        print_root_cause_report(report)


if __name__ == "__main__":
    main()
