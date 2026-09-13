"""
Semiconductor Yield Optimization — CLI Pipeline
=================================================
Standalone CLI that runs the full analysis pipeline against the
UCI SECOM dataset (1,567 wafers × 590 sensors).

Usage
-----
    # Full pipeline on real SECOM data (default)
    python src/app.py

    # Quick demo on tiny synthetic wafer_lots (30 samples — NOT for evaluation)
    python src/app.py --demo
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Ensure src/ is on the path so that `analysis.*` imports resolve
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from analysis.data_loader import (
    load_secom_dataset,
    preprocess_data,
    get_feature_columns,
    get_dataset_summary,
)
from analysis.feature_engineering import select_features
from analysis.anomaly_detection import detect_anomalies, compute_correlations
from analysis.predictive_model import train_and_evaluate, predict_sample
from analysis.root_cause_analyzer import analyze_root_causes


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _header(title: str):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(title)
    print('=' * 70)


def _phase(label: str):
    """Print a phase marker."""
    print(f"\n▶ {label}")


# ---------------------------------------------------------------------------
# Full SECOM pipeline
# ---------------------------------------------------------------------------

def run_secom_pipeline(top_k: int = 30):
    """Execute every analysis phase on the real SECOM dataset."""

    print("╔" + "═" * 68 + "╗")
    print("║   SEMICONDUCTOR YIELD OPTIMIZATION PIPELINE                        ║")
    print("║   Real SECOM Dataset · Stratified 5-Fold CV · SMOTE               ║")
    print("╚" + "═" * 68 + "╝")

    # ── Phase 2a: Load & preprocess ─────────────────────────────────────
    _phase("PHASE 2a: Load & Preprocess SECOM Dataset")
    df = load_secom_dataset()
    raw_shape = df.shape
    print(f"  Raw shape: {raw_shape[0]} samples × {raw_shape[1]} columns")

    df = preprocess_data(df)
    summary = get_dataset_summary(df)
    print(f"  Cleaned shape: {df.shape[0]} samples × {df.shape[1]} columns")
    print(f"  Pass / Fail: {summary['pass_count']} / {summary['fail_count']}")
    print(f"  Failure rate: {summary['failure_rate']}%")
    print(f"  Features remaining: {summary['total_features']}")

    # ── Phase 2b: Feature selection ─────────────────────────────────────
    _phase(f"PHASE 2b: Feature Selection (top {top_k})")
    all_features = get_feature_columns(df)
    fs = select_features(df, all_features, top_k=top_k)
    selected = fs["selected_features"]

    print(f"  Original features : {fs['total_original']}")
    print(f"  Dropped low-var   : {fs['dropped_low_variance']}")
    print(f"  Dropped corr>0.95 : {fs['dropped_high_correlation']}")
    print(f"  Selected          : {fs['total_after_selection']}")

    print(f"\n  Top 10 features by |correlation| with failure_flag:")
    for name, val in list(fs["importances"].items())[:10]:
        bar = "█" * int(abs(val) * 60)
        print(f"    {name:20s}  {val:+.4f}  {bar}")

    # ── Phase 3a: Anomaly detection ─────────────────────────────────────
    _phase("PHASE 3a: Anomaly Detection (Isolation Forest)")
    anomaly_result = detect_anomalies(df, selected)
    print(f"  Total samples   : {anomaly_result['total_samples']}")
    print(f"  Anomalies found : {anomaly_result['num_anomalies']}")
    print(f"  Normal          : {anomaly_result['num_normal']}")

    if anomaly_result["anomaly_details"]:
        print(f"\n  Top 3 anomalies:")
        for det in anomaly_result["anomaly_details"][:3]:
            label = "FAIL" if det["failure_flag"] == 1 else "PASS"
            print(f"    Sample {det['index']} ({label}): "
                  f"score={det['anomaly_score']:.4f}")
            for feat in det["top_features"][:3]:
                print(f"      → {feat['feature']}: z={feat['z_score']:.1f}")

    # ── Phase 3c: Sensor-failure correlation ────────────────────────────
    _phase("PHASE 3c: Feature–Failure Correlation")
    corr_result = compute_correlations(df, selected)
    top_corrs = corr_result["correlations"][:10]
    for c in top_corrs:
        bar = "█" * int(c["abs_correlation"] * 40)
        sign = "+" if c["correlation"] > 0 else "−"
        print(f"  {c['feature']:20s}  {sign}{c['abs_correlation']:.4f}  {bar}")

    # ── Phase 4: Predictive model + SHAP ────────────────────────────────
    _phase("PHASE 4: Failure Prediction — Stratified 5-Fold CV + SMOTE")
    model_result = train_and_evaluate(df, selected)

    _header("MODEL EVALUATION RESULTS")
    print(f"  Best model  : {model_result['best_model']}")
    print(f"  CV Accuracy : {model_result['cv_accuracy']:.4f}")
    print(f"  CV Precision: {model_result['cv_precision']:.4f}")
    print(f"  CV Recall   : {model_result['cv_recall']:.4f}")
    print(f"  CV F1 Score : {model_result['cv_f1']:.4f}")
    print(f"  CV AUC-ROC  : {model_result['cv_auc_roc']:.4f}")
    print(f"  Folds       : {model_result['n_folds']}")

    cm = model_result["confusion_matrix"]
    print(f"\n  Confusion Matrix (summed across folds):")
    print(f"                Predicted PASS  Predicted FAIL")
    print(f"    Actual PASS    {cm[0][0]:>8}        {cm[0][1]:>8}")
    print(f"    Actual FAIL    {cm[1][0]:>8}        {cm[1][1]:>8}")

    # Model comparison
    print(f"\n  Model comparison (all tested):")
    for comp in model_result["models_comparison"]:
        print(f"    {comp['model']:25s}  "
              f"F1={comp['f1']:.4f}  "
              f"AUC={comp['auc_roc']:.4f}  "
              f"Acc={comp['accuracy']:.4f}")

    # Feature importances
    if model_result.get("feature_importances"):
        print(f"\n  Top 10 feature importances:")
        for i, (feat, imp) in enumerate(
            list(model_result["feature_importances"].items())[:10]
        ):
            bar = "█" * int(float(imp) * 50)
            print(f"    {feat:20s}  {float(imp):.4f}  {bar}")

    # SHAP summary
    if model_result.get("shap_summary") and not any(
        "error" in s for s in model_result["shap_summary"]
    ):
        print(f"\n  Top 10 SHAP importances (mean |SHAP|):")
        for s in model_result["shap_summary"][:10]:
            bar = "█" * int(s["mean_abs_shap"] * 100)
            print(f"    {s['feature']:20s}  {s['mean_abs_shap']:.6f}  {bar}")

    # ── Phase 5: Root cause analysis ────────────────────────────────────
    _phase("PHASE 5: Root Cause Analysis (cosine similarity + z-scores)")
    rca_result = analyze_root_causes(df, selected, max_failures=5)

    print(f"  Total failures in dataset : {rca_result['total_failures']}")
    print(f"  Analysed (capped)         : {rca_result['analyzed']}")

    for analysis in rca_result["analyses"]:
        print(f"\n  --- Sample {analysis['sample_index']} ---")
        if analysis["root_causes"]:
            print(f"  Probable root causes:")
            for rc in analysis["root_causes"][:3]:
                print(f"    #{rc['rank']}: {rc['feature']} "
                      f"(z={rc['z_score']:.1f}, "
                      f"confidence={rc['confidence']}%, "
                      f"level={rc['level']})")
                print(f"       failed={rc['failed_value']:.4f}  "
                      f"best_pass={rc['best_pass_value']:.4f}")
        if analysis["similar_passing"]:
            best_sim = analysis["similar_passing"][0]
            print(f"  Most similar passing sample: index {best_sim['index']} "
                  f"(similarity={best_sim['similarity']:.4f})")

    # ── Done ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ Pipeline complete — all phases executed on real SECOM data.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Demo mode (tiny synthetic dataset — NOT for evaluation)
# ---------------------------------------------------------------------------

def run_demo_pipeline():
    """Run a quick demo on the small synthetic wafer_lots dataset.

    This mode exists only for fast smoke-testing. The synthetic dataset
    has only ~30 samples, so any metrics are statistically meaningless.
    """
    print("╔" + "═" * 68 + "╗")
    print("║   ⚠️  DEMO MODE                                                    ║")
    print("║   N=30 synthetic samples — results are NOT statistically           ║")
    print("║   meaningful and must NOT be used for evaluation claims.            ║")
    print("╚" + "═" * 68 + "╝")

    # Check if demo data exists
    try:
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
        )
        from analysis.root_cause_analyzer import (
            rank_root_causes,
            print_root_cause_report,
        )
    except ImportError as e:
        print(f"\n❌ Demo mode requires factory lot functions: {e}")
        print("   These functions are only available if the codebase includes")
        print("   the synthetic wafer_lots pipeline. Use the default SECOM mode.")
        return

    raw_dir = _SRC_DIR / "data" / "raw"
    required = ["wafer_lots.csv", "sensor_data.csv",
                "defect_data.csv", "process_parameters.csv"]
    missing = [f for f in required if not (raw_dir / f).exists()]
    if missing:
        print(f"\n❌ Demo data missing: {missing}")
        print(f"   Expected in: {raw_dir}")
        print("   Use the default SECOM mode instead: python src/app.py")
        return

    print("\n⚠️  DEMO MODE: N=30 synthetic samples, results are not "
          "statistically meaningful and must not be used for evaluation claims\n")

    _phase("DEMO Phase 2: Load synthetic wafer lots")
    lot_data, sensors, defects = build_lot_dataset()
    validate_dataset(lot_data)

    _phase("DEMO Phase 3a: Anomaly Detection")
    lot_data = detect_sensor_anomalies(lot_data)
    print_anomaly_report(lot_data)
    print("\n⚠️  DEMO MODE: anomaly counts on N=30 are not meaningful")

    _phase("DEMO Phase 3b: Equipment Performance")
    analyze_equipment_performance(lot_data)

    _phase("DEMO Phase 3c: Sensor–Failure Correlation")
    sensor_failure_correlation(lot_data)

    _phase("DEMO Phase 4: Failure Prediction (simple train/test split)")
    model_artifacts = train_failure_predictor(lot_data)
    explain_failures_shap(model_artifacts, lot_data)
    print("\n⚠️  DEMO MODE: model metrics on N=30 (e.g. 100% accuracy) are "
          "artifacts of extreme overfitting on a tiny sample — do NOT cite them")

    _phase("DEMO Phase 5: Root Cause Analysis")
    failed_lots = lot_data[lot_data["failure_flag"] == 1]["lot_id"].tolist()
    for lot_id in failed_lots[:3]:  # Cap at 3 for demo
        report = rank_root_causes(lot_id, lot_data)
        print_root_cause_report(report)

    print("\n" + "=" * 70)
    print("⚠️  DEMO pipeline complete (N=30 synthetic samples).")
    print("   For real evaluation, run:  python src/app.py  (without --demo)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semiconductor Yield Optimization Pipeline"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run on tiny synthetic wafer_lots (N=30) instead of real SECOM data. "
             "Results are NOT statistically meaningful."
    )
    parser.add_argument(
        "--top-k", type=int, default=30,
        help="Number of top features to select (default: 30, empirically optimal)"
    )
    args = parser.parse_args()

    if args.demo:
        run_demo_pipeline()
    else:
        run_secom_pipeline(top_k=args.top_k)


if __name__ == "__main__":
    main()
