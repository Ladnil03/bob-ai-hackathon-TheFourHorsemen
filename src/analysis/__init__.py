"""
Semiconductor Yield Optimization Analysis Package
===================================================
Modules:
    data_loader          — Load SECOM / uploaded CSV data (Phase 2a)
    feature_engineering   — Generic feature selection (Phase 2b)
    anomaly_detection     — Isolation Forest + correlations (Phase 3)
    predictive_model      — Random Forest + SHAP + cross-validation (Phase 4)
    root_cause_analyzer   — Root cause ranking engine (Phase 5)
"""

from analysis.data_loader import (
    load_secom_dataset,
    load_uploaded_csv,
    preprocess_data,
    get_feature_columns,
    get_dataset_summary,
    load_raw_data,
    build_lot_dataset,
    validate_dataset,
)
from analysis.feature_engineering import (
    select_features,
    extract_sensor_features,
    extract_defect_features,
)
from analysis.anomaly_detection import (
    detect_anomalies,
    compute_correlations,
    analyze_groups,
    detect_sensor_anomalies,
    print_anomaly_report,
    analyze_equipment_performance,
    sensor_failure_correlation,
    SENSOR_FEATURE_COLS,
)
from analysis.predictive_model import (
    train_and_evaluate,
    predict_sample,
    train_failure_predictor,
    explain_failures_shap,
    FEATURE_COLS,
)
from analysis.root_cause_analyzer import (
    analyze_root_causes,
    rank_root_causes,
    print_root_cause_report,
    SENSOR_DEFECT_MAP,
    COMPARISON_FEATURES,
)

__all__ = [
    "load_secom_dataset",
    "load_uploaded_csv",
    "preprocess_data",
    "get_feature_columns",
    "get_dataset_summary",
    "load_raw_data",
    "build_lot_dataset",
    "validate_dataset",
    "select_features",
    "extract_sensor_features",
    "extract_defect_features",
    "detect_anomalies",
    "compute_correlations",
    "analyze_groups",
    "detect_sensor_anomalies",
    "print_anomaly_report",
    "analyze_equipment_performance",
    "sensor_failure_correlation",
    "SENSOR_FEATURE_COLS",
    "train_and_evaluate",
    "predict_sample",
    "train_failure_predictor",
    "explain_failures_shap",
    "FEATURE_COLS",
    "analyze_root_causes",
    "rank_root_causes",
    "print_root_cause_report",
    "SENSOR_DEFECT_MAP",
    "COMPARISON_FEATURES",
]
