"""
Semiconductor Yield Optimization Analysis Package
===================================================
Modules:
    data_loader          — Load & merge raw CSV data (Phase 2a)
    feature_engineering   — Sensor & defect feature extraction (Phase 2b/c)
    anomaly_detection     — Isolation Forest + equipment analysis (Phase 3)
    predictive_model      — Random Forest + SHAP explanations (Phase 4)
    root_cause_analyzer   — Root cause ranking engine (Phase 5)
"""

from analysis.data_loader import build_lot_dataset, load_raw_data
from analysis.feature_engineering import extract_sensor_features, extract_defect_features
from analysis.anomaly_detection import detect_sensor_anomalies, analyze_equipment_performance
from analysis.predictive_model import train_failure_predictor, explain_failures_shap
from analysis.root_cause_analyzer import rank_root_causes

__all__ = [
    "build_lot_dataset",
    "load_raw_data",
    "extract_sensor_features",
    "extract_defect_features",
    "detect_sensor_anomalies",
    "analyze_equipment_performance",
    "train_failure_predictor",
    "explain_failures_shap",
    "rank_root_causes",
]
