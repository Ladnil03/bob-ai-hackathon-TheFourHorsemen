"""
Feature Engineering Module — Phase 2b / 2c
============================================
Converts raw time-series sensor data and per-wafer defect records into
one-row-per-lot feature vectors suitable for ML modelling.

Key engineered features
-----------------------
**Sensor features** (from ``sensor_data.csv``):
- ``pressure_drift`` — strongest single predictor of failures.  High drift
  indicates a pressure transducer or valve problem.
- ``temp_drift``     — large swings correlate with CD shift defects.
- ``cd_drift``       — critical-dimension variation over the run.
- Per-sensor mean / std capture baseline stability.

**Defect features** (from ``defect_data.csv``):
- ``total_defects``      — overall defect burden.
- Per-type counts        — METAL_VOID, CD_SHIFT, GATE_THIN.
- ``avg_severity``       — average severity across wafers with defects.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Phase 2b — Sensor feature extraction
# ---------------------------------------------------------------------------

def extract_sensor_features(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate time-series sensor readings into per-lot statistical features.

    For every ``lot_id`` in *sensor_df* we compute:

    +-----------------------+-----------------------------------------------+
    | Feature               | Why it matters                                |
    +=======================+===============================================+
    | temp_mean / temp_std  | Baseline temperature stability                |
    | **temp_drift**        | Max − min over run; >3 °C ⇒ CD shift risk    |
    | pressure_mean / std   | Baseline chamber pressure stability           |
    | **pressure_drift**    | #1 failure indicator – valve / transducer wear |
    | flow_mean / flow_std  | Gas-flow consistency                          |
    | power_mean / power_std| RF power stability                            |
    | etch_rate_max / std   | Aggressive etching ⇒ gate thinning           |
    | **cd_drift**          | Critical-dimension variation over the run     |
    | humidity_mean         | Ambient moisture (affects oxidation rate)     |
    +-----------------------+-----------------------------------------------+

    Parameters
    ----------
    sensor_df : pd.DataFrame
        Raw sensor readings with columns ``lot_id``, ``timestamp_minutes``,
        ``chamber_temp_C``, ``pressure_pa``, ``flow_rate_sccm``, ``power_w``,
        ``humidity_percent``, ``etch_rate_nm_min``, ``critical_dimension_nm``.

    Returns
    -------
    pd.DataFrame
        One row per lot with all engineered features.
    """
    features = []

    for lot_id, group in sensor_df.groupby("lot_id"):
        group = group.sort_values("timestamp_minutes")

        # --- Temperature features ---
        temp_mean = group["chamber_temp_C"].mean()
        temp_std = group["chamber_temp_C"].std(ddof=0)
        temp_drift = group["chamber_temp_C"].max() - group["chamber_temp_C"].min()

        # --- Pressure features (STRONGEST failure predictor) ---
        pressure_mean = group["pressure_pa"].mean()
        pressure_std = group["pressure_pa"].std(ddof=0)
        pressure_drift = group["pressure_pa"].max() - group["pressure_pa"].min()

        # --- Flow-rate features ---
        flow_mean = group["flow_rate_sccm"].mean()
        flow_std = group["flow_rate_sccm"].std(ddof=0)

        # --- Power features ---
        power_mean = group["power_w"].mean()
        power_std = group["power_w"].std(ddof=0)

        # --- Etch-rate features ---
        etch_rate_max = group["etch_rate_nm_min"].max()
        etch_rate_std = group["etch_rate_nm_min"].std(ddof=0)

        # --- Critical dimension tracking ---
        cd_drift = group["critical_dimension_nm"].max() - group["critical_dimension_nm"].min()

        # --- Humidity (auxiliary) ---
        humidity_mean = group["humidity_percent"].mean()

        features.append({
            "lot_id": lot_id,
            "temp_mean": round(temp_mean, 3),
            "temp_std": round(temp_std, 4),
            "temp_drift": round(temp_drift, 2),          # KEY — >3 °C = risk
            "pressure_mean": round(pressure_mean, 3),
            "pressure_std": round(pressure_std, 4),
            "pressure_drift": round(pressure_drift, 2),  # KEY — #1 predictor
            "flow_mean": round(flow_mean, 3),
            "flow_std": round(flow_std, 4),
            "power_mean": round(power_mean, 3),
            "power_std": round(power_std, 4),
            "etch_rate_max": round(etch_rate_max, 2),
            "etch_rate_std": round(etch_rate_std, 4),
            "cd_drift": round(cd_drift, 3),
            "humidity_mean": round(humidity_mean, 3),
        })

    return pd.DataFrame(features)


# ---------------------------------------------------------------------------
# Phase 2c — Defect feature extraction
# ---------------------------------------------------------------------------

def extract_defect_features(defect_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-wafer defect records into per-lot summary features.

    Computed features:
    - ``total_defects``     — sum of all defect counts across wafers.
    - ``metal_void_count``  — METAL_VOID defects (correlate with pressure).
    - ``cd_shift_count``    — CRITICAL_DIMENSION_SHIFT (correlate with temp).
    - ``gate_thin_count``   — GATE_DIELECTRIC_THIN (correlate with etch rate).
    - ``line_roughness_count`` — LINE_ROUGHNESS defects (minor, cosmetic).
    - ``particle_count``    — PARTICLE_CONTAMINATION defects (handling).
    - ``avg_severity``      — mean severity across wafers that have defects.
    - ``max_severity``      — worst-case severity on any wafer.

    Lots with zero defects receive ``0`` for all counts and severities.

    Parameters
    ----------
    defect_df : pd.DataFrame
        Raw defect inspection data.

    Returns
    -------
    pd.DataFrame
        One row per lot.
    """
    features = []

    for lot_id, group in defect_df.groupby("lot_id"):
        total_defects = group["defect_count"].sum()

        # Per-type counts
        def _type_count(dtype):
            return group.loc[group["defect_type"] == dtype, "defect_count"].sum()

        metal_void_count = _type_count("METAL_VOID")
        cd_shift_count = _type_count("CRITICAL_DIMENSION_SHIFT")
        gate_thin_count = _type_count("GATE_DIELECTRIC_THIN")
        line_roughness_count = _type_count("LINE_ROUGHNESS")
        particle_count = _type_count("PARTICLE_CONTAMINATION")

        # Severity — only consider wafers that actually have defects
        has_defects = group[group["defect_count"] > 0]
        avg_severity = has_defects["severity_1_to_5"].mean() if len(has_defects) > 0 else 0.0
        max_severity = has_defects["severity_1_to_5"].max() if len(has_defects) > 0 else 0

        features.append({
            "lot_id": lot_id,
            "total_defects": int(total_defects),
            "metal_void_count": int(metal_void_count),
            "cd_shift_count": int(cd_shift_count),
            "gate_thin_count": int(gate_thin_count),
            "line_roughness_count": int(line_roughness_count),
            "particle_count": int(particle_count),
            "avg_severity": round(avg_severity, 2) if not np.isnan(avg_severity) else 0.0,
            "max_severity": int(max_severity),
        })

    return pd.DataFrame(features)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from analysis.data_loader import load_raw_data

    _, _, sensors, defects = load_raw_data()

    sf = extract_sensor_features(sensors)
    print("Sensor features shape:", sf.shape)
    print(sf.to_string(index=False))

    print()
    df = extract_defect_features(defects)
    print("Defect features shape:", df.shape)
    print(df.to_string(index=False))
