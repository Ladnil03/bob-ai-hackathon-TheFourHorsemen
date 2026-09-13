"""
Data Loader Module — Phase 2a
==============================
Loads all four raw CSV data sources and merges them into a single
analysis-ready DataFrame keyed on ``lot_id``.

Data sources
------------
- wafer_lots.csv      — lot metadata (equipment, yield, status)
- process_parameters.csv — recipe targets for each lot
- sensor_data.csv     — time-series readings (5-min intervals)
- defect_data.csv     — per-wafer defect inspections

The merge strategy preserves all lots (left joins) so we never silently
drop records.  Sensor and defect data are aggregated to one row per lot
before merging.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_data(data_dir=None):
    """Load the four raw CSV files and return them as individual DataFrames.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing the CSV files.  Defaults to
        ``src/data/raw/`` relative to this module.

    Returns
    -------
    lots, params, sensors, defects : tuple of pd.DataFrame
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    lots = pd.read_csv(data_dir / "wafer_lots.csv")
    params = pd.read_csv(data_dir / "process_parameters.csv")
    sensors = pd.read_csv(data_dir / "sensor_data.csv")
    defects = pd.read_csv(data_dir / "defect_data.csv")

    return lots, params, sensors, defects


def build_lot_dataset(data_dir=None):
    """Build a merged, analysis-ready DataFrame with one row per lot.

    Merge order
    -----------
    1. ``lots`` LEFT JOIN ``params``  → adds recipe target columns.
    2. Aggregate ``sensors``          → one row per lot with drift / mean / std
       features, then LEFT JOIN onto the result.
    3. Aggregate ``defects``          → one row per lot with defect counts and
       severity, then LEFT JOIN onto the result.
    4. Derive ``failure_flag`` target variable from ``final_status``.

    All joins use ``lot_id`` as the key.  Left joins ensure every lot in
    ``wafer_lots.csv`` is preserved even when sensor or defect data is
    missing (e.g., for lots that haven't been inspected yet).

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing the CSV files.

    Returns
    -------
    lot_data : pd.DataFrame
        Merged dataset with all features + ``failure_flag`` target.
    sensors : pd.DataFrame
        Raw sensor time-series (useful for plotting).
    defects : pd.DataFrame
        Raw defect records (useful for spatial analysis).
    """
    lots, params, sensors, defects = load_raw_data(data_dir)

    # ------------------------------------------------------------------
    # Step 1: Merge lots ← process parameters
    # ------------------------------------------------------------------
    # Inner join is fine here because every lot should have parameters.
    # But we use left join defensively so new lots without params still show.
    lot_data = lots.merge(
        params, on="lot_id", how="left", suffixes=("", "_param")
    )
    _validate_no_row_explosion(lots, lot_data, "lots ← params")

    # ------------------------------------------------------------------
    # Step 2: Aggregate sensor time-series → 1 row per lot
    # ------------------------------------------------------------------
    from analysis.feature_engineering import extract_sensor_features

    sensor_features = extract_sensor_features(sensors)
    lot_data = lot_data.merge(
        sensor_features, on="lot_id", how="left"
    )
    _validate_no_row_explosion(lots, lot_data, "lots ← sensor_features")

    # ------------------------------------------------------------------
    # Step 3: Aggregate defect records → 1 row per lot
    # ------------------------------------------------------------------
    from analysis.feature_engineering import extract_defect_features

    defect_features = extract_defect_features(defects)
    lot_data = lot_data.merge(
        defect_features, on="lot_id", how="left"
    )
    _validate_no_row_explosion(lots, lot_data, "lots ← defect_features")

    # ------------------------------------------------------------------
    # Step 4: Fill NaN for lots with no sensor/defect data and add target
    # ------------------------------------------------------------------
    numeric_cols = lot_data.select_dtypes(include=[np.number]).columns
    lot_data[numeric_cols] = lot_data[numeric_cols].fillna(0)

    # Binary classification target
    lot_data["failure_flag"] = (lot_data["final_status"] == "FAIL").astype(int)

    return lot_data, sensors, defects


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_no_row_explosion(original, merged, label):
    """Guard against accidental many-to-many joins doubling the row count."""
    if len(merged) > len(original):
        raise ValueError(
            f"Row explosion after merge '{label}': "
            f"{len(original)} → {len(merged)}.  "
            "Check for duplicate lot_ids in the right-hand table."
        )


def validate_dataset(lot_data):
    """Print a concise validation summary of the merged dataset."""
    print("\n" + "=" * 70)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Shape           : {lot_data.shape[0]} lots × {lot_data.shape[1]} features")
    print(f"Unique lot_ids  : {lot_data['lot_id'].nunique()}")
    print(f"Pass / Fail     : {(lot_data['failure_flag'] == 0).sum()} / "
          f"{(lot_data['failure_flag'] == 1).sum()}")

    null_counts = lot_data.isnull().sum()
    if null_counts.sum() == 0:
        print("Null values     : 0  ✓")
    else:
        print("Null values     :")
        for col, cnt in null_counts[null_counts > 0].items():
            print(f"  {col:30s}: {cnt}")

    print("\nColumn dtypes:")
    for col in lot_data.columns:
        print(f"  {col:35s}: {lot_data[col].dtype}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI entry-point for quick testing
# ---------------------------------------------------------------------------

def main():
    """Load data, merge, validate, and print summary statistics."""
    lot_data, sensors, defects = build_lot_dataset()
    validate_dataset(lot_data)

    # Quick preview
    print("First 5 rows (selected columns):")
    preview_cols = [
        "lot_id", "equipment_id", "yield_percent", "final_status",
        "failure_flag",
    ]
    # Add sensor feature columns if present
    for col in ["pressure_drift", "temp_drift", "total_defects"]:
        if col in lot_data.columns:
            preview_cols.append(col)
    print(lot_data[preview_cols].head().to_string(index=False))


if __name__ == "__main__":
    main()
