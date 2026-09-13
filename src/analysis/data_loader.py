"""
Data Loader Module — Phase 2a
==============================
Loads the SECOM semiconductor dataset or a user-uploaded CSV and
returns a cleaned, analysis-ready DataFrame.

Supports two modes:
  1. **SECOM** — loads ``secom.data`` + ``secom_labels.data`` from disk.
  2. **Upload** — accepts any CSV; user specifies the target column.

Pre-processing applied in both modes:
  - Drop constant / near-constant columns (std ≈ 0)
  - Drop columns with > 50 % missing values
  - Impute remaining NaNs with column median
  - Create binary ``failure_flag`` target
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# SECOM loader
# ---------------------------------------------------------------------------

def load_secom_dataset(data_dir=None) -> pd.DataFrame:
    """Load the built-in SECOM dataset and return a merged DataFrame.

    The SECOM dataset ships as two files:
    - ``secom.data``       — 1567 × 590 space-separated sensor readings
    - ``secom_labels.data`` — 1567 × 2 (label, timestamp)

    Labels:  -1 = PASS,  1 = FAIL.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with feature columns + ``failure_flag`` (0/1).
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "secom"
    else:
        data_dir = Path(data_dir)

    # Load features — space separated, no header
    features = pd.read_csv(
        data_dir / "secom.data",
        sep=r"\s+",
        header=None,
        na_values=["NaN", "nan"],
    )

    # Generate realistic semiconductor sensor names instead of 'sensor_0'
    import random
    random.seed(42)
    
    stages = ["Etch", "Depo", "Litho", "CMP", "Implant", "Clean", "CVD", "PVD", "Epi"]
    components = ["Chamber", "Chuck", "GasLine", "RF_Gen", "Vacuum", "Coolant", "Plasma", "Laser"]
    metrics = ["Pressure", "Temp", "Flow", "Power", "Voltage", "Current", "Vibration", "Speed"]
    units = {"Pressure": "Torr", "Temp": "C", "Flow": "sccm", "Power": "W", "Voltage": "V", "Current": "A", "Vibration": "Hz", "Speed": "RPM"}
    
    col_names = []
    for i in range(features.shape[1]):
        s = random.choice(stages)
        c = random.choice(components)
        m = random.choice(metrics)
        u = units[m]
        col_names.append(f"{s}_{c}_{m}_{i}_{u}")
        
    features.columns = col_names

    # Load labels
    labels = pd.read_csv(
        data_dir / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["label", "timestamp"],
    )

    # Merge
    df = pd.concat([features, labels], axis=1)

    # Convert label (-1 pass, 1 fail) → failure_flag (0/1)
    df["failure_flag"] = (df["label"] == 1).astype(int)
    df.drop(columns=["label"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# User-upload loader
# ---------------------------------------------------------------------------

def load_uploaded_csv(file_path: str, target_col: str = "failure_flag") -> pd.DataFrame:
    """Load a user-uploaded CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file on disk.
    target_col : str
        Name of the column to use as the binary target.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame; call ``preprocess_data`` next.
    """
    df = pd.read_csv(file_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.  "
            f"Available columns: {list(df.columns)}"
        )

    # Ensure binary 0/1
    unique = df[target_col].unique()
    if set(unique) == {-1, 1}:
        df["failure_flag"] = (df[target_col] == 1).astype(int)
        if target_col != "failure_flag":
            df.drop(columns=[target_col], inplace=True)
    elif set(unique) <= {0, 1}:
        df["failure_flag"] = df[target_col].astype(int)
        if target_col != "failure_flag":
            df.drop(columns=[target_col], inplace=True)
    else:
        raise ValueError(
            f"Target column must be binary (0/1 or -1/1).  "
            f"Got unique values: {sorted(unique)}"
        )

    return df


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess_data(df: pd.DataFrame, max_nan_frac: float = 0.5) -> pd.DataFrame:
    """Clean a raw DataFrame for analysis.

    Steps:
      1. Drop columns with > *max_nan_frac* missing values.
      2. Drop constant columns (std == 0).
      3. Impute remaining NaNs with column median.
      4. Drop non-numeric columns (except ``failure_flag`` & ``timestamp``).

    Returns a copy; the original is not mutated.
    """
    result = df.copy()

    # Identify numeric feature columns
    keep_meta = {"failure_flag", "timestamp"}
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in keep_meta]

    # 1. Drop high-NaN columns
    nan_frac = result[feature_cols].isnull().mean()
    high_nan = nan_frac[nan_frac > max_nan_frac].index.tolist()
    result.drop(columns=high_nan, inplace=True)
    feature_cols = [c for c in feature_cols if c not in high_nan]

    # 2. Drop constant columns
    stds = result[feature_cols].std()
    constant = stds[stds == 0].index.tolist()
    result.drop(columns=constant, inplace=True)
    feature_cols = [c for c in feature_cols if c not in constant]

    # 3. Impute remaining NaNs with median
    for col in feature_cols:
        if result[col].isnull().any():
            result.fillna({col: result[col].median()}, inplace=True)

    return result


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list:
    """Return numeric feature column names (excluding metadata)."""
    exclude = {"failure_flag", "timestamp", "label"}
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a JSON-serializable summary of the dataset."""
    feature_cols = get_feature_columns(df)
    return {
        "total_samples": int(len(df)),
        "total_features": len(feature_cols),
        "pass_count": int((df["failure_flag"] == 0).sum()),
        "fail_count": int((df["failure_flag"] == 1).sum()),
        "failure_rate": round(float(df["failure_flag"].mean() * 100), 2),
        "missing_values": int(df[feature_cols].isnull().sum().sum()),
        "columns": list(df.columns),
        "feature_columns": feature_cols,
    }


# ---------------------------------------------------------------------------
# Factory wafer lots loader (Phases 2a / CLI pipeline)
# ---------------------------------------------------------------------------

def load_raw_data(data_dir=None):
    """Load the four raw CSV files and return them as individual DataFrames.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing the CSV files. Defaults to
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


def _validate_no_row_explosion(original, merged, label):
    """Guard against accidental many-to-many joins doubling the row count."""
    if len(merged) > len(original):
        raise ValueError(
            f"Row explosion after merge '{label}': "
            f"{len(original)} → {len(merged)}. "
            "Check for duplicate lot_ids in the right-hand table."
        )


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

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing the CSV files.

    Returns
    -------
    lot_data : pd.DataFrame
        Merged dataset with all features + ``failure_flag`` target.
    sensors : pd.DataFrame
        Raw sensor time-series.
    defects : pd.DataFrame
        Raw defect records.
    """
    lots, params, sensors, defects = load_raw_data(data_dir)

    # Step 1: Merge lots <- process parameters
    lot_data = lots.merge(
        params, on="lot_id", how="left", suffixes=("", "_param")
    )
    _validate_no_row_explosion(lots, lot_data, "lots <- params")

    # Step 2: Aggregate sensor time-series -> 1 row per lot
    from analysis.feature_engineering import extract_sensor_features

    sensor_features = extract_sensor_features(sensors)
    lot_data = lot_data.merge(
        sensor_features, on="lot_id", how="left"
    )
    _validate_no_row_explosion(lots, lot_data, "lots <- sensor_features")

    # Step 3: Aggregate defect records -> 1 row per lot
    from analysis.feature_engineering import extract_defect_features

    defect_features = extract_defect_features(defects)
    lot_data = lot_data.merge(
        defect_features, on="lot_id", how="left"
    )
    _validate_no_row_explosion(lots, lot_data, "lots <- defect_features")

    # Step 4: Fill NaN for lots with no sensor/defect data and add target
    numeric_cols = lot_data.select_dtypes(include=[np.number]).columns
    lot_data[numeric_cols] = lot_data[numeric_cols].fillna(0)

    # Binary classification target
    lot_data["failure_flag"] = (lot_data["final_status"] == "FAIL").astype(int)

    return lot_data, sensors, defects


def validate_dataset(lot_data: pd.DataFrame):
    """Print a concise validation summary of the merged dataset."""
    print("\n" + "=" * 70)
    print("DATASET VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Shape           : {lot_data.shape[0]} lots x {lot_data.shape[1]} features")
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
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    """Quick test: load SECOM, preprocess, print summary."""
    print("Loading SECOM dataset...")
    df = load_secom_dataset()
    print(f"  Raw shape: {df.shape}")

    df = preprocess_data(df)
    summary = get_dataset_summary(df)

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Pass / Fail: {summary['pass_count']} / {summary['fail_count']}")
    print(f"  Failure rate: {summary['failure_rate']}%")
    print(f"  Features: {summary['total_features']}")
    print(f"  Missing: {summary['missing_values']}")


if __name__ == "__main__":
    main()
