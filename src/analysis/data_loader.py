"""
Data Loader Module
Loads raw CSV data sources from src/data/raw/ and merges them on lot_id.
"""

from pathlib import Path
import pandas as pd


def load_wafer_data(data_dir=None):
    """Load wafer lots, process parameters, sensors, and defects data."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    lots = pd.read_csv(data_dir / "wafer_lots.csv")
    params = pd.read_csv(data_dir / "process_parameters.csv")
    sensors = pd.read_csv(data_dir / "sensor_data.csv")
    defects = pd.read_csv(data_dir / "defect_data.csv")

    return lots, params, sensors, defects


if __name__ == "__main__":
    lots, params, sensors, defects = load_wafer_data()
    print(f"Loaded lots: {lots.shape}, params: {params.shape}, sensors: {sensors.shape}, defects: {defects.shape}")
