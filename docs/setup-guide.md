# Setup Guide

## Prerequisites

- Python 3.10 or higher
- Git
- Bash/terminal (Windows: PowerShell or CMD)

## Step-by-Step Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ladnil03/bob-ai-hackathon-Thefentasticfour.git
cd bob-ai-hackathon-Thefentasticfour
```

### 2. Create a virtual environment

On macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Expected output: "Successfully installed pandas, numpy, scikit-learn, ..."

### 4. Verify data is in place

```bash
ls -la src/data/raw/
```

You should see:
- wafer_lots.csv
- sensor_data.csv
- defect_data.csv
- process_parameters.csv

### 5. Run the full pipeline

```bash
python src/app.py
```

### Expected Output

The program should complete in 10–30 seconds and print:

```
╔════════════════════════════════════════════════════════════════╗
║   SEMICONDUCTOR YIELD OPTIMIZATION PIPELINE                    ║
║   Phases 2–6: Data → Features → Anomalies → Prediction → Action║
╚════════════════════════════════════════════════════════════════╝

▶ PHASE 2: Data Pipeline & Feature Engineering
  Loaded 30 lots with 47 features...

▶ PHASE 3a: Anomaly Detection
  Detected 5 anomalous lots...

[... more phases ...]

✅ Pipeline complete.
```

## How to Verify It Worked

If you see "Pipeline complete" and no errors → **Success!** ✓

## Troubleshooting

**Error: `ModuleNotFoundError: No module named 'pandas'`**
- Fix: Run `pip install -r requirements.txt` again

**Error: `FileNotFoundError: [Errno 2] No such file or directory: 'src/data/raw/wafer_lots.csv'`**
- Fix: Make sure you're in the repo root directory (`bob-ai-hackathon-Thefentasticfour/`)

**Error: `ImportError: No module named 'shap'`**
- Fix: Run `pip install shap`

**Slow execution (>60 seconds)**
- Normal on older machines or when running for first time
- Subsequent runs may cache, making them faster
