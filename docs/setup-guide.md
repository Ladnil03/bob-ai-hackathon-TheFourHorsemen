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

Verify the benchmark SECOM dataset:
```bash
ls -la src/data/secom/
```
You should see:
- `secom.data` (1,567 wafers × 590 sensors)
- `secom_labels.data` (labels: -1 = PASS, 1 = FAIL)

For demo/quick testing, synthetic data is also available in:
```bash
ls -la src/data/raw/
```
- `wafer_lots.csv`, `sensor_data.csv`, `defect_data.csv`, `process_parameters.csv`

### 5. Run the full pipeline

#### Real SECOM Pipeline (Default & Benchmark):
```bash
python src/app.py
```

Expected Output:
```
╔════════════════════════════════════════════════════════════════════╗
║   SEMICONDUCTOR YIELD OPTIMIZATION PIPELINE                        ║
║   Real SECOM Dataset · Stratified 5-Fold CV · SMOTE               ║
╚════════════════════════════════════════════════════════════════════╝

▶ PHASE 2a: Load & Preprocess SECOM Dataset
  Raw shape: 1567 samples × 590 columns
  Cleaned shape: 1567 samples × 443 columns
  Pass / Fail: 1463 / 104
  Failure rate: 6.64%

▶ PHASE 2b: Feature Selection (top 30)
  Original features : 442
  Selected          : 30

▶ PHASE 3a: Anomaly Detection (Isolation Forest)
  Total samples   : 1567
  Anomalies found : 79

▶ PHASE 3c: Feature–Failure Correlation
  ...

▶ PHASE 4: Failure Prediction — Stratified 5-Fold CV + SMOTE
  Best model  : LogisticRegression
  CV Accuracy : 0.7550
  CV Precision: 0.1537
  CV Recall   : 0.5967
  CV F1 Score : 0.2439
  CV AUC-ROC  : 0.7297

▶ PHASE 5: Root Cause Analysis (cosine similarity + z-scores)
  ...
✅ Pipeline complete — all phases executed on real SECOM data.
```

#### Demo Mode (Synthetic wafer_lots, N=30):
```bash
python src/app.py --demo
```
> **Note**: Uses 30 synthetic samples for quick smoke-testing only. Metrics on this tiny dataset are extreme overfitting artifacts and not statistically meaningful.

### 6. Run the FastAPI Server & Frontend

#### Start the backend API:
```bash
uvicorn src.api:app --reload --port 8000
```
Interactive Swagger docs available at: `http://localhost:8000/docs`

#### Start the React Frontend:
```bash
cd frontend
npm install
npm run dev
```
Dashboard available at: `http://localhost:5173`

## How to Verify It Worked

If you see `✅ Pipeline complete` and no errors → **Success!** ✓

## Troubleshooting

**Error: `ModuleNotFoundError: No module named 'pandas'` (or `sklearn`, `shap`, etc.)**
- Fix: Run `pip install -r requirements.txt`

**Error: `FileNotFoundError: ... secom.data`**
- Fix: Make sure you're in the repo root directory (`bob-ai-hackathon-Thefentasticfour/`) and `src/data/secom/` contains the dataset files.

**Realistic Expectation on Model Metrics:**
- In real semiconductor manufacturing, yield failure rates are typically 5–7% (14:1 class imbalance).
- Standard models without data leakage achieve CV AUC-ROC of ~0.72–0.75 and F1 ~0.24–0.26. Any claim of >95% F1 or 100% accuracy indicates severe overfitting or data leakage on synthetic data.

