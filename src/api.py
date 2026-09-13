"""
Semiconductor Yield Optimization — FastAPI Backend
=====================================================
REST API that exposes the ML pipeline for the React frontend.

Usage:
    cd src
    uvicorn api:app --reload --port 8000
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path

# Ensure src/ is on the path
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse

from analysis.data_loader import (
    load_secom_dataset,
    load_uploaded_csv,
    preprocess_data,
    get_feature_columns,
    get_dataset_summary,
)
from analysis.feature_engineering import select_features
from analysis.anomaly_detection import detect_anomalies, compute_correlations
from analysis.predictive_model import train_and_evaluate, predict_sample
from analysis.root_cause_analyzer import analyze_root_causes


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Semiconductor Yield Optimization API",
    description="AI-powered root cause analysis for semiconductor manufacturing",
    version="2.0.0",
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

state = {
    "df": None,                # Current loaded DataFrame
    "feature_cols": None,      # Selected feature columns
    "pipeline_results": None,  # Full pipeline results
    "model_artifacts": None,   # Trained model (for prediction)
    "dataset_name": None,      # "secom" or uploaded filename
}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "dataset_loaded": state["df"] is not None}


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------

@app.post("/api/dataset/use-secom")
async def use_secom():
    """Load the built-in SECOM dataset."""
    try:
        df = load_secom_dataset()
        df = preprocess_data(df)
        state["df"] = df
        state["feature_cols"] = None
        state["pipeline_results"] = None
        state["model_artifacts"] = None
        state["dataset_name"] = "SECOM (UCI ML Repository)"
        return {
            "message": "SECOM dataset loaded successfully",
            "summary": get_dataset_summary(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...), target_col: str = "failure_flag"):
    """Upload a CSV file and load it as the active dataset."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    try:
        # Save to temp file
        upload_dir = _SRC_DIR / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        df = load_uploaded_csv(str(file_path), target_col=target_col)
        df = preprocess_data(df)
        state["df"] = df
        state["feature_cols"] = None
        state["pipeline_results"] = None
        state["model_artifacts"] = None
        state["dataset_name"] = file.filename

        return {
            "message": f"Dataset '{file.filename}' loaded successfully",
            "summary": get_dataset_summary(df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/info")
async def dataset_info():
    """Get summary of the currently loaded dataset."""
    if state["df"] is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    return {
        "dataset_name": state["dataset_name"],
        "summary": get_dataset_summary(state["df"]),
    }


@app.get("/api/dataset/preview")
async def dataset_preview(rows: int = 10):
    """Get first N rows of the loaded dataset."""
    if state["df"] is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")

    df = state["df"]
    preview = df.head(rows)

    # Convert to JSON-safe format
    records = []
    for _, row in preview.iterrows():
        record = {}
        for col in preview.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                record[col] = round(float(val), 4) if not np.isnan(val) else None
            elif isinstance(val, (int, np.integer)):
                record[col] = int(val)
            else:
                record[col] = str(val) if val is not None else None
        records.append(record)

    return {
        "rows": records,
        "total_rows": len(df),
        "columns": list(preview.columns),
    }


@app.get("/api/dataset/columns")
async def dataset_columns():
    """Get column names and types for the loaded dataset."""
    if state["df"] is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")
    df = state["df"]
    columns = []
    for col in df.columns:
        columns.append({
            "name": col,
            "dtype": str(df[col].dtype),
            "is_numeric": df[col].dtype.kind in "iufb",
            "null_count": int(df[col].isnull().sum()),
        })
    return {"columns": columns}


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

@app.post("/api/pipeline/run")
async def run_pipeline(top_k_features: int = 50):
    """Run the full analysis pipeline on the loaded dataset."""
    if state["df"] is None:
        raise HTTPException(status_code=404, detail="No dataset loaded. Upload a CSV or use SECOM first.")

    df = state["df"]
    all_features = get_feature_columns(df)

    try:
        # Phase 2b — Feature selection
        fs_result = select_features(df, all_features, top_k=top_k_features)
        selected = fs_result["selected_features"]
        state["feature_cols"] = selected

        # Phase 3a — Anomaly detection
        anomaly_result = detect_anomalies(df, selected)

        # Phase 3c — Correlations
        corr_result = compute_correlations(df, selected)

        # Phase 4 — Predictive model + SHAP
        model_result = train_and_evaluate(df, selected)

        # Remove non-serializable objects for the response
        model_artifacts = {
            "model": model_result.pop("model"),
            "scaler": model_result.pop("scaler"),
            "features": model_result.pop("features"),
            "feature_importances": model_result["feature_importances"],
        }
        state["model_artifacts"] = model_artifacts

        # Phase 5 — Root cause analysis
        rca_result = analyze_root_causes(df, selected, max_failures=15)

        # Store results
        pipeline_results = {
            "feature_selection": fs_result,
            "anomaly_detection": anomaly_result,
            "correlations": corr_result,
            "model": model_result,
            "root_cause_analysis": rca_result,
        }
        state["pipeline_results"] = pipeline_results

        return {
            "message": "Pipeline completed successfully",
            "results": pipeline_results,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Results endpoints
# ---------------------------------------------------------------------------

@app.get("/api/results/overview")
async def results_overview():
    """Get a high-level overview of pipeline results."""
    if state["pipeline_results"] is None:
        raise HTTPException(status_code=404, detail="Pipeline not run yet")

    r = state["pipeline_results"]
    return {
        "dataset_name": state["dataset_name"],
        "total_samples": r["anomaly_detection"]["total_samples"],
        "features_selected": r["feature_selection"]["total_after_selection"],
        "anomalies_found": r["anomaly_detection"]["num_anomalies"],
        "cv_accuracy": r["model"]["cv_accuracy"],
        "cv_f1": r["model"]["cv_f1"],
        "cv_auc_roc": r["model"]["cv_auc_roc"],
        "cv_precision": r["model"]["cv_precision"],
        "cv_recall": r["model"]["cv_recall"],
        "total_failures": r["root_cause_analysis"]["total_failures"],
    }


@app.get("/api/results/anomalies")
async def results_anomalies():
    if state["pipeline_results"] is None:
        raise HTTPException(status_code=404, detail="Pipeline not run yet")
    return state["pipeline_results"]["anomaly_detection"]


@app.get("/api/results/model")
async def results_model():
    if state["pipeline_results"] is None:
        raise HTTPException(status_code=404, detail="Pipeline not run yet")
    return state["pipeline_results"]["model"]


@app.get("/api/results/correlations")
async def results_correlations():
    if state["pipeline_results"] is None:
        raise HTTPException(status_code=404, detail="Pipeline not run yet")
    return state["pipeline_results"]["correlations"]


def load_json_result(filename: str):
    if state.get("pipeline_results") is None:
        return None
    key = filename.replace(".json", "")
    return state["pipeline_results"].get(key)


@app.get("/api/results/root-causes")
async def get_root_causes():
    data = load_json_result("root_cause_analysis.json")
    if not data:
        raise HTTPException(status_code=404, detail="Pipeline not run yet")
    return data

class ExplainRequest(BaseModel):
    sample_index: int
    predicted_fail: bool
    root_causes: list

@app.post("/api/results/explain-llm")
async def explain_with_llm(req: ExplainRequest):
    import os
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return {"explanation": "LLM Explanation is unavailable because the GROQ_API_KEY is not set in the environment. Please add it to see natural language insights!"}
        
    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key)
        
        causes_str = ", ".join([f"{c['feature']} (Importance: {c['importance']:.3f})" for c in req.root_causes[:5]])
        prompt = f"You are an expert semiconductor manufacturing engineer. An anomaly was detected for wafer {req.sample_index}. The model predicted a failure with these top contributing root causes: {causes_str}. Write a concise, 2-3 sentence technical explanation of what likely went wrong and what the maintenance team should check."
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise semiconductor engineer assistant."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=150,
        )
        
        return {"explanation": chat_completion.choices[0].message.content}
    except Exception as e:
        return {"explanation": f"LLM Error: {str(e)}"}


@app.get("/api/results/feature-selection")
async def results_feature_selection():
    if state["pipeline_results"] is None:
        raise HTTPException(status_code=404, detail="Pipeline not run yet")
    return state["pipeline_results"]["feature_selection"]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@app.post("/api/predict")
async def predict(sample: dict):
    """Score a single sample for failure risk."""
    if state["model_artifacts"] is None:
        raise HTTPException(status_code=404, detail="Model not trained. Run the pipeline first.")

    try:
        result = predict_sample(sample, state["model_artifacts"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
