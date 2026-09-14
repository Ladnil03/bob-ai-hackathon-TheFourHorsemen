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

from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse, FileResponse
from best_model_service import best_model_service
from groq_advisor import groq_advisor
from batch_risk_analyzer import batch_analyzer
from rl_bandit import contextual_bandit
from mlflow_manager import mlflow_manager

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

# Production: Serve built React frontend from src/static/
_STATIC_DIR = _SRC_DIR / "static"
if _STATIC_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import FileResponse as StarletteFileResponse

    @app.get("/")
    async def serve_root():
        return StarletteFileResponse(str(_STATIC_DIR / "index.html"))

    # Mount static assets (JS, CSS, images) AFTER API routes are defined
    # This is done at module bottom via startup event
    @app.on_event("startup")
    async def mount_static():
        app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


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
            model="openai/gpt-oss-120b",
            temperature=0.3,
            max_tokens=300,
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
# Best Model (SOTA Ensemble) Endpoints
# ---------------------------------------------------------------------------

class BestPredictRequest(BaseModel):
    features: dict
    threshold: Optional[float] = None

@app.get("/api/best-model/status")
async def best_model_status():
    """Check readiness and summary of the Optuna-tuned Best Model."""
    return {
        "ready": best_model_service.is_ready,
        "overview": best_model_service.get_overview_metrics(),
    }

@app.get("/api/best-model/metrics")
async def best_model_metrics():
    """Retrieve full metrics, comparison table, and hyperparameters for Best Model."""
    return best_model_service.get_overview_metrics()

@app.get("/api/best-model/curves")
async def best_model_curves():
    """Serve the generated Precision-Recall and ROC curve image."""
    if best_model_service.curves_path and best_model_service.curves_path.exists():
        return FileResponse(best_model_service.curves_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Curves image not found")

@app.get("/api/best-model/sample-wafers")
async def best_model_sample_wafers():
    """Get pre-extracted real wafers (pass, fail, borderline) for 1-click testing."""
    return {"samples": best_model_service.get_sample_wafers()}

@app.post("/api/best-model/predict")
async def best_model_predict(payload: BestPredictRequest):
    """Run inference with the Best Model ensemble and get root cause attribution."""
    if not best_model_service.is_ready:
        raise HTTPException(status_code=503, detail="Best Model is not loaded")
    try:
        res = best_model_service.predict_sample(payload.features, payload.threshold)
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/best-model/batch-predict")
async def best_model_batch_predict(threshold: Optional[float] = None):
    """Batch predict on the active loaded dataset using the Best Model."""
    if not best_model_service.is_ready:
        raise HTTPException(status_code=503, detail="Best Model is not loaded")
    if state["df"] is None:
        raise HTTPException(status_code=404, detail="No dataset loaded")

    df = state["df"]
    feature_cols = [c for c in df.columns if c not in ("target", "failure_flag", "Pass/Fail", "timestamp", "time")]
    rows_to_score = df[feature_cols].to_dict(orient="records")

    total = len(rows_to_score)
    predictions = []
    fail_count = 0
    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}

    for i, row in enumerate(rows_to_score):
        p_res = best_model_service.predict_sample(row, threshold)
        pred_fail = p_res["predicted_failure"]
        if pred_fail:
            fail_count += 1
        r_level = p_res["risk_level"]
        risk_dist[r_level] = risk_dist.get(r_level, 0) + 1

        if i < 50:  # store sample predictions for display
            predictions.append({
                "sample_index": i,
                "prob": p_res["failure_probability"],
                "prob_pct": p_res["failure_probability_pct"],
                "predicted_fail": pred_fail,
                "risk_level": r_level,
                "top_cause": p_res["root_causes"][0]["feature"] if p_res["root_causes"] else "N/A",
            })

    return {
        "total_wafers": total,
        "predicted_defects": fail_count,
        "predicted_yield_pct": round((1.0 - (fail_count / max(total, 1))) * 100.0, 2),
        "risk_distribution": risk_dist,
        "sample_preview": predictions,
        "calibrated_threshold": threshold or best_model_service.model.threshold,
    }

class PrescribeRequest(BaseModel):
    wafer_id: str = "Wafer-Candidate"
    defect_prob_pct: float
    threshold: float = 0.1518
    risk_level: str = "HIGH"
    root_causes: List[dict] = []

@app.post("/api/best-model/prescribe")
async def best_model_prescribe(payload: PrescribeRequest):
    """Use Groq-powered AI to explain wafer root causes and prescribe corrective actions."""
    try:
        advice = groq_advisor.generate_prescriptive_actions(
            wafer_id=payload.wafer_id,
            defect_prob_pct=payload.defect_prob_pct,
            threshold=payload.threshold,
            risk_level=payload.risk_level,
            root_causes=payload.root_causes,
        )
        return advice
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/best-model/batch-preflight")
async def best_model_batch_preflight(
    batch_id: str = "LOT-2026-B09",
    chamber_id: str = "Chamber-Etch-04B",
    recipe_name: str = "3nm-Gate-Etch-v4.2"
):
    """Evaluate upcoming production lot parameters before processing to prevent yield drop."""
    # If a dataset is loaded, sample 25 wafers from it to represent the upcoming lot
    records = []
    if state["df"] is not None:
        feature_cols = [c for c in state["df"].columns if c not in ("target", "failure_flag", "Pass/Fail", "timestamp", "time")]
        records = state["df"][feature_cols].head(25).to_dict(orient="records")
    else:
        # Generate representative upcoming lot from demo wafers
        samples = best_model_service.get_sample_wafers()
        records = [s["features"] for s in samples] * 6  # 24 wafers

    result = batch_analyzer.evaluate_upcoming_batch(
        batch_id=batch_id,
        wafer_records=records,
        chamber_id=chamber_id,
        recipe_name=recipe_name,
    )
    return result


# ---------------------------------------------------------------------------
# Reinforcement Learning — Contextual Bandit Routes
# ---------------------------------------------------------------------------

class RLSuggestRequest(BaseModel):
    context: List[float] = []  # Sensor feature vector (top-K features)

class RLRecordRequest(BaseModel):
    context: List[float] = []
    actions_taken: Dict[str, int] = {}  # {param_id: bin_index}
    reward: float = 1.0  # 1.0=PASS, 0.0=FAIL
    wafer_id: str = "unknown"

@app.post("/api/rl/suggest-recipe-offset")
async def rl_suggest_recipe(payload: RLSuggestRequest):
    """Given current sensor readings, suggest optimal recipe parameter offsets using Thompson Sampling."""
    try:
        result = contextual_bandit.select_action(payload.context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rl/record-outcome")
async def rl_record_outcome(payload: RLRecordRequest):
    """Record yield outcome for a previously suggested recipe offset to update the bandit policy."""
    try:
        result = contextual_bandit.record_outcome(
            context=payload.context,
            actions_taken=payload.actions_taken,
            reward=payload.reward,
            wafer_id=payload.wafer_id,
        )
        # Also log to MLflow
        mlflow_manager.log_rl_experiment(
            update_num=result["total_updates"],
            reward=payload.reward,
            actions=payload.actions_taken,
            cluster=result["cluster"],
            wafer_id=payload.wafer_id,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rl/policy-dashboard")
async def rl_policy_dashboard():
    """Get the current RL bandit policy state and learned recipe offsets."""
    return contextual_bandit.get_policy_dashboard()

@app.post("/api/rl/simulate-batch")
async def rl_simulate_batch(n_samples: int = 20):
    """Simulate a batch of production outcomes for demo/training purposes."""
    try:
        result = contextual_bandit.simulate_batch_feedback(n_samples)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# MLflow Model Registry Routes
# ---------------------------------------------------------------------------

@app.post("/api/mlflow/register-current")
async def mlflow_register_current():
    """Register the currently loaded BestEnsemble model into MLflow model registry."""
    if not best_model_service.is_ready:
        raise HTTPException(status_code=503, detail="Best Model not loaded")
    try:
        result = mlflow_manager.register_model(
            metrics=best_model_service.metrics,
            hyperparams=best_model_service.metrics.get("best_params", {}),
            feature_names=best_model_service.feature_names,
            model_path=str(best_model_service.artifact_dir / "best_model.pkl"),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mlflow/model-versions")
async def mlflow_model_versions():
    """List all registered model versions with stage info."""
    return mlflow_manager.get_model_versions()

class PromoteRequest(BaseModel):
    version: int
    stage: str = "Production"

@app.post("/api/mlflow/promote")
async def mlflow_promote(payload: PromoteRequest):
    """Promote a model version to Production/Staging/Archived."""
    return mlflow_manager.promote_model(payload.version, payload.stage)

@app.get("/api/mlflow/experiments")
async def mlflow_experiments():
    """Get summary of RL experiments and training history."""
    return mlflow_manager.get_experiments_summary()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
