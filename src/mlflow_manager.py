"""
MLflow Model Registry & Experiment Tracking for Semiconductor Yield Optimization.

Provides local MLflow tracking for:
  - Registering the BestEnsemble model with metrics and hyperparameters
  - Logging RL bandit updates as experiment runs
  - Model versioning and lifecycle management (Staging -> Production)
  - Experiment comparison and history

Runs embedded within the FastAPI app — no separate MLflow server required.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("mlflow_manager")

# Default tracking URI — local filesystem
DEFAULT_TRACKING_URI = str(Path(__file__).resolve().parent / "data" / "mlruns")
DEFAULT_MODEL_NAME = "secom-yield-ensemble"


class MLflowManager:
    """
    Manages MLflow model registry and experiment tracking.
    Falls back to a lightweight JSON-based registry if MLflow is not installed.
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        self.model_name = DEFAULT_MODEL_NAME
        self.mlflow_available = False
        self.mlflow = None

        # JSON-based lightweight registry (always available as fallback)
        self.registry_path = Path(self.tracking_uri) / "model_registry.json"
        self.registry: Dict[str, Any] = {"models": [], "experiments": [], "current_production": None}

        self._init_mlflow()
        self._load_registry()

    def _init_mlflow(self) -> None:
        """Try to initialize MLflow client."""
        try:
            import mlflow
            import mlflow.pyfunc
            self.mlflow = mlflow
            self.mlflow.set_tracking_uri(self.tracking_uri)
            self.mlflow_available = True
            log.info("MLflow initialized with tracking URI: %s", self.tracking_uri)
        except ImportError:
            log.info("MLflow not installed. Using lightweight JSON registry as fallback.")
            self.mlflow_available = False

    def _load_registry(self) -> None:
        """Load JSON registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
            except Exception as e:
                log.warning("Could not load registry: %s", e)

    def _save_registry(self) -> None:
        """Persist JSON registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register_model(
        self,
        metrics: Dict[str, Any],
        hyperparams: Dict[str, Any],
        feature_names: List[str],
        model_path: Optional[str] = None,
        description: str = "Optuna-tuned 4-way ensemble (CatBoost + LightGBM + XGBoost + TabPFN)",
    ) -> Dict[str, Any]:
        """
        Register the current BestEnsemble model with metrics and hyperparameters.
        """
        version = len(self.registry["models"]) + 1
        timestamp = time.time()

        model_record = {
            "version": version,
            "name": self.model_name,
            "description": description,
            "stage": "Staging",
            "metrics": {
                "auc_roc": metrics.get("final_metrics", {}).get("auc_roc"),
                "pr_auc": metrics.get("final_metrics", {}).get("pr_auc"),
                "f1": metrics.get("final_metrics", {}).get("f1"),
                "precision": metrics.get("final_metrics", {}).get("precision"),
                "recall": metrics.get("final_metrics", {}).get("recall"),
                "accuracy": metrics.get("final_metrics", {}).get("accuracy"),
                "threshold": metrics.get("threshold"),
            },
            "hyperparams": hyperparams,
            "feature_count": len(feature_names),
            "ensemble_mode": metrics.get("ensemble_mode", "blend"),
            "model_path": model_path,
            "registered_at": timestamp,
            "registered_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            "tags": {
                "framework": "sklearn+catboost+lgbm+xgb+tabpfn",
                "dataset": "SECOM-UCI",
                "hackathon": "IBM-Bob-AI-2026",
            },
        }

        self.registry["models"].append(model_record)

        # If this is the first model or has better PR-AUC, auto-promote
        if not self.registry["current_production"] or (
            model_record["metrics"].get("pr_auc", 0) >=
            self.registry["current_production"].get("metrics", {}).get("pr_auc", 0)
        ):
            model_record["stage"] = "Production"
            self.registry["current_production"] = model_record

        # Also log to MLflow if available
        if self.mlflow_available:
            try:
                self.mlflow.set_experiment("secom-yield-optimization")
                with self.mlflow.start_run(run_name=f"best-ensemble-v{version}"):
                    # Log metrics
                    for mk, mv in model_record["metrics"].items():
                        if mv is not None:
                            self.mlflow.log_metric(mk, float(mv))

                    # Log params
                    for pk, pv in hyperparams.items():
                        if isinstance(pv, (str, int, float, bool)):
                            self.mlflow.log_param(pk, pv)

                    # Log tags
                    for tk, tv in model_record["tags"].items():
                        self.mlflow.set_tag(tk, tv)

                    self.mlflow.set_tag("version", str(version))
                    self.mlflow.set_tag("stage", model_record["stage"])

                log.info("Registered model v%d in MLflow experiment.", version)
            except Exception as e:
                log.warning("MLflow logging failed (non-critical): %s", e)

        self._save_registry()

        return {
            "status": "registered",
            "version": version,
            "stage": model_record["stage"],
            "model_name": self.model_name,
            "metrics": model_record["metrics"],
        }

    def log_rl_experiment(
        self,
        update_num: int,
        reward: float,
        actions: Dict[str, int],
        cluster: int,
        wafer_id: str = "unknown",
    ) -> Dict[str, Any]:
        """Log an RL bandit update as an experiment run."""
        experiment_record = {
            "type": "rl_bandit_update",
            "update_num": update_num,
            "reward": reward,
            "actions": actions,
            "cluster": cluster,
            "wafer_id": wafer_id,
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        self.registry["experiments"].append(experiment_record)

        # Keep last 500 experiment records
        if len(self.registry["experiments"]) > 500:
            self.registry["experiments"] = self.registry["experiments"][-500:]

        # MLflow logging
        if self.mlflow_available:
            try:
                self.mlflow.set_experiment("secom-rl-recipe-optimization")
                with self.mlflow.start_run(run_name=f"rl-update-{update_num}"):
                    self.mlflow.log_metric("reward", reward)
                    self.mlflow.log_metric("update_num", update_num)
                    self.mlflow.log_metric("cluster", cluster)
                    self.mlflow.set_tag("wafer_id", wafer_id)
            except Exception as e:
                log.debug("MLflow RL logging failed: %s", e)

        self._save_registry()
        return experiment_record

    def promote_model(self, version: int, target_stage: str = "Production") -> Dict[str, Any]:
        """Promote a model version to a new stage (Staging, Production, Archived)."""
        for model in self.registry["models"]:
            if model["version"] == version:
                old_stage = model["stage"]
                model["stage"] = target_stage

                if target_stage == "Production":
                    # Demote current production model
                    if self.registry["current_production"] and self.registry["current_production"]["version"] != version:
                        for m in self.registry["models"]:
                            if m["version"] == self.registry["current_production"]["version"]:
                                m["stage"] = "Archived"
                    self.registry["current_production"] = model

                self._save_registry()
                return {
                    "status": "promoted",
                    "version": version,
                    "old_stage": old_stage,
                    "new_stage": target_stage,
                }

        return {"status": "error", "message": f"Version {version} not found"}

    def get_model_versions(self) -> Dict[str, Any]:
        """List all registered model versions with their stages."""
        return {
            "model_name": self.model_name,
            "total_versions": len(self.registry["models"]),
            "current_production": self.registry.get("current_production"),
            "versions": self.registry["models"],
            "mlflow_available": self.mlflow_available,
            "tracking_uri": self.tracking_uri,
        }

    def get_experiments_summary(self) -> Dict[str, Any]:
        """Get summary of RL experiments and training history."""
        experiments = self.registry.get("experiments", [])
        rl_updates = [e for e in experiments if e.get("type") == "rl_bandit_update"]

        if rl_updates:
            rewards = [e["reward"] for e in rl_updates]
            recent_rewards = rewards[-20:] if len(rewards) >= 20 else rewards
            cumulative_yield = sum(recent_rewards) / len(recent_rewards) * 100 if recent_rewards else 0
        else:
            cumulative_yield = 0

        return {
            "total_experiments": len(experiments),
            "rl_updates": len(rl_updates),
            "recent_yield_rate_pct": round(cumulative_yield, 1),
            "recent_experiments": experiments[-10:],
            "mlflow_available": self.mlflow_available,
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
mlflow_manager = MLflowManager()
