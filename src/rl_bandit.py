"""
Reinforcement Learning — Contextual Bandit for Semiconductor Recipe Optimization.

Uses Thompson Sampling with Beta priors to learn optimal recipe parameter offsets
based on real-time yield feedback. The bandit treats each context cluster (derived
from sensor readings) as an independent multi-armed bandit over discrete recipe actions.

Architecture:
  - Context: Top-K sensor feature vector → hashed to cluster ID via KMeans/binning
  - Actions: Discrete recipe offset bins for each equipment parameter
  - Reward: Binary yield outcome (PASS=1 / FAIL=0) from downstream metrology
  - Update: Bayesian posterior update on Beta(alpha, beta) per (cluster, action) pair

Designed for sparse-reward, low-frequency industrial control loops (~100-500 lots/day).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger("rl_bandit")

# ---------------------------------------------------------------------------
# Recipe Action Space — discrete offset bins per equipment parameter
# ---------------------------------------------------------------------------
RECIPE_PARAMETERS = [
    {
        "id": "rf_power",
        "name": "RF Power Offset",
        "unit": "W",
        "subsystem": "Plasma RF Generator",
        "bins": [-5.0, -2.5, 0.0, +2.5, +5.0],
        "bin_labels": ["-5W", "-2.5W", "0W (baseline)", "+2.5W", "+5W"],
    },
    {
        "id": "gas_flow_cf4",
        "name": "CF4 Gas Flow Offset",
        "unit": "sccm",
        "subsystem": "MFC Gas Delivery",
        "bins": [-1.0, -0.5, 0.0, +0.5, +1.0],
        "bin_labels": ["-1.0", "-0.5", "0 (baseline)", "+0.5", "+1.0"],
    },
    {
        "id": "chamber_pressure",
        "name": "Chamber Pressure Offset",
        "unit": "mTorr",
        "subsystem": "Throttle Valve / Vacuum",
        "bins": [-3.0, -1.5, 0.0, +1.5, +3.0],
        "bin_labels": ["-3mT", "-1.5mT", "0mT (baseline)", "+1.5mT", "+3mT"],
    },
    {
        "id": "susceptor_temp",
        "name": "Susceptor Temperature Offset",
        "unit": "deg C",
        "subsystem": "Heater Controller",
        "bins": [-2.0, -1.0, 0.0, +1.0, +2.0],
        "bin_labels": ["-2C", "-1C", "0C (baseline)", "+1C", "+2C"],
    },
    {
        "id": "exposure_dose",
        "name": "DUV Exposure Dose Offset",
        "unit": "mJ/cm2",
        "subsystem": "Lithography Scanner",
        "bins": [-0.3, -0.15, 0.0, +0.15, +0.3],
        "bin_labels": ["-0.3", "-0.15", "0 (baseline)", "+0.15", "+0.3"],
    },
]

# Number of context clusters (sensor state buckets)
N_CONTEXT_CLUSTERS = 8


class ContextualBandit:
    """
    Thompson Sampling Contextual Bandit for recipe optimization.

    For each (context_cluster, parameter, action_bin) triple, maintains a
    Beta(alpha, beta) posterior that is updated with binary yield rewards.
    """

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = Path(state_path or os.environ.get(
            "RL_STATE_PATH",
            str(Path(__file__).resolve().parent / "data" / "rl_bandit_state.json")
        ))
        self.parameters = RECIPE_PARAMETERS
        self.n_clusters = N_CONTEXT_CLUSTERS
        self.n_params = len(self.parameters)

        # Initialize posteriors: {cluster_id: {param_id: {bin_idx: (alpha, beta)}}}
        self.posteriors: Dict[str, Dict[str, Dict[int, Tuple[float, float]]]] = {}
        self.history: List[Dict[str, Any]] = []
        self.total_updates = 0
        self.created_at = time.time()

        self._load_state()

    def _default_posteriors(self) -> Dict[str, Dict[str, Dict[int, Tuple[float, float]]]]:
        """Create uniform Beta(1,1) priors for all (cluster, param, bin) triples."""
        posteriors = {}
        for c in range(self.n_clusters):
            ckey = str(c)
            posteriors[ckey] = {}
            for param in self.parameters:
                posteriors[ckey][param["id"]] = {}
                for b in range(len(param["bins"])):
                    posteriors[ckey][param["id"]][str(b)] = [1.0, 1.0]  # Beta(1,1) = uniform
        return posteriors

    def _load_state(self) -> None:
        """Load persisted bandit state from JSON."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.posteriors = data.get("posteriors", self._default_posteriors())
                self.history = data.get("history", [])[-200:]  # Keep last 200
                self.total_updates = data.get("total_updates", 0)
                self.created_at = data.get("created_at", time.time())
                log.info("Loaded RL bandit state from %s (%d updates)", self.state_path, self.total_updates)
            except Exception as e:
                log.warning("Could not load RL state, starting fresh: %s", e)
                self.posteriors = self._default_posteriors()
        else:
            self.posteriors = self._default_posteriors()

    def _save_state(self) -> None:
        """Persist bandit state to JSON."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "posteriors": self.posteriors,
            "history": self.history[-200:],
            "total_updates": self.total_updates,
            "created_at": self.created_at,
            "last_updated": time.time(),
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _context_to_cluster(self, context: List[float]) -> int:
        """
        Hash a sensor context vector to a cluster ID.
        Uses a simple quantile-based binning for robustness.
        """
        if not context:
            return 0
        # Use mean of context features and bin into N clusters
        mean_val = float(np.nanmean(context[:20]))  # Use top-20 features
        # Normalize to [0, N_CLUSTERS-1] using simple hashing
        bucket = int(abs(hash(str(round(mean_val, 2)))) % self.n_clusters)
        return bucket

    def select_action(self, context: List[float]) -> Dict[str, Any]:
        """
        Given current sensor context, use Thompson Sampling to select
        the best recipe offset for each parameter.

        Returns a dict with recommended offsets and confidence scores.
        """
        cluster = self._context_to_cluster(context)
        ckey = str(cluster)

        # Ensure cluster exists
        if ckey not in self.posteriors:
            self.posteriors[ckey] = self._default_posteriors()[str(0)]

        recommendations = []
        for param in self.parameters:
            pid = param["id"]
            n_bins = len(param["bins"])

            # Thompson Sampling: draw from each Beta posterior
            sampled_rewards = []
            for b in range(n_bins):
                bkey = str(b)
                if bkey not in self.posteriors[ckey].get(pid, {}):
                    alpha, beta = 1.0, 1.0
                else:
                    alpha, beta = self.posteriors[ckey][pid][bkey]
                sample = float(np.random.beta(alpha, beta))
                sampled_rewards.append(sample)

            # Select the bin with highest sampled reward
            best_bin = int(np.argmax(sampled_rewards))
            best_value = param["bins"][best_bin]

            # Confidence = concentration of posterior
            alpha_b, beta_b = self.posteriors[ckey][pid][str(best_bin)]
            total_obs = alpha_b + beta_b - 2  # subtract priors
            confidence = min(total_obs / 20.0, 1.0)  # 20 obs = full confidence

            # Expected yield rate for best action
            expected_yield = alpha_b / (alpha_b + beta_b)

            recommendations.append({
                "parameter_id": pid,
                "parameter_name": param["name"],
                "unit": param["unit"],
                "subsystem": param["subsystem"],
                "recommended_offset": best_value,
                "bin_index": best_bin,
                "bin_label": param["bin_labels"][best_bin],
                "confidence": round(confidence, 3),
                "expected_yield_rate": round(expected_yield, 4),
                "sampled_scores": [round(s, 4) for s in sampled_rewards],
                "observations": int(total_obs),
            })

        return {
            "context_cluster": cluster,
            "recommendations": recommendations,
            "total_bandit_updates": self.total_updates,
            "policy_maturity": "Exploring" if self.total_updates < 50 else (
                "Learning" if self.total_updates < 200 else "Converged"
            ),
        }

    def record_outcome(
        self,
        context: List[float],
        actions_taken: Dict[str, int],  # {param_id: bin_index}
        reward: float,  # 1.0 = PASS, 0.0 = FAIL
        wafer_id: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Update the bandit posteriors given the yield outcome.

        Args:
            context: Sensor feature vector at time of production
            actions_taken: Dict mapping param_id -> bin_index that was used
            reward: 1.0 for PASS (yield), 0.0 for FAIL (defect)
            wafer_id: Identifier for audit trail
        """
        cluster = self._context_to_cluster(context)
        ckey = str(cluster)

        if ckey not in self.posteriors:
            self.posteriors[ckey] = self._default_posteriors()[str(0)]

        updates = []
        for pid, bin_idx in actions_taken.items():
            bkey = str(bin_idx)
            if pid in self.posteriors[ckey] and bkey in self.posteriors[ckey][pid]:
                alpha, beta = self.posteriors[ckey][pid][bkey]
                if reward > 0.5:
                    alpha += 1.0  # Success
                else:
                    beta += 1.0   # Failure
                self.posteriors[ckey][pid][bkey] = [alpha, beta]
                updates.append({
                    "param": pid,
                    "bin": bin_idx,
                    "new_alpha": alpha,
                    "new_beta": beta,
                    "expected_yield": round(alpha / (alpha + beta), 4),
                })

        self.total_updates += 1

        # Log to history
        self.history.append({
            "timestamp": time.time(),
            "wafer_id": wafer_id,
            "cluster": cluster,
            "actions": actions_taken,
            "reward": reward,
            "update_num": self.total_updates,
        })

        self._save_state()

        return {
            "status": "updated",
            "cluster": cluster,
            "reward": reward,
            "updates": updates,
            "total_updates": self.total_updates,
        }

    def get_policy_dashboard(self) -> Dict[str, Any]:
        """Return the current learned policy state for UI visualization."""
        policy_summary = []

        for param in self.parameters:
            pid = param["id"]

            # Aggregate across all clusters to find global best action
            global_scores = [0.0] * len(param["bins"])
            total_obs = 0

            for c in range(self.n_clusters):
                ckey = str(c)
                if ckey in self.posteriors and pid in self.posteriors[ckey]:
                    for b in range(len(param["bins"])):
                        bkey = str(b)
                        if bkey in self.posteriors[ckey][pid]:
                            alpha, beta = self.posteriors[ckey][pid][bkey]
                            global_scores[b] += alpha / (alpha + beta)
                            total_obs += (alpha + beta - 2)

            # Normalize
            global_scores = [s / max(self.n_clusters, 1) for s in global_scores]
            best_global = int(np.argmax(global_scores))

            policy_summary.append({
                "parameter_id": pid,
                "parameter_name": param["name"],
                "unit": param["unit"],
                "subsystem": param["subsystem"],
                "bins": param["bins"],
                "bin_labels": param["bin_labels"],
                "expected_yields": [round(s, 4) for s in global_scores],
                "best_action_idx": best_global,
                "best_action_label": param["bin_labels"][best_global],
                "best_action_value": param["bins"][best_global],
                "total_observations": int(total_obs / len(param["bins"])),
            })

        return {
            "policy": policy_summary,
            "total_updates": self.total_updates,
            "policy_maturity": "Exploring" if self.total_updates < 50 else (
                "Learning" if self.total_updates < 200 else "Converged"
            ),
            "n_context_clusters": self.n_clusters,
            "recent_history": self.history[-10:],
            "algorithm": "Thompson Sampling (Beta-Bernoulli Contextual Bandit)",
        }

    def simulate_batch_feedback(self, n_samples: int = 20) -> Dict[str, Any]:
        """
        Simulate a batch of production outcomes for demo/training purposes.
        Uses a simple ground-truth model where baseline (0 offset) has ~93% yield
        and small offsets improve yield while large offsets decrease it.
        """
        results = []
        for i in range(n_samples):
            # Random context
            context = list(np.random.randn(20).astype(float))

            # Select actions from current policy
            suggestion = self.select_action(context)

            # Build action dict
            actions = {}
            total_offset_magnitude = 0
            for rec in suggestion["recommendations"]:
                actions[rec["parameter_id"]] = rec["bin_index"]
                total_offset_magnitude += abs(rec["recommended_offset"])

            # Simulate yield: baseline 93%, small offsets help, large offsets hurt
            # Optimal is near 0 with slight adjustments
            base_yield_prob = 0.934
            offset_penalty = total_offset_magnitude * 0.008  # Small penalty for large offsets
            noise = np.random.normal(0, 0.02)
            yield_prob = np.clip(base_yield_prob - offset_penalty + noise, 0.0, 1.0)
            reward = 1.0 if np.random.random() < yield_prob else 0.0

            # Record
            result = self.record_outcome(
                context=context,
                actions_taken=actions,
                reward=reward,
                wafer_id=f"SIM-{i:04d}",
            )
            results.append({
                "wafer": f"SIM-{i:04d}",
                "reward": reward,
                "yield_prob": round(yield_prob, 4),
            })

        passes = sum(1 for r in results if r["reward"] > 0.5)
        return {
            "simulated": n_samples,
            "passes": passes,
            "fails": n_samples - passes,
            "yield_rate": round(passes / max(n_samples, 1) * 100, 1),
            "total_updates": self.total_updates,
            "results_sample": results[:5],
        }


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
contextual_bandit = ContextualBandit()
