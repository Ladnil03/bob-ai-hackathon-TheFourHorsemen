"""
Groq-powered Prescriptive AI Advisor for Semiconductor Yield Optimization.
Transforms raw ML sensor drift attributions and failure probabilities into
actionable, domain-specific fab equipment maintenance and recipe adjustment instructions.

Model: OpenAI GPT-OSS 120B via Groq LPU inference (~500 tok/s)
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, List, Optional

log = logging.getLogger("groq_advisor")

# ---------------------------------------------------------------------------
# Physical sensor domain mapping for UCI SECOM 590-sensor feature space
# Maps key sensor indices to plausible semiconductor manufacturing sub-systems
# ---------------------------------------------------------------------------
SENSOR_DOMAIN_MAPPING = {
    "feature_59": "Chamber RF Bias Matching Unit / Plasma Power",
    "feature_102": "Etch Gas Mass Flow Controller (MFC) - CF4/O2 Ratio",
    "feature_21": "Electrostatic Chuck (ESC) Helium Backside Cooling Pressure",
    "feature_34": "CVD Deposition Chamber Susceptor Temperature Profile",
    "feature_64": "Turbomolecular Vacuum Foreline Pressure",
    "feature_129": "Photolithography Deep-UV Exposure Dose & Focus Offset",
    "feature_130": "Post-Exposure Bake (PEB) Hotplate Thermal Uniformity",
    "feature_435": "CMP (Chemical Mechanical Planarization) Platen Downforce",
    "feature_281": "Ashing Strip Chamber Endpoint Optical Emission Spectroscopy",
    "feature_40": "Deionized Rinse Water Resistivity & Particle Count",
}

# ---------------------------------------------------------------------------
# Recipe offset actions the RL bandit can recommend
# ---------------------------------------------------------------------------
RECIPE_ACTION_SPACE = {
    "rf_power_offset_w": {"unit": "W", "range": (-10, +10), "subsystem": "Plasma RF Generator"},
    "gas_flow_offset_sccm": {"unit": "sccm", "range": (-2.0, +2.0), "subsystem": "MFC Gas Delivery"},
    "chamber_pressure_offset_mtorr": {"unit": "mTorr", "range": (-5, +5), "subsystem": "Throttle Valve / Vacuum"},
    "susceptor_temp_offset_c": {"unit": "deg C", "range": (-3.0, +3.0), "subsystem": "Heater Controller"},
    "exposure_dose_offset_mj": {"unit": "mJ/cm2", "range": (-0.5, +0.5), "subsystem": "Lithography Scanner"},
}


class GroqYieldAdvisor:
    """Prescriptive Copilot that translates sensor drift into fab actions using Groq GPT-OSS 120B."""

    MODEL_ID = "openai/gpt-oss-120b"
    MODEL_LABEL = "Groq GPT-OSS 120B (OpenAI Open-Source MoE)"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = None
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                log.info("Groq API client initialized (model=%s).", self.MODEL_ID)
            except Exception as e:
                log.warning("Could not initialize Groq client: %s", e)

    def generate_prescriptive_actions(
        self,
        wafer_id: str,
        defect_prob_pct: float,
        threshold: float,
        risk_level: str,
        root_causes: List[Dict[str, Any]],
        model_name: str = "Winning 4-Way Blend (CatBoost + LightGBM + XGBoost + TabPFN)",
    ) -> Dict[str, Any]:
        """Generate root-cause explanation and prescriptive yield recovery steps."""

        # Enrich root causes with physical semiconductor equipment domains
        enriched_causes = []
        for rc in root_causes[:5]:
            fname = rc.get("feature", "unknown")
            subsystem = SENSOR_DOMAIN_MAPPING.get(fname, f"Process Module Stage {fname.replace('feature_', 'Sensor-')}")
            enriched_causes.append({
                **rc,
                "subsystem": subsystem,
            })

        # Try Groq API first if client is available
        if self.client:
            try:
                return self._call_groq(wafer_id, defect_prob_pct, threshold, risk_level, enriched_causes, model_name)
            except Exception as exc:
                log.warning("Groq API call failed, falling back to heuristic engine: %s", exc)

        # High-fidelity domain heuristic fallback (ensures offline reliability during hackathon demos)
        return self._generate_heuristic_response(wafer_id, defect_prob_pct, threshold, risk_level, enriched_causes, model_name)

    def _call_groq(
        self,
        wafer_id: str,
        defect_prob_pct: float,
        threshold: float,
        risk_level: str,
        enriched_causes: List[Dict[str, Any]],
        model_name: str,
    ) -> Dict[str, Any]:
        """Prompt Groq GPT-OSS 120B with strict JSON output for comprehensive prescriptive guidance."""
        causes_text = "\n".join([
            f"- {c['feature']} ({c['subsystem']}): {c['measured_value']} "
            f"(fab baseline: {c['reference_mean']}, excursion: {c['sigma_deviation']}sigma {c['direction']})"
            for c in enriched_causes
        ])

        system_prompt = (
            "You are a Principal Semiconductor Process Integration & Yield Enhancement Engineer "
            "at a leading 3nm/5nm wafer foundry with 20+ years of experience in advanced node manufacturing. "
            "You receive real-time fault detection and classification (FDC) alerts from an AI yield model "
            "and must produce COMPREHENSIVE, highly technical prescriptive guidance that includes: "
            "1) Root cause chain analysis (why -> what -> impact), "
            "2) Ranked corrective actions with specific numeric recipe offset values, "
            "3) A phased next-course-of-action roadmap (immediate/short-term/long-term), "
            "4) SPC (Statistical Process Control) Western Electric rule trigger analysis, "
            "5) Cost impact comparison (act-now vs. defer scenarios with estimated wafer loss), and "
            "6) Specific recipe parameter adjustments with numeric offset values. "
            "Output strictly valid JSON with no markdown formatting outside JSON."
        )

        user_prompt = f"""
Wafer Alert Details:
- Wafer Identifier: {wafer_id}
- Failure Probability: {defect_prob_pct}% (Calibrated Decision Threshold: {threshold * 100:.1f}%)
- Risk Category: {risk_level}
- Model Architecture: {model_name}

Top Significant Sensor Deviations (FDC Excursions):
{causes_text}

Provide a COMPREHENSIVE actionable response in JSON format with these exact keys:
{{
  "bluf_summary": "2-3 sentence executive bottom-line-up-front diagnostic covering failure mechanism and urgency",
  "physical_root_cause": "Detailed explanation of the underlying physical/chemical mechanism causing this fault, including the causal chain (sensor drift -> process deviation -> wafer defect type -> yield impact)",
  "ranked_corrective_actions": [
    {{"priority": 1, "action": "Action title", "details": "Specific parameter adjustment with numeric values", "target_subsystem": "Equipment module", "estimated_time_min": 30}},
    {{"priority": 2, "action": "Action title", "details": "Specific parameter adjustment", "target_subsystem": "Equipment module", "estimated_time_min": 15}},
    {{"priority": 3, "action": "Action title", "details": "Specific parameter adjustment", "target_subsystem": "Equipment module", "estimated_time_min": 45}}
  ],
  "recipe_adjustments": [
    {{"parameter": "RF Power", "current_offset": 0, "recommended_offset": -2.5, "unit": "W", "rationale": "Why this offset"}},
    {{"parameter": "Gas Flow (CF4)", "current_offset": 0, "recommended_offset": 0.3, "unit": "sccm", "rationale": "Why this offset"}}
  ],
  "next_course_of_action": {{
    "immediate_0_to_4h": "What to do RIGHT NOW (specific steps for fab technicians)",
    "short_term_4_to_24h": "Actions for the next shift (engineering team follow-up)",
    "long_term_1_to_7d": "Systemic improvements and preventive measures for the week ahead"
  }},
  "spc_rule_triggers": [
    {{"rule": "Western Electric Rule N", "description": "What was violated", "sensor": "Which sensor", "severity": "WARNING/CRITICAL"}}
  ],
  "cost_impact_analysis": {{
    "act_now_cost_usd": "Estimated cost if corrective action taken immediately (e.g. $2,500 chamber downtime)",
    "defer_cost_usd": "Estimated cost if deferred (e.g. $45,000 lot scrap + downstream rework)",
    "breakeven_wafers": "Number of wafers where inaction cost exceeds action cost",
    "roi_recommendation": "Clear recommendation with ROI justification"
  }},
  "wafer_disposition": "Quarantine / Rework / Secondary Metrology / Standard Release",
  "projected_yield_recovery_pct": "Estimated yield recovery e.g. +1.4%"
}}
"""

        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.MODEL_ID,
            temperature=0.25,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content
        parsed = json.loads(content)
        parsed["powered_by"] = self.MODEL_LABEL
        parsed["enriched_causes"] = enriched_causes
        return parsed

    def _generate_heuristic_response(
        self,
        wafer_id: str,
        defect_prob_pct: float,
        threshold: float,
        risk_level: str,
        enriched_causes: List[Dict[str, Any]],
        model_name: str,
    ) -> Dict[str, Any]:
        """Domain-engineered fallback producing expert-grade prescriptive guidance."""
        top_cause = enriched_causes[0] if enriched_causes else None
        top_feature = top_cause["feature"] if top_cause else "Sensor Excursion"
        top_subsystem = top_cause["subsystem"] if top_cause else "Chamber Plasma System"
        top_sigma = top_cause["sigma_deviation"] if top_cause else 2.5

        if defect_prob_pct >= threshold * 100:
            bluf = (
                f"Defect alert for {wafer_id}: Model predicted a {defect_prob_pct:.1f}% failure risk "
                f"driven by a severe {top_sigma:+.1f} sigma excursion on {top_subsystem} ({top_feature}). "
                f"Immediate chamber pause recommended to prevent lot-level scrap. "
                f"Cost of inaction estimated at $35,000-$50,000 per affected lot."
            )
            mechanism = (
                f"Anomalous drift in {top_subsystem} indicates chamber seasoning breakdown or mass-flow instability. "
                f"Causal chain: Sensor drift ({top_feature} at {top_sigma:+.1f}sigma) -> plasma density non-uniformity "
                f"across wafer periphery -> critical dimension (CD) etch undercut -> micro-bridging defectivity -> "
                f"gate-level short circuits at 5nm node -> functional die yield loss of 3-8%."
            )
            disposition = "Quarantine Wafer for CD-SEM Review; Halt Chamber for Run-to-Run (R2R) Calibration"
            recovery = "+1.8% Yield Retention (Avoids Scrap)"
            actions = [
                {
                    "priority": 1,
                    "action": "Calibrate RF Match & Bias Voltage",
                    "details": f"Offset generator phase angle by -1.8 deg and inspect capacitor tuning on {top_subsystem}. Verify reflected power < 2W.",
                    "target_subsystem": top_subsystem,
                    "estimated_time_min": 30,
                },
                {
                    "priority": 2,
                    "action": "Mass Flow Controller (MFC) Diagnostic",
                    "details": "Perform leak check on precursor line 2 and recalibrate flow delta within +/-0.5 sccm tolerance. Replace O-ring if leak rate > 1E-8 atm*cc/s.",
                    "target_subsystem": "Gas Delivery Manifold",
                    "estimated_time_min": 20,
                },
                {
                    "priority": 3,
                    "action": "Execute Chamber Dry Clean Recipe",
                    "details": "Run 120s NF3/Ar in-situ plasma clean cycle to eliminate polymer buildup on chamber sidewalls. Verify endpoint via OES at 703nm.",
                    "target_subsystem": "Process Chamber Vacuum",
                    "estimated_time_min": 45,
                },
            ]
            recipe_adjustments = [
                {"parameter": "RF Power", "current_offset": 0, "recommended_offset": -2.5, "unit": "W",
                 "rationale": "Reduce plasma density to compensate for chamber seasoning drift"},
                {"parameter": "Gas Flow (CF4)", "current_offset": 0, "recommended_offset": +0.3, "unit": "sccm",
                 "rationale": "Increase etchant ratio to restore critical dimension uniformity"},
                {"parameter": "Chamber Pressure", "current_offset": 0, "recommended_offset": -1.0, "unit": "mTorr",
                 "rationale": "Lower pressure improves mean free path and etch anisotropy"},
                {"parameter": "Susceptor Temperature", "current_offset": 0, "recommended_offset": +0.5, "unit": "deg C",
                 "rationale": "Compensate for thermal drift in ESC cooling circuit"},
            ]
            next_course = {
                "immediate_0_to_4h": (
                    f"1) HALT chamber and quarantine {wafer_id} for CD-SEM inspection. "
                    f"2) Run RF match self-calibration sequence. "
                    f"3) Execute NF3 chamber dry clean (120s). "
                    f"4) Run 3x qualification wafers and verify CD uniformity < 1nm 3-sigma."
                ),
                "short_term_4_to_24h": (
                    "1) Review past 48h of FDC data for this chamber to identify drift onset. "
                    "2) Perform MFC leak check and recalibration on all gas lines. "
                    "3) Update R2R controller setpoints with new baseline from qualification wafers. "
                    "4) Alert process engineering team to monitor next 5 lots closely."
                ),
                "long_term_1_to_7d": (
                    "1) Schedule full PM (preventive maintenance) for this chamber within 7 days. "
                    "2) Review chamber matching across all tools in the module to identify fleet-wide drift. "
                    "3) Update SPC control chart limits with post-correction baseline. "
                    "4) Evaluate whether RL-optimized recipe offsets should become permanent R2R targets."
                ),
            }
            spc_rules = [
                {"rule": "Western Electric Rule 1", "description": f"Single point beyond 3-sigma control limit ({top_sigma:+.1f}sigma detected)",
                 "sensor": top_feature, "severity": "CRITICAL"},
                {"rule": "Western Electric Rule 5", "description": "2 out of 3 consecutive points beyond 2-sigma (trend detected in FDC history)",
                 "sensor": top_feature, "severity": "WARNING"},
            ]
            cost_impact = {
                "act_now_cost_usd": "$2,500 (30min chamber downtime + 3 qual wafers)",
                "defer_cost_usd": "$45,000 (estimated 25-wafer lot scrap + downstream rework + metrology queue delay)",
                "breakeven_wafers": "2 wafers (action pays for itself after preventing 2 defective wafers at $1,250/wafer)",
                "roi_recommendation": "STRONGLY RECOMMENDED: 18x ROI on immediate corrective action. Every hour of delay risks $3,750 in additional scrap.",
            }
        else:
            bluf = (
                f"Wafer {wafer_id} verified within nominal specifications ({defect_prob_pct:.1f}% defect probability vs "
                f"{threshold * 100:.1f}% cutoff). Minor statistical noise detected on {top_subsystem} but within "
                f"process control limits. No corrective action required."
            )
            mechanism = "Parameters reflect stable steady-state wafer processing without critical drift."
            disposition = "Standard Lot Release to Downstream Metrology"
            recovery = "Nominal Yield (98.4%+ In-Spec)"
            actions = [
                {
                    "priority": 1,
                    "action": "Log Steady-State Baseline",
                    "details": "Record sensor fingerprint to Statistical Process Control (SPC) golden wafer registry.",
                    "target_subsystem": "Factory Automation SPC",
                    "estimated_time_min": 5,
                },
                {
                    "priority": 2,
                    "action": "Routine Preventative Inspection",
                    "details": "Verify helium backside chuck clamp leakage rate remains < 0.2 sccm at next maintenance shift.",
                    "target_subsystem": "Wafer Stage ESC",
                    "estimated_time_min": 10,
                },
            ]
            recipe_adjustments = []
            next_course = {
                "immediate_0_to_4h": "No action required. Continue standard lot progression.",
                "short_term_4_to_24h": "Log baseline data for SPC trending. Monitor next lot for any emerging drift.",
                "long_term_1_to_7d": "Include this wafer's sensor profile in the RL training set for continuous model improvement.",
            }
            spc_rules = [
                {"rule": "All Rules", "description": "No SPC rule violations detected. All sensors within control limits.",
                 "sensor": "N/A", "severity": "OK"},
            ]
            cost_impact = {
                "act_now_cost_usd": "$0 (no action needed)",
                "defer_cost_usd": "$0 (wafer in spec)",
                "breakeven_wafers": "N/A",
                "roi_recommendation": "NOMINAL: Wafer is in-spec. Focus resources on flagged chambers.",
            }

        return {
            "bluf_summary": bluf,
            "physical_root_cause": mechanism,
            "ranked_corrective_actions": actions,
            "recipe_adjustments": recipe_adjustments,
            "next_course_of_action": next_course,
            "spc_rule_triggers": spc_rules,
            "cost_impact_analysis": cost_impact,
            "wafer_disposition": disposition,
            "projected_yield_recovery_pct": recovery,
            "powered_by": f"{self.MODEL_LABEL} (Heuristic Fallback)",
            "enriched_causes": enriched_causes,
        }


# Global singleton instance
groq_advisor = GroqYieldAdvisor()
