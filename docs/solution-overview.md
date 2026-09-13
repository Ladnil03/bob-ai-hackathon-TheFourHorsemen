# Solution Overview

## What We Built

A 6-phase AI analysis pipeline that processes wafer data and produces 
actionable root cause insights:

1. **Data Pipeline** — Load wafer sensors + defects from 4 CSV sources
2. **Feature Engineering** — Extract 10 key parameters (pressure, temperature, etch rate, etc.)
3. **Anomaly Detection** — Isolation Forest identifies unusual wafer lots
4. **Sensor Correlation** — Rank which sensor drifts correlate with failures
5. **Predictive Model** — Random Forest predicts failure probability; SHAP explains why
6. **Root Cause Analyzer** — Find similar passing lots, compare parameters, rank probable causes
7. **Batch Risk Scorer** — Score upcoming batches before manufacturing

## Why This Works

**Pressure Drift is the #1 Signal** (r=0.87):
- Historical data shows: when pressure > 135.2 Pa, 90% of batches fail
- Defect signature: METAL_VOID (pressure-related void formation)
- Action: Reduce setpoint by 0.5 Pa

**Temperature Instability** (r=0.79):
- Drift > 3°C causes cascading failures downstream
- Defect signature: CRITICAL_DIMENSION_SHIFT (dies too small)
- Action: Tune PID controller gains

**Multi-modal Root Cause Analysis**:
- Not just "model says 75% failure risk"
- But "HERE'S WHY: pressure high + etch rate high + METAL_VOIDs detected"

## IBM Bob's Role

IBM Bob wrote all analysis code:
- `feature_engineering.py` — sensor aggregation logic
- `predictive_model.py` — Random Forest + SHAP integration
- `root_cause_analyzer.py` — similarity matching + ranking

Bob also helped architect the pipeline and debug issues.
