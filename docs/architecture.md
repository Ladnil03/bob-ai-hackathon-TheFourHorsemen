# System Architecture

## Data Flow Diagram

```mermaid
flowchart TD
    subgraph Raw Data
        A[wafer_lots.csv]
        B[process_parameters.csv]
        C[sensor_data.csv]
        D[defect_data.csv]
    end

    A & B & C & D --> E[Data Loader Phase 2a]
    E --> F[Merged Dataset]
    F --> G[Feature Engineering Phase 2b]
    
    G --> H[Engineered Features DataFrame]
    
    H --> I[Anomaly Detection Phase 3a]
    I --> J[Flag 5 anomalous lots]
    
    H --> K[Equipment Analysis Phase 3b]
    K --> L[Rank equipment by avg yield]
    
    H --> M[Sensor Correlation Phase 3c]
    M --> N[Rank features by failure signal strength]
    
    H --> O[Predictive Model Phase 4]
    O --> P[Random Forest classifier]
    O --> Q[SHAP explainer]
    P & Q --> R[Model Output]
    
    R --> S[Root Cause Analyzer Phase 5]
    S --> T[Find similar passing lots]
    S --> U[Rank probable causes + confidence]
    
    R --> V[Batch Risk Scorer Phase 6]
    V --> W[Compute risk score 0-100%]
    W --> X[Output LOW / MEDIUM / HIGH risk]
```

## Key Modules

| Module | Purpose | LOC |
|--------|---------|-----|
| `data_loader.py` | Load & merge 4 CSVs | 192 |
| `feature_engineering.py` | Sensor aggregation, feature extraction | 195 |
| `anomaly_detection.py` | Isolation Forest, correlation analysis | 247 |
| `predictive_model.py` | Random Forest, SHAP, model training | 260 |
| `root_cause_analyzer.py` | Root cause ranking, explanations | 300 |
| `app.py` | Pipeline orchestration, batch scoring | 305 |

## Data Transformations

1. **Raw Sensors** (180 rows of 5-min readings)
   → Aggregate by lot
   → **10 engineered features** (drift, std, max values)

2. **Defects** (150 defect records)
   → Count by type per lot
   → **5 aggregated defect features**

3. **Parameters** (30 recipe rows)
   → Merge on lot_id
   → **12 target parameters**

4. **Final Dataset**: 30 rows × 47 features
