# Semiconductor Yield Optimization with AI

> AI-powered root cause analysis for advanced chip manufacturing using IBM Bob

---

## 👥 Team

| Field | Value |
|---|---|
| **Team Name** | Thefentasticfour |
| **Track** | AI |
| **Team Lead** | Team Lead — lead@example.com |
| **Members** | Manan Panchal,Tirth Bhanderi,Bhakti Ruparel,Nil Lad |

## 🎯 Problem Statement

At advanced semiconductor nodes (3nm/5nm), yield losses of just 1% cost 
$50–100M per month in revenue. When a batch fails, engineers spend weeks 
manually correlating thousands of equipment sensors and defect reports to 
find the root cause. This reactive analysis delays recovery and costs 
millions daily. Our solution aims to proactively identify these root causes.

## 💡 Solution

We built an AI system using IBM Bob that: (1) analyzes real wafer sensor 
data and defect reports, (2) trains a Random Forest to predict which lots 
will fail, (3) explains predictions via SHAP to rank probable root causes, 
and (4) scores upcoming batches before manufacturing. This enables targeted actions to avoid revenue loss.

## ✨ Key Features

- **Real-time anomaly detection**: Isolation Forest flags unusual sensor patterns
- **Failure prediction**: Random Forest classifier (78% accuracy on test set)
- **SHAP explanations**: Root causes ranked by feature importance
- **Batch risk scoring**: Predict batch outcomes before manufacturing
- **Equipment tracking**: Performance degradation detection

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python 3.10+ |
| **ML/Data** | scikit-learn, pandas, numpy, SHAP |
| **IBM** | IBM Bob (code generation & planning) |
| **Visualization** | matplotlib, seaborn |
| **Other** | GitHub Actions, dotenv |

## ⚡ How to Run

```bash
# 1. Clone & setup
git clone https://github.com/Ladnil03/bob-ai-hackathon-Thefentasticfour.git
cd bob-ai-hackathon-Thefentasticfour

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run full pipeline
python src/app.py

# Expected: 6-phase analysis pipeline completes in <30 seconds
```

## 🖥️ Demo

- 📹 **Video**: [Link coming soon]
- 🖼️ **Screenshots**: [See demo/screenshots/](demo/screenshots/)
- 📊 **Live demo**: Not deployed (use local script)

## ✨ What We're Most Proud Of

We identified pressure drift as the #1 yield predictor (r=0.87 correlation 
with failures) and built a production-ready root cause analyzer that ranks 
probable causes with SHAP feature importance scoring. The system catches 
equipment degradation patterns automatically.

## ⚠️ Known Limitations

- Dataset spans 2 months (Jan–Feb 2024) — limited for seasonal patterns
- SHAP explanations currently text-based (could add visualizations)
- No predictive maintenance scheduling (ranked causes only)
- Batch scoring uses historical recipe stats as proxy (not real physics model)
