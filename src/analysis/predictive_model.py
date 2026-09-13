"""
Predictive Model Module — Phase 4
====================================
Trains multiple classifiers with **Stratified 5-Fold Cross-Validation**
using **SMOTE** for class imbalance, evaluates them, and provides SHAP-based explanations.
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("shap not installed — SHAP explanations will be skipped.")


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "failure_flag",
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Train multiple models with stratified K-Fold CV, SMOTE, and return metrics.

    Returns
    -------
    dict with metrics for the best model and a comparison of all tested models.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state),
        "GradientBoosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=random_state),
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    model_metrics = {m: {"acc": [], "prec": [], "rec": [], "f1": [], "auc": [], "cm": np.zeros((2,2))} for m in models}

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, y_train = X_scaled.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X_scaled.iloc[test_idx], y.iloc[test_idx]

        smote = SMOTE(random_state=random_state)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        for name, clf in models.items():
            clf.fit(X_train_sm, y_train_sm)

            probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
            if not hasattr(clf, "predict_proba"):
                probs = 1 / (1 + np.exp(-probs))

            preds = clf.predict(X_test)

            model_metrics[name]["acc"].append(accuracy_score(y_test, preds))
            model_metrics[name]["prec"].append(precision_score(y_test, preds, zero_division=0))
            model_metrics[name]["rec"].append(recall_score(y_test, preds, zero_division=0))
            model_metrics[name]["f1"].append(f1_score(y_test, preds, zero_division=0))
            try:
                model_metrics[name]["auc"].append(roc_auc_score(y_test, probs))
            except:
                pass
            model_metrics[name]["cm"] += confusion_matrix(y_test, preds, labels=[0, 1])

    comparison = []
    for name in models:
        comparison.append({
            "model": name,
            "accuracy": round(float(np.mean(model_metrics[name]["acc"])), 4),
            "precision": round(float(np.mean(model_metrics[name]["prec"])), 4),
            "recall": round(float(np.mean(model_metrics[name]["rec"])), 4),
            "f1": round(float(np.mean(model_metrics[name]["f1"])), 4),
            "auc_roc": round(float(np.mean(model_metrics[name]["auc"]) if model_metrics[name]["auc"] else 0.0), 4),
            "confusion_matrix": model_metrics[name]["cm"].astype(int).tolist()
        })

    # Select best model based on F1 Score
    best_model_info = max(comparison, key=lambda x: x["f1"])
    best_model_name = best_model_info["model"]
    best_clf = models[best_model_name]

    # Retrain best model on full data for feature importance & SHAP
    smote_full = SMOTE(random_state=random_state)
    X_sm, y_sm = smote_full.fit_resample(X_scaled, y)
    best_clf.fit(X_sm, y_sm)

    # Feature importances
    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
    elif hasattr(best_clf, "coef_"):
        importances = np.abs(best_clf.coef_[0])
    else:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(best_clf, X_scaled, y, n_repeats=5, random_state=random_state)
        importances = r.importances_mean

    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(50)
    feat_imp_dict = {str(k): round(float(v), 6) for k, v in feat_imp.to_dict().items()}

    shap_summary = []
    shap_values_fail = np.zeros(X_scaled.shape)
    if HAS_SHAP:
        try:
            if best_model_name == "RandomForest":
                explainer = shap.TreeExplainer(best_clf)
                shap_values = explainer.shap_values(X_scaled)
                if isinstance(shap_values, list):
                    shap_values_fail = shap_values[1]
                else:
                    shap_values_fail = shap_values
            elif best_model_name == "LogisticRegression":
                explainer = shap.LinearExplainer(best_clf, X_scaled)
                shap_values_fail = explainer.shap_values(X_scaled)
            else:
                explainer = shap.Explainer(best_clf.predict, X_scaled)
                shap_values_fail = explainer(X_scaled).values
                if shap_values_fail.ndim == 3:
                    shap_values_fail = shap_values_fail[:, :, 1]
            
            mean_abs_shap = np.abs(shap_values_fail).mean(axis=0)
            for i, col in enumerate(feature_cols):
                shap_summary.append({
                    "feature": col,
                    "mean_abs_shap": round(float(mean_abs_shap[i]), 6),
                })
            shap_summary.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
            shap_summary = shap_summary[:30]
        except Exception as e:
            shap_summary = [{"error": str(e)}]

    # Full dataset predictions to find some failed cases for details
    probs_full = best_clf.predict_proba(X_scaled)[:, 1] if hasattr(best_clf, "predict_proba") else best_clf.decision_function(X_scaled)
    if not hasattr(best_clf, "predict_proba"):
        probs_full = 1 / (1 + np.exp(-probs_full))
    preds_full = best_clf.predict(X_scaled)
    
    predictions = []
    failed_indices = np.where(preds_full == 1)[0]
    for idx in failed_indices[:20]:
        sv = shap_values_fail[idx]
        top_shap = sorted(zip(feature_cols, sv), key=lambda x: abs(x[1]), reverse=True)[:5]
        predictions.append({
            "index": int(idx),
            "actual": int(y.iloc[idx]),
            "predicted": 1,
            "probability": round(float(probs_full[idx]), 4),
            "shap_details": [{"feature": str(f), "shap_value": round(float(v), 6)} for f, v in top_shap]
        })

    clf_report = {
        "PASS": {"precision": best_model_info["confusion_matrix"][0][0] / max(1, (best_model_info["confusion_matrix"][0][0] + best_model_info["confusion_matrix"][1][0])), 
                 "recall": best_model_info["confusion_matrix"][0][0] / max(1, sum(best_model_info["confusion_matrix"][0])), 
                 "f1-score": 0.0},
        "FAIL": {"precision": best_model_info["precision"], 
                 "recall": best_model_info["recall"], 
                 "f1-score": best_model_info["f1"]}
    }
    clf_report["PASS"]["f1-score"] = 2 * (clf_report["PASS"]["precision"] * clf_report["PASS"]["recall"]) / max(0.001, (clf_report["PASS"]["precision"] + clf_report["PASS"]["recall"]))

    return {
        "best_model": best_model_name,
        "cv_accuracy": best_model_info["accuracy"],
        "cv_precision": best_model_info["precision"],
        "cv_recall": best_model_info["recall"],
        "cv_f1": best_model_info["f1"],
        "cv_auc_roc": best_model_info["auc_roc"],
        "n_folds": n_folds,
        "confusion_matrix": best_model_info["confusion_matrix"],
        "classification_report": clf_report,
        "feature_importances": feat_imp_dict,
        "shap_summary": shap_summary,
        "failed_predictions": predictions,
        "models_comparison": comparison,
        "model": best_clf,
        "scaler": scaler,
        "features": feature_cols,
    }

def predict_sample(sample: dict, model_artifacts: dict) -> dict:
    """Predict failure risk for a single sample."""
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    features = model_artifacts["features"]

    vec = []
    for feat in features:
        vec.append(float(sample.get(feat, 0.0)))

    X = np.array([vec])
    X_scaled = scaler.transform(X)

    proba = float(model.predict_proba(X_scaled)[0][1]) if hasattr(model, "predict_proba") else float(1 / (1 + np.exp(-model.decision_function(X_scaled)[0])))
    risk_pct = round(proba * 100, 1)

    if risk_pct < 25:
        classification = "LOW"
    elif risk_pct < 50:
        classification = "MEDIUM"
    elif risk_pct < 75:
        classification = "HIGH"
    else:
        classification = "CRITICAL"

    imps = model_artifacts.get("feature_importances", {})
    top_feats = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "risk_score": risk_pct,
        "classification": classification,
        "probability": round(proba, 4),
        "top_features": [{"feature": f, "importance": v} for f, v in top_feats],
    }

def main():
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

    from analysis.data_loader import load_secom_dataset, preprocess_data, get_feature_columns
    from analysis.feature_engineering import select_features

    df = load_secom_dataset()
    df = preprocess_data(df)
    feature_cols = get_feature_columns(df)

    fs = select_features(df, feature_cols, top_k=40)
    selected = fs["selected_features"]

    result = train_and_evaluate(df, selected)

    print(f"Best Model   : {result['best_model']}")
    print(f"CV Accuracy  : {result['cv_accuracy']}")
    print(f"CV Precision : {result['cv_precision']}")
    print(f"CV Recall    : {result['cv_recall']}")
    print(f"CV F1        : {result['cv_f1']}")
    print(f"CV AUC-ROC   : {result['cv_auc_roc']}")
    print(f"Confusion Matrix: {result['confusion_matrix']}")

if __name__ == "__main__":
    main()
