"""
Feature Engineering Module — Phase 2b
=======================================
Generic feature selection for any tabular dataset (SECOM or user-uploaded).

Strategy:
  1. Remove near-zero variance features (< 0.01 threshold).
  2. Remove highly correlated pairs (|r| > 0.95) — keeps the one with
     higher correlation to the target.
  3. Rank remaining features by absolute correlation with the target.
  4. Return top-K features (default 50).
"""

import pandas as pd
import numpy as np


def remove_low_variance(df: pd.DataFrame, feature_cols: list,
                        threshold: float = 0.01) -> list:
    """Drop features whose normalised variance is below *threshold*.

    Normalised variance = std / (|mean| + 1e-8) so that features on
    different scales are comparable.
    """
    kept = []
    for col in feature_cols:
        series = df[col]
        norm_var = series.std() / (abs(series.mean()) + 1e-8)
        if norm_var >= threshold:
            kept.append(col)
    return kept


def remove_high_correlation(df: pd.DataFrame, feature_cols: list,
                            target_col: str = "failure_flag",
                            threshold: float = 0.95) -> list:
    """Drop one feature from each highly-correlated pair.

    For each pair with |r| > *threshold*, the feature with lower
    absolute correlation to the target is dropped.
    """
    if len(feature_cols) < 2:
        return feature_cols

    corr_matrix = df[feature_cols].corr().abs()
    target_corr = df[feature_cols].corrwith(df[target_col]).abs()

    to_drop = set()
    for i in range(len(feature_cols)):
        if feature_cols[i] in to_drop:
            continue
        for j in range(i + 1, len(feature_cols)):
            if feature_cols[j] in to_drop:
                continue
            if corr_matrix.iloc[i, j] > threshold:
                # Drop the one less correlated with target
                if target_corr.get(feature_cols[i], 0) < target_corr.get(feature_cols[j], 0):
                    to_drop.add(feature_cols[i])
                else:
                    to_drop.add(feature_cols[j])

    return [c for c in feature_cols if c not in to_drop]


def select_features(df: pd.DataFrame, feature_cols: list,
                    target_col: str = "failure_flag",
                    top_k: int = 50) -> dict:
    """Run the full feature-selection pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-processed dataset.
    feature_cols : list
        Candidate feature column names.
    target_col : str
        Binary target column.
    top_k : int
        Number of final features to keep.

    Returns
    -------
    dict
        selected_features : list[str]
        importances : dict[str, float]  — absolute correlation with target
        dropped_low_variance : int
        dropped_high_correlation : int
        total_original : int
    """
    original_count = len(feature_cols)

    # Step 1 — Remove low variance
    after_var = remove_low_variance(df, feature_cols)
    dropped_lv = original_count - len(after_var)

    # Step 2 — Remove highly correlated pairs
    after_corr = remove_high_correlation(df, after_var, target_col)
    dropped_hc = len(after_var) - len(after_corr)

    # Step 3 — Rank by correlation with target
    correlations = {}
    for col in after_corr:
        r = df[col].corr(df[target_col])
        if not np.isnan(r):
            correlations[col] = round(float(r), 6)

    # Sort by absolute correlation
    ranked = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top = ranked[:top_k]

    selected = [name for name, _ in top]
    importances = {name: val for name, val in top}

    return {
        "selected_features": selected,
        "importances": importances,
        "dropped_low_variance": dropped_lv,
        "dropped_high_correlation": dropped_hc,
        "total_original": original_count,
        "total_after_selection": len(selected),
    }


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

    from analysis.data_loader import load_secom_dataset, preprocess_data, get_feature_columns

    df = load_secom_dataset()
    df = preprocess_data(df)
    feature_cols = get_feature_columns(df)

    result = select_features(df, feature_cols, top_k=30)
    print(f"Original features : {result['total_original']}")
    print(f"Dropped low-var   : {result['dropped_low_variance']}")
    print(f"Dropped corr>0.95 : {result['dropped_high_correlation']}")
    print(f"Selected features : {result['total_after_selection']}")
    print("\nTop 15 features by |correlation| with failure_flag:")
    for name, val in list(result["importances"].items())[:15]:
        bar = "█" * int(abs(val) * 60)
        print(f"  {name:20s}  {val:+.4f}  {bar}")
