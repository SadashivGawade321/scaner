"""
ml/explain.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — SHAP Explainability Pipeline (Week 9)
─────────────────────────────────────────────────────────────────────────────

Uses SHAP TreeExplainer to:
  1. Compute SHAP values for every product
  2. Build per-product explanation context (for Gemini)
  3. Print a global feature importance summary

USAGE
─────
    python ml/explain.py

OUTPUT
──────
    models/shap_values.pkl    (full SHAP matrix)
    models/feature_names.pkl  (ordered feature list)
"""

import os, sys, pickle
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")

sys.path.insert(0, ROOT)
from ml.features import build_feature_matrix


def load_model():
    import xgboost as xgb
    path = os.path.join(MODELS_DIR, "xgboost_v1.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Model not found. Run: python ml/train.py first.")
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m


def compute_shap(model, X):
    import shap
    print("Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Handle both scalar and array expected_value (newer SHAP versions)
    ev = explainer.expected_value
    base_value  = float(ev[0]) if hasattr(ev, '__len__') else float(ev)
    print(f"  SHAP base value (expected score): {base_value:.2f}")
    return shap_values, base_value, explainer


def build_explanation_context(product_row: pd.Series,
                               shap_row: np.ndarray,
                               feature_names: list,
                               predicted_score: float,
                               base_value: float) -> dict:
    """
    Build a structured explanation context dict for one product.
    This is what gets passed to Gemini for natural language generation.
    """
    shap_dict = dict(zip(feature_names, shap_row))

    # Top positive (strengths) and negative (concerns) features
    sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    strengths = [k for k, v in sorted_shap if v > 0.5][:3]
    concerns  = [k for k, v in sorted_shap if v < -0.5][:3]

    # Map feature names to readable labels
    label_map = {
        "sugar_g_100g":            "sugar",
        "protein_g_100g":          "protein",
        "fiber_g_100g":            "fiber",
        "sodium_mg_100g":          "sodium",
        "saturated_fat_g_100g":    "saturated fat",
        "energy_kcal_100g":        "energy density",
        "trans_fat_g_100g":        "trans fat",
        "added_sugar_g_100g":      "added sugar",
        "processing_level_enc":    "processing level",
        "contains_whole_grain":    "whole grain content",
        "contains_artificial_sweetener": "artificial sweeteners",
        "sugar_to_carbs_ratio":    "sugar-to-carb ratio",
        "fiber_to_carbs_ratio":    "fiber-to-carb ratio",
        "sodium_density":          "sodium density",
        "protein_per_calorie":     "protein density",
        "saturated_fat_ratio":     "saturated fat ratio",
    }

    def clean(lst):
        return [label_map.get(f, f.replace("_", " ")) for f in lst]

    return {
        "product_name":    str(product_row.get("product_name", "Unknown")),
        "category":        str(product_row.get("category", "")),
        "processing_level":str(product_row.get("processing_level", "")),
        "predicted_score": round(float(predicted_score), 1),
        "base_value":      round(base_value, 1),
        "score_version":   "v1.0",
        "strengths":       clean(strengths),
        "concerns":        clean(concerns),
        "shap_top_features": {
            label_map.get(k, k): round(v, 2)
            for k, v in sorted_shap[:6]
        },
    }


def print_global_importance(shap_values, feature_names, top_n=10):
    """Print global feature importance (mean |SHAP|)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])

    print(f"\nGlobal Feature Importance (mean |SHAP|), top {top_n}:")
    print(f"  {'Feature':<35} {'Importance':>12}")
    print("  " + "-" * 49)
    for feat, imp in importance[:top_n]:
        bar = "█" * int(imp / importance[0][1] * 20)
        print(f"  {feat:<35} {imp:>8.3f}  {bar}")


def main():
    print("=" * 60)
    print("AAROGYA — SHAP Explainability")
    print("=" * 60)

    model = load_model()

    # Load dataset
    for path in [
        os.path.join(ROOT, "data", "raw", "aarogya_products_500.csv"),
        os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv"),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("No dataset found.")

    X, _ = build_feature_matrix(df)
    feature_names = X.columns.tolist()

    # Predict
    y_pred = model.predict(X)

    # SHAP
    shap_values, base_value, explainer = compute_shap(model, X)

    # Save SHAP values
    shap_path = os.path.join(MODELS_DIR, "shap_values.pkl")
    fn_path   = os.path.join(MODELS_DIR, "feature_names.pkl")
    with open(shap_path, "wb") as f:
        pickle.dump({"shap_values": shap_values, "base_value": base_value,
                     "feature_names": feature_names}, f)
    with open(fn_path, "wb") as f:
        pickle.dump(feature_names, f)
    print(f"SHAP values saved: {shap_path}")

    # Global importance
    print_global_importance(shap_values, feature_names)

    # Example explanation for one product
    print("\n--- Example: Explanation for Product 1 ---")
    ctx = build_explanation_context(
        df.iloc[0], shap_values[0], feature_names, y_pred[0], base_value
    )
    print(f"  Product   : {ctx['product_name']}")
    print(f"  Score     : {ctx['predicted_score']}/100")
    print(f"  Strengths : {ctx['strengths']}")
    print(f"  Concerns  : {ctx['concerns']}")
    print(f"  Top SHAP  : {ctx['shap_top_features']}")

    print("\n[SHAP EXPLAINABILITY OK]")


if __name__ == "__main__":
    main()
