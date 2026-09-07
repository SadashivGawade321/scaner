"""
ml/evaluate.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Model Evaluation Pipeline (Week 8)
─────────────────────────────────────────────────────────────────────────────

Loads the trained XGBoost model and runs rigorous evaluation:
  - MAE, RMSE, R² on test set
  - Per-category error analysis
  - Residual distribution
  - Prediction vs actual scatter summary

USAGE
─────
    python ml/evaluate.py
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


def load_model_and_data():
    model_path = os.path.join(MODELS_DIR, "xgboost_v1.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model not found. Run: python ml/train.py first."
        )

    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # Try 500-product dataset first
    for path in [
        os.path.join(ROOT, "data", "raw", "aarogya_products_500.csv"),
        os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv"),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("No dataset found.")

    return model, df


def evaluate(model, df):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    X, _ = build_feature_matrix(df)
    y    = df["food_score"].values.astype(float)

    # Re-create same split as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    _, df_test            = train_test_split(df, test_size=0.15, random_state=42)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    residuals = y_test - y_pred

    print("\n" + "=" * 60)
    print("TEST SET METRICS")
    print("=" * 60)
    print(f"  N test products : {len(y_test)}")
    print(f"  MAE             : {mae:.2f}  (avg error in score points)")
    print(f"  RMSE            : {rmse:.2f}")
    print(f"  R²              : {r2:.3f}  (1.0 = perfect)")
    print(f"  Max error       : {np.abs(residuals).max():.1f}")
    print(f"  % within ±5 pts : {(np.abs(residuals) <= 5).mean()*100:.1f}%")
    print(f"  % within ±10 pts: {(np.abs(residuals) <= 10).mean()*100:.1f}%")

    # Per-category breakdown
    if "category" in df_test.columns:
        print("\nPer-category MAE:")
        df_eval = df_test.copy().reset_index(drop=True)
        X_test_reset = X_test.reset_index(drop=True)
        df_eval["predicted"] = y_pred
        df_eval["actual"]    = y_test
        df_eval["abs_error"] = np.abs(y_pred - y_test)

        cat_mae = df_eval.groupby("category")["abs_error"].agg(["mean","count"])
        cat_mae.columns = ["MAE", "N"]
        cat_mae = cat_mae.sort_values("MAE")
        for cat, row in cat_mae.iterrows():
            print(f"  {cat:<35} MAE={row['MAE']:.2f}  (n={int(row['N'])})")

    # Residual summary
    print("\nResidual Distribution:")
    print(f"  Mean residual  : {residuals.mean():.2f}  (should be ~0)")
    print(f"  Std  residual  : {residuals.std():.2f}")
    print(f"  25th percentile: {np.percentile(residuals, 25):.2f}")
    print(f"  75th percentile: {np.percentile(residuals, 75):.2f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    print("=" * 60)
    print("AAROGYA — Model Evaluation")
    print("=" * 60)
    model, df = load_model_and_data()
    metrics = evaluate(model, df)
    print("\n[EVALUATION OK]")
    return metrics


if __name__ == "__main__":
    main()
