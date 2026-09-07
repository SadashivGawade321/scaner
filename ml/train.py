"""
ml/train.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — ML Training Pipeline (Week 7)
─────────────────────────────────────────────────────────────────────────────

Trains three models:
  1. Linear Regression  (baseline)
  2. Random Forest      (ensemble baseline)
  3. XGBoost            (primary model)

Logs results to console and saves best model to models/.

USAGE
─────
    python ml/train.py

OUTPUT
──────
    models/xgboost_v1.json
    models/linear_regression_v1.pkl
    models/random_forest_v1.pkl
    models/training_summary.txt
"""

import os, sys, json, pickle
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(ROOT, "models")
DATASET_500 = os.path.join(ROOT, "data", "raw", "aarogya_products_500.csv")
DATASET_100 = os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv")

os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from ml.features import build_feature_matrix


def load_dataset():
    """Load the largest available dataset."""
    if os.path.exists(DATASET_500):
        df = pd.read_csv(DATASET_500)
        print(f"Using 500-product dataset: {len(df)} rows")
    elif os.path.exists(DATASET_100):
        df = pd.read_csv(DATASET_100)
        print(f"Using 100-product demo dataset: {len(df)} rows")
        print("  NOTE: 100 products is insufficient for robust ML — run generate_large_dataset.py")
    else:
        raise FileNotFoundError("No dataset found. Run data/generate_large_dataset.py first.")
    return df


def train_evaluate_all(X_train, X_test, y_train, y_test, feature_names):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb

    results = {}

    # ── 1. Linear Regression (baseline) ──────────────────────────────────
    print("\n[1/3] Linear Regression (baseline)...")
    X_train_imp = X_train.fillna(X_train.median())
    X_test_imp  = X_test.fillna(X_train.median())

    lr = LinearRegression()
    lr.fit(X_train_imp, y_train)
    y_pred_lr = lr.predict(X_test_imp)

    results["linear_regression"] = {
        "model": lr,
        "mae":   float(mean_absolute_error(y_test, y_pred_lr)),
        "rmse":  float(np.sqrt(mean_squared_error(y_test, y_pred_lr))),
        "r2":    float(r2_score(y_test, y_pred_lr)),
    }
    print(f"   MAE={results['linear_regression']['mae']:.2f}  "
          f"RMSE={results['linear_regression']['rmse']:.2f}  "
          f"R²={results['linear_regression']['r2']:.3f}")

    # Save
    with open(os.path.join(MODELS_DIR, "linear_regression_v1.pkl"), "wb") as f:
        pickle.dump(lr, f)

    # ── 2. Random Forest ──────────────────────────────────────────────────
    print("[2/3] Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10,
                               random_state=42, n_jobs=-1)
    rf.fit(X_train_imp, y_train)
    y_pred_rf = rf.predict(X_test_imp)

    results["random_forest"] = {
        "model": rf,
        "mae":   float(mean_absolute_error(y_test, y_pred_rf)),
        "rmse":  float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
        "r2":    float(r2_score(y_test, y_pred_rf)),
    }
    print(f"   MAE={results['random_forest']['mae']:.2f}  "
          f"RMSE={results['random_forest']['rmse']:.2f}  "
          f"R²={results['random_forest']['r2']:.3f}")

    with open(os.path.join(MODELS_DIR, "random_forest_v1.pkl"), "wb") as f:
        pickle.dump(rf, f)

    # ── 3. XGBoost (primary) ──────────────────────────────────────────────
    print("[3/3] XGBoost (primary model)...")
    xgb_model = xgb.XGBRegressor(
        n_estimators     = 300,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        reg_alpha        = 0.1,    # L1
        reg_lambda       = 1.0,    # L2
        random_state     = 42,
        n_jobs           = -1,
        eval_metric      = "rmse",
    )
    eval_set = [(X_test, y_test)]
    xgb_model.fit(X_train, y_train,
                  eval_set=eval_set,
                  verbose=False)

    y_pred_xgb = xgb_model.predict(X_test)

    results["xgboost"] = {
        "model": xgb_model,
        "mae":   float(mean_absolute_error(y_test, y_pred_xgb)),
        "rmse":  float(np.sqrt(mean_squared_error(y_test, y_pred_xgb))),
        "r2":    float(r2_score(y_test, y_pred_xgb)),
    }
    print(f"   MAE={results['xgboost']['mae']:.2f}  "
          f"RMSE={results['xgboost']['rmse']:.2f}  "
          f"R²={results['xgboost']['r2']:.3f}")

    # Save XGBoost (JSON format — portable)
    xgb_model.save_model(os.path.join(MODELS_DIR, "xgboost_v1.json"))
    print(f"   Saved: models/xgboost_v1.json")

    return results


def cross_validate_xgb(X, y):
    """5-fold cross-validation on full dataset."""
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, mean_absolute_error
    import xgboost as xgb

    print("\nRunning 5-fold cross-validation on XGBoost...")
    model = xgb.XGBRegressor(n_estimators=200, max_depth=5,
                              learning_rate=0.05, random_state=42, n_jobs=-1)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    scores = cross_val_score(model, X, y, cv=5, scoring=mae_scorer)
    cv_mae = -scores.mean()
    cv_std = scores.std()
    print(f"   CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
    return cv_mae, cv_std


def save_summary(results, cv_mae, cv_std, n_train, n_test):
    """Save training results to a text file."""
    lines = [
        "AAROGYA — Training Summary",
        "=" * 50,
        f"Training samples : {n_train}",
        f"Test samples     : {n_test}",
        "",
        "Model Performance on Test Set:",
        "-" * 50,
    ]
    for name, r in results.items():
        lines += [
            f"  {name}:",
            f"    MAE  = {r['mae']:.2f}",
            f"    RMSE = {r['rmse']:.2f}",
            f"    R²   = {r['r2']:.3f}",
            "",
        ]
    lines += [
        f"XGBoost 5-fold CV MAE: {cv_mae:.2f} ± {cv_std:.2f}",
        "",
        "Best Model: XGBoost",
        "Saved to: models/xgboost_v1.json",
        "",
        "IMPORTANT: Model trained on SYNTHETIC data.",
        "Performance will improve with real product data.",
        "Label circularity: model learns to imitate scoring rules.",
    ]
    path = os.path.join(MODELS_DIR, "training_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nTraining summary saved: {path}")


def main():
    print("=" * 60)
    print("AAROGYA — ML Training Pipeline")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    df = load_dataset()
    print(f"Loaded {len(df)} products.")

    # ── Feature engineering ────────────────────────────────────────────────
    print("\nBuilding feature matrix...")
    X, encoders = build_feature_matrix(df)
    y = df["food_score"].values.astype(float)

    print(f"Feature matrix: {X.shape[0]} x {X.shape[1]}")
    print(f"Target (food_score): min={y.min():.1f}, max={y.max():.1f}, mean={y.mean():.1f}")

    # Save encoders for inference
    encoder_path = os.path.join(MODELS_DIR, "encoders.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(encoders, f)
    print(f"Encoders saved: {encoder_path}")

    # ── Train/test split ───────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")

    # ── Train all models ───────────────────────────────────────────────────
    print("\n--- Training Models ---")
    results = train_evaluate_all(X_train, X_test, y_train, y_test, X.columns.tolist())

    # ── Cross-validation ───────────────────────────────────────────────────
    cv_mae, cv_std = cross_validate_xgb(X, y)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — Best Model: XGBoost")
    print(f"  Test MAE  = {results['xgboost']['mae']:.2f} score points")
    print(f"  Test RMSE = {results['xgboost']['rmse']:.2f}")
    print(f"  Test R²   = {results['xgboost']['r2']:.3f}")
    print(f"  CV MAE    = {cv_mae:.2f} ± {cv_std:.2f}")
    print("=" * 60)

    save_summary(results, cv_mae, cv_std, len(X_train), len(X_test))
    print("\n[TRAINING OK]")


if __name__ == "__main__":
    main()
