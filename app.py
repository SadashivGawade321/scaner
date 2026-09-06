"""
app.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Main Entry Point (Day 1)
─────────────────────────────────────────────────────────────────────────────

Day 1 version: verifies the environment is working, runs baseline scoring
on the demo dataset, and prints a summary.

This file will evolve into the Streamlit application entry point in Week 14.
Do NOT build the Streamlit UI until the ML foundation is complete.

USAGE
─────
    python app.py
"""

import os
import sys
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_DATASET = os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv")


def main():
    print("=" * 65)
    print("  AAROGYA — AI Food Product Intelligence")
    print("  Day 1 Environment Check")
    print("=" * 65)

    # 1. Python version
    print(f"\n[1] Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("    WARNING: Python 3.9+ recommended.")

    # 2. Core imports
    print("\n[2] Checking core imports...")
    imports_ok = True
    for pkg, import_name in [
        ("pandas",       "pandas"),
        ("numpy",        "numpy"),
        ("scikit-learn", "sklearn"),
        ("xgboost",      "xgboost"),
        ("shap",         "shap"),
    ]:
        try:
            mod = __import__(import_name)
            ver = getattr(mod, "__version__", "?")
            print(f"    {pkg:<20} OK  (v{ver})")
        except ImportError:
            print(f"    {pkg:<20} MISSING — run: pip install {pkg}")
            imports_ok = False

    if not imports_ok:
        print("\n  Some packages missing. Run: pip install -r requirements.txt")
        sys.exit(1)

    # 3. Dataset check
    print("\n[3] Checking demo dataset...")
    if os.path.exists(DEMO_DATASET):
        df = pd.read_csv(DEMO_DATASET)
        print(f"    Found: {DEMO_DATASET}")
        print(f"    Rows:  {len(df)}  |  Columns: {len(df.columns)}")
    else:
        print(f"    NOT FOUND: {DEMO_DATASET}")
        print("    Generate the dataset before proceeding.")
        sys.exit(1)

    # 4. Run baseline scoring
    print("\n[4] Running baseline scoring on demo dataset...")
    try:
        from ml.scoring import score_dataframe, SCORE_VERSION
        df_scored = score_dataframe(df)
        print(f"    Score version : {SCORE_VERSION}")
        print(f"    Products scored: {len(df_scored)}")
        print(f"    Score range   : {df_scored['computed_score'].min():.1f} – "
              f"{df_scored['computed_score'].max():.1f}")
        print(f"    Mean score    : {df_scored['computed_score'].mean():.1f}")

        print("\n    Top 3 products (highest Aarogya score):")
        top3 = df_scored.nlargest(3, "computed_score")[
            ["product_name", "category", "computed_score"]
        ]
        for _, row in top3.iterrows():
            print(f"      {row['product_name'][:45]:<45}  {row['computed_score']:.1f}")

        print("\n    Bottom 3 products (lowest Aarogya score):")
        bot3 = df_scored.nsmallest(3, "computed_score")[
            ["product_name", "category", "computed_score"]
        ]
        for _, row in bot3.iterrows():
            print(f"      {row['product_name'][:45]:<45}  {row['computed_score']:.1f}")

    except ImportError as e:
        print(f"    Scoring import failed: {e}")

    # 5. Feature engineering check
    print("\n[5] Running feature engineering on demo dataset...")
    try:
        from ml.features import build_feature_matrix
        X, encoders = build_feature_matrix(df)
        print(f"    Feature matrix: {X.shape[0]} products × {X.shape[1]} features")
        null_pct = X.isnull().mean().mean() * 100
        print(f"    Overall NaN %: {null_pct:.1f}% (intentional missing values are OK)")
    except ImportError as e:
        print(f"    Feature engineering import failed: {e}")

    # 6. MongoDB check (optional)
    print("\n[6] Checking MongoDB connection (optional for Day 1)...")
    try:
        from database.mongodb import AarogyaDB
        db = AarogyaDB()
        health = db.health_check()
        status = health.get("status", "unknown")
        if status == "connected":
            print(f"    MongoDB: CONNECTED")
        else:
            print(f"    MongoDB: {status.upper()} (OK for Day 1 — not required yet)")
        db.close()
    except Exception as e:
        print(f"    MongoDB check skipped: {e}")

    print("\n" + "=" * 65)
    print("  DAY 1 ENVIRONMENT: OK")
    print("  Next step: python data/validate_dataset.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
