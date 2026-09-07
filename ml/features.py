"""
ml/features.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Feature Engineering Pipeline
─────────────────────────────────────────────────────────────────────────────

WHAT THIS FILE DOES
───────────────────
Transforms raw product data (from the dataset) into a clean feature matrix
that can be passed to scikit-learn / XGBoost for training.

Feature groups:
  1. Raw Numerical   — directly from nutritional label (per 100g)
  2. Derived Ratios  — engineered from raw features
  3. Boolean         — ingredient flags (converted to 0/1)
  4. Categorical     — encoded category/processing level

IMPORTANT: UNKNOWN ≠ ZERO
──────────────────────────
Missing nutritional values are kept as NaN.
When passed to tree models (XGBoost/Random Forest), NaN is handled natively.
For linear models, imputation is applied separately (see note in impute_features).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# 1. RAW NUMERICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

RAW_NUMERICAL_FEATURES = [
    "energy_kcal_100g",
    "protein_g_100g",
    "carbs_g_100g",
    "sugar_g_100g",
    "added_sugar_g_100g",
    "total_fat_g_100g",
    "saturated_fat_g_100g",
    "trans_fat_g_100g",
    "fiber_g_100g",
    "sodium_mg_100g",
    "salt_g_100g",
    "ingredient_count",
    "serving_size_g",
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. BOOLEAN FEATURES
# ─────────────────────────────────────────────────────────────────────────────

BOOLEAN_FEATURES = [
    "contains_whole_grain",
    "contains_added_sugar",
    "contains_artificial_sweetener",
    "contains_allergen",
]

# ─────────────────────────────────────────────────────────────────────────────
# 3. CATEGORICAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

CATEGORICAL_FEATURES = [
    "category",
    "processing_level",
]

# Canonical list of features that build_feature_matrix must produce
EXPECTED_FEATURES = [
    "energy_kcal_100g", "protein_g_100g", "carbs_g_100g",
    "sugar_g_100g", "added_sugar_g_100g", "total_fat_g_100g",
    "saturated_fat_g_100g", "trans_fat_g_100g", "fiber_g_100g",
    "sodium_mg_100g", "ingredient_count",
    "protein_per_calorie", "sugar_to_carbs_ratio", "fiber_to_carbs_ratio",
    "sodium_density", "fat_density",
    "contains_whole_grain", "contains_added_sugar",
    "contains_artificial_sweetener", "contains_allergen",
    "category_enc", "processing_level_enc",
]

# ─────────────────────────────────────────────────────────────────────────────
# 4. DERIVED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
# WHY derived features?
# Raw nutritional values have limited standalone interpretive power.
# Ratios and density features capture relationships between nutrients
# that are informative for a model predicting nutritional quality.

def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived ratio / density features.

    All derived features use safe division (returns NaN where denominator = 0
    or either operand is NaN) rather than silently dividing by zero.

    Parameters
    ----------
    df : pd.DataFrame  (must contain the raw columns)

    Returns
    -------
    df : pd.DataFrame  (original df + new derived columns)

    Derived features created:
    ──────────────────────────────────────────────────────────────────────────
    protein_per_calorie      WHY: Separates calorie-dense products from
                                  nutrient-dense ones. A protein bar at
                                  400kcal is different from chips at 400kcal.

    sugar_to_carbs_ratio     WHY: Proportion of carbohydrates that are sugars.
                                  High ratio → fast-digesting carbs → relevant
                                  for blood sugar concerns.

    fiber_to_carbs_ratio     WHY: Proportion of carbs that is fiber.
                                  Higher = more complex, slower-release carbs.

    sodium_density           WHY: Sodium relative to energy. Catches
                                  high-sodium, moderate-calorie products.

    saturated_fat_ratio      WHY: Saturated fat as proportion of total fat.
                                  High ratio = less healthy fat profile.

    fat_density              WHY: Total fat relative to calories.

    added_sugar_ratio        WHY: Added sugar as fraction of total sugar.
                                  Distinguishes naturally-occurring from
                                  manufacturer-added sugar.
    ──────────────────────────────────────────────────────────────────────────

    LEAKAGE WARNING:
    ─────────────────
    Do NOT include any feature derived from food_score or computed_score
    in the training feature set. That would be label leakage.

    REDUNDANCY NOTE:
    ─────────────────
    salt_g_100g and sodium_mg_100g encode the same information
    (1g salt ≈ 393mg sodium). Use sodium_mg_100g as the primary feature.
    salt_g_100g should be DROPPED from the ML feature set to avoid
    redundancy. It is kept in the dataset for validation purposes.
    """
    df = df.copy()

    def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
        """Return a/b, with NaN where b==0 or either value is NaN."""
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                (b == 0) | b.isna() | a.isna(),
                np.nan,
                a / b
            )
        return pd.Series(result, index=a.index)

    # protein per calorie
    df["protein_per_calorie"] = safe_divide(
        df["protein_g_100g"],
        df["energy_kcal_100g"]
    )

    # sugar : carbs ratio (range 0–1)
    df["sugar_to_carbs_ratio"] = safe_divide(
        df["sugar_g_100g"],
        df["carbs_g_100g"]
    )

    # fiber : carbs ratio (range 0–1)
    df["fiber_to_carbs_ratio"] = safe_divide(
        df["fiber_g_100g"],
        df["carbs_g_100g"]
    )

    # sodium density: sodium per 100kcal
    df["sodium_density"] = safe_divide(
        df["sodium_mg_100g"],
        df["energy_kcal_100g"]
    ) * 100

    # saturated fat ratio: sat_fat / total_fat (range 0–1)
    df["saturated_fat_ratio"] = safe_divide(
        df["saturated_fat_g_100g"],
        df["total_fat_g_100g"]
    )

    # fat density: total_fat per 100kcal
    df["fat_density"] = safe_divide(
        df["total_fat_g_100g"],
        df["energy_kcal_100g"]
    ) * 100

    # added sugar ratio: added_sugar / sugar (range 0–1)
    df["added_sugar_ratio"] = safe_divide(
        df["added_sugar_g_100g"],
        df["sugar_g_100g"]
    )

    return df


DERIVED_FEATURES = [
    "protein_per_calorie",
    "sugar_to_carbs_ratio",
    "fiber_to_carbs_ratio",
    "sodium_density",
    "saturated_fat_ratio",
    "fat_density",
    "added_sugar_ratio",
]

# ─────────────────────────────────────────────────────────────────────────────
# 5. BOOLEAN ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_boolean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string booleans ('true'/'false') to integer 0/1.
    NaN values are kept as NaN (not converted to 0).
    """
    df = df.copy()
    bool_map = {"true": 1, "false": 0, "1": 1, "0": 0,
                "yes": 1, "no": 0, True: 1, False: 0}
    for col in BOOLEAN_FEATURES:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: bool_map.get(str(x).lower().strip(), np.nan)
                if not pd.isna(x) else np.nan
            )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. CATEGORICAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_categorical_features(df: pd.DataFrame,
                                 fit_encoders: dict = None) -> tuple:
    """
    Label-encode categorical features.

    Parameters
    ----------
    df            : DataFrame with raw categorical columns
    fit_encoders  : dict of {col_name: fitted LabelEncoder}
                    If None, new encoders are fitted on df (training mode).
                    If provided, existing encoders are used (inference mode).

    Returns
    -------
    df            : DataFrame with encoded columns
    encoders      : dict of fitted LabelEncoder objects (save these with model)

    NOTE
    ────
    For tree models (XGBoost, Random Forest), label encoding is acceptable
    because the model splits on values, not ordinal distances.
    For linear models, one-hot encoding is preferable.
    We use label encoding here as XGBoost is the primary candidate.
    """
    df = df.copy()
    if fit_encoders is None:
        fit_encoders = {}
        fitting = True
    else:
        fitting = False

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()
        if fitting:
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col])
            fit_encoders[col] = le
        else:
            le = fit_encoders[col]
            # Handle unseen categories at inference time
            df[col + "_enc"] = df[col].apply(
                lambda x: le.transform([x])[0]
                if x in le.classes_
                else -1
            )

    return df, fit_encoders


# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE SELECTION — FINAL ML FEATURE LIST
# ─────────────────────────────────────────────────────────────────────────────

def get_ml_feature_columns() -> list:
    """
    Returns the canonical list of feature column names to be used in
    ML model training (after all encoding steps).

    salt_g_100g is EXCLUDED (redundant with sodium_mg_100g).
    serving_size_g is EXCLUDED from core features (metadata, not nutrient).
    product_id, barcode, product_name, brand are EXCLUDED (identifiers).
    food_score, computed_score are EXCLUDED (these are the TARGET, not features).
    """
    numerical = [
        "energy_kcal_100g",
        "protein_g_100g",
        "carbs_g_100g",
        "sugar_g_100g",
        "added_sugar_g_100g",
        "total_fat_g_100g",
        "saturated_fat_g_100g",
        "trans_fat_g_100g",
        "fiber_g_100g",
        "sodium_mg_100g",
        "ingredient_count",
    ]
    derived = DERIVED_FEATURES
    boolean = BOOLEAN_FEATURES      # will be 0/1 encoded
    categorical_enc = [f + "_enc" for f in CATEGORICAL_FEATURES]

    return numerical + derived + boolean + categorical_enc


def build_feature_matrix(df: pd.DataFrame,
                          fit_encoders: dict = None) -> tuple:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df           : Raw product DataFrame
    fit_encoders : dict of fitted encoders (None for training, required for inference)

    Returns
    -------
    X            : pd.DataFrame — feature matrix ready for ML
    encoders     : dict of fitted LabelEncoders (save with model artifacts)
    """
    df = build_derived_features(df)
    df = encode_boolean_features(df)
    df, encoders = encode_categorical_features(df, fit_encoders)

    feature_cols = get_ml_feature_columns()
    # Only include columns that actually exist in the dataframe
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols]

    return X, encoders


# ─────────────────────────────────────────────────────────────────────────────
# IMPUTATION NOTE (for linear models)
# ─────────────────────────────────────────────────────────────────────────────

def impute_features(X: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing values for linear regression models.

    WHY NOT USE THIS FOR TREE MODELS?
    ──────────────────────────────────
    XGBoost and Random Forest handle NaN natively — they create a separate
    branch for missing values during tree splits. Imputing before feeding to
    XGBoost can introduce artificial patterns and reduce model integrity.

    USE THIS ONLY for:
      - Linear Regression (baseline)
      - Logistic Regression
      - SVMs
      - Models that cannot handle NaN

    DO NOT use this as a default pre-processing step for XGBoost.
    """
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    return X_imputed


# ─────────────────────────────────────────────────────────────────────────────
# Quick Test — Run: python ml/features.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "raw", "aarogya_food_products_demo.csv"
    )

    print("=" * 60)
    print("AAROGYA Feature Engineering — Quick Test")
    print("=" * 60)

    if not os.path.exists(demo_path):
        print(f"Demo dataset not found: {demo_path}")
        print("Place aarogya_food_products_demo.csv in data/raw/ first.")
        sys.exit(1)

    df = pd.read_csv(demo_path)
    print(f"Loaded {len(df)} products, {len(df.columns)} raw columns.")

    X, encoders = build_feature_matrix(df)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Features ({len(X.columns)}):")
    for col in X.columns:
        null_pct = X[col].isna().mean() * 100
        print(f"  {col:<35} NaN%: {null_pct:.1f}")

    print(f"\nEncoders fitted for: {list(encoders.keys())}")
    for col, le in encoders.items():
        print(f"  {col}: {list(le.classes_)}")

    print("\n[FEATURE ENGINEERING OK]")
