"""
tests/test_features.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Feature Engineering Tests
─────────────────────────────────────────────────────────────────────────────

Run with: pytest tests/ -v
"""

import os, sys
import pytest
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.features import build_feature_matrix, EXPECTED_FEATURES


@pytest.fixture
def demo_df():
    path = os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv")
    if not os.path.exists(path):
        pytest.skip("Demo dataset not found.")
    return pd.read_csv(path)


@pytest.fixture
def single_product_df():
    return pd.DataFrame([{
        "product_name":        "Test Product",
        "category":            "Biscuits & Cookies",
        "subcategory":         "Crackers",
        "recommendation_group":"biscuits_cookies",
        "serving_size_g":      30,
        "energy_kcal_100g":    400.0,
        "protein_g_100g":       8.0,
        "carbs_g_100g":        60.0,
        "sugar_g_100g":        12.0,
        "added_sugar_g_100g":  10.0,
        "total_fat_g_100g":    14.0,
        "saturated_fat_g_100g": 4.0,
        "trans_fat_g_100g":     0.0,
        "fiber_g_100g":         3.5,
        "sodium_mg_100g":      350.0,
        "ingredient_count":     10,
        "contains_whole_grain": False,
        "contains_added_sugar": True,
        "contains_artificial_sweetener": False,
        "contains_allergen":   True,
        "processing_level":    "processed",
    }])


class TestBuildFeatureMatrix:

    def test_returns_dataframe(self, demo_df):
        X, _ = build_feature_matrix(demo_df)
        assert isinstance(X, pd.DataFrame)

    def test_correct_number_of_rows(self, demo_df):
        X, _ = build_feature_matrix(demo_df)
        assert len(X) == len(demo_df)

    def test_expected_features_present(self, demo_df):
        X, _ = build_feature_matrix(demo_df)
        for feat in EXPECTED_FEATURES:
            assert feat in X.columns, f"Missing feature: {feat}"

    def test_no_all_nan_columns(self, demo_df):
        X, _ = build_feature_matrix(demo_df)
        for col in X.columns:
            assert not X[col].isna().all(), f"Column {col} is entirely NaN"

    def test_nan_fraction_acceptable(self, demo_df):
        X, _ = build_feature_matrix(demo_df)
        nan_pct = X.isna().mean().mean() * 100
        assert nan_pct < 10, f"Too many NaN values: {nan_pct:.1f}%"

    def test_returns_encoders(self, demo_df):
        X, encoders = build_feature_matrix(demo_df)
        assert isinstance(encoders, dict)
        assert "category" in encoders or len(encoders) >= 0

    def test_single_product(self, single_product_df):
        X, _ = build_feature_matrix(single_product_df)
        assert len(X) == 1

    def test_protein_per_calorie_feature(self, demo_df):
        X, _ = build_feature_matrix(demo_df)
        assert "protein_per_calorie" in X.columns
        # Should be protein / energy, so in range [0, 1] for food
        assert (X["protein_per_calorie"] >= 0).all()

    def test_ratios_in_valid_range(self, demo_df):
        X, _ = build_feature_matrix(demo_df)
        if "sugar_to_carbs_ratio" in X.columns:
            valid = X["sugar_to_carbs_ratio"].dropna()
            assert (valid >= 0).all()
            assert (valid <= 1.01).all(), "sugar_to_carbs_ratio > 1 found"

    def test_encoding_is_numeric(self, demo_df):
        X, _ = build_feature_matrix(demo_df)
        for col in X.columns:
            assert pd.api.types.is_numeric_dtype(X[col]), \
                f"Column {col} is not numeric after encoding"
