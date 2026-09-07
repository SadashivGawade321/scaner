"""
tests/test_scoring.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Scoring Engine Tests (Week 14)
─────────────────────────────────────────────────────────────────────────────

Run with: pytest tests/ -v
"""

import os, sys
import pytest
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ml.scoring import compute_aarogya_score as score_product, score_dataframe as score_dataset

# ─── Fixtures ─────────────────────────────────────────────────────────────

HEALTHY_PRODUCT = {
    "product_name":          "Test Oat Porridge",
    "category":              "Breakfast Cereals",
    "processing_level":      "minimally_processed",
    "energy_kcal_100g":      350.0,
    "protein_g_100g":        12.0,
    "carbs_g_100g":          60.0,
    "sugar_g_100g":           3.0,
    "added_sugar_g_100g":     1.0,
    "total_fat_g_100g":       5.0,
    "saturated_fat_g_100g":   1.0,
    "trans_fat_g_100g":       0.0,
    "fiber_g_100g":          10.0,
    "sodium_mg_100g":        100.0,
    "ingredient_count":        5,
    "contains_whole_grain":  True,
    "contains_artificial_sweetener": False,
    "recommendation_group":  "breakfast_cereals",
}

UNHEALTHY_PRODUCT = {
    "product_name":          "Test Sugar Cookie",
    "category":              "Biscuits & Cookies",
    "processing_level":      "ultra_processed",
    "energy_kcal_100g":      520.0,
    "protein_g_100g":         5.0,
    "carbs_g_100g":          70.0,
    "sugar_g_100g":          40.0,
    "added_sugar_g_100g":    38.0,
    "total_fat_g_100g":      22.0,
    "saturated_fat_g_100g":  12.0,
    "trans_fat_g_100g":       0.5,
    "fiber_g_100g":           1.0,
    "sodium_mg_100g":        450.0,
    "ingredient_count":       18,
    "contains_whole_grain":  False,
    "contains_artificial_sweetener": True,
    "recommendation_group":  "biscuits_cookies",
}


# ─── Tests ────────────────────────────────────────────────────────────────

class TestScoreProduct:

    def test_returns_dict(self):
        result = score_product(HEALTHY_PRODUCT)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = score_product(HEALTHY_PRODUCT)
        assert "final_score" in result
        assert "score_version" in result
        assert "component_scores" in result

    def test_score_in_range(self):
        for product in [HEALTHY_PRODUCT, UNHEALTHY_PRODUCT]:
            result = score_product(product)
            score = result["final_score"]
            assert 0 <= score <= 100, f"Score {score} out of range"

    def test_healthy_scores_higher(self):
        h_result = score_product(HEALTHY_PRODUCT)
        u_result = score_product(UNHEALTHY_PRODUCT)
        assert h_result["final_score"] > u_result["final_score"], (
            f"Healthy ({h_result['final_score']}) should score "
            f"higher than unhealthy ({u_result['final_score']})"
        )

    def test_healthy_score_above_60(self):
        result = score_product(HEALTHY_PRODUCT)
        assert result["final_score"] > 60, \
            f"Expected healthy product > 60, got {result['final_score']}"

    def test_unhealthy_score_below_50(self):
        result = score_product(UNHEALTHY_PRODUCT)
        assert result["final_score"] < 50, \
            f"Expected unhealthy product < 50, got {result['final_score']}"

    def test_score_version(self):
        result = score_product(HEALTHY_PRODUCT)
        assert result["score_version"] == "v1.0"

    def test_score_components_present(self):
        result = score_product(HEALTHY_PRODUCT)
        components = result["component_scores"]
        assert "sugar" in components
        assert "protein" in components
        assert "fiber" in components
        assert "sodium" in components

    def test_deterministic(self):
        """Same input must always give same output."""
        r1 = score_product(HEALTHY_PRODUCT)
        r2 = score_product(HEALTHY_PRODUCT)
        assert r1["final_score"] == r2["final_score"]

    def test_high_sugar_low_score(self):
        """Very high sugar should pull score down significantly vs low-sugar baseline."""
        p_lo = {**HEALTHY_PRODUCT, "sugar_g_100g": 3.0,  "added_sugar_g_100g": 1.0}
        p_hi = {**HEALTHY_PRODUCT, "sugar_g_100g": 80.0, "added_sugar_g_100g": 75.0}
        lo = score_product(p_lo)["final_score"]
        hi = score_product(p_hi)["final_score"]
        assert hi < lo, f"High sugar ({hi}) should score lower than low sugar ({lo})"
        assert (lo - hi) > 10, f"Sugar increase should drop score by >10 pts, got {lo-hi:.1f}"

    def test_high_fiber_boosts_score(self):
        """Higher fiber should improve score."""
        p_lo = {**UNHEALTHY_PRODUCT, "fiber_g_100g": 1.0}
        p_hi = {**UNHEALTHY_PRODUCT, "fiber_g_100g": 15.0}
        lo = score_product(p_lo)["final_score"]
        hi = score_product(p_hi)["final_score"]
        assert hi > lo

    def test_whole_grain_bonus(self):
        """Whole grain flag should add a bonus."""
        p_no  = {**HEALTHY_PRODUCT, "contains_whole_grain": False}
        p_yes = {**HEALTHY_PRODUCT, "contains_whole_grain": True}
        no  = score_product(p_no)["final_score"]
        yes = score_product(p_yes)["final_score"]
        assert yes >= no

    def test_ultra_processed_penalty(self):
        """Ultra-processed should score lower than minimally processed."""
        p_min = {**HEALTHY_PRODUCT, "processing_level": "minimally_processed"}
        p_ult = {**HEALTHY_PRODUCT, "processing_level": "ultra_processed"}
        min_score = score_product(p_min)["final_score"]
        ult_score = score_product(p_ult)["final_score"]
        assert min_score > ult_score


class TestScoreDataset:

    @pytest.fixture
    def demo_df(self):
        path = os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv")
        if not os.path.exists(path):
            pytest.skip("Demo dataset not found.")
        return pd.read_csv(path)

    def test_score_dataset_returns_dataframe(self, demo_df):
        result = score_dataset(demo_df)
        assert isinstance(result, pd.DataFrame)

    def test_score_dataset_has_computed_score(self, demo_df):
        result = score_dataset(demo_df)
        assert "computed_score" in result.columns

    def test_score_dataset_correct_length(self, demo_df):
        result = score_dataset(demo_df)
        assert len(result) == len(demo_df)

    def test_all_scores_in_range(self, demo_df):
        result = score_dataset(demo_df)
        assert result["computed_score"].between(0, 100).all(), \
            "Not all scores in [0, 100]"

    def test_score_distribution_reasonable(self, demo_df):
        result = score_dataset(demo_df)
        mean = result["computed_score"].mean()
        assert 30 < mean < 80, f"Mean score {mean:.1f} seems unreasonable"

    def test_no_nan_scores(self, demo_df):
        result = score_dataset(demo_df)
        assert not result["computed_score"].isna().any(), \
            "NaN scores found — scoring engine produced NaN"
