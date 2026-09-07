"""
tests/test_recommender.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Recommender Engine Tests
─────────────────────────────────────────────────────────────────────────────

Run with: pytest tests/ -v
"""

import os, sys
import pytest
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from recommendation.recommender import AarogyaRecommender, DEFAULT_PREFERENCE_PROFILES


@pytest.fixture
def catalog_df():
    path = os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv")
    if not os.path.exists(path):
        pytest.skip("Demo dataset not found.")
    return pd.read_csv(path)


@pytest.fixture
def recommender(catalog_df):
    return AarogyaRecommender(catalog_df)


class TestRecommender:

    def test_instantiation(self, catalog_df):
        r = AarogyaRecommender(catalog_df)
        assert r is not None

    def test_recommend_returns_dataframe(self, recommender):
        recs = recommender.recommend("AAR004", top_n=3)
        assert isinstance(recs, pd.DataFrame)

    def test_top_n_respected(self, recommender):
        for n in [1, 3, 5]:
            recs = recommender.recommend("AAR004", top_n=n)
            assert len(recs) <= n

    def test_recommendations_not_current_product(self, recommender):
        recs = recommender.recommend("AAR004", top_n=5)
        assert "AAR004" not in recs["product_id"].tolist()

    def test_recommendations_same_group(self, recommender, catalog_df):
        product_id = "AAR004"
        current_group = catalog_df[catalog_df["product_id"] == product_id].iloc[0]["recommendation_group"]
        recs = recommender.recommend(product_id, top_n=5)
        for _, row in recs.iterrows():
            rec_group = catalog_df[catalog_df["product_id"] == row["product_id"]].iloc[0]["recommendation_group"]
            assert rec_group == current_group, \
                f"Recommendation {row['product_id']} is from a different group"

    def test_required_output_columns(self, recommender):
        recs = recommender.recommend("AAR004", top_n=3)
        for col in ["product_id", "product_name", "food_score", "ranking_score", "recommendation_reason"]:
            assert col in recs.columns, f"Missing column: {col}"

    def test_recommendations_sorted_by_ranking_score(self, recommender):
        recs = recommender.recommend("AAR004", top_n=5)
        scores = recs["ranking_score"].tolist()
        assert scores == sorted(scores, reverse=True), \
            "Recommendations not sorted by ranking_score"

    def test_invalid_product_id_raises(self, recommender):
        with pytest.raises(ValueError):
            recommender.recommend("INVALID_ID_9999", top_n=3)

    def test_all_preference_profiles_work(self, recommender):
        for profile in DEFAULT_PREFERENCE_PROFILES.keys():
            recs = recommender.recommend("AAR004", preference_profile=profile, top_n=3)
            assert isinstance(recs, pd.DataFrame), \
                f"Profile {profile} returned non-DataFrame"

    def test_reason_is_nonempty_string(self, recommender):
        recs = recommender.recommend("AAR004", top_n=3)
        for _, row in recs.iterrows():
            assert isinstance(row["recommendation_reason"], str)
            assert len(row["recommendation_reason"]) > 0

    def test_recommend_by_data(self, recommender):
        product = {
            "product_name":        "Test Cookie",
            "energy_kcal_100g":    490.0,
            "protein_g_100g":       5.5,
            "carbs_g_100g":        68.0,
            "sugar_g_100g":        32.0,
            "added_sugar_g_100g":  30.0,
            "total_fat_g_100g":    19.0,
            "saturated_fat_g_100g": 8.5,
            "trans_fat_g_100g":     0.3,
            "fiber_g_100g":         1.8,
            "sodium_mg_100g":      380.0,
            "ingredient_count":     14,
        }
        recs = recommender.recommend_by_data(
            current_product=product,
            recommendation_group="biscuits_cookies",
            preference_profile="sugar_focused",
            top_n=3,
        )
        assert isinstance(recs, pd.DataFrame)
        assert len(recs) > 0
