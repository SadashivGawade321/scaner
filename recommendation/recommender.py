"""
recommendation/recommender.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Preference-Weighted Recommendation Engine (v1 — Rule-Based)
─────────────────────────────────────────────────────────────────────────────

DESIGN
──────
This is the Phase 1 recommendation engine. It is deliberately rule-based
and transparent. Phase 2 (Learning-to-Rank) will replace/extend this once
sufficient data and user feedback is available.

PIPELINE
────────
  Current Product
       ↓
  recommendation_group filter (same product family)
       ↓
  Candidate Products (remove current product)
       ↓
  Compute preference-weighted nutrition gap scores
       ↓
  Rank candidates
       ↓
  Top 3 alternatives

WHY recommendation_group INSTEAD OF category?
──────────────────────────────────────────────
category = "Chips & Namkeen" is broad.
recommendation_group = "chips_namkeen" is the matching key.
This ensures a biscuit → biscuit comparison, not biscuit → protein bar.

WHY NOT RECOMMEND THE BEST OVERALL PRODUCT?
────────────────────────────────────────────
Aarogya must recommend comparatively suitable alternatives, not simply
the globally highest-scoring product. A user eating biscuits should be
recommended a better biscuit, not a green salad.

HOW PREFERENCE WEIGHTS WORK
────────────────────────────
Each user preference (e.g., "lower sugar") translates into a direction:
  - Lower Sugar       → sugar_g_100g should be LOWER in the alternative
  - Higher Protein    → protein_g_100g should be HIGHER
  - Higher Fiber      → fiber_g_100g should be HIGHER
  - Lower Sodium      → sodium_mg_100g should be LOWER
  - Lower Calories    → energy_kcal_100g should be LOWER
  - Lower Sat Fat     → saturated_fat_g_100g should be LOWER

Each nutrient gets a "gap" score = (current_value - alternative_value) for
"lower is better" nutrients, or (alternative_value - current_value) for
"higher is better" nutrients.

A positive gap means the alternative is BETTER for that preference.

These gaps are then normalized and weighted by user preference weights.
The final ranking score = sum of (weight × normalized_gap) across all nutrients.

Higher ranking score = better match for user preferences.
"""

import numpy as np
import pandas as pd
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# PREFERENCE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Maps preference dimension → (column, direction)
# direction: "lower" = lower value is better for user
#            "higher" = higher value is better for user
PREFERENCE_MAP = {
    "sugar":         ("sugar_g_100g",          "lower"),
    "protein":       ("protein_g_100g",         "higher"),
    "fiber":         ("fiber_g_100g",           "higher"),
    "sodium":        ("sodium_mg_100g",         "lower"),
    "calorie":       ("energy_kcal_100g",       "lower"),
    "saturated_fat": ("saturated_fat_g_100g",   "lower"),
}

# Default normalization ranges for gap normalization
# Used to map raw gaps to [-1, 1] range before weighting.
# These are rough typical ranges for packaged food per 100g.
GAP_NORMALIZATION_RANGES = {
    "sugar_g_100g":          60.0,   # typical max delta in sugar
    "protein_g_100g":        30.0,   # typical max delta in protein
    "fiber_g_100g":          15.0,   # typical max delta in fiber
    "sodium_mg_100g":      2000.0,   # typical max delta in sodium
    "energy_kcal_100g":     400.0,   # typical max delta in energy
    "saturated_fat_g_100g":  30.0,   # typical max delta in sat fat
}


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT PREFERENCE PROFILES
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PREFERENCE_PROFILES = {
    "balanced": {
        "sugar_weight":         0.20,
        "protein_weight":       0.20,
        "fiber_weight":         0.20,
        "sodium_weight":        0.20,
        "calorie_weight":       0.10,
        "saturated_fat_weight": 0.10,
    },
    "sugar_focused": {
        "sugar_weight":         0.40,
        "protein_weight":       0.10,
        "fiber_weight":         0.15,
        "sodium_weight":        0.15,
        "calorie_weight":       0.10,
        "saturated_fat_weight": 0.10,
    },
    "protein_focused": {
        "sugar_weight":         0.10,
        "protein_weight":       0.40,
        "fiber_weight":         0.15,
        "sodium_weight":        0.10,
        "calorie_weight":       0.10,
        "saturated_fat_weight": 0.15,
    },
    "fiber_focused": {
        "sugar_weight":         0.15,
        "protein_weight":       0.15,
        "fiber_weight":         0.40,
        "sodium_weight":        0.10,
        "calorie_weight":       0.10,
        "saturated_fat_weight": 0.10,
    },
    "low_sodium": {
        "sugar_weight":         0.15,
        "protein_weight":       0.15,
        "fiber_weight":         0.15,
        "sodium_weight":        0.40,
        "calorie_weight":       0.10,
        "saturated_fat_weight": 0.05,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AarogyaRecommender:
    """
    Rule-based preference-weighted recommendation engine.

    Parameters
    ----------
    products_df : pd.DataFrame
        Product catalog with nutritional columns and recommendation_group.
    """

    def __init__(self, products_df: pd.DataFrame):
        self.df = products_df.copy()
        self._validate_catalog()

    def _validate_catalog(self):
        """Check that required columns exist."""
        required = ["product_id", "product_name", "recommendation_group",
                    "food_score"] + list(GAP_NORMALIZATION_RANGES.keys())
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Product catalog missing required columns: {missing}\n"
                "Ensure you are passing the full product DataFrame."
            )

    def recommend(
        self,
        product_id: str,
        preference_weights: Optional[dict] = None,
        preference_profile: Optional[str] = "balanced",
        top_n: int = 3,
    ) -> pd.DataFrame:
        """
        Recommend top_n alternatives to a given product.

        Parameters
        ----------
        product_id          : str  — product_id of the current product
        preference_weights  : dict — custom weights dict, e.g.:
                                     {"sugar_weight": 0.4, "protein_weight": 0.3, ...}
                                     If None, uses preference_profile.
        preference_profile  : str  — name of a default profile (ignored if
                                     preference_weights is provided)
        top_n               : int  — number of recommendations to return

        Returns
        -------
        pd.DataFrame with columns:
            product_id, product_name, category, food_score,
            ranking_score, recommendation_reason
        """
        # ── Resolve preference weights ─────────────────────────────────────
        if preference_weights is None:
            weights = DEFAULT_PREFERENCE_PROFILES.get(preference_profile)
            if weights is None:
                weights = DEFAULT_PREFERENCE_PROFILES["balanced"]
        else:
            weights = preference_weights

        self._validate_weights(weights)

        # ── Find current product ───────────────────────────────────────────
        current = self.df[self.df["product_id"] == product_id]
        if current.empty:
            raise ValueError(f"Product ID '{product_id}' not found in catalog.")
        current = current.iloc[0]

        # ── Filter to same recommendation group ────────────────────────────
        rec_group = current["recommendation_group"]
        candidates = self.df[
            (self.df["recommendation_group"] == rec_group) &
            (self.df["product_id"] != product_id)
        ].copy()

        if candidates.empty:
            return pd.DataFrame(columns=[
                "product_id", "product_name", "category",
                "food_score", "ranking_score", "recommendation_reason"
            ])

        # ── Compute preference-weighted ranking score ──────────────────────
        ranking_scores = []
        for _, candidate in candidates.iterrows():
            score = self._compute_ranking_score(current, candidate, weights)
            ranking_scores.append(score)

        candidates["ranking_score"] = ranking_scores

        # ── Sort and select top_n ──────────────────────────────────────────
        top = candidates.nlargest(top_n, "ranking_score")

        # ── Build explanation strings ──────────────────────────────────────
        top = top.copy()
        top["recommendation_reason"] = top.apply(
            lambda row: self._build_reason(current, row, weights), axis=1
        )

        return top[["product_id", "product_name", "category", "brand",
                    "food_score", "ranking_score",
                    "recommendation_reason"]].reset_index(drop=True)

    def _compute_ranking_score(
        self,
        current: pd.Series,
        candidate: pd.Series,
        weights: dict,
    ) -> float:
        """
        Compute preference-weighted ranking score for one candidate.

        For each preference dimension:
          gap = improvement in the direction the user prefers
                (positive = candidate is better for that preference)
          normalized_gap = gap / normalization_range (capped to [-1, 1])
          contribution = weight × normalized_gap

        ranking_score = sum of all contributions
        """
        total = 0.0

        for pref_key, (col, direction) in PREFERENCE_MAP.items():
            weight_key = f"{pref_key}_weight"
            w = weights.get(weight_key, 0.0)
            if w == 0.0:
                continue

            cur_val = pd.to_numeric(current.get(col), errors="coerce")
            can_val = pd.to_numeric(candidate.get(col), errors="coerce")

            if pd.isna(cur_val) or pd.isna(can_val):
                # Cannot compute gap when either value is missing
                continue

            # Raw gap: positive means candidate is better
            if direction == "lower":
                raw_gap = cur_val - can_val    # positive = candidate is lower (better)
            else:
                raw_gap = can_val - cur_val    # positive = candidate is higher (better)

            # Normalize gap
            norm_range = GAP_NORMALIZATION_RANGES.get(col, 1.0)
            normalized_gap = float(np.clip(raw_gap / norm_range, -1.0, 1.0))

            total += w * normalized_gap

        return round(total, 4)

    def _build_reason(
        self,
        current: pd.Series,
        candidate: pd.Series,
        weights: dict,
    ) -> str:
        """
        Build a short human-readable reason for recommendation.
        Used as input context for Gemini's natural language generation.
        """
        improvements = []
        for pref_key, (col, direction) in PREFERENCE_MAP.items():
            weight_key = f"{pref_key}_weight"
            w = weights.get(weight_key, 0.0)
            if w < 0.10:
                continue  # ignore low-weight preferences in explanation

            cur_val = pd.to_numeric(current.get(col), errors="coerce")
            can_val = pd.to_numeric(candidate.get(col), errors="coerce")

            if pd.isna(cur_val) or pd.isna(can_val):
                continue

            if direction == "lower" and can_val < cur_val - 0.5:
                unit = "mg" if "sodium" in col else "g"
                improvements.append(
                    f"Lower {pref_key.replace('_', ' ')} "
                    f"({can_val:.1f} vs {cur_val:.1f}{unit}/100g)"
                )
            elif direction == "higher" and can_val > cur_val + 0.5:
                unit = "g"
                improvements.append(
                    f"Higher {pref_key.replace('_', ' ')} "
                    f"({can_val:.1f} vs {cur_val:.1f}{unit}/100g)"
                )

        if not improvements:
            return "Comparable alternative in same product category."
        return "Better match for your preferences: " + "; ".join(improvements[:3]) + "."

    @staticmethod
    def _validate_weights(weights: dict):
        required_keys = [f"{k}_weight" for k in PREFERENCE_MAP.keys()]
        missing = [k for k in required_keys if k not in weights]
        if missing:
            raise ValueError(f"Preference weights missing keys: {missing}")
        total = sum(weights[k] for k in required_keys)
        if abs(total - 1.0) > 0.05:
            raise ValueError(
                f"Preference weights must sum to ~1.0 (got {total:.3f}). "
                "Normalize your weights."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "raw", "aarogya_food_products_demo.csv"
    )

    if not os.path.exists(demo_path):
        print("Demo dataset not found. Run from aarogya/ root.")
        sys.exit(1)

    df = pd.read_csv(demo_path)
    recommender = AarogyaRecommender(df)

    # Test: recommend alternatives for a biscuit (AAR004 = Choco Cream Sandwich)
    # with sugar-focused preference
    print("=" * 60)
    print("AAROGYA Recommender — Quick Test")
    print("=" * 60)
    print("Current Product: AAR004 (Choco Cream Sandwich Biscuits)")
    print("Preference Profile: sugar_focused")
    print()

    recs = recommender.recommend(
        product_id="AAR004",
        preference_profile="sugar_focused",
        top_n=3,
    )

    if recs.empty:
        print("No recommendations found (only product in its group).")
    else:
        for i, row in recs.iterrows():
            print(f"Recommendation #{i+1}: {row['product_name']} (Score: {row['food_score']})")
            print(f"  Ranking Score: {row['ranking_score']}")
            print(f"  Reason: {row['recommendation_reason']}")
            print()

    print("[RECOMMENDER OK]")
