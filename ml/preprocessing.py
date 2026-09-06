"""
ml/preprocessing.py — [WEEK 3]
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Data Cleaning and Preprocessing Pipeline
─────────────────────────────────────────────────────────────────────────────

This module will be implemented in Week 3 when real data from Open Food Facts
and USDA is collected. For now it contains the function signatures and
documented design decisions.

DESIGN DECISIONS
────────────────
1. All normalization is done to per-100g.
2. UNKNOWN ≠ ZERO — missing values remain NaN.
3. Deduplication uses barcode as primary key.
4. Conflict resolution prefers manually_verified > open_food_facts > usda.
5. All cleaning decisions are logged to a processing_log column.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def normalize_to_per_100g(df: pd.DataFrame) -> pd.DataFrame:
    """
    [WEEK 3] Normalize all nutritional values to per-100g basis.
    Some sources report per-serving values — these need conversion.
    Requires serving_size_g column to be present.
    """
    raise NotImplementedError("Implement in Week 3 after real data collection.")


def deduplicate_by_barcode(df: pd.DataFrame) -> pd.DataFrame:
    """
    [WEEK 3] Remove duplicate products keeping highest-confidence source.
    Source priority: verified > open_food_facts > usda > other
    """
    raise NotImplementedError("Implement in Week 3 after real data collection.")


def resolve_conflicting_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    [WEEK 3] When two sources give different values for the same barcode,
    flag the conflict and prefer the higher-confidence source.
    """
    raise NotImplementedError("Implement in Week 3 after real data collection.")


def clean_ingredients_text(text: str) -> str:
    """
    [WEEK 3] Standardize ingredient text: lowercase, remove extra whitespace,
    expand common abbreviations.
    """
    raise NotImplementedError("Implement in Week 3.")
