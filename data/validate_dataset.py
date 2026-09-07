"""
data/validate_dataset.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Dataset Validation Script
─────────────────────────────────────────────────────────────────────────────

Validates aarogya_food_products_demo.csv against schema, consistency rules,
and data quality requirements.

USAGE
─────
    python data/validate_dataset.py

From project root (aarogya/).

VALIDATION CATEGORIES
─────────────────────
1.  File existence
2.  Required columns present
3.  Duplicate product IDs
4.  Duplicate barcodes
5.  Numeric range checks (no negatives, plausible ceilings)
6.  Boolean column validity
7.  Category distribution
8.  Score range (0–100)
9.  Nutritional consistency (sugar ≤ carbs, sat_fat ≤ total_fat, etc.)
10. Missing values report
11. Processing level validity
12. Serving size plausibility

All checks emit PASS / WARN / FAIL status.
Script exits with code 0 if no FAILs, code 1 if any FAILs.
"""

import os
import sys
import pandas as pd
import numpy as np

# Force UTF-8 output on Windows (fixes UnicodeEncodeError for ≤ ≥ → ─ etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv")
PREFS_PATH   = os.path.join(ROOT, "data", "raw", "aarogya_user_preferences_demo.csv")

REQUIRED_COLUMNS = [
    "product_id", "barcode", "product_name", "brand",
    "category", "subcategory", "recommendation_group", "serving_size_g",
    "energy_kcal_100g", "protein_g_100g", "carbs_g_100g",
    "sugar_g_100g", "added_sugar_g_100g",
    "total_fat_g_100g", "saturated_fat_g_100g", "trans_fat_g_100g",
    "fiber_g_100g", "sodium_mg_100g", "salt_g_100g",
    "ingredients_text", "ingredient_count",
    "contains_whole_grain", "contains_added_sugar",
    "contains_artificial_sweetener", "contains_allergen", "allergens",
    "processing_level", "food_score", "score_version",
    "data_source", "verified",
]

VALID_CATEGORIES = {
    "Biscuits & Cookies",
    "Chips & Namkeen",
    "Breakfast Cereals",
    "Beverages",
    "Instant Foods",
    "Chocolates & Sweet Snacks",
    "Snack/Protein Bars",
    "Bread & Bakery",
    "Dairy & Dairy Drinks",
}

VALID_PROCESSING_LEVELS = {
    "minimally_processed",
    "processed",
    "ultra_processed",
}

EXPECTED_CATEGORY_COUNTS = {
    "Biscuits & Cookies":       15,
    "Chips & Namkeen":          13,
    "Breakfast Cereals":        12,
    "Beverages":                13,
    "Instant Foods":            10,
    "Chocolates & Sweet Snacks": 10,
    "Snack/Protein Bars":        5,
    "Bread & Bakery":            5,
    "Dairy & Dairy Drinks":      7,
}

BOOLEAN_COLUMNS = [
    "contains_whole_grain",
    "contains_added_sugar",
    "contains_artificial_sweetener",
    "contains_allergen",
    "verified",
]

# Numeric columns with their (min, max) plausible range per 100g
# NaN values are ALLOWED (UNKNOWN ≠ ZERO).
NUMERIC_RANGES = {
    "energy_kcal_100g":      (0,   900),
    "protein_g_100g":        (0,   100),
    "carbs_g_100g":          (0,   100),
    "sugar_g_100g":          (0,   100),
    "added_sugar_g_100g":    (0,   100),
    "total_fat_g_100g":      (0,   100),
    "saturated_fat_g_100g":  (0,   100),
    "trans_fat_g_100g":      (0,    10),
    "fiber_g_100g":          (0,    50),
    "sodium_mg_100g":        (0,  5000),
    "salt_g_100g":           (0,    15),
    "ingredient_count":      (1,    50),
    "serving_size_g":        (5,  1000),
    "food_score":            (0,   100),
}


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

class ValidationReport:
    def __init__(self):
        self.passes = 0
        self.warnings = 0
        self.failures = 0
        self.lines = []

    def _log(self, level: str, check: str, message: str):
        line = f"[{level:4s}] {check:<45} {message}"
        self.lines.append(line)
        print(line)
        if level == "PASS":
            self.passes += 1
        elif level == "WARN":
            self.warnings += 1
        elif level == "FAIL":
            self.failures += 1

    def passed(self, check: str, message: str = ""):
        self._log("PASS", check, message)

    def warn(self, check: str, message: str):
        self._log("WARN", check, message)

    def fail(self, check: str, message: str):
        self._log("FAIL", check, message)

    def info(self, message: str):
        line = f"       {message}"
        self.lines.append(line)
        print(line)

    def summary(self):
        print("\n" + "=" * 70)
        print(f"VALIDATION SUMMARY  |  PASS: {self.passes}  "
              f"WARN: {self.warnings}  FAIL: {self.failures}")
        print("=" * 70)
        if self.failures == 0 and self.warnings == 0:
            print("DATASET STATUS: CLEAN — No issues found.")
        elif self.failures == 0:
            print(f"DATASET STATUS: ACCEPTABLE — {self.warnings} warning(s), no failures.")
        else:
            print(f"DATASET STATUS: NEEDS ATTENTION — {self.failures} failure(s) found.")
        return self.failures


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_file_exists(report: ValidationReport) -> bool:
    if os.path.exists(DATASET_PATH):
        size_kb = os.path.getsize(DATASET_PATH) / 1024
        report.passed("File Exists", f"{DATASET_PATH} ({size_kb:.1f} KB)")
        return True
    else:
        report.fail("File Exists", f"NOT FOUND: {DATASET_PATH}")
        return False


def check_required_columns(df: pd.DataFrame, report: ValidationReport):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if not missing:
        report.passed("Required Columns", f"All {len(REQUIRED_COLUMNS)} columns present")
    else:
        report.fail("Required Columns", f"Missing: {missing}")
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        report.warn("Extra Columns", f"Non-schema columns found: {extra}")


def check_row_count(df: pd.DataFrame, report: ValidationReport):
    n = len(df)
    if n == 100:
        report.passed("Row Count", f"Exactly 100 products found")
    elif 90 <= n < 100:
        report.warn("Row Count", f"Expected 100, found {n}")
    else:
        report.fail("Row Count", f"Expected 100, found {n}")


def check_duplicate_ids(df: pd.DataFrame, report: ValidationReport):
    dupes = df["product_id"].duplicated().sum()
    if dupes == 0:
        report.passed("Unique Product IDs", "No duplicates")
    else:
        report.fail("Unique Product IDs", f"{dupes} duplicate IDs found")
        report.info(str(df[df["product_id"].duplicated(keep=False)]["product_id"].tolist()))


def check_duplicate_barcodes(df: pd.DataFrame, report: ValidationReport):
    dupes = df["barcode"].duplicated().sum()
    if dupes == 0:
        report.passed("Unique Barcodes", "No duplicates")
    else:
        report.fail("Unique Barcodes", f"{dupes} duplicate barcodes found")


def check_numeric_ranges(df: pd.DataFrame, report: ValidationReport):
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            report.warn(f"Range Check: {col}", "Column missing — skipped")
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        below = (series < lo).sum()
        above = (series > hi).sum()
        nans  = series.isna().sum()

        issues = []
        if below > 0:
            issues.append(f"{below} values < {lo}")
        if above > 0:
            issues.append(f"{above} values > {hi}")

        if issues:
            report.fail(f"Range Check: {col}", " | ".join(issues))
        else:
            nan_note = f"({nans} NaN — intentional missing values OK)"
            report.passed(f"Range Check: {col}", nan_note if nans else "All in range")


def check_boolean_columns(df: pd.DataFrame, report: ValidationReport):
    valid_bool_values = {"true", "false", "True", "False", "TRUE", "FALSE",
                         "0", "1", 0, 1, True, False}
    for col in BOOLEAN_COLUMNS:
        if col not in df.columns:
            report.warn(f"Boolean Check: {col}", "Column missing — skipped")
            continue
        invalid = df[col].apply(
            lambda x: str(x).strip() not in {str(v) for v in valid_bool_values}
            if not pd.isna(x) else False
        )
        n_invalid = invalid.sum()
        if n_invalid == 0:
            report.passed(f"Boolean Check: {col}", "All valid")
        else:
            report.fail(f"Boolean Check: {col}", f"{n_invalid} invalid values")
            report.info(str(df.loc[invalid, col].unique().tolist()))


def check_categories(df: pd.DataFrame, report: ValidationReport):
    if "category" not in df.columns:
        report.fail("Categories", "Column missing")
        return
    found_cats = set(df["category"].dropna().unique())
    unknown_cats = found_cats - VALID_CATEGORIES
    if unknown_cats:
        report.warn("Categories", f"Unexpected categories: {unknown_cats}")
    else:
        report.passed("Categories", f"All categories recognized: {len(found_cats)}")

    report.info("Category distribution:")
    dist = df["category"].value_counts()
    for cat, count in dist.items():
        expected = EXPECTED_CATEGORY_COUNTS.get(cat, "?")
        flag = "" if count == expected else f" (expected ~{expected})"
        report.info(f"  {cat:<35} {count:>3}{flag}")


def check_processing_levels(df: pd.DataFrame, report: ValidationReport):
    if "processing_level" not in df.columns:
        report.fail("Processing Level", "Column missing")
        return
    invalid = df[~df["processing_level"].isin(VALID_PROCESSING_LEVELS)]["processing_level"]
    if len(invalid) == 0:
        report.passed("Processing Level", "All values valid")
    else:
        report.fail("Processing Level", f"Invalid values: {invalid.unique().tolist()}")


def check_nutritional_consistency(df: pd.DataFrame, report: ValidationReport):
    """
    Consistency rules:
    1. sugar ≤ carbs         (sugar is a subset of carbohydrates)
    2. added_sugar ≤ sugar   (added sugar cannot exceed total sugar)
    3. sat_fat ≤ total_fat   (saturated fat is a subset of total fat)
    4. energy > 0            (all food products must have positive energy)
    5. sodium ≈ salt * 393   (1g salt = ~393mg sodium; check within 20% tolerance)
    """
    issues_total = 0

    # 1. sugar ≤ carbs
    mask = (df["sugar_g_100g"] > df["carbs_g_100g"] + 0.5)
    n = mask.sum()
    if n > 0:
        report.fail("Consistency: sugar ≤ carbs", f"{n} products violate this")
        issues_total += n
    else:
        report.passed("Consistency: sugar ≤ carbs", "OK")

    # 2. added_sugar ≤ sugar
    mask2 = (df["added_sugar_g_100g"] > df["sugar_g_100g"] + 0.5)
    n2 = mask2.sum()
    if n2 > 0:
        report.fail("Consistency: added_sugar ≤ sugar", f"{n2} violations")
        issues_total += n2
    else:
        report.passed("Consistency: added_sugar ≤ sugar", "OK")

    # 3. sat_fat ≤ total_fat
    mask3 = (df["saturated_fat_g_100g"] > df["total_fat_g_100g"] + 0.5)
    n3 = mask3.sum()
    if n3 > 0:
        report.fail("Consistency: sat_fat ≤ total_fat", f"{n3} violations")
        issues_total += n3
    else:
        report.passed("Consistency: sat_fat ≤ total_fat", "OK")

    # 4. energy > 0
    zero_energy = (pd.to_numeric(df["energy_kcal_100g"], errors="coerce") <= 0).sum()
    if zero_energy > 0:
        report.warn("Consistency: energy > 0",
                    f"{zero_energy} products with 0/negative energy (check if valid)")
    else:
        report.passed("Consistency: energy > 0", "OK")

    # 5. sodium ≈ salt * 393 (loose tolerance check)
    df_num = df[["sodium_mg_100g", "salt_g_100g"]].apply(pd.to_numeric, errors="coerce")
    both_present = df_num.dropna()
    if len(both_present) > 0:
        expected_sodium = both_present["salt_g_100g"] * 393
        ratio = (both_present["sodium_mg_100g"] / expected_sodium).replace([np.inf, -np.inf], np.nan).dropna()
        inconsistent = ((ratio < 0.70) | (ratio > 1.30)).sum()
        if inconsistent > 0:
            report.warn("Consistency: sodium ≈ salt×393",
                        f"{inconsistent} products may have inconsistent sodium/salt values")
        else:
            report.passed("Consistency: sodium ≈ salt×393",
                          f"Checked {len(both_present)} products — within tolerance")


def check_missing_values(df: pd.DataFrame, report: ValidationReport):
    null_summary = df.isnull().sum()
    null_cols = null_summary[null_summary > 0]

    if len(null_cols) == 0:
        report.passed("Missing Values", "No NaN values found")
    else:
        report.info(f"Missing value summary ({len(null_cols)} columns with NaN):")
        for col, count in null_cols.items():
            pct = count / len(df) * 100
            level = "WARN" if pct > 20 else "INFO"
            report.info(f"  {col:<35} {count:>3} NaN ({pct:.1f}%)")

    # Ensure core nutrition columns don't have excessive missing
    core_cols = ["energy_kcal_100g", "protein_g_100g", "carbs_g_100g",
                 "sugar_g_100g", "total_fat_g_100g", "sodium_mg_100g"]
    for col in core_cols:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            if pct > 30:
                report.warn(f"High Missing: {col}", f"{pct:.1f}% NaN — consider imputation strategy")
            elif pct > 0:
                report.info(f"  Note: {col} has {pct:.1f}% NaN (intentional per UNKNOWN≠ZERO rule)")


def check_score_range(df: pd.DataFrame, report: ValidationReport):
    if "food_score" not in df.columns:
        report.fail("Score Range", "food_score column missing")
        return
    scores = pd.to_numeric(df["food_score"], errors="coerce")
    out_of_range = ((scores < 0) | (scores > 100)).sum()
    nans = scores.isna().sum()
    if out_of_range > 0:
        report.fail("Score Range [0–100]", f"{out_of_range} out-of-range scores")
    else:
        report.passed("Score Range [0–100]",
                      f"Min: {scores.min():.0f}, Max: {scores.max():.0f}, "
                      f"Mean: {scores.mean():.1f}")
    if nans > 0:
        report.warn("Score NaN", f"{nans} products with missing food_score")


def check_user_preferences(report: ValidationReport):
    if not os.path.exists(PREFS_PATH):
        report.warn("User Prefs File", f"NOT FOUND: {PREFS_PATH} — generate it")
        return
    prefs = pd.read_csv(PREFS_PATH)
    report.passed("User Prefs File", f"{len(prefs)} users loaded")
    weight_cols = [c for c in prefs.columns if c.endswith("_weight")]
    if weight_cols:
        weights_sum = prefs[weight_cols].sum(axis=1)
        bad = (abs(weights_sum - 1.0) > 0.01).sum()
        if bad > 0:
            report.warn("Preference Weights", f"{bad} users have weights not summing to 1.0")
        else:
            report.passed("Preference Weights", "All user weights sum to 1.0")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    report = ValidationReport()

    print("=" * 70)
    print("AAROGYA — Dataset Validation")
    print("=" * 70)
    print(f"Dataset: {DATASET_PATH}")
    print()

    # 1. File existence
    if not check_file_exists(report):
        report.summary()
        sys.exit(1)

    # Load dataset
    try:
        df = pd.read_csv(DATASET_PATH)
        report.passed("CSV Parseable", f"{len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        report.fail("CSV Parseable", str(e))
        report.summary()
        sys.exit(1)

    print()
    # 2. Required columns
    check_required_columns(df, report)
    print()
    # 3. Row count
    check_row_count(df, report)
    print()
    # 4. Duplicate IDs
    check_duplicate_ids(df, report)
    # 5. Duplicate barcodes
    check_duplicate_barcodes(df, report)
    print()
    # 6. Numeric ranges
    check_numeric_ranges(df, report)
    print()
    # 7. Boolean columns
    check_boolean_columns(df, report)
    print()
    # 8. Categories
    check_categories(df, report)
    print()
    # 9. Processing levels
    check_processing_levels(df, report)
    print()
    # 10. Nutritional consistency
    check_nutritional_consistency(df, report)
    print()
    # 11. Missing values
    check_missing_values(df, report)
    print()
    # 12. Score range
    check_score_range(df, report)
    print()
    # 13. User preferences file
    check_user_preferences(report)

    # Final summary
    n_failures = report.summary()
    sys.exit(0 if n_failures == 0 else 1)


if __name__ == "__main__":
    main()
