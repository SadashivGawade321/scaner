"""
data/generate_large_dataset.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Large Synthetic Dataset Generator (Week 2)
─────────────────────────────────────────────────────────────────────────────
Generates 500 synthetic products for ML training.
Nutrition values are drawn from realistic distributions per category.

USAGE
─────
    python data/generate_large_dataset.py

OUTPUT
──────
    data/raw/aarogya_products_500.csv   (500 products)
"""

import os, sys, csv, random
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)
np.random.seed(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "data", "raw", "aarogya_products_500.csv")

# ─── Category blueprints ───────────────────────────────────────────────────
# Each entry: (category, subcategory, rec_group, processing_level_weights,
#              nutrition_means, nutrition_stds)
# nutrition order: energy, protein, carbs, sugar, added_sugar, fat, sat_fat,
#                  trans, fiber, sodium, serving, ingredient_count
CATEGORIES = [
    {
        "category": "Biscuits & Cookies",
        "subcategories": ["Digestive Biscuits","Butter Cookies","Crackers",
                          "Oatmeal Cookies","Cream Sandwich","Millet Biscuits",
                          "High Fiber","Glucose Biscuits"],
        "group": "biscuits_cookies",
        "processing_dist": {"minimally_processed":0.2,"processed":0.3,"ultra_processed":0.5},
        "nutrition": {
            "energy":    (435, 30),  "protein":  (8.0, 2.0),
            "carbs":     (64,  8),   "sugar":    (14,  10),
            "added_sugar":(12, 9),   "fat":      (16,  5),
            "sat_fat":   (6,  3),    "trans":    (0.1, 0.1),
            "fiber":     (3.5, 2.5), "sodium":   (380, 150),
            "serving":   (30, 5),    "ingr_cnt": (11, 3),
        },
        "whole_grain_prob": 0.35,
    },
    {
        "category": "Chips & Namkeen",
        "subcategories": ["Potato Chips","Puffed Rice","Makhana","Bhujia",
                          "Lentil Snacks","Tortilla Chips","Roasted Nuts",
                          "Popcorn","Roasted Legumes"],
        "group": "chips_namkeen",
        "processing_dist": {"minimally_processed":0.25,"processed":0.45,"ultra_processed":0.30},
        "nutrition": {
            "energy":    (470, 60),  "protein":  (9.0, 5.5),
            "carbs":     (52,  18),  "sugar":    (3,   3),
            "added_sugar":(2,  2.5), "fat":      (28,  15),
            "sat_fat":   (6,  4),    "trans":    (0.15,0.12),
            "fiber":     (4.5, 3.5), "sodium":   (450, 180),
            "serving":   (30, 5),    "ingr_cnt": (7, 3),
        },
        "whole_grain_prob": 0.20,
    },
    {
        "category": "Breakfast Cereals",
        "subcategories": ["Corn Flakes","Muesli","Granola","Oat Porridge",
                          "Wheat Puffs","Bran Flakes","Millet Flakes","Ancient Grain"],
        "group": "breakfast_cereals",
        "processing_dist": {"minimally_processed":0.35,"processed":0.35,"ultra_processed":0.30},
        "nutrition": {
            "energy":    (368, 25),  "protein":  (9.5, 2.5),
            "carbs":     (68,  10),  "sugar":    (13,  11),
            "added_sugar":(10, 10),  "fat":      (5.5, 4.5),
            "sat_fat":   (1.5, 1.5), "trans":    (0.0, 0.05),
            "fiber":     (7.5, 3.5), "sodium":   (280, 180),
            "serving":   (40, 8),    "ingr_cnt": (10, 3),
        },
        "whole_grain_prob": 0.55,
    },
    {
        "category": "Beverages",
        "subcategories": ["Fruit Drink","Tea RTD","Coffee Drink","Coconut Water",
                          "Sports Drink","Plant Milk","Protein Shake",
                          "Traditional Drink","Carbonated Soft Drink"],
        "group": "beverages",
        "processing_dist": {"minimally_processed":0.30,"processed":0.30,"ultra_processed":0.40},
        "nutrition": {
            "energy":    (55, 60),   "protein":  (2.0, 5.0),
            "carbs":     (10, 9),    "sugar":    (8,   8),
            "added_sugar":(6, 7),    "fat":      (1.0, 2.0),
            "sat_fat":   (0.5, 1.0), "trans":    (0.0, 0.02),
            "fiber":     (0.3, 0.5), "sodium":   (80,  80),
            "serving":   (250, 60),  "ingr_cnt": (7, 3),
        },
        "whole_grain_prob": 0.05,
    },
    {
        "category": "Instant Foods",
        "subcategories": ["Instant Noodles","Instant Oats","Instant Mix",
                          "Instant Soup","Instant Curry Mix","Instant Poha",
                          "Instant Khichdi","Instant Idli Mix"],
        "group": "instant_foods",
        "processing_dist": {"minimally_processed":0.20,"processed":0.35,"ultra_processed":0.45},
        "nutrition": {
            "energy":    (340, 30),  "protein":  (10, 4),
            "carbs":     (62,  10),  "sugar":    (4,  4),
            "added_sugar":(3,  3.5), "fat":      (7,  5),
            "sat_fat":   (2.5, 1.5), "trans":    (0.1, 0.1),
            "fiber":     (4.5, 3.0), "sodium":   (680, 380),
            "serving":   (50, 15),   "ingr_cnt": (11, 3),
        },
        "whole_grain_prob": 0.25,
    },
    {
        "category": "Chocolates & Sweet Snacks",
        "subcategories": ["Dark Chocolate","Milk Chocolate","White Chocolate",
                          "Chikki","Fudge","Natural Confection","Chocolate Bar"],
        "group": "chocolates_sweets",
        "processing_dist": {"minimally_processed":0.20,"processed":0.20,"ultra_processed":0.60},
        "nutrition": {
            "energy":    (510, 55),  "protein":  (6.5, 3.0),
            "carbs":     (54,  14),  "sugar":    (44,  16),
            "added_sugar":(30, 18),  "fat":      (28,  14),
            "sat_fat":   (14, 8),    "trans":    (0.15,0.1),
            "fiber":     (3.5, 3.5), "sodium":   (90,  90),
            "serving":   (30, 8),    "ingr_cnt": (9, 3),
        },
        "whole_grain_prob": 0.05,
    },
    {
        "category": "Snack/Protein Bars",
        "subcategories": ["Protein Bar","Granola Bar","Energy Bar",
                          "Wellness Bar","Vegan Bar","Keto Bar"],
        "group": "snack_protein_bars",
        "processing_dist": {"minimally_processed":0.30,"processed":0.45,"ultra_processed":0.25},
        "nutrition": {
            "energy":    (400, 45),  "protein":  (14, 7),
            "carbs":     (52,  14),  "sugar":    (22, 14),
            "added_sugar":(16, 14),  "fat":      (16, 9),
            "sat_fat":   (5.5, 4),   "trans":    (0.05,0.05),
            "fiber":     (6.5, 3.5), "sodium":   (170, 90),
            "serving":   (45, 10),   "ingr_cnt": (10, 2),
        },
        "whole_grain_prob": 0.45,
    },
    {
        "category": "Bread & Bakery",
        "subcategories": ["Whole Wheat Bread","White Bread","Multigrain Bread",
                          "Millet Bread","Sweet Loaf"],
        "group": "bread_bakery",
        "processing_dist": {"minimally_processed":0.25,"processed":0.45,"ultra_processed":0.30},
        "nutrition": {
            "energy":    (255, 25),  "protein":  (9.5, 1.5),
            "carbs":     (47,  5),   "sugar":    (4.5, 3),
            "added_sugar":(3.5, 3),  "fat":      (4.0, 2),
            "sat_fat":   (1.2, 1.0), "trans":    (0.05,0.05),
            "fiber":     (5.0, 2.5), "sodium":   (400, 100),
            "serving":   (40, 5),    "ingr_cnt": (9, 2),
        },
        "whole_grain_prob": 0.55,
    },
    {
        "category": "Dairy & Dairy Drinks",
        "subcategories": ["Plain Yoghurt","Greek Yoghurt","Flavored Yoghurt",
                          "Full Cream Milk","Lassi","Paneer","Flavored Milk"],
        "group": "dairy_drinks",
        "processing_dist": {"minimally_processed":0.50,"processed":0.35,"ultra_processed":0.15},
        "nutrition": {
            "energy":    (80, 45),   "protein":  (6.5, 4.5),
            "carbs":     (8,  5),    "sugar":    (7,   5),
            "added_sugar":(4, 5),    "fat":      (4.5, 4.5),
            "sat_fat":   (2.5, 2.5), "trans":    (0.0, 0.02),
            "fiber":     (0.0, 0.1), "sodium":   (60,  50),
            "serving":   (150, 50),  "ingr_cnt": (5, 3),
        },
        "whole_grain_prob": 0.00,
    },
    {
        "category": "Sauces & Condiments",
        "subcategories": ["Tomato Ketchup","Green Chutney","Pickle",
                          "Mayonnaise","Hot Sauce","Salad Dressing"],
        "group": "sauces_condiments",
        "processing_dist": {"minimally_processed":0.15,"processed":0.35,"ultra_processed":0.50},
        "nutrition": {
            "energy":    (150, 100), "protein":  (1.5, 1.5),
            "carbs":     (18,  14),  "sugar":    (14,  12),
            "added_sugar":(12, 11),  "fat":      (8,   12),
            "sat_fat":   (1.5, 2.0), "trans":    (0.05,0.05),
            "fiber":     (1.0, 1.0), "sodium":   (900, 400),
            "serving":   (20, 10),   "ingr_cnt": (12, 4),
        },
        "whole_grain_prob": 0.05,
    },
]

# ─── Ingredient templates ──────────────────────────────────────────────────
INGREDIENTS = {
    "minimally_processed": [
        "Whole wheat flour, oats, seeds, jaggery, sunflower oil, salt",
        "Brown rice, lentils, vegetables, spices, salt",
        "Puffed quinoa, amaranth, seeds, coconut sugar, cinnamon",
        "Almonds, cashews, raisins, sunflower seeds, salt",
        "Rolled oats, dried fruits, seeds, honey, coconut oil",
    ],
    "processed": [
        "Whole wheat flour, refined flour, sugar, vegetable oil, salt, yeast, emulsifiers",
        "Rice flour, corn grits, sunflower oil, spice mix, salt, citric acid",
        "Oats, milk powder, sugar, malt extract, salt, vitamins",
        "Skimmed milk, sugar, stabilizers, flavours, live cultures",
        "Chickpeas, sunflower oil, spice blend, salt, lemon juice powder",
    ],
    "ultra_processed": [
        "Refined wheat flour, sugar, palm oil, glucose syrup, emulsifiers, artificial flavours, salt",
        "Corn grits, sugar, artificial colours, sodium benzoate, artificial flavours",
        "Sugar, vegetable fat, skim milk powder, glucose syrup, emulsifiers, artificial vanilla",
        "Refined flour, palm oil, MSG, hydrolyzed soy protein, artificial flavours, salt",
        "Corn syrup, sugar, artificial colours, sodium benzoate, citric acid, preservatives",
    ],
}

ALLERGEN_POOL = ["wheat,gluten","wheat,gluten,milk","milk,soy","nuts","peanuts,sesame",
                 "wheat,gluten,soy","oats,nuts","milk","soy",""]

BRANDS = ["NutriCo","GrainMart","HealthFirst","FitBite","DesiGood","PureNature",
          "QuickMeal","SlimChoice","EarthFoods","ProSnack","MorningBest","CrunchWave",
          "SpiceBlend","SweetNature","DairyPure","GreenLeaf","PowerFuel","BakeMaster",
          "TrailBlaze","HarvestGold","CleanEat","SwiftGrain","NutriPulse","SunriseFarm"]

HEADER = ["product_id","barcode","product_name","brand","category","subcategory",
          "recommendation_group","serving_size_g","energy_kcal_100g","protein_g_100g",
          "carbs_g_100g","sugar_g_100g","added_sugar_g_100g","total_fat_g_100g",
          "saturated_fat_g_100g","trans_fat_g_100g","fiber_g_100g","sodium_mg_100g",
          "salt_g_100g","ingredients_text","ingredient_count","contains_whole_grain",
          "contains_added_sugar","contains_artificial_sweetener","contains_allergen",
          "allergens","processing_level","food_score","score_version","data_source","verified"]


def sample_nutrition(blueprint, proc_level):
    n = blueprint["nutrition"]
    def pos(mean, std, lo=0.0, hi=None):
        v = max(lo, np.random.normal(mean, std))
        return round(v if hi is None else min(v, hi), 1)

    energy   = pos(n["energy"][0],   n["energy"][1],   10, 650)
    protein  = pos(n["protein"][0],  n["protein"][1],  0,  80)
    carbs    = pos(n["carbs"][0],    n["carbs"][1],    0, 100)
    sugar    = pos(n["sugar"][0],    n["sugar"][1],    0,  min(carbs, 80))
    added    = pos(n["added_sugar"][0], n["added_sugar"][1], 0, sugar)
    fat      = pos(n["fat"][0],      n["fat"][1],      0,  70)
    sat_fat  = pos(n["sat_fat"][0],  n["sat_fat"][1],  0,  fat)
    trans    = pos(n["trans"][0],    n["trans"][1],    0,  min(fat*0.2, 2.0))
    fiber    = pos(n["fiber"][0],    n["fiber"][1],    0,  30)
    sodium   = pos(n["sodium"][0],   n["sodium"][1],   0, 2500)
    salt     = round(sodium / 393, 2)
    serving  = pos(n["serving"][0],  n["serving"][1],  10,  500)
    ingr_cnt = max(2, int(np.random.normal(n["ingr_cnt"][0], n["ingr_cnt"][1])))

    # Ultra-processed: slightly bump sugar and sodium
    if proc_level == "ultra_processed":
        sugar   = min(sugar * 1.2, carbs)
        sodium  = min(sodium * 1.3, 2500)
        added   = min(added * 1.2, sugar)
        salt    = round(sodium / 393, 2)

    return {
        "energy": energy, "protein": protein, "carbs": round(carbs,1),
        "sugar": round(sugar,1), "added_sugar": round(added,1),
        "fat": fat, "sat_fat": sat_fat, "trans": trans,
        "fiber": fiber, "sodium": round(sodium,0), "salt": salt,
        "serving": round(serving,0), "ingr_cnt": ingr_cnt,
    }


def compute_score(n, proc_level, whole_grain, artificial_sweetener):
    """Simplified deterministic score (mirrors ml/scoring.py logic)."""
    def lerp(val, lo, hi):
        return max(0.0, min(100.0, (val - lo) / (hi - lo) * 100))

    sugar_score    = 100 - lerp(n["sugar"],    0, 50)
    protein_score  = lerp(n["protein"],        0, 30)
    fiber_score    = lerp(n["fiber"],           0, 15)
    sodium_score   = 100 - lerp(n["sodium"],   0, 1500)
    sat_fat_score  = 100 - lerp(n["sat_fat"],  0, 30)
    energy_score   = 100 - lerp(n["energy"],   50, 600)

    weighted = (
        sugar_score   * 0.25 +
        protein_score * 0.20 +
        fiber_score   * 0.20 +
        sodium_score  * 0.15 +
        sat_fat_score * 0.12 +
        energy_score  * 0.08
    )
    mod = 0.0
    if proc_level == "minimally_processed": mod += 5.0
    elif proc_level == "ultra_processed":   mod -= 8.0
    if whole_grain:       mod += 3.0
    if artificial_sweetener: mod -= 3.0

    return round(max(0.0, min(100.0, weighted + mod)), 0)


def generate(n_products=500):
    rows = []
    # Distribute products proportionally across categories
    per_cat = {}
    total_weight = sum(len(c["subcategories"]) for c in CATEGORIES)
    remaining = n_products
    for i, cat in enumerate(CATEGORIES):
        if i == len(CATEGORIES) - 1:
            per_cat[cat["category"]] = remaining
        else:
            count = max(5, round(len(cat["subcategories"]) / total_weight * n_products))
            per_cat[cat["category"]] = count
            remaining -= count

    pid = 1
    for cat in CATEGORIES:
        cat_count = per_cat[cat["category"]]
        proc_choices = list(cat["processing_dist"].keys())
        proc_weights = list(cat["processing_dist"].values())

        for _ in range(cat_count):
            proc = random.choices(proc_choices, weights=proc_weights)[0]
            sub  = random.choice(cat["subcategories"])
            brand = random.choice(BRANDS)
            name  = f"{brand} {sub} {'Premium' if random.random()>0.7 else ''}"
            name  = name.strip()

            n = sample_nutrition(cat, proc)
            whole_grain  = random.random() < cat["whole_grain_prob"]
            has_sweetener = proc == "ultra_processed" and random.random() < 0.15
            allergen_str  = random.choice(ALLERGEN_POOL)
            has_allergen  = bool(allergen_str)
            has_added_sugar = n["added_sugar"] > 1.0
            ingredients   = random.choice(INGREDIENTS[proc])

            score = int(compute_score(n, proc, whole_grain, has_sweetener))

            rows.append([
                f"AAR{pid:04d}",
                f"890100{pid:07d}",
                name, brand,
                cat["category"], sub, cat["group"],
                n["serving"],
                n["energy"], n["protein"], n["carbs"],
                n["sugar"], n["added_sugar"], n["fat"],
                n["sat_fat"], n["trans"], n["fiber"],
                n["sodium"], n["salt"],
                ingredients, n["ingr_cnt"],
                str(whole_grain).lower(),
                str(has_added_sugar).lower(),
                str(has_sweetener).lower(),
                str(has_allergen).lower(),
                allergen_str, proc,
                score, "v1.0", "synthetic_v2", "false"
            ])
            pid += 1

    return rows


def main():
    print("Generating 500-product synthetic dataset...")
    rows = generate(500)
    os.makedirs(os.path.join(ROOT, "data", "raw"), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow(row)

    # Verify
    try:
        import pandas as pd
        df = pd.read_csv(OUT)
        print(f"SUCCESS: {len(df)} rows, {len(df.columns)} cols")
        print(f"Score range: {df['food_score'].min()} – {df['food_score'].max()}")
        print(f"Score mean:  {df['food_score'].mean():.1f}")
        print("\nCategory distribution:")
        print(df["category"].value_counts().to_string())
    except ImportError:
        print(f"Written {len(rows)} rows to {OUT}")

if __name__ == "__main__":
    main()
