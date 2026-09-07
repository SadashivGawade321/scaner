"""
database/seed_mongodb.py
========================
Seeds the Aarogya MongoDB Atlas database with:
  1. Full product catalog (aarogya_products_500.csv)
  2. Demo products (hardcoded)

Run once:
    cd aarogya
    python database/seed_mongodb.py
"""
import os, sys, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

import pymongo

MONGO_URI = os.getenv("MONGO_URI","")
DB_NAME   = os.getenv("MONGO_DB_NAME","aarogya_db")

if not MONGO_URI:
    print("ERROR: MONGO_URI not set in .env"); sys.exit(1)

print(f"Connecting to MongoDB Atlas...")
client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
client.admin.command("ping")
print("Connected!")

db = client[DB_NAME]

# ── 1. Seed Product Catalog ────────────────────────────────────────────────
catalog_col = db["products"]

for csv_path in [
    os.path.join(ROOT,"data","raw","aarogya_products_500.csv"),
    os.path.join(ROOT,"data","raw","aarogya_food_products_demo.csv"),
]:
    if not os.path.exists(csv_path): continue
    df = pd.read_csv(csv_path)
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    print(f"Upserting {len(records)} products from {os.path.basename(csv_path)}...")
    for rec in records:
        name = rec.get("product_name","")
        if not name: continue
        catalog_col.update_one(
            {"product_name": name},
            {"$set": rec},
            upsert=True
        )
    print(f"  Done.")

# ── 2. Seed Demo & Popular Indian Products ─────────────────────────────────
demo_products = [
    {
        "product_name": "ChocoBite Cream Biscuits", "brand": "SnackTime",
        "category": "Biscuits & Cookies", "recommendation_group": "biscuits_cookies",
        "processing_level": "ultra_processed", "food_score": 22,
        "energy_kcal_100g": 520, "protein_g_100g": 5, "carbs_g_100g": 68,
        "sugar_g_100g": 38, "added_sugar_g_100g": 35, "total_fat_g_100g": 22,
        "saturated_fat_g_100g": 12, "fiber_g_100g": 1, "sodium_mg_100g": 420,
        "is_demo": True,
    },
    {
        "product_name": "NutriOats Instant Porridge", "brand": "HealthFirst",
        "category": "Breakfast Cereals", "recommendation_group": "breakfast_cereals",
        "processing_level": "minimally_processed", "food_score": 86,
        "energy_kcal_100g": 380, "protein_g_100g": 13, "carbs_g_100g": 62,
        "sugar_g_100g": 4, "added_sugar_g_100g": 0, "total_fat_g_100g": 7,
        "saturated_fat_g_100g": 1.5, "fiber_g_100g": 9, "sodium_mg_100g": 95,
        "is_demo": True,
    },
    {
        "product_name": "MasalaKing Spicy Chips", "brand": "CrunchCo",
        "category": "Chips & Namkeen", "recommendation_group": "chips_namkeen",
        "processing_level": "ultra_processed", "food_score": 18,
        "energy_kcal_100g": 550, "protein_g_100g": 6, "carbs_g_100g": 58,
        "sugar_g_100g": 3, "added_sugar_g_100g": 2, "total_fat_g_100g": 32,
        "saturated_fat_g_100g": 8, "fiber_g_100g": 3, "sodium_mg_100g": 680,
        "is_demo": True,
    },
    {
        "product_name": "Maggi 2-Minute Masala Instant Noodles", "brand": "Nestle",
        "category": "Instant Noodles", "recommendation_group": "instant_noodles",
        "processing_level": "ultra_processed", "food_score": 28,
        "energy_kcal_100g": 427, "protein_g_100g": 8, "carbs_g_100g": 63.5,
        "sugar_g_100g": 2.2, "added_sugar_g_100g": 1.5, "total_fat_g_100g": 15.7,
        "saturated_fat_g_100g": 6.8, "fiber_g_100g": 3.5, "sodium_mg_100g": 1050,
        "is_demo": False,
    },
    {
        "product_name": "Maggi Nutri-licious Atta Noodles", "brand": "Nestle",
        "category": "Instant Noodles", "recommendation_group": "instant_noodles",
        "processing_level": "processed", "food_score": 58,
        "energy_kcal_100g": 410, "protein_g_100g": 9.5, "carbs_g_100g": 65,
        "sugar_g_100g": 2.0, "added_sugar_g_100g": 1.0, "total_fat_g_100g": 13.0,
        "saturated_fat_g_100g": 5.2, "fiber_g_100g": 7.0, "sodium_mg_100g": 780,
        "is_demo": False,
    },
    {
        "product_name": "Parle-G Original Gluco Biscuits", "brand": "Parle",
        "category": "Biscuits & Cookies", "recommendation_group": "biscuits_cookies",
        "processing_level": "ultra_processed", "food_score": 32,
        "energy_kcal_100g": 454, "protein_g_100g": 6.5, "carbs_g_100g": 78.2,
        "sugar_g_100g": 25.5, "added_sugar_g_100g": 24.0, "total_fat_g_100g": 13.0,
        "saturated_fat_g_100g": 6.0, "fiber_g_100g": 1.5, "sodium_mg_100g": 280,
        "is_demo": False,
    },
    {
        "product_name": "Britannia Good Day Butter Cookies", "brand": "Britannia",
        "category": "Biscuits & Cookies", "recommendation_group": "biscuits_cookies",
        "processing_level": "ultra_processed", "food_score": 25,
        "energy_kcal_100g": 490, "protein_g_100g": 5.8, "carbs_g_100g": 68.0,
        "sugar_g_100g": 31.0, "added_sugar_g_100g": 28.0, "total_fat_g_100g": 22.0,
        "saturated_fat_g_100g": 10.5, "fiber_g_100g": 1.0, "sodium_mg_100g": 340,
        "is_demo": False,
    },
    {
        "product_name": "Kurkure Masala Munch", "brand": "PepsiCo",
        "category": "Chips & Namkeen", "recommendation_group": "chips_namkeen",
        "processing_level": "ultra_processed", "food_score": 20,
        "energy_kcal_100g": 560, "protein_g_100g": 5.8, "carbs_g_100g": 55.4,
        "sugar_g_100g": 1.5, "added_sugar_g_100g": 0.5, "total_fat_g_100g": 35.0,
        "saturated_fat_g_100g": 14.5, "fiber_g_100g": 2.0, "sodium_mg_100g": 920,
        "is_demo": False,
    },
    {
        "product_name": "Lay's India's Magic Masala", "brand": "PepsiCo",
        "category": "Chips & Namkeen", "recommendation_group": "chips_namkeen",
        "processing_level": "ultra_processed", "food_score": 21,
        "energy_kcal_100g": 544, "protein_g_100g": 6.8, "carbs_g_100g": 51.5,
        "sugar_g_100g": 2.5, "added_sugar_g_100g": 1.5, "total_fat_g_100g": 34.8,
        "saturated_fat_g_100g": 12.0, "fiber_g_100g": 3.8, "sodium_mg_100g": 790,
        "is_demo": False,
    },
    {
        "product_name": "Amul Pasteurised Butter", "brand": "Amul",
        "category": "Dairy", "recommendation_group": "dairy",
        "processing_level": "processed", "food_score": 45,
        "energy_kcal_100g": 720, "protein_g_100g": 0.5, "carbs_g_100g": 0.0,
        "sugar_g_100g": 0.0, "added_sugar_g_100g": 0.0, "total_fat_g_100g": 80.0,
        "saturated_fat_g_100g": 51.0, "fiber_g_100g": 0.0, "sodium_mg_100g": 830,
        "is_demo": False,
    },
]
print(f"Upserting {len(demo_products)} demo/staple products...")
for rec in demo_products:
    catalog_col.update_one(
        {"product_name": rec["product_name"]},
        {"$set": rec}, upsert=True)
print("  Done.")

# ── 3. Create indexes ──────────────────────────────────────────────────────
catalog_col.create_index("product_name")
catalog_col.create_index("category")
catalog_col.create_index("food_score")
db["scan_history"].create_index([("scanned_at", pymongo.DESCENDING)])

# ── Summary ────────────────────────────────────────────────────────────────
total = catalog_col.count_documents({})
print("\n[OK] MongoDB seeded successfully!")
print(f"   Database:       {DB_NAME}")
print(f"   Products total: {total}")
print(f"   Collections:    products, scan_history")
client.close()
