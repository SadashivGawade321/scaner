"""
database/mongodb.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — MongoDB Connection and Collection Manager
─────────────────────────────────────────────────────────────────────────────

DATABASE: aarogya_db

COLLECTIONS
───────────
  products          — Product catalog with nutrition & ingredient data
  users             — User accounts and preference profiles
  scans             — History of product scans (barcode/OCR)
  recommendations   — Recommendation results (logged per session)
  model_predictions — ML model outputs per product per version
  scoring_versions  — Scoring methodology metadata

NOTE
────
This module is OPTIONAL for Day 1 (dataset + scoring work does NOT require
MongoDB). Include this when you are ready to wire up the FastAPI backend.

Day 1: You may test the connection but do not need to load data yet.
Week 13+: Full MongoDB integration.

USAGE
─────
    from database.mongodb import AarogyaDB
    db = AarogyaDB()                  # connects using MONGO_URI from .env
    db.upsert_product(product_dict)   # insert/update a product
    db.get_product_by_barcode("...")  # barcode lookup

REQUIRES: pymongo, python-dotenv
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports — MongoDB is not a Day 1 hard requirement
# ─────────────────────────────────────────────────────────────────────────────

try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.errors import ConnectionFailure, DuplicateKeyError
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False
    logger.warning(
        "pymongo not installed. MongoDB features unavailable. "
        "Install with: pip install pymongo"
    )

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # dotenv is optional; use system env vars directly


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA DEFINITIONS (document templates — for documentation & validation)
# ─────────────────────────────────────────────────────────────────────────────

PRODUCT_SCHEMA_EXAMPLE = {
    "_id":                   "AAR001",              # = product_id
    "barcode":               "8901001000001",
    "product_name":          "Wheatfield Digestive Biscuits",
    "brand":                 "Crunchway",
    "category":              "Biscuits & Cookies",
    "subcategory":           "Digestive Biscuits",
    "recommendation_group":  "biscuits_cookies",
    "nutrition": {
        "energy_kcal_100g":      430.0,
        "protein_g_100g":          7.2,
        "carbs_g_100g":           68.0,
        "sugar_g_100g":           18.0,
        "added_sugar_g_100g":     16.5,
        "total_fat_g_100g":       14.5,
        "saturated_fat_g_100g":    5.8,
        "trans_fat_g_100g":        0.1,
        "fiber_g_100g":            4.2,
        "sodium_mg_100g":        420.0,
        "salt_g_100g":             1.05,
        "serving_size_g":         30.0,
    },
    "ingredients": {
        "ingredients_text":       "Whole wheat flour, sugar...",
        "ingredient_count":       14,
        "contains_whole_grain":   True,
        "contains_added_sugar":   True,
        "contains_artificial_sweetener": False,
        "contains_allergen":      True,
        "allergens":              "wheat,gluten",
        "processing_level":       "ultra_processed",
    },
    "scores": {
        "food_score":         58.0,
        "computed_score":     57.4,
        "score_version":      "v1.0",
        "score_strengths":    ["fiber"],
        "score_concerns":     ["sugar", "sodium"],
    },
    "metadata": {
        "data_source":        "demo_synthetic",
        "verified":           False,
        "created_at":         "2024-01-01T00:00:00Z",
        "updated_at":         "2024-01-01T00:00:00Z",
    }
}

USER_SCHEMA_EXAMPLE = {
    "_id":               "USR001",
    "preference_profile": "balanced",
    "preferences": {
        "sugar_weight":         0.20,
        "protein_weight":       0.20,
        "fiber_weight":         0.20,
        "sodium_weight":        0.20,
        "calorie_weight":       0.10,
        "saturated_fat_weight": 0.10,
    },
    "scan_history": [],       # list of product_ids scanned
    "created_at": "2024-01-01T00:00:00Z",
}

SCAN_SCHEMA_EXAMPLE = {
    "user_id":      "USR001",
    "product_id":   "AAR001",
    "barcode":      "8901001000001",
    "scan_method":  "barcode",   # "barcode" or "ocr"
    "scanned_at":   "2024-01-01T00:00:00Z",
    "ocr_raw_text": None,        # populated only for OCR scans
    "confidence":   None,
}

MODEL_PREDICTION_SCHEMA_EXAMPLE = {
    "product_id":       "AAR001",
    "model_version":    "xgb_v1.0",
    "score_version":    "v1.0",
    "predicted_score":  57.2,
    "shap_values": {
        "sugar":         -8.5,
        "protein":        2.1,
        "fiber":          5.3,
        "sodium":        -3.2,
        "saturated_fat": -1.8,
        "energy":        -0.5,
    },
    "prediction_at": "2024-01-01T00:00:00Z",
}

SCORING_VERSION_SCHEMA_EXAMPLE = {
    "version":    "v1.0",
    "description": "Initial deterministic scoring methodology",
    "weights": {
        "sugar":         0.25,
        "protein":       0.20,
        "fiber":         0.20,
        "sodium":        0.15,
        "saturated_fat": 0.12,
        "energy":        0.08,
    },
    "created_at": "2024-01-01T00:00:00Z",
    "active":     True,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class AarogyaDB:
    """
    MongoDB interface for Aarogya.

    All methods are safe to call even if MongoDB is unavailable —
    they log warnings and return None rather than crashing.
    This allows Day 1 work (scoring, feature engineering) to proceed
    without a running MongoDB instance.
    """

    def __init__(self):
        self.client = None
        self.db = None
        self._connected = False

        if not MONGO_AVAILABLE:
            logger.warning("AarogyaDB: pymongo not available. All DB calls will no-op.")
            return

        uri     = os.getenv("MONGO_URI",     "mongodb://localhost:27017")
        db_name = os.getenv("MONGO_DB_NAME", "aarogya_db")

        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command("ping")
            self.db = self.client[db_name]
            self._connected = True
            logger.info(f"AarogyaDB connected to {uri} / {db_name}")
            self._ensure_indexes()
        except Exception as e:
            logger.warning(f"AarogyaDB: Could not connect to MongoDB: {e}")
            logger.warning("Continuing without database. Set MONGO_URI in .env to fix.")

    @property
    def connected(self) -> bool:
        return self._connected

    def _ensure_indexes(self):
        """Create indexes for efficient lookups."""
        if not self._connected:
            return
        # Products: barcode lookup (most common query)
        self.db.products.create_index([("barcode", ASCENDING)], unique=True, sparse=True)
        # Products: recommendation group (for filtering)
        self.db.products.create_index([("recommendation_group", ASCENDING)])
        # Scans: user history
        self.db.scans.create_index([("user_id", ASCENDING), ("scanned_at", ASCENDING)])
        # Model predictions: product + version
        self.db.model_predictions.create_index(
            [("product_id", ASCENDING), ("model_version", ASCENDING)]
        )
        logger.info("MongoDB indexes ensured.")

    # ── Products ──────────────────────────────────────────────────────────────

    def upsert_product(self, product: dict) -> bool:
        """Insert or update a product by product_id."""
        if not self._connected:
            return False
        try:
            product["metadata"]["updated_at"] = _now()
            self.db.products.update_one(
                {"_id": product["product_id"]},
                {"$set": product},
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"upsert_product failed: {e}")
            return False

    def get_product_by_barcode(self, barcode: str) -> Optional[dict]:
        """Look up a product by barcode. Returns None if not found."""
        if not self._connected:
            return None
        try:
            return self.db.products.find_one({"barcode": barcode})
        except Exception as e:
            logger.error(f"get_product_by_barcode failed: {e}")
            return None

    def get_product_by_id(self, product_id: str) -> Optional[dict]:
        """Look up a product by product_id."""
        if not self._connected:
            return None
        try:
            return self.db.products.find_one({"_id": product_id})
        except Exception as e:
            logger.error(f"get_product_by_id failed: {e}")
            return None

    def get_products_by_group(self, recommendation_group: str) -> list:
        """Get all products in a recommendation group (for recommendations)."""
        if not self._connected:
            return []
        try:
            return list(self.db.products.find(
                {"recommendation_group": recommendation_group}
            ))
        except Exception as e:
            logger.error(f"get_products_by_group failed: {e}")
            return []

    # ── Scans ─────────────────────────────────────────────────────────────────

    def log_scan(self, user_id: str, product_id: str,
                 barcode: str, method: str = "barcode",
                 ocr_text: str = None) -> bool:
        """Log a product scan event."""
        if not self._connected:
            return False
        try:
            self.db.scans.insert_one({
                "user_id":      user_id,
                "product_id":   product_id,
                "barcode":      barcode,
                "scan_method":  method,
                "scanned_at":   _now(),
                "ocr_raw_text": ocr_text,
            })
            return True
        except Exception as e:
            logger.error(f"log_scan failed: {e}")
            return False

    # ── Model Predictions ─────────────────────────────────────────────────────

    def save_prediction(self, prediction: dict) -> bool:
        """Save an ML model prediction (with SHAP values)."""
        if not self._connected:
            return False
        try:
            prediction["prediction_at"] = _now()
            self.db.model_predictions.insert_one(prediction)
            return True
        except Exception as e:
            logger.error(f"save_prediction failed: {e}")
            return False

    # ── Utilities ─────────────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """Return connection health status."""
        if not self._connected:
            return {"status": "disconnected", "pymongo": MONGO_AVAILABLE}
        try:
            self.client.admin.command("ping")
            counts = {
                "products":          self.db.products.count_documents({}),
                "users":             self.db.users.count_documents({}),
                "scans":             self.db.scans.count_documents({}),
                "model_predictions": self.db.model_predictions.count_documents({}),
            }
            return {"status": "connected", "collections": counts}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def close(self):
        if self.client:
            self.client.close()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Quick Test — Run: python database/mongodb.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("AAROGYA MongoDB — Connection Test")
    print("=" * 60)

    db = AarogyaDB()
    health = db.health_check()
    print(f"Status: {health}")

    if health["status"] == "connected":
        print("\nMongoDB connection successful!")
        print(f"Collection counts: {health.get('collections', {})}")
    elif health["status"] == "disconnected":
        print("\nMongoDB not connected.")
        print("This is EXPECTED on Day 1 if MongoDB is not yet installed.")
        print("Install MongoDB or set MONGO_URI in .env")
        print("Day 1 scoring and validation work does NOT require MongoDB.")
    else:
        print(f"\nConnection error: {health.get('detail')}")

    db.close()
    print("\n[DATABASE MODULE OK]")
