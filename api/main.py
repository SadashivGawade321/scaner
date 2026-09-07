"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — FastAPI REST Backend (Week 13)
─────────────────────────────────────────────────────────────────────────────

ENDPOINTS
─────────
  GET  /                          Health check
  GET  /health                    Detailed health status
  POST /analyze                   Score + explain a product
  POST /recommend                 Get category-matched recommendations
  POST /compare                   Compare two products side by side
  GET  /product/{product_id}      Lookup product from catalog
  GET  /categories                List available categories
  GET  /top/{category}            Top 5 products in a category

USAGE
─────
    pip install fastapi uvicorn
    uvicorn api.main:app --reload --port 8000
    # Then: http://localhost:8000/docs
"""

import os, sys, json, logging
from typing import Optional, List
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aarogya.api")

# ─── FastAPI (lazy import — graceful if not installed) ─────────────────────
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not installed. Run: pip install fastapi uvicorn")

# ─── Project imports ───────────────────────────────────────────────────────
from ml.scoring      import score_product
from ml.features     import build_feature_matrix
from recommendation.recommender import AarogyaRecommender
from llm.gemini      import AarogyaExplainer

# ─── Load catalog ─────────────────────────────────────────────────────────
def _load_catalog() -> pd.DataFrame:
    for path in [
        os.path.join(ROOT, "data", "raw", "aarogya_products_500.csv"),
        os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv"),
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("No dataset found.")

_catalog   : pd.DataFrame  = None
_recommender: AarogyaRecommender = None
_explainer  : AarogyaExplainer   = None
_predictor                         = None


def get_catalog():
    global _catalog
    if _catalog is None:
        _catalog = _load_catalog()
    return _catalog


def get_recommender():
    global _recommender
    if _recommender is None:
        _recommender = AarogyaRecommender(get_catalog())
    return _recommender


def get_explainer():
    global _explainer
    if _explainer is None:
        _explainer = AarogyaExplainer()
    return _explainer


def get_predictor():
    global _predictor
    if _predictor is None:
        try:
            from ml.predict import AarogyaPredictor
            _predictor = AarogyaPredictor()
        except Exception as e:
            logger.warning(f"ML predictor not loaded: {e}. Using scoring fallback.")
    return _predictor


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title        = "AAROGYA API",
        description  = "AI Food Product Intelligence — Score, Explain, Recommend",
        version      = "1.0.0",
        docs_url     = "/docs",
        redoc_url    = "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],
        allow_methods  = ["*"],
        allow_headers  = ["*"],
    )

    # ── Pydantic Models ────────────────────────────────────────────────────

    class ProductInput(BaseModel):
        product_name            : str   = "My Product"
        category                : str   = "Biscuits & Cookies"
        subcategory             : str   = ""
        recommendation_group    : str   = "biscuits_cookies"
        energy_kcal_100g        : float = Field(..., ge=0, le=900)
        protein_g_100g          : float = Field(..., ge=0, le=100)
        carbs_g_100g            : float = Field(..., ge=0, le=100)
        sugar_g_100g            : float = Field(..., ge=0, le=100)
        added_sugar_g_100g      : float = Field(0.0, ge=0)
        total_fat_g_100g        : float = Field(..., ge=0, le=100)
        saturated_fat_g_100g    : float = Field(0.0, ge=0)
        trans_fat_g_100g        : float = Field(0.0, ge=0)
        fiber_g_100g            : float = Field(0.0, ge=0)
        sodium_mg_100g          : float = Field(0.0, ge=0)
        ingredient_count        : int   = Field(5, ge=1)
        contains_whole_grain    : bool  = False
        contains_added_sugar    : bool  = False
        contains_artificial_sweetener: bool = False
        contains_allergen       : bool  = False
        processing_level        : str   = "processed"

    class AnalyzeRequest(BaseModel):
        product   : ProductInput
        user_prefs: Optional[str] = "balanced"

    class RecommendRequest(BaseModel):
        product_id  : str
        preferences : Optional[str] = "balanced"
        top_n       : int = 3

    class CompareRequest(BaseModel):
        product_id_a: str
        product_id_b: str

    # ── Endpoints ──────────────────────────────────────────────────────────

    @app.get("/", tags=["Health"])
    def root():
        return {"status": "ok", "service": "AAROGYA API", "version": "1.0.0"}

    @app.get("/health", tags=["Health"])
    def health():
        cat = get_catalog()
        return {
            "status"       : "ok",
            "catalog_size" : len(cat),
            "categories"   : int(cat["category"].nunique()),
            "gemini_ready" : get_explainer()._ready,
        }

    @app.post("/analyze", tags=["Analysis"])
    def analyze(req: AnalyzeRequest):
        """Score and explain a product based on its nutrition data."""
        p = req.product.dict()

        # Step 1: Deterministic score
        det_score = score_product(p)

        # Step 2: ML score (if model available)
        predictor = get_predictor()
        if predictor:
            try:
                ml_result      = predictor.predict(p)
                predicted_score = ml_result["predicted_score"]
                shap_ctx        = ml_result["explanation_context"]
            except Exception as e:
                logger.warning(f"ML predict failed: {e}")
                predicted_score = det_score["computed_score"]
                shap_ctx        = {
                    "product_name": p["product_name"],
                    "predicted_score": predicted_score,
                    "strengths": det_score.get("score_strengths", []),
                    "concerns":  det_score.get("score_concerns", []),
                }
        else:
            predicted_score = det_score["computed_score"]
            shap_ctx = {
                "product_name":    p["product_name"],
                "category":        p["category"],
                "processing_level":p["processing_level"],
                "predicted_score": predicted_score,
                "base_value":      55.0,
                "strengths":       det_score.get("score_strengths", []),
                "concerns":        det_score.get("score_concerns", []),
                "shap_top_features": {},
            }

        # Step 3: Gemini explanation
        explanation = get_explainer().explain_score(shap_ctx)

        # Step 4: Recommendations
        try:
            rec_df = get_recommender().recommend(
                recommendation_group = p["recommendation_group"],
                current_product      = p,
                preference_profile   = req.user_prefs,
                top_n                = 3,
            )
            recs = rec_df.to_dict("records") if rec_df is not None else []
        except Exception:
            recs = []

        return {
            "product_name"      : p["product_name"],
            "deterministic_score": det_score["computed_score"],
            "predicted_score"   : predicted_score,
            "score_version"     : "v1.0",
            "explanation"       : explanation,
            "strengths"         : shap_ctx.get("strengths", []),
            "concerns"          : shap_ctx.get("concerns", []),
            "shap_features"     : shap_ctx.get("shap_top_features", {}),
            "recommendations"   : recs[:3],
        }

    @app.post("/recommend", tags=["Recommendations"])
    def recommend(req: RecommendRequest):
        """Get top N alternatives for a product from the catalog."""
        cat = get_catalog()
        row = cat[cat["product_id"] == req.product_id]
        if row.empty:
            raise HTTPException(404, f"Product {req.product_id} not found in catalog.")

        recs = get_recommender().recommend(
            product_id         = req.product_id,
            preference_profile = req.preferences,
            top_n              = req.top_n,
        )
        if recs is None or len(recs) == 0:
            return {"product_id": req.product_id, "recommendations": []}

        return {
            "product_id"     : req.product_id,
            "product_name"   : row.iloc[0]["product_name"],
            "recommendations": recs.to_dict("records"),
        }

    @app.post("/compare", tags=["Analysis"])
    def compare(req: CompareRequest):
        """Compare two products side by side."""
        cat = get_catalog()
        a = cat[cat["product_id"] == req.product_id_a]
        b = cat[cat["product_id"] == req.product_id_b]
        if a.empty:
            raise HTTPException(404, f"Product {req.product_id_a} not found.")
        if b.empty:
            raise HTTPException(404, f"Product {req.product_id_b} not found.")

        a_row = a.iloc[0]
        b_row = b.iloc[0]
        nutrients = ["protein_g_100g","sugar_g_100g","fiber_g_100g",
                     "sodium_mg_100g","saturated_fat_g_100g","energy_kcal_100g"]

        comparison = {}
        for n in nutrients:
            av = float(a_row.get(n, 0)) if pd.notna(a_row.get(n)) else None
            bv = float(b_row.get(n, 0)) if pd.notna(b_row.get(n)) else None
            comparison[n] = {"a": av, "b": bv}

        return {
            "product_a": {
                "id": req.product_id_a,
                "name": a_row["product_name"],
                "score": float(a_row.get("food_score", 0)),
            },
            "product_b": {
                "id": req.product_id_b,
                "name": b_row["product_name"],
                "score": float(b_row.get("food_score", 0)),
            },
            "nutrient_comparison": comparison,
            "winner": req.product_id_a if float(a_row.get("food_score",0)) >= float(b_row.get("food_score",0)) else req.product_id_b,
        }

    @app.get("/product/{product_id}", tags=["Catalog"])
    def get_product(product_id: str):
        """Look up a product by ID."""
        cat = get_catalog()
        row = cat[cat["product_id"] == product_id]
        if row.empty:
            raise HTTPException(404, f"Product {product_id} not found.")
        return row.iloc[0].dropna().to_dict()

    @app.get("/categories", tags=["Catalog"])
    def list_categories():
        """List all available categories."""
        cat = get_catalog()
        result = {}
        for cat_name, group in cat.groupby("category"):
            result[cat_name] = int(len(group))
        return result

    @app.get("/top/{category}", tags=["Catalog"])
    def top_in_category(category: str, n: int = Query(5, ge=1, le=20)):
        """Get top N products in a category by Aarogya score."""
        cat = get_catalog()
        filtered = cat[cat["category"].str.lower() == category.lower()]
        if filtered.empty:
            raise HTTPException(404, f"Category '{category}' not found.")
        top = filtered.nlargest(n, "food_score")[
            ["product_id","product_name","brand","food_score","processing_level"]
        ]
        return top.to_dict("records")

else:
    # Stub when FastAPI not installed
    class app:
        pass


if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
    else:
        import uvicorn
        print("Starting AAROGYA API on http://localhost:8000")
        print("API docs at: http://localhost:8000/docs")
        uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
