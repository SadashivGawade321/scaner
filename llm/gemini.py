"""
llm/gemini.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Gemini Explanation & Multimodal Vision Layer
─────────────────────────────────────────────────────────────────────────────
Uses Google Gemini (Gemini 2.5 Flash / Gemini Flash Latest) for:
  1. Score and recommendation natural-language explanations
  2. Direct multimodal vision parsing of packaged food nutrition labels & ingredients
  3. Interactive product AI chat
  4. Real-time product nutrient estimation

Falls back gracefully to rule-based logic if GEMINI_API_KEY is not configured.
"""

import os, sys, logging, json, re, io
from typing import Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import google.generativeai as genai
    from PIL import Image
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class AarogyaExplainer:
    """
    Translates structured ML analysis into natural language via Gemini,
    and performs multimodal vision analysis on product photos.
    """

    SCORE_TO_TIER = {
        (0,  39): ("Needs Consideration", "red"),
        (40, 59): ("Moderate",            "amber"),
        (60, 74): ("Good",                "light-green"),
        (75, 100):("Excellent",           "green"),
    }

    def __init__(self):
        self._model = None
        self._ready = False
        self._model_name = "gemini-2.5-flash"

        if not GEMINI_AVAILABLE:
            logger.info("google-generativeai not installed — using fallback.")
            return

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                    api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                pass

        if not api_key or api_key == "your_gemini_api_key_here":
            logger.info("GEMINI_API_KEY not set — using rule-based fallback.")
            return

        try:
            genai.configure(api_key=api_key)
            candidate_models = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash", "gemini-2.5-flash-lite"]
            for m in candidate_models:
                try:
                    self._model = genai.GenerativeModel(m)
                    self._model_name = m
                    self._ready = True
                    logger.info(f"Gemini connected using {m}.")
                    break
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def model_name(self) -> str:
        return self._model_name if self._ready else "Rule-based Fallback"

    # ── Vision Analysis ────────────────────────────────────────────────────────
    def analyze_product_image(self, image_bytes: bytes) -> dict:
        """Analyze packaged food label using Gemini Vision."""
        if not self._ready or not self._model:
            return {"error": "Gemini not configured", "confidence": "low"}
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            prompt = """You are an expert AI food label analyzer for Aarogya.
Analyze this food product image or nutrition facts label.
Extract nutrients per 100g. Return ONLY a single valid JSON object without markdown formatting:
{
  "product_name": "Product Name",
  "brand": "Brand Name",
  "category": "Biscuits & Cookies|Breakfast Cereals|Chips & Namkeen|Dairy|Beverages|Other",
  "processing_level": "minimally_processed|processed|ultra_processed",
  "energy_kcal_100g": 0.0,
  "protein_g_100g": 0.0,
  "carbs_g_100g": 0.0,
  "sugar_g_100g": 0.0,
  "total_fat_g_100g": 0.0,
  "saturated_fat_g_100g": 0.0,
  "fiber_g_100g": 0.0,
  "sodium_mg_100g": 0.0,
  "confidence": "high",
  "harmful_additives": [],
  "allergens_found": [],
  "health_claims": [],
  "contains_whole_grain": false,
  "contains_added_sugar": false,
  "ingredients_list": []
}"""
            res = self._model.generate_content([prompt, img])
            if res and res.text:
                txt = res.text.strip()
                txt = re.sub(r"^```(?:json)?\s*", "", txt)
                txt = re.sub(r"\s*```$", "", txt)
                parsed = json.loads(txt)
                parsed["_source"] = f"Gemini Vision ({self._model_name})"
                return parsed
        except Exception as e:
            logger.warning(f"Gemini image analysis error: {e}")
            return {"error": str(e), "confidence": "low"}
        return {"error": "No response from Gemini", "confidence": "low"}

    def analyze_ingredients(self, image_bytes: bytes) -> dict:
        """Analyze ingredients list photo using Gemini Vision."""
        if not self._ready or not self._model:
            return {"error": "Gemini not configured"}
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            prompt = """You are an expert food safety auditor.
Analyze this photo of an ingredients list on packaged food.
Return ONLY a single valid JSON object without markdown formatting:
{
  "ingredients_raw": "full raw text",
  "ingredients_list": ["ingredient 1", "ingredient 2"],
  "total_ingredients_count": 0,
  "ingredient_quality_score": 7,
  "summary": "1-2 sentence overview of ingredient quality",
  "red_flag_ingredients": [{"name": "name", "severity": "high|medium|low", "reason": "why harmful"}]
}"""
            res = self._model.generate_content([prompt, img])
            if res and res.text:
                txt = res.text.strip()
                txt = re.sub(r"^```(?:json)?\s*", "", txt)
                txt = re.sub(r"\s*```$", "", txt)
                parsed = json.loads(txt)
                parsed["_source"] = f"Gemini Vision ({self._model_name})"
                return parsed
        except Exception as e:
            logger.warning(f"Gemini ingredients analysis error: {e}")
            return {"error": str(e)}
        return {"error": "No response"}

    def analyze_product_by_name(self, product_name: str, language: str = "en") -> dict:
        """Estimate nutritional facts and safety score for any product by name."""
        if not self._ready or not self._model:
            return {
                "product_name": product_name,
                "category": "Other",
                "energy_kcal_100g": 300,
                "protein_g_100g": 5,
                "carbs_g_100g": 50,
                "sugar_g_100g": 15,
                "total_fat_g_100g": 10,
                "saturated_fat_g_100g": 3,
                "fiber_g_100g": 2,
                "sodium_mg_100g": 300,
                "processing_level": "processed",
                "ingredients_list": [],
                "harmful_additives": [],
                "healthier_alternatives": [],
            }
        try:
            prompt = f"""Estimate standard nutritional values per 100g for this packaged food item: '{product_name}'.
Return ONLY a single valid JSON object (no markdown):
{{
  "product_name": "{product_name}",
  "brand": "Popular Brand",
  "category": "Biscuits & Cookies|Breakfast Cereals|Chips & Namkeen|Dairy|Beverages|Other",
  "processing_level": "minimally_processed|processed|ultra_processed",
  "energy_kcal_100g": 350.0,
  "protein_g_100g": 5.0,
  "carbs_g_100g": 60.0,
  "sugar_g_100g": 20.0,
  "total_fat_g_100g": 12.0,
  "saturated_fat_g_100g": 5.0,
  "fiber_g_100g": 2.5,
  "sodium_mg_100g": 300.0,
  "ingredients_list": ["ingredient 1", "ingredient 2"],
  "harmful_additives": [],
  "healthier_alternatives": ["healthier option 1", "healthier option 2"]
}}"""
            res = self._model.generate_content(prompt)
            if res and res.text:
                txt = res.text.strip()
                txt = re.sub(r"^```(?:json)?\s*", "", txt)
                txt = re.sub(r"\s*```$", "", txt)
                return json.loads(txt)
        except Exception as e:
            logger.warning(f"Product analysis by name error: {e}")
        return {"product_name": product_name, "category": "Other"}

    def chat(self, user_question: str, product_ctx: Optional[dict] = None) -> str:
        """Answer user questions with Gemini."""
        if not self._ready or not self._model:
            return "AI chat is currently in fallback mode. Connect your Gemini API key to enable interactive chat."
        try:
            prompt = f"You are Aarogya, a helpful food safety assistant.\n"
            if product_ctx:
                prompt += f"Product Context:\nName: {product_ctx.get('product_name')}\nScore: {product_ctx.get('predicted_score')}/100\nCategory: {product_ctx.get('category')}\n"
            prompt += f"User Question: {user_question}\nAnswer clearly in 2-3 sentences."
            res = self._model.generate_content(prompt)
            return res.text.strip()
        except Exception as e:
            return f"Chat error: {e}"

    # ── Explanation Methods ───────────────────────────────────────────────────
    def explain_score(self, ctx: dict, language: str = "en") -> str:
        if self._ready:
            try:
                return self._gemini_explain_score(ctx)
            except Exception as e:
                logger.warning(f"Gemini call failed: {e}. Using fallback.")
        return self._fallback_explain_score(ctx)

    def explain_recommendation(self, current: dict, recs: list) -> str:
        if self._ready:
            try:
                return self._gemini_explain_recs(current, recs)
            except Exception as e:
                logger.warning(f"Gemini call failed: {e}. Using fallback.")
        return self._fallback_explain_recs(current, recs)

    def _gemini_explain_score(self, ctx: dict) -> str:
        tier, _ = self._get_tier(ctx.get("predicted_score", 50))
        prompt = f"""You are AAROGYA, a packaged food intelligence assistant.
Product: {ctx.get('product_name','Unknown')}
Category: {ctx.get('category','')}
Processing: {ctx.get('processing_level','')}
Aarogya Score: {ctx.get('predicted_score', 50)}/100 ({tier})
Key factors:
- Positive: {', '.join(ctx.get('strengths', [])) if ctx.get('strengths') else 'None'}
- Concerns: {', '.join(ctx.get('concerns', [])) if ctx.get('concerns') else 'None'}

Instructions:
- Write exactly 2-3 concise sentences explaining this food score.
- Do NOT make medical claims or diagnoses.
- Be objective and friendly."""
        res = self._model.generate_content(prompt)
        return res.text.strip()

    def _gemini_explain_recs(self, current: dict, recs: list) -> str:
        rec_lines = [f"{i}. {r.get('product_name','?')} (Score: {r.get('food_score', r.get('computed_score','?'))})" for i, r in enumerate(recs[:3], 1)]
        prompt = f"""You are AAROGYA, a packaged food assistant.
Current Product: {current.get('product_name','Unknown')} (Score: {current.get('predicted_score','?')}/100)
Recommended alternatives:
{chr(10).join(rec_lines)}

Write 2 concise sentences explaining why these alternatives offer better nutritional balance."""
        res = self._model.generate_content(prompt)
        return res.text.strip()

    def _fallback_explain_score(self, ctx: dict) -> str:
        score = ctx.get("predicted_score", 50)
        tier, _ = self._get_tier(score)
        name = ctx.get("product_name", "This product")
        parts = [f"{name} has an Aarogya score of {score}/100, rated as {tier}."]
        if ctx.get("strengths"):
            parts.append(f"It performs well on {', '.join(ctx['strengths'][:2])}.")
        if ctx.get("concerns"):
            parts.append(f"Keep in mind it has higher {', '.join(ctx['concerns'][:2])}.")
        return " ".join(parts)

    def _fallback_explain_recs(self, current: dict, recs: list) -> str:
        if not recs:
            return "No alternative products found in the same category."
        top = recs[0]
        name = top.get("product_name", "the recommended alternative")
        score = top.get("food_score", top.get("computed_score", "?"))
        return f"Based on nutritional values, {name} (Score: {score}) offers a healthier profile compared to {current.get('product_name', 'this product')}."

    def _get_tier(self, score: float):
        for (lo, hi), (tier, color) in self.SCORE_TO_TIER.items():
            if lo <= score <= hi:
                return tier, color
        return "Unknown", "grey"
