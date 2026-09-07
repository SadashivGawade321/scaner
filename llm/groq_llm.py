"""
llm/groq_llm.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Groq LLM Explanation Layer (Week 13)
─────────────────────────────────────────────────────────────────────────────

Uses Groq's blazing-fast inference (Llama 3.1, Mixtral) to convert
structured ML output into natural language explanations.

DESIGN PRINCIPLE — Groq is a TRANSLATION LAYER only.
  - It receives: score + SHAP context (structured dict)
  - It outputs:  natural language explanation (string)
  - It does NOT: generate scores, look up nutrition, make medical claims.

USAGE
─────
    from llm.groq_llm import AarogyaGroqExplainer

    exp  = AarogyaGroqExplainer()
    text = exp.explain_score(ctx)             # ctx from ml/explain.py
    text = exp.explain_recommendation(ctx, recs)
    text = exp.chat(user_question, product_ctx)

    # Falls back to rule-based if GROQ_API_KEY not set or call fails.
"""

import os, sys, logging, json
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
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.info("groq package not installed. Run: pip install groq")


class AarogyaGroqExplainer:
    """
    Translates structured ML analysis into natural language via Groq.
    Falls back to rule-based if Groq is unavailable or key is missing.
    """

    SCORE_TIERS = {
        (75, 100): ("Excellent",            "green"),
        (60,  74): ("Good",                 "light-green"),
        (40,  59): ("Moderate",             "amber"),
        (0,   39): ("Needs Consideration",  "red"),
    }

    SYSTEM_PROMPT = """You are AAROGYA, an AI food product information assistant built for Indian consumers.

Your role:
- Translate structured nutrition analysis into clear, helpful explanations.
- Be friendly, concise, and non-judgmental.
- Use "for your selected preferences" instead of "healthy/unhealthy".
- NEVER make medical claims or diagnoses.
- NEVER invent numbers — only use data provided to you.
- Keep responses to 2-3 sentences unless asked for more.
- When you don't know something, say so clearly."""

    def __init__(self):
        self._client = None
        self._model  = os.getenv("GROQ_MODEL", "groq/compound-mini")
        self._ready  = False

        if not GROQ_AVAILABLE:
            logger.info("groq not installed — rule-based fallback active.")
            return

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
                    api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                pass

        if not api_key:
            logger.info("GROQ_API_KEY not set — rule-based fallback active.")
            return

        try:
            self._client = Groq(api_key=api_key)
            # Quick connectivity test
            self._ready  = True
            logger.info(f"Groq connected. Model: {self._model}")
        except Exception as e:
            logger.warning(f"Groq init failed: {e}. Using fallback.")

    # ─── Public API ────────────────────────────────────────────────────────

    def explain_score(self, ctx: dict, language: str = "en") -> str:
        """
        Generate a natural language score explanation.

        Parameters
        ----------
        ctx      : dict — output of ml/explain.py build_explanation_context()
        language : str — 'en', 'hi', 'mr', 'gu', 'ta', 'es'

        Returns
        -------
        str — 2-3 sentence explanation
        """
        if self._ready:
            try:
                return self._call_groq(self._build_score_messages(ctx, language=language))
            except Exception as e:
                logger.warning(f"Groq call failed: {e}. Using fallback.")
        return self._fallback_score(ctx)

    def explain_recommendation(self, current: dict, recs: list) -> str:
        """
        Generate a recommendation explanation.

        Parameters
        ----------
        current : dict — explanation context of current product
        recs    : list of recommendation dicts from recommender

        Returns
        -------
        str — recommendation narrative
        """
        if self._ready:
            try:
                return self._call_groq(self._build_rec_messages(current, recs))
            except Exception as e:
                logger.warning(f"Groq call failed: {e}. Using fallback.")
        return self._fallback_recs(current, recs)

    def chat(self, user_question: str, product_ctx: Optional[dict] = None) -> str:
        """
        Answer a user's free-form question about a product.

        Parameters
        ----------
        user_question : str  — what the user asked
        product_ctx   : dict — optional product context

        Returns
        -------
        str — Groq's answer
        """
        if not self._ready:
            return (
                "I can answer questions about products once the AI connection is configured. "
                "Please check your GROQ_API_KEY."
            )

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        if product_ctx:
            context_text = (
                f"Product: {product_ctx.get('product_name', 'Unknown')}\n"
                f"Category: {product_ctx.get('category', '')}\n"
                f"Aarogya Score: {product_ctx.get('predicted_score', '?')}/100\n"
                f"Strengths: {', '.join(product_ctx.get('strengths', []) or ['None'])}\n"
                f"Concerns: {', '.join(product_ctx.get('concerns', []) or ['None'])}\n"
                f"Processing: {product_ctx.get('processing_level', '')}"
            )
            messages.append({
                "role": "system",
                "content": f"Current product context:\n{context_text}"
            })

        messages.append({"role": "user", "content": user_question})

        try:
            return self._call_groq(messages)
        except Exception as e:
            logger.warning(f"Groq chat failed: {e}")
            return "Sorry, I couldn't process that question right now."

    # ─── Groq call ─────────────────────────────────────────────────────────

    def _call_groq(self, messages: list, max_tokens: int = 300) -> str:
        response = self._client.chat.completions.create(
            model       = self._model,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = 0.4,   # low temp = factual, consistent
            top_p       = 0.9,
        )
        return response.choices[0].message.content.strip()

    # ─── Message builders ──────────────────────────────────────────────────

    def _build_score_messages(self, ctx: dict, language: str = "en") -> list:
        tier, _ = self._get_tier(ctx.get("predicted_score", 50))
        strengths = ", ".join(ctx.get("strengths", [])) or "None identified"
        concerns  = ", ".join(ctx.get("concerns",  [])) or "None identified"

        shap_lines = ""
        if ctx.get("shap_top_features"):
            shap_lines = "\nTop SHAP feature contributions:\n"
            for feat, val in list(ctx["shap_top_features"].items())[:5]:
                direction = "positive" if val > 0 else "negative"
                shap_lines += f"  - {feat}: {direction} impact ({val:+.2f})\n"

        lang_instruction = ""
        if language == "hi":
            lang_instruction = "\nCRITICAL: Write the entire explanation in natural, easy-to-understand Hindi (हिन्दी)."
        elif language == "mr":
            lang_instruction = "\nCRITICAL: Write the entire explanation in natural, easy-to-understand Marathi (मराठी)."
        elif language == "gu":
            lang_instruction = "\nCRITICAL: Write the entire explanation in natural, easy-to-understand Gujarati (ગુજરાતી)."
        elif language == "ta":
            lang_instruction = "\nCRITICAL: Write the entire explanation in natural, easy-to-understand Tamil (தமிழ்)."
        elif language == "es":
            lang_instruction = "\nCRITICAL: Write the entire explanation in natural, easy-to-understand Spanish (Español)."

        user_msg = f"""Explain this product's Aarogya nutritional score:

Product: {ctx.get('product_name', 'Unknown')}
Category: {ctx.get('category', '')}
Processing Level: {ctx.get('processing_level', '')}
Aarogya Score: {ctx.get('predicted_score', '?')}/100 ({tier})
Score baseline (average): {ctx.get('base_value', 55)}/100

Key positive factors: {strengths}
Key concerns: {concerns}
{shap_lines}
Write exactly 2-3 sentences. Do not invent any numbers.{lang_instruction}"""

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

    def _build_rec_messages(self, current: dict, recs: list) -> list:
        rec_lines = []
        for i, r in enumerate(recs[:3], 1):
            score = r.get("food_score", r.get("computed_score", "?"))
            rec_lines.append(
                f"  {i}. {r.get('product_name', 'Unknown')} (Score: {score}/100)"
            )

        user_msg = f"""Explain why these alternatives may be better for the user.

Current product: {current.get('product_name', 'Unknown')} (Score: {current.get('predicted_score', '?')}/100)

Recommended alternatives (same category):
{chr(10).join(rec_lines)}

Write 2 sentences. Use "for your selected preferences". Do not make medical claims."""

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

    # ─── Vision Analysis (OCR + Text Model) ──────────────────────────────────

    @staticmethod
    def _extract_text_from_image(image_bytes: bytes) -> str:
        """Extract text from image using RapidOCR (primary) and pytesseract (fallback)."""
        # 1. Primary: RapidOCR (zero external dependency, high accuracy, ONNX)
        try:
            from rapidocr_onnxruntime import RapidOCR
            from PIL import Image
            import io, numpy as np
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            engine = RapidOCR()
            res, _ = engine(np.array(img))
            if res:
                lines = [r[1] for r in res if r and len(r) > 1 and r[1].strip()]
                if lines:
                    return "\n".join(lines).strip()
        except Exception as e:
            logger.warning(f"RapidOCR failed: {e}")

        # 2. Secondary: pytesseract
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import pytesseract, io
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            w, h = img.size
            if w < 800:
                img = img.resize((w*2, h*2), Image.LANCZOS)
            img = img.convert("L")
            img = ImageEnhance.Contrast(img).enhance(2.5)
            img = ImageEnhance.Sharpness(img).enhance(2.0)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            text = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
            return text.strip()
        except Exception as e:
            logger.warning(f"pytesseract OCR fallback failed: {e}")
            return ""

    def analyze_product_image(self, image_bytes: bytes) -> dict:
        """
        Analyze a product/nutrition label image.
        Pipeline: pytesseract OCR -> Groq text model parses the text.
        No vision-capable model required.
        """
        import json, re
        ocr_text = self._extract_text_from_image(image_bytes)
        if not ocr_text or len(ocr_text) < 20:
            return {"confidence": "low",
                    "error": "Could not extract text. Try better lighting or a clearer photo."}

        prompt = f"""You are an AI nutrition label parser for Aarogya food intelligence.

The following text was extracted via OCR from a packaged food nutrition label.
Parse it carefully and return ONLY a JSON object (no explanation):

OCR TEXT:
\"\"\"
{ocr_text[:2000]}
\"\"\"

Return ONLY this JSON structure (use null for missing values):
{{
  "product_name": "...",
  "brand": "...",
  "category": "Biscuits & Cookies|Breakfast Cereals|Chips & Namkeen|Dairy|Beverages|Other",
  "processing_level": "minimally_processed|processed|ultra_processed",
  "energy_kcal_100g": number_or_null,
  "protein_g_100g": number_or_null,
  "carbs_g_100g": number_or_null,
  "sugar_g_100g": number_or_null,
  "total_fat_g_100g": number_or_null,
  "saturated_fat_g_100g": number_or_null,
  "fiber_g_100g": number_or_null,
  "sodium_mg_100g": number_or_null,
  "ingredients_list": ["..."],
  "contains_whole_grain": true_or_false,
  "contains_added_sugar": true_or_false,
  "contains_artificial_sweetener": true_or_false,
  "contains_allergen": true_or_false,
  "allergens_found": ["..."],
  "harmful_additives": ["..."],
  "health_claims": ["..."],
  "confidence": "high|medium|low"
}}"""

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=700, temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```[a-z]*\n?","",raw)
            raw = re.sub(r"\n?```$","",raw)
            result = json.loads(raw)
            # Mark confidence based on how much text we got
            if not result.get("confidence"):
                result["confidence"] = "high" if len(ocr_text)>200 else "medium"
            result["_ocr_text"] = ocr_text[:500]
            return result
        except Exception as e:
            logger.warning(f"Product image analysis failed: {e}")
            # Final fallback: regex parse directly
            return self._regex_parse_nutrition(ocr_text)

    def analyze_ingredients(self, image_bytes: bytes) -> dict:
        """
        Analyze an ingredients list image.
        Pipeline: pytesseract OCR -> Groq text model analyses ingredients.
        """
        import json, re
        ocr_text = self._extract_text_from_image(image_bytes)
        if not ocr_text or len(ocr_text) < 10:
            return {"error": "Could not read text. Try a clearer/closer photo.",
                    "summary": "Image too unclear to read."}

        prompt = f"""You are an ingredients list analyser for Aarogya, an Indian food intelligence app.

OCR text from ingredients label:
\"\"\"
{ocr_text[:2000]}
\"\"\"

Analyse this ingredients list and return ONLY this JSON (no explanation):
{{
  "ingredients_raw": "full text as extracted",
  "ingredients_list": ["ingredient1","ingredient2",...],
  "total_ingredients_count": number,
  "red_flag_ingredients": [
    {{"name":"...","reason":"why concerning","severity":"high|medium|low"}}
  ],
  "additives_found": ["INS 322","E471",...],
  "allergens": ["gluten","milk",...],
  "contains_palm_oil": true_or_false,
  "contains_artificial_colours": true_or_false,
  "contains_artificial_flavours": true_or_false,
  "contains_preservatives": true_or_false,
  "contains_added_sugar": true_or_false,
  "processing_level_guess": "minimally_processed|processed|ultra_processed",
  "ingredient_quality_score": 0_to_10,
  "summary": "2 sentence plain English summary for Indian consumer"
}}"""

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=700, temperature=0.1,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```[a-z]*\n?","",raw)
            raw = re.sub(r"\n?```$","",raw)
            result = json.loads(raw)
            result["ingredients_raw"] = result.get("ingredients_raw","") or ocr_text[:500]
            return result
        except Exception as e:
            logger.warning(f"Ingredients analysis failed: {e}")
            return {"error": str(e),
                    "summary": "Could not analyse ingredients.",
                    "ingredients_raw": ocr_text[:300]}

    def analyze_product_by_name(self, product_name: str, language: str = "en") -> dict:
        """
        Generate complete nutritional facts, category, processing level,
        red flags, and healthy alternatives for any food queried by name.
        """
        import json, re

        lang_instruction = ""
        if language == "hi":
            lang_instruction = "Respond in Hindi for 'summary', 'red_flags', and 'healthier_alternatives'."
        elif language == "mr":
            lang_instruction = "Respond in Marathi for 'summary', 'red_flags', and 'healthier_alternatives'."
        elif language == "gu":
            lang_instruction = "Respond in Gujarati for 'summary', 'red_flags', and 'healthier_alternatives'."
        elif language == "ta":
            lang_instruction = "Respond in Tamil for 'summary', 'red_flags', and 'healthier_alternatives'."
        elif language == "es":
            lang_instruction = "Respond in Spanish for 'summary', 'red_flags', and 'healthier_alternatives'."

        prompt = f"""You are an expert food nutritionist for Aarogya, an Indian food intelligence platform.
Analyze the packaged food product named: "{product_name}".
Provide estimated realistic nutritional facts per 100g, category, processing level, red flags, and healthier alternatives.
{lang_instruction}

Return ONLY a valid JSON object matching this structure:
{{
  "product_name": "{product_name.title()}",
  "brand": "Popular Brand or Indian Manufacturer",
  "category": "Biscuits & Cookies|Breakfast Cereals|Chips & Namkeen|Dairy|Beverages|Instant Noodles|Chocolates & Sweets|Bakery|Other",
  "recommendation_group": "biscuits_cookies|breakfast_cereals|chips_namkeen|dairy|beverages|instant_noodles|other",
  "processing_level": "minimally_processed|processed|ultra_processed",
  "energy_kcal_100g": 400.0,
  "protein_g_100g": 7.0,
  "carbs_g_100g": 60.0,
  "sugar_g_100g": 3.0,
  "added_sugar_g_100g": 2.0,
  "total_fat_g_100g": 16.0,
  "saturated_fat_g_100g": 7.0,
  "trans_fat_g_100g": 0.1,
  "fiber_g_100g": 3.0,
  "sodium_mg_100g": 950.0,
  "ingredient_count": 16,
  "contains_whole_grain": false,
  "contains_added_sugar": true,
  "contains_artificial_sweetener": false,
  "contains_allergen": true,
  "allergens": ["Gluten", "Soy"],
  "red_flags": ["High Sodium", "Palm Oil", "Refined Flour (Maida)"],
  "healthier_alternatives": ["Millet Noodles", "Oats / Quinoa Upma", "Whole Wheat Roti"],
  "summary": "2 concise sentences explaining health verdict for consumers."
}}"""

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=650,
                temperature=0.2,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            data = json.loads(raw)
            return data
        except Exception as e:
            logger.warning(f"AI product analysis failed: {e}")
            return {
                "product_name": product_name.title(),
                "brand": "Popular Brand",
                "category": "Other",
                "recommendation_group": "other",
                "processing_level": "ultra_processed",
                "energy_kcal_100g": 420.0,
                "protein_g_100g": 6.0,
                "carbs_g_100g": 58.0,
                "sugar_g_100g": 5.0,
                "added_sugar_g_100g": 4.0,
                "total_fat_g_100g": 18.0,
                "saturated_fat_g_100g": 8.0,
                "fiber_g_100g": 2.5,
                "sodium_mg_100g": 850.0,
                "ingredient_count": 14,
                "red_flags": ["High Sodium / Salt", "Refined Ingredients"],
                "healthier_alternatives": ["Fresh Fruits & Home-cooked Meals"],
                "summary": f"Nutritional estimate for {product_name}.",
            }

    @staticmethod
    def _regex_parse_nutrition(text: str) -> dict:
        """Fallback: pure regex extraction from OCR text."""
        import re
        t = text.lower()
        def fv(pats):
            for p in pats:
                m = re.search(p, t)
                if m:
                    try: return float(m.group(1))
                    except: pass
            return None
        result = {
            "product_name": None, "brand": None,
            "category": "Biscuits & Cookies",
            "processing_level": "processed",
            "confidence": "low",
            "energy_kcal_100g":     fv([r"energy[:\s]+(\d+\.?\d*)\s*kcal",r"(\d+\.?\d*)\s*kcal"]),
            "protein_g_100g":       fv([r"protein[:\s]+(\d+\.?\d*)\s*g"]),
            "carbs_g_100g":         fv([r"carbohydrate[s]?[:\s]+(\d+\.?\d*)\s*g"]),
            "sugar_g_100g":         fv([r"sugar[s]?[:\s]+(\d+\.?\d*)\s*g"]),
            "total_fat_g_100g":     fv([r"total fat[:\s]+(\d+\.?\d*)\s*g",r"\bfat[:\s]+(\d+\.?\d*)\s*g"]),
            "saturated_fat_g_100g": fv([r"saturated fat[:\s]+(\d+\.?\d*)\s*g"]),
            "fiber_g_100g":         fv([r"fi(?:b|br)(?:er|re)[:\s]+(\d+\.?\d*)\s*g"]),
            "sodium_mg_100g":       fv([r"sodium[:\s]+(\d+\.?\d*)\s*mg"]),
        }
        found = sum(1 for k in ["energy_kcal_100g","sugar_g_100g","protein_g_100g"] if result.get(k))
        if found == 0:
            result["error"] = "Could not parse nutrition values from image."
        return result


        """
        Analyze a product front/back image using Groq vision.
        Returns dict with: product_name, brand, category, processing_level,
                           estimated_nutrition, concerns, confidence
        """
        import base64
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = """You are analyzing a packaged food product image for the Aarogya food intelligence system.

Look at this image carefully. It may show:
- The product front (name, brand, claims)
- A nutrition facts label
- An ingredients list
- A barcode

Extract and return ONLY a JSON object with these fields (use null if not found):
{
  "product_name": "...",
  "brand": "...",
  "category": "...",
  "processing_level": "minimally_processed|processed|ultra_processed",
  "energy_kcal_100g": number_or_null,
  "protein_g_100g": number_or_null,
  "carbs_g_100g": number_or_null,
  "sugar_g_100g": number_or_null,
  "total_fat_g_100g": number_or_null,
  "saturated_fat_g_100g": number_or_null,
  "fiber_g_100g": number_or_null,
  "sodium_mg_100g": number_or_null,
  "ingredients_list": ["ingredient1","ingredient2",...],
  "contains_whole_grain": true/false,
  "contains_added_sugar": true/false,
  "contains_artificial_sweetener": true/false,
  "contains_allergen": true/false,
  "allergens_found": ["..."],
  "harmful_additives": ["E-number or additive name if concerning"],
  "health_claims": ["..."],
  "confidence": "high|medium|low"
}

Return ONLY the JSON. No explanation."""

        try:
            response = self._client.chat.completions.create(
                model=self._vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                max_tokens=600,
                temperature=0.1,
            )
            import json, re
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Vision analysis failed: {e}")
            return {"confidence": "low", "error": str(e)}

    def analyze_ingredients(self, image_bytes: bytes) -> dict:
        """
        Analyze an ingredients list image.
        Returns structured ingredient analysis with red flags.
        """
        import base64
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt = """You are analyzing an ingredients list photo from a packaged food product for Aarogya food intelligence.

Read ALL text visible in this image carefully.

Return ONLY a JSON object:
{
  "ingredients_raw": "full ingredients text as written",
  "ingredients_list": ["ingredient1", "ingredient2", ...],
  "total_ingredients_count": number,
  "red_flag_ingredients": [
    {"name": "...", "reason": "why it's concerning", "severity": "high|medium|low"}
  ],
  "additives_found": ["E471", "INS 322", ...],
  "allergens": ["gluten", "milk", ...],
  "contains_palm_oil": true/false,
  "contains_artificial_colours": true/false,
  "contains_artificial_flavours": true/false,
  "contains_preservatives": true/false,
  "contains_added_sugar": true/false,
  "processing_level_guess": "minimally_processed|processed|ultra_processed",
  "ingredient_quality_score": 0_to_10,
  "summary": "2 sentence plain English summary of ingredient quality"
}

Return ONLY the JSON. No extra text."""

        try:
            response = self._client.chat.completions.create(
                model=self._vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }],
                max_tokens=700,
                temperature=0.1,
            )
            import json, re
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"Ingredients analysis failed: {e}")
            return {"error": str(e), "summary": "Could not analyze ingredients."}


    def _fallback_score(self, ctx: dict) -> str:
        score = ctx.get("predicted_score", 0)
        tier, _ = self._get_tier(score)
        name = ctx.get("product_name", "This product")
        parts = [f"{name} has an Aarogya Score of {score}/100 — rated as {tier}."]

        strengths = ctx.get("strengths", [])
        concerns  = ctx.get("concerns",  [])

        if strengths:
            parts.append(f"For your selected preferences, it performs well on {', '.join(strengths[:2])}.")
        if concerns:
            parts.append(f"You may want to note that it has relatively high {', '.join(concerns[:2])}.")
        if not strengths and not concerns:
            parts.append("It has a balanced nutritional profile within its category.")
        return " ".join(parts)

    def _fallback_recs(self, current: dict, recs: list) -> str:
        if not recs:
            return "No similar alternatives were found in this category."
        top  = recs[0]
        name = top.get("product_name", "the recommended product")
        score = top.get("food_score", top.get("computed_score", "?"))
        return (
            f"Based on your selected preferences, {name} (Score: {score}/100) "
            f"may be a better fit. It scores higher on the nutrients you care about most."
        )

    def _get_tier(self, score: float):
        for (lo, hi), (tier, color) in self.SCORE_TIERS.items():
            if lo <= score <= hi:
                return tier, color
        return "Unknown", "grey"

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def model_name(self) -> str:
        return self._model if self._ready else "rule-based fallback"


# ─── Quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AAROGYA — Groq Explainer Test")
    print("=" * 60)

    exp = AarogyaGroqExplainer()
    print(f"\nGroq ready: {exp.is_ready}")
    print(f"Model:      {exp.model_name}")

    ctx = {
        "product_name":     "Oat Bran Porridge Instant",
        "category":         "Breakfast Cereals",
        "processing_level": "minimally_processed",
        "predicted_score":  94.0,
        "base_value":       53.95,
        "strengths":        ["fiber", "sodium", "protein density"],
        "concerns":         [],
        "shap_top_features": {
            "fiber": 12.5, "sodium density": 8.2,
            "protein density": 6.1, "sugar": -1.2
        },
    }

    print("\n[Test 1] Score explanation:")
    explanation = exp.explain_score(ctx)
    print(f"  {explanation}")

    recs = [
        {"product_name": "Quinoa Amaranth Pops",     "food_score": 80},
        {"product_name": "High Protein Granola Mix",  "food_score": 68},
    ]
    print("\n[Test 2] Recommendation explanation:")
    rec_text = exp.explain_recommendation(ctx, recs)
    print(f"  {rec_text}")

    print("\n[Test 3] Free-form chat:")
    answer = exp.chat("Is this product suitable for someone watching their sodium intake?", ctx)
    print(f"  {answer}")

    print("\n[GROQ EXPLAINER OK]")
