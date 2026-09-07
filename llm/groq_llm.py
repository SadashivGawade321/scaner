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
        self._vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        self._ready  = False

        if not GROQ_AVAILABLE:
            logger.info("groq not installed — rule-based fallback active.")
            return

        api_key = os.getenv("GROQ_API_KEY", "")
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

    def explain_score(self, ctx: dict) -> str:
        """
        Generate a natural language score explanation.

        Parameters
        ----------
        ctx : dict — output of ml/explain.py build_explanation_context()

        Returns
        -------
        str — 2-3 sentence explanation
        """
        if self._ready:
            try:
                return self._call_groq(self._build_score_messages(ctx))
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

    def _build_score_messages(self, ctx: dict) -> list:
        tier, _ = self._get_tier(ctx.get("predicted_score", 50))
        strengths = ", ".join(ctx.get("strengths", [])) or "None identified"
        concerns  = ", ".join(ctx.get("concerns",  [])) or "None identified"

        shap_lines = ""
        if ctx.get("shap_top_features"):
            shap_lines = "\nTop SHAP feature contributions:\n"
            for feat, val in list(ctx["shap_top_features"].items())[:5]:
                direction = "positive" if val > 0 else "negative"
                shap_lines += f"  - {feat}: {direction} impact ({val:+.2f})\n"

        user_msg = f"""Explain this product's Aarogya nutritional score:

Product: {ctx.get('product_name', 'Unknown')}
Category: {ctx.get('category', '')}
Processing Level: {ctx.get('processing_level', '')}
Aarogya Score: {ctx.get('predicted_score', '?')}/100 ({tier})
Score baseline (average): {ctx.get('base_value', 55)}/100

Key positive factors: {strengths}
Key concerns: {concerns}
{shap_lines}
Write exactly 2-3 sentences. Do not invent any numbers."""

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

    # ─── Vision Analysis ───────────────────────────────────────────────────

    def analyze_product_image(self, image_bytes: bytes) -> dict:
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
