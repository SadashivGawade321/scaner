"""
llm/gemini.py — [WEEK 13]
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Gemini Explanation Layer
─────────────────────────────────────────────────────────────────────────────

DESIGN PRINCIPLE
────────────────
Gemini is a TRANSLATION LAYER only.

It converts structured ML analysis output (score + SHAP values) into
user-friendly natural language. It does NOT:
  - Calculate or override the numerical Aarogya score
  - Look up nutrition information independently
  - Make medical claims
  - Invent ingredients or nutritional values

SECURITY
────────
All prompts use structured JSON input. Never pass raw user text directly
to Gemini without sanitization. Always specify exact output format.

[WEEK 13 IMPLEMENTATION]
"""

import os
import logging

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Install in Week 13.")


class AarogyaExplainer:
    """
    [WEEK 13] Gemini-based explanation layer.
    Takes structured analysis context and returns natural language.
    """

    def __init__(self):
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini not available. Explanations will be rule-based fallback.")
            return

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set in environment. "
                "Add it to your .env file."
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def explain_score(self, analysis_context: dict) -> str:
        """
        [WEEK 13] Generate a natural language explanation of the score.

        Parameters
        ----------
        analysis_context : dict with keys:
            product_name, predicted_score, strengths, concerns, shap_values

        Returns
        -------
        str — 2-3 sentence explanation
        """
        raise NotImplementedError("Implement in Week 13.")

    def explain_recommendation(self, current: dict, recommendations: list) -> str:
        """
        [WEEK 13] Generate natural language recommendation explanation.
        """
        raise NotImplementedError("Implement in Week 13.")

    @staticmethod
    def _fallback_explanation(analysis_context: dict) -> str:
        """
        Rule-based fallback when Gemini is unavailable.
        Used in development without API key.
        """
        score = analysis_context.get("predicted_score", "?")
        strengths = analysis_context.get("strengths", [])
        concerns = analysis_context.get("concerns", [])

        parts = [
            f"This product has an Aarogya score of {score}/100."
        ]
        if strengths:
            parts.append(
                f"For your selected preferences, it performs well on: "
                f"{', '.join(strengths)}."
            )
        if concerns:
            parts.append(
                f"You may want to consider: {', '.join(concerns)}."
            )
        return " ".join(parts)
