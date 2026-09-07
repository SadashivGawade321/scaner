"""
ml/predict.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Inference Pipeline (Week 8)
─────────────────────────────────────────────────────────────────────────────

Loads the trained XGBoost model and produces a full analysis
(score + SHAP explanation) for a single product.

USAGE
─────
    python ml/predict.py                        # demo mode
    from ml.predict import AarogyaPredictor     # import mode
"""

import os, sys, pickle
import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)

from ml.features import build_feature_matrix
from ml.explain  import build_explanation_context


class AarogyaPredictor:
    """
    Full prediction pipeline: product dict -> score + SHAP explanation.

    USAGE
    ─────
        predictor = AarogyaPredictor()
        result    = predictor.predict(product_dict)
        print(result["predicted_score"])
        print(result["explanation_context"])
    """

    def __init__(self):
        self._model     = None
        self._explainer = None
        self._base_val  = None
        self._loaded    = False
        self._feat_names = None

    def _load(self):
        if self._loaded:
            return
        import xgboost as xgb
        import shap

        model_path = os.path.join(MODELS_DIR, "xgboost_v1.json")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Run: python ml/train.py first."
            )
        self._model = xgb.XGBRegressor()
        self._model.load_model(model_path)

        fn_path = os.path.join(MODELS_DIR, "feature_names.pkl")
        if os.path.exists(fn_path):
            with open(fn_path, "rb") as f:
                self._feat_names = pickle.load(f)

        self._explainer = shap.TreeExplainer(self._model)
        # Handle both scalar and array expected_value (newer SHAP versions)
        ev = self._explainer.expected_value
        self._base_val = float(ev[0]) if hasattr(ev, '__len__') else float(ev)
        self._loaded    = True

    def predict(self, product: dict) -> dict:
        """
        Parameters
        ----------
        product : dict — must contain all required nutrition columns.

        Returns
        -------
        dict with keys:
            predicted_score, explanation_context, raw_shap
        """
        self._load()

        row = pd.DataFrame([product])
        X, _ = build_feature_matrix(row)

        pred        = float(self._model.predict(X)[0])
        shap_vals   = self._explainer.shap_values(X)[0]
        feat_names  = X.columns.tolist()

        explanation = build_explanation_context(
            row.iloc[0], shap_vals, feat_names, pred, self._base_val
        )

        return {
            "predicted_score":      round(pred, 1),
            "explanation_context":  explanation,
            "raw_shap":             dict(zip(feat_names, shap_vals.tolist())),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict scores for all products in a DataFrame."""
        self._load()
        X, _ = build_feature_matrix(df)
        preds = self._model.predict(X)
        df = df.copy()
        df["predicted_score"] = np.round(preds, 1)
        return df


# ─── Quick demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AAROGYA — Inference Demo")
    print("=" * 60)

    # Load dataset for demo
    for path in [
        os.path.join(ROOT, "data", "raw", "aarogya_products_500.csv"),
        os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv"),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("No dataset found.")

    predictor = AarogyaPredictor()

    # Test on first 3 products
    for i in range(min(3, len(df))):
        product = df.iloc[i].to_dict()
        result  = predictor.predict(product)
        ctx     = result["explanation_context"]
        print(f"\nProduct: {ctx['product_name']}")
        print(f"  Actual score   : {product.get('food_score', '?')}")
        print(f"  Predicted score: {result['predicted_score']}")
        print(f"  Strengths      : {ctx['strengths']}")
        print(f"  Concerns       : {ctx['concerns']}")

    print("\n[PREDICT OK]")
