# AAROGYA — AI Food Product Intelligence

> A packaged-food intelligence and comparative decision-support system.

**AAROGYA** analyzes packaged food products, generates an explainable Aarogya Food Insight Score (0–100), and recommends comparatively suitable alternatives based on user preferences — using a full ML pipeline from data collection through SHAP-explained predictions.

---

## ⚠️ Important Disclaimers

- This is an **academic / research project**.
- Aarogya is **not** a medical device, clinical nutrition system, or diagnostic tool.
- It does **not** determine whether a product is "healthy" or "unhealthy" in a clinical sense.
- All scores represent **comparative nutritional profiles** relative to defined methodology, not medical advice.
- The demo dataset (`aarogya_food_products_demo.csv`) is **100% synthetic** — fictional brands, fabricated values. It is intended for development and testing only.
- 100 synthetic products are **not sufficient** for a production ML model. Real data collection (1,000–5,000+ products) is required for Week 2+.

---

## Core Design Principle

```
Rules Calculate → ML Learns → SHAP Explains → AI Communicates → Preferences Personalize
```

| Component | Role |
|-----------|------|
| **Deterministic Scoring** | Creates transparent, auditable target labels for ML training |
| **XGBoost ML Model** | Learns relationships between nutritional features and score |
| **SHAP** | Explains which features drove each prediction |
| **Gemini API** | Converts structured analysis into user-friendly natural language |
| **User Preferences** | Personalize recommendation ranking |

**Gemini does NOT generate the numerical score.** The score comes from the ML model. Gemini only translates the structured result into readable language.

---

## Project Status

| Phase | Status |
|-------|--------|
| Project Foundation | ✅ Day 1 |
| Demo Dataset | ✅ Day 1 |
| Data Validation | ✅ Day 1 |
| Baseline Scoring | ✅ Day 1 |
| Feature Engineering | ✅ Day 1 |
| Recommendation Engine (Rule-based) | ✅ Day 1 |
| Real Data Collection | 🔲 Week 2 |
| ML Training | 🔲 Week 7 |
| SHAP Explainability | 🔲 Week 9 |
| Gemini Integration | 🔲 Week 13 |
| Streamlit UI | 🔲 Week 14 |

---

## Quick Start (Day 1)

### 1. Create virtual environment

```powershell
cd aarogya
python -m venv venv
venv\Scripts\activate
```

### 2. Install Day 1 dependencies

```powershell
pip install -r requirements.txt
```

### 3. Copy environment file

```powershell
copy .env.example .env
```

### 4. Verify environment

```powershell
python app.py
```

### 5. Validate dataset

```powershell
python data/validate_dataset.py
```

### 6. Run baseline scoring

```powershell
python ml/scoring.py
```

### 7. Test feature engineering

```powershell
python ml/features.py
```

---

## Project Structure

```
aarogya/
│
├── app.py                          # Main entry point
├── requirements.txt                # Phased dependencies
├── README.md
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/
│   │   ├── aarogya_food_products_demo.csv     ← 100 synthetic products
│   │   └── aarogya_user_preferences_demo.csv  ← 10 demo user profiles
│   ├── validate_dataset.py         # Dataset validation script
│   ├── cleaned/                    # Cleaned data (Week 3)
│   └── processed/                  # ML-ready features (Week 5)
│
├── ml/
│   ├── scoring.py                  # ✅ Deterministic scoring engine (v1.0)
│   ├── features.py                 # ✅ Feature engineering pipeline
│   ├── preprocessing.py            # (Week 3) Data cleaning
│   ├── train.py                    # (Week 7) Model training
│   ├── evaluate.py                 # (Week 8) Model evaluation
│   └── predict.py                  # (Week 8) Inference
│
├── models/                         # Saved model artifacts (.pkl/.joblib)
│
├── recommendation/
│   ├── recommender.py              # ✅ Rule-based recommendation engine
│   ├── similarity.py               # (Week 11) Cosine/Euclidean similarity
│   └── ranking.py                  # (Future) Learning-to-Rank model
│
├── ocr/
│   ├── preprocess.py               # (Week 12) OpenCV image preprocessing
│   ├── extract.py                  # (Week 12) PaddleOCR text extraction
│   └── parser.py                   # (Week 12) Nutrition/ingredient parser
│
├── llm/
│   └── gemini.py                   # (Week 13) Gemini explanation layer
│
├── database/
│   └── mongodb.py                  # ✅ MongoDB connection manager
│
├── api/
│   └── main.py                     # (Week 13) FastAPI backend
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # (Week 4) EDA
│   ├── 02_feature_engineering.ipynb # (Week 5)
│   ├── 03_model_training.ipynb     # (Week 7)
│   └── 04_model_evaluation.ipynb   # (Week 8)
│
├── tests/                          # (Week 2+) pytest test suite
└── utils/                          # Shared utility functions
```

---

## ML Problem Definition

| Task | Type | Target | When |
|------|------|--------|------|
| **Score Regression** (Primary) | Regression | Aarogya Food Insight Score (0–100) | Week 7 |
| **Food Category Classification** (Secondary) | Multi-class | processing_level or tier | Week 8+ |
| **Preference-based Ranking** (Future) | Learning-to-Rank | Preferred product given user preferences | Week 11+ |

### Why Regression first?

Regression produces a **continuous** output (0–100) that preserves the granularity of the scoring methodology. Classification bins the score and loses information. The continuous score also enables natural ranking for recommendations.

---

## Scoring Methodology (v1.0)

```
Raw Nutrition (per 100g)
        ↓
Per-nutrient component scores (each 0–100)
        ↓
Weighted combination
        ↓
Processing level modifier (±5 to ±8 points)
        ↓
Ingredient modifiers (±3 points)
        ↓
Aarogya Food Insight Score (clipped to 0–100)
```

| Component | Weight | Direction |
|-----------|--------|-----------|
| Sugar | 25% | Lower is better |
| Protein | 20% | Higher is better |
| Fiber | 20% | Higher is better |
| Sodium | 15% | Lower is better |
| Saturated Fat | 12% | Lower is better |
| Calories (energy density) | 8% | Lower is better |

**Score Version:** `v1.0` — Track versions in `scoring_versions` MongoDB collection.

### Label Limitation (Academic Disclosure — Required)

The ML model is trained on scores generated by these deterministic rules. This creates a **circularity risk**: the model may simply learn to imitate the scoring rules rather than discovering independent nutritional patterns. This is a fundamental limitation of rule-generated labels and must be disclosed in all documentation and viva presentations.

**Future improvements:** Expert annotation, validated frameworks (NutriScore, FSSAI), pairwise human preference labels.

---

## Dataset

### Demo Dataset (Day 1)

| Property | Value |
|----------|-------|
| File | `data/raw/aarogya_food_products_demo.csv` |
| Type | **Synthetic — NOT real product data** |
| Rows | 100 |
| Source | `demo_synthetic` |
| Verified | `false` |
| Purpose | Development / Testing / Demonstration only |

**This dataset is NOT sufficient for production ML training.**

### Real Dataset (Week 2+)

Target: **1,000–5,000+ products** from:

1. **Open Food Facts** (Primary) — `world.openfoodfacts.org/data`
2. **USDA FoodData Central** (Secondary) — `fdc.nal.usda.gov/api-guide.html`
3. **ICMR-NIN Indian RDA** (Reference/Validation)
4. **Manual verification** of Indian packaged food labels

---

## Technology Stack

| Layer | Technology | Phase |
|-------|-----------|-------|
| Data | pandas, numpy | Day 1 |
| ML | scikit-learn, XGBoost | Week 7 |
| Explainability | SHAP | Week 9 |
| NLP | Sentence Transformers | Week 10 |
| OCR | OpenCV + PaddleOCR | Week 12 |
| LLM | Gemini API | Week 13 |
| Database | MongoDB (pymongo) | Week 13 |
| Backend | FastAPI | Week 13 |
| Frontend | Streamlit | Week 14 |
| Experiments | MLflow | Week 7+ |
| Testing | pytest | Week 2+ |

---

## 12-Week Roadmap

| Week | Focus | Key Output |
|------|-------|-----------|
| 1 | **Foundation** | Project structure, demo dataset, scoring, features |
| 2 | **Real Data Collection** | Open Food Facts + USDA API data pull |
| 3 | **Data Cleaning** | Deduplication, unit normalization, missing value strategy |
| 4 | **EDA** | Data quality report, distribution plots, correlation analysis |
| 5 | **Feature Engineering** | Validated feature matrix, derived ratios |
| 6 | **Baseline Scoring Validation** | Score distribution analysis, threshold tuning |
| 7 | **ML Training** | Linear baseline + XGBoost model |
| 8 | **Evaluation** | MAE, RMSE, R², cross-validation, leakage check |
| 9 | **SHAP** | Feature importance, explanation pipeline |
| 10 | **Ingredient NLP** | Sentence Transformer embeddings |
| 11 | **Recommendation** | Enhanced recommendation engine |
| 12 | **Barcode + OCR** | Barcode lookup, PaddleOCR integration |
| 13 | **Gemini + FastAPI + MongoDB** | Full backend integration |
| 14 | **Streamlit + Testing + Deployment** | Production-ready UI |

---

## Recommendation System

```
Current Product
       ↓
Same recommendation_group
       ↓
Candidate Products
       ↓
Preference-Weighted Gap Scores
       ↓
Rank
       ↓
Top 3 Alternatives
```

### Example: Sugar-focused user

```python
from recommendation.recommender import AarogyaRecommender
import pandas as pd

df = pd.read_csv("data/raw/aarogya_food_products_demo.csv")
rec = AarogyaRecommender(df)

results = rec.recommend(
    product_id="AAR004",              # Choco Cream Sandwich Biscuits
    preference_profile="sugar_focused",
    top_n=3,
)
print(results)
```

---

## Viva Key Points

1. **Why not just use Gemini to score products?**
   Gemini is a language model, not a nutrition analysis system. It would hallucinate scores, have no reproducibility, and couldn't be audited. The ML model produces deterministic, explainable scores.

2. **What is the primary limitation?**
   Label circularity — the model learns to imitate rules, not discover nutritional truth. Solution: expert labels + larger real-world dataset.

3. **Why XGBoost?**
   Handles tabular nutritional data well, natively handles NaN, fast, interpretable via SHAP, industry standard for tabular ML.

4. **Why SHAP?**
   Model-agnostic explainability that decomposes each prediction into per-feature contributions. Enables "this product scored 62 primarily because of high sugar and low fiber."

5. **Why recommendation_group, not category?**
   Prevents cross-category recommendations. A user with chips should get a better chip alternative, not a salad recommendation.

---

## Contributing

This is a college project. Contributions welcome from collaborators.

```
main → develop → feature/week-N-<feature-name>
```

**Never commit:**
- `.env` files
- API keys
- Trained model files (`.pkl`, `.joblib`)
- Large data files (Open Food Facts dumps)

---

## License

Academic project — MIT License

---

*Built as a B.Tech IT final-year project demonstrating end-to-end ML engineering.*
