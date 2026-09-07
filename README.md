# 🌿 AAROGYA — AI Food Product Intelligence

> **B.Tech IT Final Project** | End-to-end AI/ML system for packaged food analysis

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io)
[![Groq](https://img.shields.io/badge/LLM-Groq-purple.svg)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)](#testing)

---

## What Is Aarogya?

Aarogya is a packaged-food intelligence system that:

1. **Scores** any packaged food product (0–100) using a deterministic nutrition algorithm
2. **Predicts** scores with a trained XGBoost model (R² = 0.955, MAE = 2.0)
3. **Explains** every score using SHAP values → natural language via Groq LLM
4. **Recommends** category-matched healthier alternatives
5. **Compares** two products side-by-side
6. **Chats** about any product using Groq AI

---

## User Flow

```
SCAN / SEARCH
     ↓
IDENTIFY product (catalog or manual entry)
     ↓
EXTRACT nutrition facts
     ↓
SCORE  (deterministic v1.0 + XGBoost ML)
     ↓
EXPLAIN  (SHAP features → Groq LLM → plain English)
     ↓
COMPARE  (two products side-by-side)
     ↓
RECOMMEND  (preference-weighted alternatives)
```

---

## Project Structure

```
aarogya/
│
├── data/
│   ├── raw/                          ← CSV datasets
│   │   ├── aarogya_food_products_demo.csv   (100 products)
│   │   └── aarogya_products_500.csv          (500 products, generated)
│   ├── validate_dataset.py           ← 35-check validation suite
│   └── generate_large_dataset.py     ← synthetic data generator
│
├── ml/
│   ├── scoring.py      ← Deterministic nutrition scoring (v1.0)
│   ├── features.py     ← Feature engineering pipeline (24 features)
│   ├── train.py        ← XGBoost + RF + Linear training + CV
│   ├── evaluate.py     ← MAE/RMSE/R² + per-category breakdown
│   ├── explain.py      ← SHAP TreeExplainer + global importance
│   └── predict.py      ← AarogyaPredictor class (score + SHAP)
│
├── recommendation/
│   └── recommender.py  ← Preference-weighted recommendation engine
│
├── llm/
│   ├── groq_llm.py     ← Groq LLM integration (score explain + chat)
│   └── gemini.py       ← Google Gemini fallback
│
├── api/
│   └── main.py         ← FastAPI REST backend (8 endpoints)
│
├── tests/
│   ├── conftest.py
│   ├── test_scoring.py     ← 14 scoring tests
│   ├── test_features.py    ← 10 feature engineering tests
│   └── test_recommender.py ← 11 recommender tests
│
├── notebooks/              ← Jupyter EDA notebooks
├── models/                 ← Trained model files (gitignored)
├── app.py                  ← Day 1 environment checker
├── app_streamlit.py        ← Full Streamlit UI
├── generate_clean_csv.py   ← CSV fixer utility
├── requirements.txt
├── .env.example
└── README.md
```

---

## Aarogya Score System

| Score | Tier | Meaning |
|-------|------|---------|
| 75–100 | 🟢 Excellent | Great nutritional profile |
| 60–74 | 🟡 Good | Above average |
| 40–59 | 🟠 Moderate | Balanced trade-offs |
| 0–39 | 🔴 Needs Consideration | Notable concerns |

**Score components:**
- 🍬 Sugar (lower = better)
- 💪 Protein (higher = better)
- 🌾 Fiber (higher = better)
- 🧂 Sodium (lower = better)
- 🧈 Saturated Fat (lower = better)
- ⚡ Energy Density
- ⚙️ Processing Level penalty/bonus
- 🌾 Whole Grain bonus

---

## ML Model Results

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression (baseline) | 1.01 | 1.31 | 0.991 |
| Random Forest | 3.16 | 4.39 | 0.895 |
| **XGBoost (primary)** | **2.00** | **2.86** | **0.955** |

**XGBoost 5-fold CV MAE: 3.16 ± 0.68**

Top SHAP features (by |mean SHAP|):
1. Processing Level (5.69)
2. Sugar (3.08)
3. Protein per Calorie (2.44)
4. Fiber (1.83)
5. Sodium (1.63)

---

## Quick Start

### 1. Setup
```bash
git clone <your-repo-url>
cd aarogya

python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Generate Dataset & Train Model
```bash
python data/generate_large_dataset.py   # creates 500-product dataset
python ml/train.py                       # trains XGBoost (R²=0.955)
```

### 4. Run Tests
```bash
pytest tests/ -v                         # 40 tests → all pass
```

### 5. Launch Streamlit UI
```bash
streamlit run app_streamlit.py
# Opens at http://localhost:8501
```

### 6. Launch FastAPI (optional)
```bash
uvicorn api.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed status |
| POST | `/analyze` | Score + explain a product |
| POST | `/recommend` | Get alternatives |
| POST | `/compare` | Compare two products |
| GET | `/product/{id}` | Lookup from catalog |
| GET | `/categories` | List all categories |
| GET | `/top/{category}` | Top products in category |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Pandas, NumPy |
| ML | XGBoost, scikit-learn |
| Explainability | SHAP (TreeExplainer) |
| LLM | Groq (compound-mini) |
| UI | Streamlit, Plotly |
| API | FastAPI, Uvicorn |
| Testing | pytest (40 tests) |

---

## Dataset

- **100 products** (demo): `data/raw/aarogya_food_products_demo.csv`
- **500 products** (generated): `data/raw/aarogya_products_500.csv`
- **10 categories**: Biscuits & Cookies, Breakfast Cereals, Chips & Namkeen, Beverages, Chocolates & Sweet Snacks, Snack/Protein Bars, Instant Foods, Bread & Bakery, Dairy & Dairy Drinks, Sauces & Condiments
- **31 columns** per product including full nutrition label, processing level, allergens

---

## Important Disclaimer

> ⚠️ Aarogya is an academic project built on synthetic data.
> Scores are **comparative**, not clinical.
> This is **not medical advice**.
> Do not use for dietary decisions without consulting a qualified nutritionist.

---

## Author

**B.Tech IT Student** | AAROGYA Project | 2026
