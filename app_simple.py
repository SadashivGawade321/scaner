"""
app_simple.py  —  AAROGYA Smart Food Intelligence Platform  (Presentation Build)
================================================================================
Features:
  • Animated gradient splash screen on load
  • 6-Language Multilingual Support: English | Hindi | Marathi | Gujarati | Tamil | Spanish
  • 5 Scan Modes: 📷 Camera | 🤖 AI Vision | 📤 Upload | 🔍 Search | 🎯 Demo
  • RapidOCR (Local ONNX) + Groq AI: 100% reliable vision & label reading (No 404 errors)
  • MongoDB Atlas Cloud Sync: 570+ products seeded, auto-saves every scan & new AI analysis
  • Smart Search with Fuzzy Matching + "✨ Instant AI Analysis" for ANY product on earth
  • Food Safety Score (0-100) + Red Flags + Better Alternatives + Localized AI Doctor Explanation
"""
import os, sys, re, io, base64, datetime
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st
from utils.translations import t, LANGUAGES, get_lang_code

st.set_page_config(page_title="Aarogya Food Intelligence", page_icon="🌿",
                   layout="centered", initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════════════════════════════
# ANIMATED SPLASH SCREEN  (shown once per session)
# ══════════════════════════════════════════════════════════════════════════════
if "splash_done" not in st.session_state:
    st.session_state["splash_done"] = True
    splash = st.empty()
    splash.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body,.stApp{background:#0d1117 !important}
.splash{
  position:fixed;top:0;left:0;width:100vw;height:100vh;
  background:#0d1117;display:flex;flex-direction:column;
  align-items:center;justify-content:center;z-index:99999;
  animation:fadeOut 0.5s ease-in-out 2.4s forwards;
}
.splash-logo{
  font-size:4.5rem;font-weight:900;font-family:'Inter',sans-serif;
  background:linear-gradient(135deg,#3fb950 0%,#58a6ff 50%,#a371f7 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:popIn 0.6s cubic-bezier(.175,.885,.32,1.275) both;
}
.splash-sub{
  color:#8b949e;font-family:'Inter',sans-serif;font-size:1.1rem;
  margin-top:.5rem;letter-spacing:2px;text-transform:uppercase;
  animation:popIn 0.6s 0.2s cubic-bezier(.175,.885,.32,1.275) both;
}
.splash-bar-wrap{
  margin-top:2rem;width:240px;height:4px;background:#21262d;
  border-radius:99px;overflow:hidden;
  animation:popIn 0.5s 0.4s ease both;
}
.splash-bar{
  height:100%;width:0;background:linear-gradient(90deg,#3fb950,#58a6ff);
  border-radius:99px;animation:grow 1.8s 0.5s ease-in-out forwards;
}
.splash-tagline{
  color:#58a6ff;font-family:'Inter',sans-serif;font-size:.85rem;
  margin-top:1rem;letter-spacing:1px;font-weight:600;
  animation:popIn 0.5s 0.7s ease both;
}
.splash-icons{
  display:flex;gap:1.8rem;margin-top:1.8rem;font-size:2rem;
  animation:popIn 0.5s 0.9s ease both;
}
.splash-icon{animation:pulse 1.4s infinite alternate;}
.splash-icon:nth-child(2){animation-delay:.2s}
.splash-icon:nth-child(3){animation-delay:.4s}
@keyframes popIn{from{opacity:0;transform:scale(.7) translateY(20px)}to{opacity:1;transform:none}}
@keyframes grow{from{width:0}to{width:100%}}
@keyframes pulse{from{transform:scale(1)}to{transform:scale(1.2)}}
@keyframes fadeOut{from{opacity:1;pointer-events:all}to{opacity:0;pointer-events:none;display:none}}
</style>
<div class="splash">
  <div class="splash-logo">🌿 Aarogya</div>
  <div class="splash-sub">AI Food Intelligence</div>
  <div class="splash-bar-wrap"><div class="splash-bar"></div></div>
  <div class="splash-tagline">Scan · Score · Multilingual · Cloud Sync</div>
  <div class="splash-icons">
    <span class="splash-icon">📷</span>
    <span class="splash-icon">🤖</span>
    <span class="splash-icon">☁️</span>
  </div>
</div>
""", unsafe_allow_html=True)
    import time; time.sleep(2.6)
    splash.empty()

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#0d1117;color:#e6edf3}
.stApp{background:#0d1117}
.stTabs [data-baseweb="tab-list"]{background:#161b22;border-radius:12px;padding:4px;gap:4px}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#8b949e;padding:.5rem 1rem}
.stTabs [aria-selected="true"]{background:#21262d;color:#e6edf3;font-weight:700}

.scan-title{font-size:2.8rem;font-weight:900;text-align:center;
  background:linear-gradient(135deg,#3fb950,#58a6ff,#a371f7);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:shine 3s linear infinite}
.scan-sub{text-align:center;color:#8b949e;font-size:0.95rem;margin-bottom:1.2rem}

.verdict-safe    {background:linear-gradient(135deg,#0a2a15,#0d3320);border:2px solid #3fb950;
  border-radius:20px;padding:2rem;text-align:center;
  box-shadow:0 0 40px #3fb95022;animation:glow-green 2s ease-in-out infinite alternate}
.verdict-moderate{background:linear-gradient(135deg,#1f1a08,#2a2208);border:2px solid #d29922;
  border-radius:20px;padding:2rem;text-align:center;
  box-shadow:0 0 40px #d2992222}
.verdict-harmful {background:linear-gradient(135deg,#2a0a0a,#330d0d);border:2px solid #f85149;
  border-radius:20px;padding:2rem;text-align:center;
  box-shadow:0 0 40px #f8514922;animation:glow-red 2s ease-in-out infinite alternate}

.rec-card{background:#161b22;border:1px solid #30363d;border-radius:14px;
  padding:.9rem 1.2rem;margin-bottom:.65rem;transition:border-color .2s;
  animation:slideUp .4s ease both}
.rec-card:hover{border-color:#58a6ff}

.nutrient-row{display:flex;justify-content:space-between;padding:.4rem 0;
  border-bottom:1px solid #21262d;font-size:.88rem}
.nutrient-label{color:#8b949e}.nutrient-value{font-weight:600}
.tag-good{background:#0d2b1a;color:#3fb950;border-radius:6px;padding:2px 9px;font-size:.78rem;font-weight:700}
.tag-bad {background:#2d0f0f;color:#f85149;border-radius:6px;padding:2px 9px;font-size:.78rem;font-weight:700}

.history-card{background:#161b22;border:1px solid #21262d;border-radius:10px;
  padding:.7rem 1rem;margin-bottom:.5rem;display:flex;align-items:center;gap:.8rem}

.disclaimer{text-align:center;color:#484f58;font-size:.72rem;margin-top:2rem;
  padding-top:1rem;border-top:1px solid #21262d}

@keyframes shine{0%{filter:hue-rotate(0deg)}100%{filter:hue-rotate(30deg)}}
@keyframes glow-green{from{box-shadow:0 0 20px #3fb95022}to{box-shadow:0 0 50px #3fb95055}}
@keyframes glow-red{from{box-shadow:0 0 20px #f8514922}to{box-shadow:0 0 50px #f8514955}}
@keyframes slideUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE SELECTION (Top Bar)
# ══════════════════════════════════════════════════════════════════════════════
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

top_c1, top_c2 = st.columns([3, 2])
with top_c2:
    lang_labels = list(LANGUAGES.values())
    curr_idx = list(LANGUAGES.keys()).index(st.session_state["lang"])
    selected_label = st.selectbox(
        "Language",
        options=lang_labels,
        index=curr_idx,
        label_visibility="collapsed",
        key="lang_selector"
    )
    st.session_state["lang"] = get_lang_code(selected_label)

lang = st.session_state["lang"]

# ══════════════════════════════════════════════════════════════════════════════
# DEMO DATA
# ══════════════════════════════════════════════════════════════════════════════
DEMO_DIR = os.path.join(ROOT, "static", "demo")
DEMO_PRODUCTS = {
    "chocobite.jpg": {
        "product_name":"ChocoBite Cream Biscuits","brand":"SnackTime",
        "category":"Biscuits & Cookies","processing_level":"ultra_processed",
        "recommendation_group":"biscuits_cookies",
        "energy_kcal_100g":520,"protein_g_100g":5,"carbs_g_100g":68,
        "sugar_g_100g":38,"added_sugar_g_100g":36,"total_fat_g_100g":22,
        "saturated_fat_g_100g":12,"trans_fat_g_100g":0,"fiber_g_100g":1,
        "sodium_mg_100g":420,"ingredient_count":14,
        "contains_whole_grain":False,"contains_added_sugar":True,
        "contains_artificial_sweetener":False,"contains_allergen":True,
    },
    "nutrioats.jpg": {
        "product_name":"NutriOats Instant Porridge","brand":"HealthFirst",
        "category":"Breakfast Cereals","processing_level":"minimally_processed",
        "recommendation_group":"breakfast_cereals",
        "energy_kcal_100g":380,"protein_g_100g":13,"carbs_g_100g":62,
        "sugar_g_100g":4,"added_sugar_g_100g":0,"total_fat_g_100g":7,
        "saturated_fat_g_100g":1.5,"trans_fat_g_100g":0,"fiber_g_100g":9,
        "sodium_mg_100g":95,"ingredient_count":4,
        "contains_whole_grain":True,"contains_added_sugar":False,
        "contains_artificial_sweetener":False,"contains_allergen":False,
    },
    "masalachips.jpg": {
        "product_name":"MasalaKing Spicy Chips","brand":"CrunchCo",
        "category":"Chips & Namkeen","processing_level":"ultra_processed",
        "recommendation_group":"chips_namkeen",
        "energy_kcal_100g":550,"protein_g_100g":6,"carbs_g_100g":58,
        "sugar_g_100g":3,"added_sugar_g_100g":1,"total_fat_g_100g":32,
        "saturated_fat_g_100g":8,"trans_fat_g_100g":0,"fiber_g_100g":3,
        "sodium_mg_100g":680,"ingredient_count":16,
        "contains_whole_grain":False,"contains_added_sugar":False,
        "contains_artificial_sweetener":False,"contains_allergen":False,
    },
}
BARCODE_MAP = {
    "8901001000042":"chocobite.jpg",
    "8901001000001":"nutrioats.jpg",
    "8901001000023":"masalachips.jpg",
}

# ══════════════════════════════════════════════════════════════════════════════
# MONGODB ATLAS & RESOURCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_mongo_db():
    """Return MongoDB Atlas database instance or None."""
    try:
        from dotenv import load_dotenv; load_dotenv()
        import pymongo
        uri  = os.getenv("MONGO_URI","")
        if not uri and hasattr(st, "secrets") and "MONGO_URI" in st.secrets:
            uri = st.secrets["MONGO_URI"]
        name = os.getenv("MONGO_DB_NAME","aarogya_db")
        if hasattr(st, "secrets") and not os.getenv("MONGO_DB_NAME") and "MONGO_DB_NAME" in st.secrets:
            name = st.secrets["MONGO_DB_NAME"]
        if not uri: return None
        cli  = pymongo.MongoClient(uri, serverSelectionTimeoutMS=4000)
        cli.admin.command("ping")
        return cli[name]
    except Exception:
        return None

def get_mongo_collection(col_name="scan_history"):
    db = get_mongo_db()
    if db is not None:
        return db[col_name]
    return None

@st.cache_data(ttl=120)
def load_catalog():
    """Load product catalog: MongoDB Atlas primary, local CSV fallback."""
    # 1. Try MongoDB Atlas
    try:
        col = get_mongo_collection("products")
        if col is not None:
            docs = list(col.find({}, {"_id": 0}))
            if docs and len(docs) >= 10:
                df = pd.DataFrame(docs)
                return df
    except Exception:
        pass

    # 2. Fallback to local CSV
    for p in [
        os.path.join(ROOT,"data","raw","aarogya_products_500.csv"),
        os.path.join(ROOT,"data","raw","aarogya_food_products_demo.csv"),
    ]:
        if os.path.exists(p): return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_resource
def load_recommender():
    from recommendation.recommender import AarogyaRecommender
    df = load_catalog()
    return AarogyaRecommender(df) if not df.empty else None

@st.cache_resource
def load_explainer():
    # Try Groq first (primary), then fall back to Gemini
    try:
        from llm.groq_llm import AarogyaGroqExplainer
        exp = AarogyaGroqExplainer()
        if exp.is_ready:
            return exp
    except Exception:
        pass
    # Fallback: Gemini
    try:
        from llm.gemini import AarogyaExplainer
        exp = AarogyaExplainer()
        if exp.is_ready:
            return exp
    except Exception:
        pass
    # Return whatever we have (even if not ready, for fallback mode)
    try:
        from llm.groq_llm import AarogyaGroqExplainer
        return AarogyaGroqExplainer()
    except Exception:
        from llm.gemini import AarogyaExplainer
        return AarogyaExplainer()

def save_scan(product, score, verdict):
    """Save scanned product to MongoDB Atlas scan_history."""
    try:
        col = get_mongo_collection("scan_history")
        if col is not None:
            col.insert_one({
                "product_name": product.get("product_name","Unknown"),
                "brand":        product.get("brand",""),
                "category":     product.get("category",""),
                "score":        round(float(score), 2),
                "verdict":      verdict,
                "scanned_at":   datetime.datetime.utcnow(),
            })
    except Exception:
        pass

def save_product_to_mongo(product):
    """Save or update product in MongoDB Atlas products collection."""
    try:
        col = get_mongo_collection("products")
        if col is not None and product.get("product_name"):
            clean_prod = {k: v for k, v in product.items() if not k.startswith("_")}
            col.update_one(
                {"product_name": clean_prod["product_name"]},
                {"$set": clean_prod},
                upsert=True
            )
            load_catalog.clear()
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# SCAN & VISION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def scan_barcode(img_bytes):
    # Try zxing-cpp first (fast, pure barcode reader)
    try:
        import zxingcpp
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        results = zxingcpp.read_barcodes(img)
        if results:
            return results[0].text.strip()
    except Exception:
        pass
    # Fallback: use Gemini Vision to read barcode from image
    try:
        from llm.gemini import AarogyaExplainer
        import json, re
        gem = AarogyaExplainer()
        if gem.is_ready and gem._model:
            from PIL import Image
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            res = gem._model.generate_content([
                "Look at this image. If there is a barcode or QR code visible, "
                "return ONLY the barcode number as plain text (digits only, no spaces). "
                "If no barcode is found, return exactly: NONE",
                img
            ])
            if res and res.text:
                txt = res.text.strip()
                # Extract just digits
                digits = re.sub(r'[^0-9]', '', txt)
                if digits and len(digits) >= 8:
                    return digits
    except Exception:
        pass
    return None

def groq_vision_scan(img_bytes, mode="product"):
    """
    AI Vision extraction pipeline.
    Uses Groq RapidOCR (primary) with Gemini multimodal Vision fallback.
    If Groq OCR fails or returns error, automatically retries with Gemini Vision.
    """
    result = None

    # Step 1: Try primary explainer (Groq RapidOCR)
    exp = load_explainer()
    if exp.is_ready:
        try:
            if mode == "ingredients":
                result = exp.analyze_ingredients(img_bytes)
            else:
                result = exp.analyze_product_image(img_bytes)
        except Exception:
            result = None

    # Step 2: If primary failed or returned error, try Gemini Vision directly
    if not result or result.get("error"):
        try:
            from llm.gemini import AarogyaExplainer
            gem = AarogyaExplainer()
            if gem.is_ready:
                if mode == "ingredients":
                    gem_result = gem.analyze_ingredients(img_bytes)
                else:
                    gem_result = gem.analyze_product_image(img_bytes)
                if gem_result and not gem_result.get("error"):
                    return gem_result
        except Exception:
            pass

    # Return whatever we got (could be error dict or None)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_score(p):
    from ml.scoring import compute_aarogya_score
    r = compute_aarogya_score(p)
    return r["final_score"], r.get("breakdown", {})

def get_verdict(s, l="en"):
    if s >= 65: return t("verdict_safe", l), "#3fb950", "✅", "verdict-safe"
    if s >= 40: return t("verdict_mod",  l), "#d29922", "⚠️", "verdict-moderate"
    return           t("verdict_harm", l), "#f85149", "❌", "verdict-harmful"

def score_color(s):
    return "#3fb950" if s>=65 else ("#d29922" if s>=40 else "#f85149")

def safe_float(val, default=0.0):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        f = float(val)
        return default if pd.isna(f) else f
    except (ValueError, TypeError):
        return default

def safe_list(val):
    """Safely convert any value (NaN, None, string, list, tuple) to a clean list of strings."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    if isinstance(val, str):
        v = val.strip()
        if not v or v.lower() in ("nan", "none", "[]", "null"):
            return []
        if v.startswith("[") and v.endswith("]"):
            import ast
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if x and not (isinstance(x, float) and pd.isna(x))]
            except Exception:
                pass
        return [x.strip() for x in v.split(",") if x.strip()]
    if isinstance(val, (list, tuple, set)):
        return [str(x).strip() for x in val if x and not (isinstance(x, float) and pd.isna(x))]
    return [str(val)]

def nut_row(label, val, unit, bad_thr, hi_good=False):
    fval = safe_float(val)
    is_bad = (fval < bad_thr) if hi_good else (fval > bad_thr)
    tag = (f"<span class='tag-bad'>Low</span>"  if is_bad and hi_good
      else f"<span class='tag-bad'>High</span>" if is_bad
      else f"<span class='tag-good'>OK</span>")
    return (f"<div class='nutrient-row'><span class='nutrient-label'>{label}</span>"
            f"<span class='nutrient-value'>{fval:.1f} {unit}&nbsp;{tag}</span></div>")

def blank_product(name="Scanned Product", cat="Biscuits & Cookies"):
    return {
        "product_name":name,"category":cat,
        "recommendation_group":"biscuits_cookies","processing_level":"processed",
        "energy_kcal_100g":0,"protein_g_100g":0,"carbs_g_100g":0,
        "sugar_g_100g":0,"added_sugar_g_100g":0,"total_fat_g_100g":0,
        "saturated_fat_g_100g":0,"trans_fat_g_100g":0,"fiber_g_100g":0,
        "sodium_mg_100g":0,"ingredient_count":10,
        "contains_whole_grain":False,"contains_added_sugar":False,
        "contains_artificial_sweetener":False,"contains_allergen":False,
    }

# ══════════════════════════════════════════════════════════════════════════════
# RESULT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_result(product, source=""):
    score, breakdown = get_score(product)
    verdict, color, icon, css = get_verdict(score, lang)
    name  = product.get("product_name","Unknown")
    brand = product.get("brand","")
    cat   = product.get("category","")
    proc  = product.get("processing_level","").replace("_"," ").title()

    save_scan(product, score, verdict)

    if source:
        st.caption(f"🔎 Source: {source}")

    # Verdict card
    st.markdown(f"""
    <div class='{css}'>
      <div style='font-size:3.8rem;'>{icon}</div>
      <div style='font-size:2rem;font-weight:900;color:{color};margin:.3rem 0'>{verdict}</div>
      <div style='font-size:3.5rem;font-weight:900;color:{color};line-height:1'>{score:.0f}
        <span style='font-size:1.2rem;color:#8b949e;font-weight:400'>/100</span></div>
      <div style='font-size:1.15rem;font-weight:700;margin-top:.7rem'>{name}</div>
      <div style='color:#8b949e;font-size:.85rem;margin-top:.2rem'>
        {brand}{" · " if brand else ""}{cat}{" · " if cat else ""}{proc}</div>
    </div>""", unsafe_allow_html=True)

    # Localized AI explanation
    with st.spinner(t("analyzing_ai", lang)):
        try:
            ex  = load_explainer()
            ai  = ex.explain_score({
                "product_name":name,"category":cat,
                "processing_level":product.get("processing_level",""),
                "predicted_score":score,"base_value":55.0,
                "strengths":breakdown.get("strengths",[]),
                "concerns": breakdown.get("concerns",[]),
                "shap_top_features":{},
            }, language=lang)
        except Exception: ai = None

    if ai:
        st.markdown(f"#### {t('why_score', lang)}")
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:14px;"
            f"padding:1.1rem 1.3rem;color:#c9d1d9;"
            f"font-size:.95rem;line-height:1.8'>💬 {ai}</div>",
            unsafe_allow_html=True)

    # Red flag ingredients if present
    flags = safe_list(product.get("red_flags"))
    if flags:
        st.markdown(f"#### {t('red_flags', lang)}")
        flag_html = " ".join([f"<span class='tag-bad' style='display:inline-block;margin:3px 4px;font-size:0.85rem'>⚠️ {fl}</span>" for fl in flags])
        st.markdown(f"<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;padding:.8rem 1rem'>{flag_html}</div>", unsafe_allow_html=True)

    # Nutrition facts
    st.markdown(f"#### 📊 {t('food_safety_score', lang)}: {t('calories', lang)} & Nutrients (per 100g)")
    rows = [
        (f"⚡ {t('calories', lang)}",     "energy_kcal_100g",    "kcal", 400, False),
        (f"🍬 {t('sugar', lang)}",        "sugar_g_100g",         "g",    15,  False),
        (f"💪 {t('protein', lang)}",      "protein_g_100g",       "g",    5,   True),
        (f"🌾 {t('fiber', lang)}",        "fiber_g_100g",         "g",    3,   True),
        (f"🧂 {t('sodium', lang)}",       "sodium_mg_100g",       "mg",   400, False),
        (f"🧈 {t('sat_fat', lang)}",      "saturated_fat_g_100g", "g",    5,   False),
        (f"🫒 {t('fat', lang)}",          "total_fat_g_100g",     "g",    20,  False),
        (f"🍞 {t('carbs', lang)}",        "carbs_g_100g",         "g",    60,  False),
    ]
    html = "".join(nut_row(l, product.get(c), u, th, hg) for l, c, u, th, hg in rows)
    st.markdown(
        f"<div style='background:#161b22;border:1px solid #30363d;"
        f"border-radius:12px;padding:.8rem 1.2rem'>{html}</div>",
        unsafe_allow_html=True)

    # Recommendations
    if score < 65:
        st.markdown("---")
        st.markdown(f"### {t('alternatives', lang)}")
        _show_recs(product, score)
    else:
        st.markdown("")
        st.success(f"✅ {t('verdict_safe', lang)} — Great choice!")

    st.markdown("---")
    if st.button(t("scan_another", lang), use_container_width=True, type="primary"):
        del st.session_state["result"]
        st.rerun()

def _show_recs(product, cur):
    catalog = load_catalog()
    recs = None
    try:
        rec = load_recommender()
        pid = product.get("product_id")
        if rec and pid:
            recs = rec.recommend(pid, preference_profile="balanced", top_n=3)
        elif rec and product.get("recommendation_group"):
            recs = rec.recommend_by_data(
                current_product=product,
                recommendation_group=product["recommendation_group"],
                preference_profile="balanced", top_n=3)
    except Exception: pass

    if (recs is None or recs.empty) and not catalog.empty and "food_score" in catalog.columns:
        same = catalog[
            (catalog["category"]==product.get("category","")) &
            (catalog["food_score"]>cur)
        ].nlargest(3,"food_score")
        if not same.empty:
            recs = same.copy()
            recs["recommendation_reason"] = "Higher scoring product in same category"

    # Also check if product has explicit healthier_alternatives from Groq
    alts = safe_list(product.get("healthier_alternatives"))
    if (recs is None or recs.empty) and alts:
        for alt_name in alts:
            st.markdown(f"""
            <div class='rec-card'>
              <div style='display:flex;align-items:center;gap:1rem'>
                <div style='font-size:1.5rem;font-weight:900;color:#3fb950;min-width:48px;text-align:center'>🌱</div>
                <div style='flex:1'>
                  <div style='font-weight:700'>{alt_name}</div>
                  <div style='color:#8b949e;font-size:.82rem;margin-top:.1rem'>Recommended whole food alternative</div>
                </div>
                <div style='background:#3fb95022;color:#3fb950;border-radius:8px;
                  padding:.25rem .7rem;font-size:.78rem;font-weight:700;white-space:nowrap'>HEALTHIER</div>
              </div>
            </div>""", unsafe_allow_html=True)
        return

    if recs is not None and not recs.empty:
        for _, r in recs.iterrows():
            rs  = float(r.get("food_score", r.get("ranking_score",0)))
            rc  = score_color(rs)
            rv,_,ri,_ = get_verdict(rs, lang)
            st.markdown(f"""
            <div class='rec-card'>
              <div style='display:flex;align-items:center;gap:1rem'>
                <div style='font-size:1.8rem;font-weight:900;color:{rc};min-width:52px;text-align:center'>{rs:.0f}</div>
                <div style='flex:1'>
                  <div style='font-weight:700'>{ri} {r.get("product_name","?")}</div>
                  <div style='color:#8b949e;font-size:.82rem;margin-top:.1rem'>{r.get("recommendation_reason","")}</div>
                </div>
                <div style='background:{rc}22;color:{rc};border-radius:8px;
                  padding:.25rem .7rem;font-size:.78rem;font-weight:700;white-space:nowrap'>{rv}</div>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No alternatives found in catalog for this category.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='scan-title'>🌿 {t('app_title', lang)}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='scan-sub'>{t('app_subtitle', lang)}</div>", unsafe_allow_html=True)

# Show MongoDB status badge
db_inst = get_mongo_db()
if db_inst is not None:
    st.markdown(
        "<div style='text-align:center;color:#3fb950;font-size:.8rem;margin-bottom:.8rem'>"
        "☁️ <b>MongoDB Atlas Connected</b> — Real-time catalog & scan persistence active</div>",
        unsafe_allow_html=True)
else:
    st.markdown(
        "<div style='text-align:center;color:#8b949e;font-size:.8rem;margin-bottom:.8rem'>"
        "💾 Local catalog mode (MongoDB Atlas offline)</div>",
        unsafe_allow_html=True)

if "result" in st.session_state:
    show_result(st.session_state["result"]["product"],
                st.session_state["result"].get("source",""))
    st.markdown("<div class='disclaimer'>⚠️ Aarogya scores are comparative, not clinical. "
                "Academic presentation project.</div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABS (Multilingual)
# ══════════════════════════════════════════════════════════════════════════════
tab_cam, tab_vision, tab_upload, tab_search, tab_demo = st.tabs([
    t("tab_camera", lang),
    t("tab_vision", lang),
    t("tab_upload", lang),
    t("tab_search", lang),
    t("tab_demo",   lang),
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Camera
# ─────────────────────────────────────────────────────────────────────────────
with tab_cam:
    st.markdown(f"**{t('camera_instruction', lang)}**")
    cam_img = st.camera_input("Capture", label_visibility="collapsed")
    if cam_img:
        ib = cam_img.getvalue()

        # Step 1: Try barcode
        with st.spinner("🔢 Scanning barcode..."):
            bc = scan_barcode(ib)

        if bc:
            st.success(f"Barcode detected: `{bc}`")
            if bc in BARCODE_MAP:
                st.session_state["result"] = {
                    "product": DEMO_PRODUCTS[BARCODE_MAP[bc]].copy(),
                    "source": f"Barcode {bc}"}
                st.rerun()
            else:
                st.info(f"Barcode `{bc}` not in presets — running AI Vision...")
                bc = None

        # Step 2: RapidOCR + Groq AI Vision
        if not bc:
            exp = load_explainer()
            if exp.is_ready:
                with st.spinner("🤖 RapidOCR + Groq AI analyzing label..."):
                    vr = groq_vision_scan(ib, "product")
                if vr and not vr.get("error"):
                    conf = vr.get("confidence","low")
                    st.success(f"AI read the label! Confidence: {conf.upper()}")
                    p = blank_product(
                        vr.get("product_name","Camera Scan"),
                        vr.get("category","Biscuits & Cookies"))
                    p["processing_level"] = vr.get("processing_level","processed") or "processed"
                    for k in ["energy_kcal_100g","protein_g_100g","carbs_g_100g","sugar_g_100g",
                               "total_fat_g_100g","saturated_fat_g_100g","fiber_g_100g","sodium_mg_100g"]:
                        if vr.get(k) is not None: p[k] = vr[k]
                    p["added_sugar_g_100g"] = p["sugar_g_100g"]
                    p["ingredient_count"] = len(vr.get("ingredients_list",[]) or []) or 10
                    p["red_flags"] = vr.get("harmful_additives",[])
                    save_product_to_mongo(p)
                    st.session_state["result"] = {"product":p,"source":"Camera + AI Vision"}
                    st.rerun()
                else:
                    st.error("AI could not extract text from this image. Try holding closer or use 🤖 AI Vision tab.")
            else:
                st.warning("Groq AI not connected. Add GROQ_API_KEY to .env and restart.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — AI Vision  (RapidOCR + Groq AI)
# ─────────────────────────────────────────────────────────────────────────────
with tab_vision:
    st.markdown(f"### {t('tab_vision', lang)}")
    st.markdown("Upload **any** product packaging or ingredients list — Local RapidOCR & Groq AI read and score it instantly.")

    exp_v = load_explainer()
    if not exp_v.is_ready:
        st.warning("⚠️ Set GROQ_API_KEY in .env to enable AI Vision.")

    scan_mode = st.radio("What are you scanning?",
                         ["📦 Product / Nutrition Label","📋 Ingredients List"],
                         horizontal=True)

    vis_up = st.file_uploader("Upload image",
                               type=["jpg","jpeg","png","webp"],
                               label_visibility="collapsed", key="vis_up")

    # Demo shortcuts
    st.caption("Or click a demo:")
    dc = st.columns(3)
    _vd = [
        ("ChocoBite Label",   os.path.join(DEMO_DIR,"chocobite.jpg")),
        ("NutriOats Label",   os.path.join(DEMO_DIR,"nutrioats.jpg")),
        ("Ingredients List",  os.path.join(DEMO_DIR,"ingredients_chocobite.jpg")),
    ]
    for col,(dlbl,dpath) in zip(dc,_vd):
        with col:
            if os.path.exists(dpath): st.image(dpath, use_container_width=True)
            if st.button(dlbl, key=f"vd_{dlbl}", use_container_width=True):
                with open(dpath,"rb") as f:
                    st.session_state["_vb"] = f.read()
                    st.session_state["_vl"] = dlbl

    vbytes, vlabel = None, ""
    if vis_up:
        vbytes = vis_up.getvalue(); vlabel = vis_up.name
        st.image(vis_up, caption="Your image", use_container_width=True)
    elif "_vb" in st.session_state:
        vbytes = st.session_state.pop("_vb")
        vlabel = st.session_state.pop("_vl","Demo")
        from PIL import Image as _PI
        st.image(_PI.open(io.BytesIO(vbytes)), caption=vlabel, use_container_width=True)

    if vbytes and exp_v.is_ready:
        if "📦" in scan_mode:
            with st.spinner("🤖 RapidOCR + Groq AI parsing product label..."):
                vr = groq_vision_scan(vbytes, "product")

            if not vr or (vr.get("error") and not vr.get("product_name")):
                st.error(f"Vision failed: {vr.get('error','No text detected') if vr else 'No response'}")
            else:
                conf = vr.get("confidence","low")
                cc = {"high":"#3fb950","medium":"#d29922","low":"#f85149"}.get(conf,"#8b949e")
                st.markdown(
                    f"<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;"
                    f"padding:.8rem 1rem;margin-bottom:.8rem'>"
                    f"<b>AI OCR Confidence:</b> <span style='color:{cc};font-weight:700'>{conf.upper()}</span>"
                    f"</div>", unsafe_allow_html=True)

                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**📦 Product Info**")
                    for lbl,key in [("Name","product_name"),("Brand","brand"),("Category","category")]:
                        st.markdown(f"**{lbl}:** {vr.get(key,'—')}")
                    st.markdown(f"**Processing:** {(vr.get('processing_level') or '').replace('_',' ').title()}")
                with cb:
                    st.markdown("**🏷 Signals**")
                    for items,ico,lbl in [
                        (vr.get("health_claims",[])    or [],"✅","Claims"),
                        (vr.get("allergens_found",[])  or [],"⚠️","Allergens"),
                        (vr.get("harmful_additives",[]) or [],"🔴","Additives"),
                    ]:
                        if items: st.markdown(f"{ico} **{lbl}:** {', '.join(str(x) for x in items[:3])}")
                    if vr.get("contains_whole_grain"): st.markdown("🌾 Whole Grain")
                    if vr.get("contains_added_sugar"): st.markdown("🍬 Added Sugar")

                # Show extracted OCR text expander
                if vr.get("_ocr_text"):
                    with st.expander("📄 View Extracted Label Text (RapidOCR)", expanded=False):
                        st.code(vr["_ocr_text"])

                nut_keys = ["energy_kcal_100g","protein_g_100g","sugar_g_100g","fiber_g_100g"]
                if any(vr.get(k) is not None for k in nut_keys):
                    st.divider()
                    ext = blank_product(vr.get("product_name","AI Scanned"), vr.get("category","Biscuits & Cookies"))
                    ext["processing_level"] = vr.get("processing_level","processed") or "processed"
                    for k in ["energy_kcal_100g","protein_g_100g","carbs_g_100g","sugar_g_100g",
                               "total_fat_g_100g","saturated_fat_g_100g","fiber_g_100g","sodium_mg_100g"]:
                        if vr.get(k) is not None: ext[k] = vr[k]
                    ext["added_sugar_g_100g"] = ext["sugar_g_100g"]
                    ext["ingredient_count"] = len(vr.get("ingredients_list",[]) or []) or 10
                    ext["red_flags"] = vr.get("harmful_additives",[])

                    if st.button("🔬 " + t("scan_now", lang) + " (Full Score + Verdict)",
                                 type="primary", use_container_width=True, key="vscore"):
                        save_product_to_mongo(ext)
                        st.session_state["result"] = {"product":ext,"source":f"AI Vision: {vlabel}"}
                        st.rerun()
        else:
            with st.spinner("🤖 RapidOCR + Groq AI analyzing ingredients list..."):
                ir = groq_vision_scan(vbytes, "ingredients")

            if not ir or (ir.get("error") and not ir.get("ingredients_raw")):
                st.error(f"Could not read: {ir.get('error','No text found') if ir else 'No response'}")
            else:
                iq     = int(ir.get("ingredient_quality_score") or 5)
                iq_col = "#3fb950" if iq>=7 else ("#d29922" if iq>=4 else "#f85149")
                st.markdown(
                    f"<div style='background:#161b22;border:1px solid #30363d;"
                    f"border-radius:14px;padding:1.2rem;margin-bottom:1rem'>"
                    f"<div style='font-size:.82rem;color:#8b949e'>Ingredient Quality Score</div>"
                    f"<div style='font-size:3rem;font-weight:900;color:{iq_col}'>{iq}"
                    f"<span style='font-size:1rem;color:#8b949e'>/10</span></div>"
                    f"<div style='color:#c9d1d9;font-size:.93rem;margin-top:.4rem'>"
                    f"{ir.get('summary','')}</div></div>", unsafe_allow_html=True)

                ci,cr = st.columns(2)
                with ci:
                    st.markdown("**📋 Ingredients List**")
                    ings = ir.get("ingredients_list",[]) or []
                    st.caption(f"Total: {ir.get('total_ingredients_count',len(ings))}")
                    for ing in ings[:12]: st.markdown(f"• {ing}")
                    if len(ings)>12: st.caption(f"...+{len(ings)-12} more")
                with cr:
                    st.markdown(f"**{t('red_flags', lang)}**")
                    flags = ir.get("red_flag_ingredients",[]) or []
                    if flags:
                        for fl in flags:
                            ico = {"high":"🔴","medium":"🟡","low":"🟠"}.get(fl.get("severity","medium"),"⚠️")
                            st.markdown(
                                f"{ico} **{fl.get('name','')}**  \n"
                                f"<span style='color:#8b949e;font-size:.82rem'>{fl.get('reason','')}</span>",
                                unsafe_allow_html=True)
                    else:
                        st.success("No major red flags detected!")
                    sigs=[]
                    if ir.get("contains_palm_oil"):           sigs.append("🌴 Palm Oil")
                    if ir.get("contains_artificial_colours"): sigs.append("🎨 Colours")
                    if ir.get("contains_artificial_flavours"):sigs.append("🧪 Flavours")
                    if ir.get("contains_preservatives"):      sigs.append("🧊 Preservatives")
                    if ir.get("contains_added_sugar"):        sigs.append("🍬 Added Sugar")
                    if sigs: st.markdown("  ".join(sigs))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Upload
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown(f"**{t('tab_upload', lang)}: Barcode or Nutrition Label Photo**")
    uploaded = st.file_uploader("Upload image",
                                type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed")
    if uploaded:
        ib = uploaded.getvalue()
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
        c1,c2 = st.columns(2)
        if c1.button("🔢 Scan Barcode", use_container_width=True, type="primary"):
            with st.spinner("Scanning barcode..."):
                bc = scan_barcode(ib)
            if bc:
                st.success(f"Barcode: `{bc}`")
                if bc in BARCODE_MAP:
                    st.session_state["result"] = {
                        "product":DEMO_PRODUCTS[BARCODE_MAP[bc]].copy(),
                        "source":f"Barcode {bc}"}
                    st.rerun()
                else:
                    st.info("Barcode not in demo list. Try 'AI Read Label' button.")
            else:
                st.warning("No barcode detected in this image.")
        if c2.button("🤖 AI Read Label", use_container_width=True):
            exp = load_explainer()
            if exp.is_ready:
                with st.spinner("RapidOCR + Groq AI reading label..."):
                    vr = groq_vision_scan(ib, "product")
                if vr and not vr.get("error"):
                    p = blank_product(vr.get("product_name","Uploaded Product"),
                                      vr.get("category","Biscuits & Cookies"))
                    p["processing_level"] = vr.get("processing_level","processed") or "processed"
                    for k in ["energy_kcal_100g","protein_g_100g","carbs_g_100g","sugar_g_100g",
                               "total_fat_g_100g","saturated_fat_g_100g","fiber_g_100g","sodium_mg_100g"]:
                        if vr.get(k) is not None: p[k] = vr[k]
                    p["added_sugar_g_100g"] = p["sugar_g_100g"]
                    p["red_flags"] = vr.get("harmful_additives",[])
                    save_product_to_mongo(p)
                    st.session_state["result"] = {"product":p,"source":f"AI Vision: {uploaded.name}"}
                    st.rerun()
                else:
                    st.error(f"AI read failed: {vr.get('error','No text found') if vr else 'No response'}")
            else:
                st.warning("Groq AI not connected.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Search with Fuzzy Matching & Instant AI Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.markdown(f"**{t('search_prompt', lang)}**")
    catalog = load_catalog()
    q = st.text_input("Search product name",
                      placeholder=t("search_placeholder", lang),
                      label_visibility="collapsed")
    if q and not catalog.empty:
        q_clean = q.strip().lower()

        # 1. Exact or Substring match
        matches = catalog[catalog["product_name"].str.contains(q_clean, case=False, na=False)]

        # 2. Fuzzy match if no direct match (e.g. 'maggie' -> 'Maggi 2-Minute Masala Instant Noodles')
        if matches.empty:
            import difflib
            all_names = catalog["product_name"].dropna().tolist()
            close_matches = difflib.get_close_matches(q_clean, [n.lower() for n in all_names], n=5, cutoff=0.28)
            if close_matches:
                matches = catalog[catalog["product_name"].str.lower().isin(close_matches)]
            else:
                # Token-based partial search
                tokens = q_clean.replace("-"," ").split()
                matched_rows = []
                for _, row in catalog.iterrows():
                    p_name_lower = str(row["product_name"]).lower()
                    if any(tok in p_name_lower or tok[:4] in p_name_lower for tok in tokens if len(tok) >= 3):
                        matched_rows.append(row)
                if matched_rows:
                    matches = pd.DataFrame(matched_rows).drop_duplicates(subset=["product_name"])

        # Display matches if found
        if not matches.empty:
            st.success(f"Found {len(matches)} matching product(s):")
            sel = st.selectbox("Select product:", matches["product_name"].tolist(), label_visibility="collapsed")
            row = matches[matches["product_name"]==sel].iloc[0]
            cl,cr = st.columns([3,1])
            cl.markdown(f"**{row['product_name']}**")
            cl.caption(f"{row.get('brand','')} · {row.get('category','')} · Score: {row.get('food_score','?')}/100")
            if cr.button(t("scan_now", lang), use_container_width=True, type="primary"):
                st.session_state["result"] = {"product":row.to_dict(),"source":f"Catalog: {sel}"}
                st.rerun()
        else:
            st.warning(t("no_products_found", lang, query=q))

        # 3. Always offer Instant AI Analysis for ANY custom/unknown food query!
        st.markdown("---")
        st.markdown(f"**⚡ Want live AI Food Intelligence for '{q}'?**")
        st.caption("Aarogya will estimate nutrients, compute food safety score, detect red flags, and save to MongoDB Atlas.")
        if st.button(t("ai_analyze_btn", lang, query=q), type="primary", use_container_width=True, key=f"ai_btn_{q}"):
            with st.spinner(t("analyzing_ai", lang)):
                exp = load_explainer()
                ai_prod = exp.analyze_product_by_name(q, language=lang)
                score, breakdown = get_score(ai_prod)
                ai_prod["food_score"] = score
                save_product_to_mongo(ai_prod)
                st.session_state["result"] = {"product": ai_prod, "source": f"Groq AI Intelligence: {q.title()}"}
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Demo Products
# ─────────────────────────────────────────────────────────────────────────────
with tab_demo:
    st.markdown(f"### {t('tab_demo', lang)}")
    st.caption("Tap Scan on any preset product to see verdict + recommendations instantly")

    items = [
        ("chocobite.jpg",   "ChocoBite Cream Biscuits","🍪 Ultra-Processed · High Sugar"),
        ("nutrioats.jpg",   "NutriOats Porridge",      "🌾 Minimally Processed · High Fiber"),
        ("masalachips.jpg", "MasalaKing Spicy Chips",  "🌶️ Ultra-Processed · High Sodium"),
    ]
    cols = st.columns(3)
    for col,(fn,dn,dt) in zip(cols,items):
        ip = os.path.join(DEMO_DIR,fn)
        with col:
            if os.path.exists(ip): st.image(ip, use_container_width=True)
            st.markdown(f"**{dn}**")
            st.caption(dt)
            if st.button(t("scan_now", lang),key=f"d_{fn}",use_container_width=True,type="primary"):
                st.session_state["result"] = {
                    "product":DEMO_PRODUCTS[fn].copy(),"source":f"Demo: {dn}"}
                st.rerun()

    st.divider()
    st.markdown("**🤖 Try AI Vision on this ingredients label:**")
    ing = os.path.join(DEMO_DIR,"ingredients_chocobite.jpg")
    if os.path.exists(ing):
        st.image(ing,caption="ChocoBite Ingredients — try in AI Vision tab",use_container_width=True)
    st.info("Go to **🤖 AI Vision** tab → select 'Ingredients List' → click **Ingredients List** demo")

# ── Scan History from MongoDB Atlas ──────────────────────────────────────────
with st.expander(t("scan_history_title", lang), expanded=False):
    try:
        mc = get_mongo_collection("scan_history")
        if mc is not None:
            history = list(mc.find({}, {"_id":0}).sort("scanned_at",-1).limit(10))
            if history:
                for h in history:
                    sc = float(h.get("score",0))
                    vc = score_color(sc)
                    ts = h.get("scanned_at","")
                    if hasattr(ts,"strftime"): ts = ts.strftime("%d %b %H:%M")
                    st.markdown(
                        f"<div class='history-card'>"
                        f"<div style='font-size:1.4rem;font-weight:800;color:{vc};min-width:40px'>{sc:.0f}</div>"
                        f"<div style='flex:1'><b>{h.get('product_name','?')}</b>"
                        f"<span style='color:#8b949e;font-size:.82rem;margin-left:.5rem'>{h.get('category','')}</span></div>"
                        f"<div style='color:#8b949e;font-size:.78rem'>{ts}</div>"
                        f"</div>", unsafe_allow_html=True)
            else:
                st.caption("No scans yet. Scan a product to see cloud history here.")
        else:
            st.caption("MongoDB not connected. Check MONGO_URI in .env.")
    except Exception as e:
        st.caption(f"History unavailable: {e}")

st.markdown("<div class='disclaimer'>⚠️ Aarogya scores are comparative, not clinical. "
            "Not medical advice. Academic project.</div>",
            unsafe_allow_html=True)
