"""
app_simple.py  —  AAROGYA Smart Product Scanner  (Presentation Build)
======================================================================
Features:
  • Animated splash screen on first load
  • 5 scan modes: Camera | AI Vision | Upload | Search | Demo
  • Camera → barcode → Groq Vision fallback (no OCR dependency)
  • AI Vision tab: product image + ingredients list via Groq llama-4-scout
  • MongoDB Atlas: saves every scan to cloud history
  • SAFE / MODERATE / HARMFUL verdict + score + AI explanation + alternatives

Run:  streamlit run app_simple.py
"""
import os, sys, re, io, base64, datetime
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st
from streamlit.components.v1 import html as st_html

st.set_page_config(page_title="Aarogya Scanner", page_icon="🌿",
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
  animation:fadeOut 0.6s ease-in-out 3.2s forwards;
}
.splash-logo{
  font-size:5rem;font-weight:900;font-family:'Inter',sans-serif;
  background:linear-gradient(135deg,#3fb950 0%,#58a6ff 50%,#a371f7 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:popIn 0.7s cubic-bezier(.175,.885,.32,1.275) both;
}
.splash-sub{
  color:#8b949e;font-family:'Inter',sans-serif;font-size:1.1rem;
  margin-top:.5rem;letter-spacing:2px;text-transform:uppercase;
  animation:popIn 0.7s 0.3s cubic-bezier(.175,.885,.32,1.275) both;
}
.splash-bar-wrap{
  margin-top:2.5rem;width:260px;height:4px;background:#21262d;
  border-radius:99px;overflow:hidden;
  animation:popIn 0.5s 0.6s ease both;
}
.splash-bar{
  height:100%;width:0;background:linear-gradient(90deg,#3fb950,#58a6ff);
  border-radius:99px;animation:grow 2.4s 0.7s ease-in-out forwards;
}
.splash-tagline{
  color:#484f58;font-family:'Inter',sans-serif;font-size:.85rem;
  margin-top:1rem;letter-spacing:1px;
  animation:popIn 0.5s 1s ease both;
}
.splash-icons{
  display:flex;gap:2rem;margin-top:2rem;font-size:2rem;
  animation:popIn 0.5s 1.2s ease both;
}
.splash-icon{animation:pulse 1.5s infinite alternate;}
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
  <div class="splash-tagline">Scan · Score · Recommend</div>
  <div class="splash-icons">
    <span class="splash-icon">📷</span>
    <span class="splash-icon">🤖</span>
    <span class="splash-icon">🌾</span>
  </div>
</div>
""", unsafe_allow_html=True)
    import time; time.sleep(3.5)
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
.scan-sub{text-align:center;color:#8b949e;font-size:1rem;margin-bottom:1.5rem}

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
# RESOURCES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_catalog():
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
    from llm.groq_llm import AarogyaGroqExplainer
    return AarogyaGroqExplainer()

@st.cache_resource
def get_mongo():
    """Return MongoDB collection or None."""
    try:
        from dotenv import load_dotenv; load_dotenv()
        import pymongo, os
        uri  = os.getenv("MONGO_URI","")
        name = os.getenv("MONGO_DB_NAME","aarogya_db")
        if not uri: return None
        cli  = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
        cli.admin.command("ping")
        return cli[name]["scan_history"]
    except Exception:
        return None

def save_scan(product, score, verdict):
    try:
        col = get_mongo()
        if col:
            col.insert_one({
                "product_name": product.get("product_name","Unknown"),
                "category":     product.get("category",""),
                "score":        round(score, 2),
                "verdict":      verdict,
                "scanned_at":   datetime.datetime.utcnow(),
            })
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_score(p):
    from ml.scoring import compute_aarogya_score
    r = compute_aarogya_score(p)
    return r["final_score"], r.get("breakdown", {})

def get_verdict(s):
    if s >= 65: return "SAFE",     "#3fb950", "✅", "verdict-safe"
    if s >= 40: return "MODERATE", "#d29922", "⚠️", "verdict-moderate"
    return           "HARMFUL",    "#f85149", "❌", "verdict-harmful"

def score_color(s):
    return "#3fb950" if s>=65 else ("#d29922" if s>=40 else "#f85149")

def nut_row(label, val, unit, bad_thr, hi_good=False):
    is_bad = (val < bad_thr) if hi_good else (val > bad_thr)
    tag = (f"<span class='tag-bad'>Low</span>"  if is_bad and hi_good
      else f"<span class='tag-bad'>High</span>" if is_bad
      else f"<span class='tag-good'>OK</span>")
    return (f"<div class='nutrient-row'><span class='nutrient-label'>{label}</span>"
            f"<span class='nutrient-value'>{val:.1f} {unit}&nbsp;{tag}</span></div>")

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

def scan_barcode(img_bytes):
    try:
        from pyzbar.pyzbar import decode
        from PIL import Image
        decoded = decode(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        return decoded[0].data.decode("utf-8").strip() if decoded else None
    except Exception:
        return None

def groq_vision_scan(img_bytes, mode="product"):
    """Use Groq llama-4-scout to read a product image or ingredients list."""
    exp = load_explainer()
    if not exp.is_ready:
        return None
    try:
        if mode == "ingredients":
            return exp.analyze_ingredients(img_bytes)
        return exp.analyze_product_image(img_bytes)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# RESULT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def show_result(product, source=""):
    score, breakdown = get_score(product)
    verdict, color, icon, css = get_verdict(score)
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
      <div style='font-size:4rem;'>{icon}</div>
      <div style='font-size:2.2rem;font-weight:900;color:{color};margin:.3rem 0'>{verdict}</div>
      <div style='font-size:3.5rem;font-weight:900;color:{color};line-height:1'>{score:.0f}
        <span style='font-size:1.2rem;color:#8b949e;font-weight:400'>/100</span></div>
      <div style='font-size:1.1rem;font-weight:700;margin-top:.6rem'>{name}</div>
      <div style='color:#8b949e;font-size:.85rem;margin-top:.2rem'>
        {brand}{" · " if brand else ""}{cat}{" · " if cat else ""}{proc}</div>
    </div>""", unsafe_allow_html=True)

    # AI explanation
    with st.spinner("💬 Getting AI explanation..."):
        try:
            ex  = load_explainer()
            ai  = ex.explain_score({
                "product_name":name,"category":cat,
                "processing_level":product.get("processing_level",""),
                "predicted_score":score,"base_value":55.0,
                "strengths":breakdown.get("strengths",[]),
                "concerns": breakdown.get("concerns",[]),
                "shap_top_features":{},
            })
        except Exception: ai = None
    if ai:
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:14px;"
            f"padding:1.1rem 1.3rem;margin-top:.8rem;color:#c9d1d9;"
            f"font-size:.93rem;line-height:1.8'>💬 {ai}</div>",
            unsafe_allow_html=True)

    # Nutrition facts
    st.markdown("#### 📊 Nutrition Facts (per 100g)")
    rows = [
        ("⚡ Energy",       "energy_kcal_100g",    "kcal",400,False),
        ("🍬 Sugar",        "sugar_g_100g",         "g",  15, False),
        ("💪 Protein",      "protein_g_100g",       "g",  5,  True),
        ("🌾 Fiber",        "fiber_g_100g",         "g",  3,  True),
        ("🧂 Sodium",       "sodium_mg_100g",       "mg", 400,False),
        ("🧈 Saturated Fat","saturated_fat_g_100g", "g",  5,  False),
    ]
    html = "".join(nut_row(l,float(product.get(c) or 0),u,t,h) for l,c,u,t,h in rows)
    st.markdown(
        f"<div style='background:#161b22;border:1px solid #30363d;"
        f"border-radius:12px;padding:.8rem 1.2rem'>{html}</div>",
        unsafe_allow_html=True)

    # Recommendations
    if verdict in ("HARMFUL","MODERATE"):
        st.markdown("---")
        st.markdown("### 🔄 Better Alternatives")
        st.caption("Same category — higher Aarogya score")
        _show_recs(product, score)
    else:
        st.markdown("")
        st.success("✅ This product scores well — keep it up!")

    st.markdown("---")
    if st.button("🔄 Scan Another Product", use_container_width=True):
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

    if recs is not None and not recs.empty:
        for _, r in recs.iterrows():
            rs  = float(r.get("food_score", r.get("ranking_score",0)))
            rc  = score_color(rs)
            rv,_,ri,_ = get_verdict(rs)
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
st.markdown("<div class='scan-title'>🌿 Aarogya</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='scan-sub'>AI Food Intelligence · Scan · Score · Recommend</div>",
    unsafe_allow_html=True)

# Show MongoDB status quietly
col_db = get_mongo()
if col_db is not None:
    st.markdown(
        "<div style='text-align:center;color:#3fb950;font-size:.75rem;margin-bottom:.5rem'>"
        "☁️ MongoDB Atlas connected — scan history saved</div>",
        unsafe_allow_html=True)

if "result" in st.session_state:
    show_result(st.session_state["result"]["product"],
                st.session_state["result"].get("source",""))
    st.markdown("<div class='disclaimer'>⚠️ Scores are comparative, not clinical. "
                "Academic project.</div>", unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_cam, tab_vision, tab_upload, tab_search, tab_demo = st.tabs([
    "📷 Camera", "🤖 AI Vision", "🖼 Upload", "🔍 Search", "🎯 Demo"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Camera  (barcode → Groq vision fallback)
# ─────────────────────────────────────────────────────────────────────────────
with tab_cam:
    st.markdown("**Point camera at barcode or nutrition label**")
    st.caption("Tip: hold steady, make sure label text is sharp")
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
                st.info(f"Barcode `{bc}` not in catalog — trying AI Vision...")
                bc = None

        # Step 2: Groq Vision (reads nutrition label automatically)
        if not bc:
            exp = load_explainer()
            if exp.is_ready:
                with st.spinner("🤖 AI reading nutrition label..."):
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
                    st.session_state["result"] = {"product":p,"source":"Camera + AI Vision"}
                    st.rerun()
                else:
                    st.error("AI could not read this image. Try better lighting or use 🤖 AI Vision tab.")
            else:
                st.warning("Groq AI not connected. Add GROQ_API_KEY to .env and restart.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — AI Vision  (upload → Groq llama-4-scout)
# ─────────────────────────────────────────────────────────────────────────────
with tab_vision:
    st.markdown("### 🤖 AI Vision Analysis")
    st.markdown("Upload **any** product photo or ingredients list — Groq AI reads and scores it.")

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
            with st.spinner("🤖 AI reading product image..."):
                vr = groq_vision_scan(vbytes, "product")

            if not vr or (vr.get("error") and not vr.get("product_name")):
                st.error(f"Vision failed: {vr.get('error','Check your internet connection') if vr else 'No response'}")
            else:
                conf = vr.get("confidence","low")
                cc = {"high":"#3fb950","medium":"#d29922","low":"#f85149"}.get(conf,"#8b949e")
                st.markdown(
                    f"<div style='background:#161b22;border:1px solid #30363d;border-radius:10px;"
                    f"padding:.8rem 1rem;margin-bottom:.8rem'>"
                    f"<b>AI Confidence:</b> <span style='color:{cc};font-weight:700'>{conf.upper()}</span>"
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

                nut_keys = ["energy_kcal_100g","protein_g_100g","sugar_g_100g","fiber_g_100g"]
                if any(vr.get(k) is not None for k in nut_keys):
                    st.divider()
                    st.markdown("**📊 Nutrition extracted — get your full Aarogya score:**")
                    ext = blank_product(vr.get("product_name","AI Scanned"), vr.get("category","Biscuits & Cookies"))
                    ext["processing_level"] = vr.get("processing_level","processed") or "processed"
                    for k in ["energy_kcal_100g","protein_g_100g","carbs_g_100g","sugar_g_100g",
                               "total_fat_g_100g","saturated_fat_g_100g","fiber_g_100g","sodium_mg_100g"]:
                        if vr.get(k) is not None: ext[k] = vr[k]
                    ext["added_sugar_g_100g"] = ext["sugar_g_100g"]
                    ext["ingredient_count"] = len(vr.get("ingredients_list",[]) or []) or 10
                    if st.button("🔬 Get Full Score + Recommendations",
                                 type="primary", use_container_width=True, key="vscore"):
                        st.session_state["result"] = {"product":ext,"source":f"AI Vision: {vlabel}"}
                        st.rerun()
        else:
            with st.spinner("🤖 AI reading ingredients list..."):
                ir = groq_vision_scan(vbytes, "ingredients")

            if not ir or (ir.get("error") and not ir.get("ingredients_raw")):
                st.error(f"Could not read: {ir.get('error','No response') if ir else 'No response'}")
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
                    st.markdown("**📋 Ingredients**")
                    ings = ir.get("ingredients_list",[]) or []
                    st.caption(f"Total: {ir.get('total_ingredients_count',len(ings))}")
                    for ing in ings[:12]: st.markdown(f"• {ing}")
                    if len(ings)>12: st.caption(f"...+{len(ings)-12} more")
                with cr:
                    st.markdown("**🚩 Red Flags**")
                    flags = ir.get("red_flag_ingredients",[]) or []
                    if flags:
                        for fl in flags:
                            ico = {"high":"🔴","medium":"🟡","low":"🟠"}.get(fl.get("severity","medium"),"⚠️")
                            st.markdown(
                                f"{ico} **{fl.get('name','')}**  \n"
                                f"<span style='color:#8b949e;font-size:.82rem'>{fl.get('reason','')}</span>",
                                unsafe_allow_html=True)
                    else:
                        st.success("No major red flags!")
                    sigs=[]
                    if ir.get("contains_palm_oil"):           sigs.append("🌴 Palm Oil")
                    if ir.get("contains_artificial_colours"): sigs.append("🎨 Colours")
                    if ir.get("contains_artificial_flavours"):sigs.append("🧪 Flavours")
                    if ir.get("contains_preservatives"):      sigs.append("🧊 Preservatives")
                    if ir.get("contains_added_sugar"):        sigs.append("🍬 Sugar")
                    if sigs: st.markdown("  ".join(sigs))
                    adds = ir.get("additives_found",[]) or []
                    algs = ir.get("allergens",[]) or []
                    if adds: st.caption(f"Additives: {', '.join(adds[:5])}")
                    if algs: st.caption(f"Allergens: {', '.join(algs)}")
    elif vbytes and not exp_v.is_ready:
        st.warning("Groq AI not connected. Add GROQ_API_KEY to .env.")
    else:
        st.info("📤 Upload an image or click a demo above to start AI analysis.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Upload (barcode or Groq Vision)
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("**Upload barcode or nutrition label photo**")
    uploaded = st.file_uploader("Upload image",
                                type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed")
    if uploaded:
        ib = uploaded.getvalue()
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
        c1,c2 = st.columns(2)
        if c1.button("🔢 Scan Barcode", use_container_width=True, type="primary"):
            with st.spinner("Scanning..."):
                bc = scan_barcode(ib)
            if bc:
                st.success(f"Barcode: `{bc}`")
                if bc in BARCODE_MAP:
                    st.session_state["result"] = {
                        "product":DEMO_PRODUCTS[BARCODE_MAP[bc]].copy(),
                        "source":f"Barcode {bc}"}
                    st.rerun()
                else:
                    st.info("Barcode not in demo catalog. Try AI Vision button.")
            else:
                st.warning("No barcode found.")
        if c2.button("🤖 AI Read Label", use_container_width=True):
            exp = load_explainer()
            if exp.is_ready:
                with st.spinner("AI reading label..."):
                    vr = groq_vision_scan(ib, "product")
                if vr and not vr.get("error"):
                    p = blank_product(vr.get("product_name","Uploaded Product"),
                                      vr.get("category","Biscuits & Cookies"))
                    p["processing_level"] = vr.get("processing_level","processed") or "processed"
                    for k in ["energy_kcal_100g","protein_g_100g","carbs_g_100g","sugar_g_100g",
                               "total_fat_g_100g","saturated_fat_g_100g","fiber_g_100g","sodium_mg_100g"]:
                        if vr.get(k) is not None: p[k] = vr[k]
                    p["added_sugar_g_100g"] = p["sugar_g_100g"]
                    st.session_state["result"] = {"product":p,"source":f"AI Vision: {uploaded.name}"}
                    st.rerun()
                else:
                    st.error(f"AI read failed: {vr.get('error','') if vr else 'No response'}")
            else:
                st.warning("Groq AI not connected.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Search by Name
# ─────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("**Search product database by name**")
    catalog = load_catalog()
    q = st.text_input("Search product name",
                      placeholder="e.g. Maggi, Oreo, Muesli, Chips...",
                      label_visibility="collapsed")
    if q and not catalog.empty:
        matches = catalog[catalog["product_name"].str.contains(q, case=False, na=False)]
        if matches.empty:
            st.warning(f"No products matching '{q}'")
        else:
            sel = st.selectbox(f"Found {len(matches)} result(s):",
                               matches["product_name"].tolist())
            row = matches[matches["product_name"]==sel].iloc[0]
            cl,cr = st.columns([3,1])
            cl.markdown(f"**{row['product_name']}**")
            cl.caption(f"{row.get('brand','')} · {row.get('category','')} · Score: {row.get('food_score','?')}/100")
            if cr.button("🔬 Scan", use_container_width=True, type="primary"):
                st.session_state["result"] = {"product":row.to_dict(),"source":f"Catalog: {sel}"}
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Demo Products
# ─────────────────────────────────────────────────────────────────────────────
with tab_demo:
    st.markdown("### 🎯 Demo Products")
    st.caption("Tap Scan on any product to see verdict + recommendations instantly")

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
            if st.button("🔬 Scan",key=f"d_{fn}",use_container_width=True,type="primary"):
                st.session_state["result"] = {
                    "product":DEMO_PRODUCTS[fn].copy(),"source":f"Demo: {dn}"}
                st.rerun()

    st.divider()
    st.markdown("**🤖 Try AI Vision on this ingredients label:**")
    ing = os.path.join(DEMO_DIR,"ingredients_chocobite.jpg")
    if os.path.exists(ing):
        st.image(ing,caption="ChocoBite Ingredients — try in AI Vision tab",use_container_width=True)
    st.info("Go to **🤖 AI Vision** tab → select 'Ingredients List' → click **Ingredients List** demo")

# ── Scan History from MongoDB ─────────────────────────────────────────────────
with st.expander("📊 Recent Scan History (MongoDB Atlas)", expanded=False):
    try:
        mc = get_mongo()
        if mc:
            history = list(mc.find({}, {"_id":0}).sort("scanned_at",-1).limit(10))
            if history:
                for h in history:
                    sc = h.get("score",0)
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
                st.caption("No scans yet. Scan a product to see history here.")
        else:
            st.caption("MongoDB not connected.")
    except Exception as e:
        st.caption(f"History unavailable: {e}")

st.markdown("<div class='disclaimer'>⚠️ Aarogya scores are comparative, not clinical. "
            "Not medical advice. Academic project using synthetic data.</div>",
            unsafe_allow_html=True)
