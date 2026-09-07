"""
app_simple.py  -  AAROGYA Smart Product Scanner
=================================================
5 ways to analyse a packaged food product:
  1. Camera        - point at barcode or nutrition label
  2. Upload Image  - barcode scan or OCR
  3. AI Vision     - product photo / ingredients list -> Groq reads it
  4. Search Name   - catalog lookup
  5. Demo Products - one-click demo

Run:  streamlit run app_simple.py
"""
import os, sys, re, io, base64
import pandas as pd
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st

st.set_page_config(page_title="Aarogya Scanner", page_icon="🌿", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#0d1117;color:#e6edf3}
.stApp{background:#0d1117}
.scan-title{font-size:2.6rem;font-weight:800;text-align:center;
  background:linear-gradient(135deg,#3fb950,#58a6ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.scan-sub{text-align:center;color:#8b949e;font-size:1rem;margin-bottom:1.5rem}
.verdict-safe    {background:linear-gradient(135deg,#0d2b1a,#0f2e1f);border:2px solid #3fb950;border-radius:20px;padding:1.8rem;text-align:center}
.verdict-moderate{background:linear-gradient(135deg,#1f1a0a,#221d0c);border:2px solid #d29922;border-radius:20px;padding:1.8rem;text-align:center}
.verdict-harmful {background:linear-gradient(135deg,#2d0f0f,#2b1010);border:2px solid #f85149;border-radius:20px;padding:1.8rem;text-align:center}
.rec-card{background:#161b22;border:1px solid #30363d;border-radius:14px;padding:.9rem 1.2rem;margin-bottom:.65rem}
.nutrient-row{display:flex;justify-content:space-between;padding:.35rem 0;border-bottom:1px solid #21262d;font-size:.88rem}
.nutrient-label{color:#8b949e}.nutrient-value{font-weight:600}
.tag-good{background:#0d2b1a;color:#3fb950;border-radius:6px;padding:2px 8px;font-size:.78rem}
.tag-bad {background:#2d0f0f;color:#f85149;border-radius:6px;padding:2px 8px;font-size:.78rem}
.disclaimer{text-align:center;color:#484f58;font-size:.75rem;margin-top:2rem;
  padding-top:1rem;border-top:1px solid #21262d}
</style>
""", unsafe_allow_html=True)

# ── Demo data ─────────────────────────────────────────────────────────────────
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

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_data
def load_catalog():
    for p in [
        os.path.join(ROOT,"data","raw","aarogya_products_500.csv"),
        os.path.join(ROOT,"data","raw","aarogya_food_products_demo.csv"),
    ]:
        if os.path.exists(p):
            return pd.read_csv(p)
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

# ── Pure helpers ──────────────────────────────────────────────────────────────
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

def nutrient_row_html(label, val, unit, bad_thr, higher_good=False):
    is_bad = (val < bad_thr) if higher_good else (val > bad_thr)
    tag = ("<span class='tag-bad'>Low</span>" if is_bad and higher_good
           else "<span class='tag-bad'>High</span>" if is_bad
           else "<span class='tag-good'>OK</span>")
    return (f"<div class='nutrient-row'><span class='nutrient-label'>{label}</span>"
            f"<span class='nutrient-value'>{val:.1f} {unit}&nbsp;{tag}</span></div>")

def scan_barcode(img_bytes):
    try:
        from pyzbar.pyzbar import decode
        from PIL import Image
        decoded = decode(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        return decoded[0].data.decode("utf-8").strip() if decoded else None
    except Exception:
        return None

def ocr_nutrition(img_bytes):
    try:
        import easyocr, numpy as np
        from PIL import Image
        arr = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        text = " ".join(reader.readtext(arr, detail=0)).lower()
        def fv(pats):
            for pat in pats:
                m = re.search(pat, text)
                if m:
                    try: return float(m.group(1))
                    except: pass
            return None
        found = {k:v for k,v in {
            "energy_kcal_100g": fv([r"energy[:\s]+(\d+\.?\d*)\s*kcal",r"(\d+\.?\d*)\s*kcal"]),
            "protein_g_100g":   fv([r"protein[:\s]+(\d+\.?\d*)\s*g"]),
            "carbs_g_100g":     fv([r"carbohydrate[s]?[:\s]+(\d+\.?\d*)\s*g"]),
            "sugar_g_100g":     fv([r"sugar[s]?[:\s]+(\d+\.?\d*)\s*g"]),
            "total_fat_g_100g": fv([r"total fat[:\s]+(\d+\.?\d*)\s*g",r"fat[:\s]+(\d+\.?\d*)\s*g"]),
            "saturated_fat_g_100g": fv([r"saturated fat[:\s]+(\d+\.?\d*)\s*g"]),
            "fiber_g_100g":     fv([r"fi(?:b|br)er[:\s]+(\d+\.?\d*)\s*g"]),
            "sodium_mg_100g":   fv([r"sodium[:\s]+(\d+\.?\d*)\s*mg"]),
        }.items() if v is not None}
        return found if len(found) >= 3 else None
    except Exception:
        return None

def blank_product(name="Scanned Product", category="Biscuits & Cookies"):
    return {
        "product_name":name,"category":category,
        "recommendation_group":"biscuits_cookies","processing_level":"processed",
        "energy_kcal_100g":0,"protein_g_100g":0,"carbs_g_100g":0,
        "sugar_g_100g":0,"added_sugar_g_100g":0,"total_fat_g_100g":0,
        "saturated_fat_g_100g":0,"trans_fat_g_100g":0,"fiber_g_100g":0,
        "sodium_mg_100g":0,"ingredient_count":10,
        "contains_whole_grain":False,"contains_added_sugar":False,
        "contains_artificial_sweetener":False,"contains_allergen":False,
    }

# ── Result renderer ───────────────────────────────────────────────────────────
def show_result(product, source=""):
    score, breakdown = get_score(product)
    verdict, color, icon, css = get_verdict(score)
    name  = product.get("product_name","Unknown")
    brand = product.get("brand","")
    cat   = product.get("category","")
    proc  = product.get("processing_level","").replace("_"," ").title()

    if source:
        st.caption(f"Source: {source}")

    st.markdown(f"""
    <div class='{css}'>
      <div style='font-size:3.5rem'>{icon}</div>
      <div style='font-size:2rem;font-weight:800;color:{color};margin:.3rem 0'>{verdict}</div>
      <div style='font-size:3rem;font-weight:800;color:{color}'>{score:.0f}
        <span style='font-size:1rem;color:#8b949e'>/100</span></div>
      <div style='font-size:1.1rem;font-weight:700;margin-top:.5rem'>{name}</div>
      <div style='color:#8b949e;font-size:.85rem'>{brand}{" | " if brand else ""}{cat}{" | " if cat else ""}{proc}</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Getting AI explanation..."):
        try:
            ex = load_explainer()
            ai = ex.explain_score({
                "product_name":name,"category":cat,
                "processing_level":product.get("processing_level",""),
                "predicted_score":score,"base_value":55.0,
                "strengths":breakdown.get("strengths",[]),
                "concerns": breakdown.get("concerns",[]),
                "shap_top_features":{},
            })
        except Exception:
            ai = None
    if ai:
        st.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
            f"padding:1rem;margin-top:.75rem;color:#c9d1d9;font-size:.92rem;line-height:1.7'>"
            f"💬 {ai}</div>", unsafe_allow_html=True)

    st.markdown("#### 📊 Nutrition Facts (per 100g)")
    rows = [
        ("⚡ Energy",       "energy_kcal_100g",     "kcal",400,False),
        ("🍬 Sugar",        "sugar_g_100g",          "g",  15, False),
        ("💪 Protein",      "protein_g_100g",        "g",  5,  True),
        ("🌾 Fiber",        "fiber_g_100g",          "g",  3,  True),
        ("🧂 Sodium",       "sodium_mg_100g",        "mg", 400,False),
        ("🧈 Saturated Fat","saturated_fat_g_100g",  "g",  5,  False),
    ]
    html = "".join(nutrient_row_html(l,float(product.get(c) or 0),u,t,h) for l,c,u,t,h in rows)
    st.markdown(
        f"<div style='background:#161b22;border:1px solid #30363d;"
        f"border-radius:12px;padding:.8rem 1.2rem'>{html}</div>",
        unsafe_allow_html=True)

    if verdict in ("HARMFUL","MODERATE"):
        st.markdown("---")
        st.markdown("### 🔄 Better Alternatives")
        st.caption("Same category — higher Aarogya score")
        _show_recs(product, score)
    else:
        st.success("✅ This product scores well — no alternatives needed.")

    st.markdown("---")
    if st.button("🔄 Scan Another Product", use_container_width=True):
        del st.session_state["result"]
        st.rerun()

def _show_recs(product, cur_score):
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
    except Exception:
        pass

    if (recs is None or recs.empty) and not catalog.empty:
        cat = product.get("category","")
        if "food_score" in catalog.columns:
            same = catalog[
                (catalog["category"]==cat) &
                (catalog["food_score"]>cur_score)
            ].nlargest(3,"food_score")
            if not same.empty:
                recs = same
                recs = recs.assign(recommendation_reason="Higher scoring in same category")

    if recs is not None and not recs.empty:
        for _, r in recs.iterrows():
            rs = float(r.get("food_score", r.get("ranking_score",0)))
            rc = score_color(rs)
            rv,_,ri,_ = get_verdict(rs)
            rn = r.get("product_name","Unknown")
            rr = r.get("recommendation_reason","")
            st.markdown(f"""
            <div class='rec-card'>
              <div style='display:flex;align-items:center;gap:1rem'>
                <div style='font-size:1.8rem;font-weight:800;color:{rc};min-width:50px'>{rs:.0f}</div>
                <div style='flex:1'>
                  <div style='font-weight:700'>{ri} {rn}</div>
                  <div style='color:#8b949e;font-size:.82rem;margin-top:.15rem'>{rr}</div>
                </div>
                <div style='background:{rc}22;color:{rc};border-radius:8px;
                  padding:.25rem .6rem;font-size:.78rem;font-weight:700'>{rv}</div>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No alternatives found in this category.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='scan-title'>🌿 Aarogya</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='scan-sub'>Scan any packaged food → SAFE / HARMFUL verdict + alternatives</div>",
    unsafe_allow_html=True)

# Show result page if already scanned
if "result" in st.session_state:
    show_result(st.session_state["result"]["product"],
                st.session_state["result"].get("source",""))
    st.markdown("<div class='disclaimer'>⚠️ Scores are comparative, not clinical. "
                "Not medical advice. Academic project.</div>", unsafe_allow_html=True)
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_cam, tab_vision, tab_upload, tab_search, tab_demo = st.tabs([
    "📷 Camera", "🤖 AI Vision", "🖼 Upload", "🔍 Search", "🎯 Demo Products"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Camera
# ─────────────────────────────────────────────────────────────────────────────
with tab_cam:
    st.markdown("**Point camera at a barcode or nutrition label**")
    cam_img = st.camera_input("Capture", label_visibility="collapsed")
    if cam_img:
        ib = cam_img.getvalue()
        with st.spinner("Scanning barcode..."):
            bc = scan_barcode(ib)
        if bc:
            st.success(f"Barcode: `{bc}`")
            if bc in BARCODE_MAP:
                st.session_state["result"] = {
                    "product": DEMO_PRODUCTS[BARCODE_MAP[bc]].copy(),
                    "source": f"Barcode {bc}"}
                st.rerun()
            else:
                st.info("Barcode not in demo catalog. Trying OCR...")
                bc = None
        if not bc:
            with st.spinner("Reading nutrition label (OCR)..."):
                parsed = ocr_nutrition(ib)
            if parsed:
                p = blank_product("Camera Scan")
                p.update(parsed)
                st.session_state["result"] = {"product":p,"source":"Camera OCR"}
                st.rerun()
            else:
                st.error("Could not read label. Try better lighting or use AI Vision tab.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — AI Vision (Groq multimodal)
# ─────────────────────────────────────────────────────────────────────────────
with tab_vision:
    st.markdown("### 🤖 AI Vision Analysis")
    st.markdown("Upload a **product photo**, **nutrition label**, or **ingredients list** — "
                "Groq AI reads and analyses it automatically.")

    exp_v = load_explainer()
    if not exp_v.is_ready:
        st.warning("⚠️ Groq AI not connected. Add GROQ_API_KEY to .env to enable this feature.")

    scan_mode = st.radio("What are you scanning?",
                         ["📦 Product / Nutrition Label", "📋 Ingredients List"],
                         horizontal=True)

    vis_upload = st.file_uploader("Upload image for AI Vision",
                                  type=["jpg","jpeg","png","webp"],
                                  label_visibility="collapsed", key="vis_up")

    # Demo shortcuts
    st.caption("Or try a demo image:")
    dc = st.columns(3)
    _vdemos = [
        ("ChocoBite Label",  os.path.join(DEMO_DIR,"chocobite.jpg")),
        ("NutriOats Label",  os.path.join(DEMO_DIR,"nutrioats.jpg")),
        ("Ingredients List", os.path.join(DEMO_DIR,"ingredients_chocobite.jpg")),
    ]
    for col,(dlbl,dpath) in zip(dc,_vdemos):
        with col:
            if os.path.exists(dpath): st.image(dpath, use_container_width=True)
            if st.button(dlbl, key=f"vd_{dlbl}", use_container_width=True):
                with open(dpath,"rb") as f:
                    st.session_state["_vb"] = f.read()
                    st.session_state["_vl"] = dlbl

    vbytes, vlabel = None, ""
    if vis_upload:
        vbytes = vis_upload.getvalue(); vlabel = vis_upload.name
        st.image(vis_upload, caption="Your image", use_container_width=True)
    elif "_vb" in st.session_state:
        vbytes = st.session_state.pop("_vb")
        vlabel = st.session_state.pop("_vl","Demo")
        from PIL import Image as PILImg
        st.image(PILImg.open(io.BytesIO(vbytes)), caption=vlabel, use_container_width=True)

    if vbytes and exp_v.is_ready:
        if "📦" in scan_mode:
            with st.spinner("🤖 AI reading product image..."):
                vr = exp_v.analyze_product_image(vbytes)

            if not vr or (vr.get("error") and not vr.get("product_name")):
                st.error(f"Vision failed: {vr.get('error','Unknown error')}")
            else:
                conf = vr.get("confidence","low")
                cc = {"high":"#3fb950","medium":"#d29922","low":"#f85149"}.get(conf,"#8b949e")
                st.markdown(f"**AI Confidence:** <span style='color:{cc};font-weight:700'>"
                            f"{conf.upper()}</span>", unsafe_allow_html=True)

                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**📦 Product Info**")
                    for lbl,key in [("Name","product_name"),("Brand","brand"),
                                    ("Category","category")]:
                        st.markdown(f"**{lbl}:** {vr.get(key,'—')}")
                    st.markdown(f"**Processing:** "
                                f"{(vr.get('processing_level') or '').replace('_',' ').title()}")
                with cb:
                    st.markdown("**🏷 Signals**")
                    for items, icon, label in [
                        (vr.get("health_claims",[])    or [], "✅", "Claims"),
                        (vr.get("allergens_found",[])  or [], "⚠️", "Allergens"),
                        (vr.get("harmful_additives",[]) or [], "🔴", "Additives"),
                    ]:
                        if items: st.markdown(f"{icon} **{label}:** {', '.join(items[:3])}")
                    if vr.get("contains_whole_grain"): st.markdown("🌾 Whole Grain")
                    if vr.get("contains_added_sugar"): st.markdown("🍬 Added Sugar")

                nut_keys = ["energy_kcal_100g","protein_g_100g","sugar_g_100g","fiber_g_100g"]
                if any(vr.get(k) is not None for k in nut_keys):
                    st.divider()
                    st.markdown("**📊 Nutrition extracted — get full score:**")
                    ext = blank_product(
                        vr.get("product_name","AI Scanned"),
                        vr.get("category","Biscuits & Cookies"))
                    ext["processing_level"] = vr.get("processing_level","processed") or "processed"
                    for k in ["energy_kcal_100g","protein_g_100g","carbs_g_100g",
                               "sugar_g_100g","total_fat_g_100g","saturated_fat_g_100g",
                               "fiber_g_100g","sodium_mg_100g"]:
                        if vr.get(k) is not None: ext[k] = vr[k]
                    ext["added_sugar_g_100g"] = ext["sugar_g_100g"]
                    ext["ingredient_count"] = len(vr.get("ingredients_list",[]) or []) or 10
                    if st.button("🔬 Get Full Score + Recommendations",
                                 type="primary", use_container_width=True, key="vscore"):
                        st.session_state["result"] = {"product":ext,"source":f"AI Vision: {vlabel}"}
                        st.rerun()
        else:
            # Ingredients list
            with st.spinner("🤖 AI reading ingredients..."):
                ir = exp_v.analyze_ingredients(vbytes)

            if not ir or (ir.get("error") and not ir.get("ingredients_raw")):
                st.error(f"Could not read: {ir.get('error','Unknown')}")
            else:
                iq = int(ir.get("ingredient_quality_score") or 5)
                iq_col = "#3fb950" if iq>=7 else ("#d29922" if iq>=4 else "#f85149")
                st.markdown(
                    f"<div style='background:#161b22;border:1px solid #30363d;"
                    f"border-radius:14px;padding:1.2rem;margin-bottom:1rem'>"
                    f"<div style='font-size:.82rem;color:#8b949e'>Ingredient Quality Score</div>"
                    f"<div style='font-size:3rem;font-weight:800;color:{iq_col}'>{iq}"
                    f"<span style='font-size:1rem;color:#8b949e'>/10</span></div>"
                    f"<div style='color:#c9d1d9;font-size:.92rem;margin-top:.4rem'>"
                    f"{ir.get('summary','')}</div></div>", unsafe_allow_html=True)

                ci, cr = st.columns(2)
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
                            icon = {"high":"🔴","medium":"🟡","low":"🟠"}.get(fl.get("severity","medium"),"⚠️")
                            st.markdown(
                                f"{icon} **{fl.get('name','')}**\n"
                                f"<span style='color:#8b949e;font-size:.82rem'>{fl.get('reason','')}</span>",
                                unsafe_allow_html=True)
                    else:
                        st.success("No major red flags!")
                    sigs = []
                    if ir.get("contains_palm_oil"):           sigs.append("🌴 Palm Oil")
                    if ir.get("contains_artificial_colours"): sigs.append("🎨 Colours")
                    if ir.get("contains_artificial_flavours"):sigs.append("🧪 Flavours")
                    if ir.get("contains_preservatives"):      sigs.append("🧊 Preservatives")
                    if ir.get("contains_added_sugar"):        sigs.append("🍬 Added Sugar")
                    if sigs: st.markdown("  ".join(sigs))
                    adds = ir.get("additives_found",[]) or []
                    algs = ir.get("allergens",[]) or []
                    if adds: st.caption(f"Additives: {', '.join(adds[:5])}")
                    if algs: st.caption(f"Allergens: {', '.join(algs)}")

    elif vbytes and not exp_v.is_ready:
        st.warning("Groq AI not connected. Please set GROQ_API_KEY in .env file.")
    else:
        st.info("📤 Upload an image or click a demo above to start AI analysis.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Upload Image (barcode + OCR)
# ─────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("**Upload a photo of the barcode or nutrition label**")
    uploaded = st.file_uploader("Upload product image",
                                type=["jpg","jpeg","png","webp"],
                                label_visibility="collapsed")
    if uploaded:
        ib = uploaded.getvalue()
        st.image(uploaded, caption="Uploaded image", use_container_width=True)
        c1, c2 = st.columns(2)
        if c1.button("🔢 Scan Barcode", use_container_width=True, type="primary"):
            with st.spinner("Scanning barcode..."):
                bc = scan_barcode(ib)
            if bc:
                st.success(f"Barcode: `{bc}`")
                if bc in BARCODE_MAP:
                    st.session_state["result"] = {
                        "product": DEMO_PRODUCTS[BARCODE_MAP[bc]].copy(),
                        "source": f"Barcode {bc}"}
                    st.rerun()
                else:
                    st.info(f"Barcode `{bc}` not in demo catalog.")
            else:
                st.warning("No barcode found. Try reading the nutrition label.")
        if c2.button("🔤 Read Nutrition Label", use_container_width=True):
            with st.spinner("Running OCR..."):
                parsed = ocr_nutrition(ib)
            if parsed:
                p = blank_product(uploaded.name.split(".")[0].replace("_"," ").title())
                p.update(parsed)
                st.session_state["result"] = {"product":p,"source":f"OCR: {uploaded.name}"}
                st.rerun()
            else:
                st.error("Could not extract nutrition. Try the 🤖 AI Vision tab for better results.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Search by Name
# ─────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("**Search the product database by name**")
    catalog = load_catalog()
    query = st.text_input("Search product name",
                          placeholder="e.g. Maggi, Oreo, Muesli, Chips...",
                          label_visibility="collapsed")
    if query and not catalog.empty:
        matches = catalog[catalog["product_name"].str.contains(query, case=False, na=False)]
        if matches.empty:
            st.warning(f"No products matching '{query}'")
        else:
            sel = st.selectbox(f"Found {len(matches)} product(s):",
                               matches["product_name"].tolist())
            row = matches[matches["product_name"]==sel].iloc[0]
            cl, cr = st.columns([2,1])
            cl.markdown(f"**{row['product_name']}**")
            cl.caption(f"{row.get('brand','')} | {row.get('category','')} | "
                       f"Score: {row.get('food_score','?')}/100")
            if cr.button("🔬 Scan", use_container_width=True, type="primary"):
                st.session_state["result"] = {
                    "product":row.to_dict(),"source":f"Catalog: {sel}"}
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Demo Products
# ─────────────────────────────────────────────────────────────────────────────
with tab_demo:
    st.markdown("**Try these demo products — click Scan to analyse**")
    st.caption("Real-looking product labels with barcodes and nutrition facts")

    demo_items = [
        ("chocobite.jpg",   "ChocoBite Cream Biscuits", "🍪 Ultra-Processed · High Sugar"),
        ("nutrioats.jpg",   "NutriOats Instant Porridge","🌾 Minimally Processed · High Fiber"),
        ("masalachips.jpg", "MasalaKing Spicy Chips",    "🌶️ Ultra-Processed · High Sodium"),
    ]
    cols = st.columns(3)
    for col,(fname,dname,dtag) in zip(cols,demo_items):
        ipath = os.path.join(DEMO_DIR, fname)
        with col:
            if os.path.exists(ipath): st.image(ipath, use_container_width=True)
            st.markdown(f"**{dname}**")
            st.caption(dtag)
            if st.button("🔬 Scan", key=f"d_{fname}", use_container_width=True, type="primary"):
                st.session_state["result"] = {
                    "product": DEMO_PRODUCTS[fname].copy(),
                    "source": f"Demo: {dname}"}
                st.rerun()

    st.divider()
    st.markdown("**Ingredients demo:**")
    ing_path = os.path.join(DEMO_DIR, "ingredients_chocobite.jpg")
    if os.path.exists(ing_path):
        st.image(ing_path, caption="ChocoBite Ingredients List — try in 🤖 AI Vision tab",
                 use_container_width=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("<div class='disclaimer'>⚠️ Aarogya scores are comparative, not clinical. "
            "Not medical advice. Academic project using synthetic data.</div>",
            unsafe_allow_html=True)
