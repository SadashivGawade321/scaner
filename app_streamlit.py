"""
app_streamlit.py
─────────────────────────────────────────────────────────────────────────────
AAROGYA — Streamlit UI (Week 14)
─────────────────────────────────────────────────────────────────────────────

Full user interface with:
  - Product search from catalog
  - Manual nutrition input for unrecognized products
  - Score display with color coding
  - Explanation text (Gemini or rule-based fallback)
  - Top 3 recommendations
  - Product comparison
  - User preference configuration

USAGE
─────
    pip install streamlit plotly
    streamlit run app_streamlit.py
"""

import os, sys
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    STREAMLIT_OK = True
except ImportError:
    STREAMLIT_OK = False
    print("Install streamlit: pip install streamlit plotly")
    sys.exit(1)

from ml.scoring          import compute_aarogya_score as score_product
from llm.groq_llm        import AarogyaGroqExplainer
from recommendation.recommender import AarogyaRecommender

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Aarogya — Food Intelligence",
    page_icon  = "🌿",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0f1117;
    color: #e0e0e0;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4ade80, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.sub-title {
    color: #6b7280;
    font-size: 1rem;
    margin-top: 0;
}

.score-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

.score-number {
    font-size: 4rem;
    font-weight: 700;
    line-height: 1;
}

.score-label {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-top: 0.5rem;
}

.tier-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

.metric-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

.explanation-box {
    background: linear-gradient(135deg, #1e3a2f, #0f2318);
    border: 1px solid #166534;
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #bbf7d0;
}

.rec-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}

.rec-card:hover {
    border-color: #4ade80;
}

.disclaimer {
    color: #6b7280;
    font-size: 0.78rem;
    border-top: 1px solid #1e293b;
    padding-top: 0.5rem;
    margin-top: 1rem;
}

div[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ───────────────────────────────────────────────────────────────

@st.cache_data
def load_catalog():
    for path in [
        os.path.join(ROOT, "data", "raw", "aarogya_products_500.csv"),
        os.path.join(ROOT, "data", "raw", "aarogya_food_products_demo.csv"),
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_resource
def get_explainer():
    return AarogyaGroqExplainer()


@st.cache_resource
def get_recommender():
    df = load_catalog()
    if df.empty:
        return None
    return AarogyaRecommender(df)


def get_predictor():
    try:
        from ml.predict import AarogyaPredictor
        p = AarogyaPredictor()
        p._load()
        return p
    except Exception:
        return None


def score_color(score):
    if score >= 75: return "#4ade80"
    if score >= 60: return "#86efac"
    if score >= 40: return "#fbbf24"
    return "#f87171"


def score_tier(score):
    if score >= 75: return "Excellent"
    if score >= 60: return "Good"
    if score >= 40: return "Moderate"
    return "Needs Consideration"


def make_gauge(score, title="Aarogya Score"):
    color = score_color(score)
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        title = {"text": title, "font": {"color": "#94a3b8", "size": 14}},
        number = {"font": {"color": color, "size": 48}},
        gauge = {
            "axis": {"range": [0, 100], "tickcolor": "#475569",
                     "tickfont": {"color": "#475569"}},
            "bar": {"color": color},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0,  40], "color": "#3f1515"},
                {"range": [40, 60], "color": "#3f3015"},
                {"range": [60, 75], "color": "#1a3f25"},
                {"range": [75,100], "color": "#0f2c1a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        }
    ))
    fig.update_layout(
        height=260, margin=dict(l=20,r=20,t=30,b=20),
        paper_bgcolor="#0f1117", font_color="#e0e0e0",
    )
    return fig


def make_shap_bar(shap_ctx):
    feats  = shap_ctx.get("shap_top_features", {})
    if not feats:
        return None
    names  = list(feats.keys())
    values = list(feats.values())
    colors = ["#4ade80" if v > 0 else "#f87171" for v in values]

    fig = go.Figure(go.Bar(
        x          = values,
        y          = names,
        orientation = "h",
        marker_color= colors,
    ))
    fig.update_layout(
        title_text = "Feature Contributions (SHAP)",
        title_font = {"color": "#94a3b8"},
        xaxis_title= "SHAP Value",
        height     = 280,
        margin     = dict(l=10,r=10,t=40,b=10),
        paper_bgcolor="#1e293b",
        plot_bgcolor ="#1e293b",
        font_color   ="#e0e0e0",
        xaxis        = dict(gridcolor="#334155"),
        yaxis        = dict(gridcolor="#334155"),
    )
    return fig


# ─── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌿 Aarogya")
    st.markdown("*AI Food Product Intelligence*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Analyze Product", "⚖️ Compare Products", "💬 AI Chat", "🎛️ Preferences"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Preference Profile**")
    pref_profile = st.selectbox(
        "Your focus",
        ["balanced","sugar_focused","protein_focused","fiber_focused",
         "sodium_focused","calorie_focused","satfat_focused"],
        label_visibility="collapsed"
    )

    st.divider()
    exp = get_explainer()
    groq_status = "🟢 Groq AI Connected" if exp.is_ready else "🟡 Rule-based (No API key)"
    st.markdown(f"**LLM Status:** {groq_status}")
    if exp.is_ready:
        st.caption(f"Model: `{exp.model_name}`")
    st.divider()
    st.markdown(
        "<span class='disclaimer'>⚠️ This is an academic project. "
        "Not medical advice. Scores are comparative, not clinical.</span>",
        unsafe_allow_html=True
    )


# ─── HOME ──────────────────────────────────────────────────────────────────

if "Home" in page:
    st.markdown("<h1 class='main-title'>Aarogya</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>AI Food Product Intelligence — Scan. Understand. Choose Better.</p>",
                unsafe_allow_html=True)
    st.divider()

    catalog = load_catalog()
    if catalog.empty:
        st.warning("Dataset not found. Please generate the dataset first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Products",    len(catalog))
        c2.metric("🏷️ Categories",  catalog["category"].nunique())
        c3.metric("📊 Avg Score",   f"{catalog['food_score'].mean():.1f}")
        c4.metric("🏆 Top Score",   f"{catalog['food_score'].max():.0f}")

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("🏆 Top Scoring Products")
            top5 = catalog.nlargest(5, "food_score")[
                ["product_name","category","food_score"]
            ].reset_index(drop=True)
            top5.index += 1
            top5.columns = ["Product","Category","Score"]
            st.dataframe(top5, use_container_width=True)

        with col_r:
            st.subheader("📊 Score by Category")
            cat_avg = catalog.groupby("category")["food_score"].mean().sort_values()
            fig = px.bar(
                x=cat_avg.values, y=cat_avg.index,
                orientation="h",
                color=cat_avg.values,
                color_continuous_scale=["#f87171","#fbbf24","#4ade80"],
                labels={"x":"Avg Score","y":"Category"}
            )
            fig.update_layout(
                height=320, coloraxis_showscale=False,
                paper_bgcolor="#0f1117", plot_bgcolor="#1e293b",
                font_color="#e0e0e0", margin=dict(l=0,r=0,t=0,b=0),
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155"),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("📈 Score Distribution")
        fig2 = px.histogram(
            catalog, x="food_score", nbins=20,
            color_discrete_sequence=["#4ade80"],
            labels={"food_score":"Aarogya Score"}
        )
        fig2.update_layout(
            height=200, paper_bgcolor="#0f1117", plot_bgcolor="#1e293b",
            font_color="#e0e0e0", margin=dict(l=0,r=0,t=0,b=0),
            bargap=0.05, showlegend=False,
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─── ANALYZE ───────────────────────────────────────────────────────────────

elif "Analyze" in page:
    st.markdown("<h1 class='main-title'>Analyze a Product</h1>", unsafe_allow_html=True)
    st.divider()

    catalog = load_catalog()
    input_mode = st.radio(
        "Input Method",
        ["Search Catalog", "Enter Nutrition Manually"],
        horizontal=True
    )

    product_data = None

    if input_mode == "Search Catalog" and not catalog.empty:
        search = st.text_input("🔍 Search product name", placeholder="e.g. Oat Muesli")
        if search:
            matches = catalog[catalog["product_name"].str.contains(search, case=False, na=False)]
            if matches.empty:
                st.info("No matches found. Try manual entry.")
            else:
                selected_name = st.selectbox(
                    "Select product",
                    matches["product_name"].tolist()
                )
                row = matches[matches["product_name"] == selected_name].iloc[0]
                product_data = row.to_dict()
                st.success(f"Selected: **{selected_name}** (Category: {row['category']})")

    else:
        st.subheader("Enter Nutrition Facts (per 100g)")
        col1, col2, col3 = st.columns(3)
        with col1:
            energy  = st.number_input("Energy (kcal)", 0.0, 900.0, 350.0)
            protein = st.number_input("Protein (g)",   0.0, 100.0,  8.0)
            carbs   = st.number_input("Carbs (g)",     0.0, 100.0, 60.0)
        with col2:
            sugar   = st.number_input("Sugar (g)",     0.0, 100.0, 12.0)
            fat     = st.number_input("Total Fat (g)", 0.0, 100.0, 12.0)
            sat_fat = st.number_input("Sat. Fat (g)",  0.0, 100.0,  4.0)
        with col3:
            fiber   = st.number_input("Fiber (g)",     0.0,  50.0,  4.0)
            sodium  = st.number_input("Sodium (mg)",   0.0, 3000.0, 400.0)
            pname   = st.text_input("Product Name", "My Product")

        cat_opts = catalog["category"].unique().tolist() if not catalog.empty else ["Biscuits & Cookies"]
        category = st.selectbox("Category", cat_opts)
        proc     = st.selectbox("Processing Level",
                                ["minimally_processed","processed","ultra_processed"])
        whole    = st.checkbox("Contains Whole Grain")

        group_map = {
            "Biscuits & Cookies":"biscuits_cookies",
            "Chips & Namkeen":"chips_namkeen",
            "Breakfast Cereals":"breakfast_cereals",
            "Beverages":"beverages",
            "Instant Foods":"instant_foods",
            "Chocolates & Sweet Snacks":"chocolates_sweets",
            "Snack/Protein Bars":"snack_protein_bars",
            "Bread & Bakery":"bread_bakery",
            "Dairy & Dairy Drinks":"dairy_drinks",
        }
        group = group_map.get(category, "biscuits_cookies")

        product_data = {
            "product_name": pname, "category": category,
            "recommendation_group": group,
            "energy_kcal_100g": energy, "protein_g_100g": protein,
            "carbs_g_100g": carbs, "sugar_g_100g": sugar,
            "added_sugar_g_100g": sugar * 0.8,
            "total_fat_g_100g": fat, "saturated_fat_g_100g": sat_fat,
            "trans_fat_g_100g": 0.0, "fiber_g_100g": fiber,
            "sodium_mg_100g": sodium, "ingredient_count": 10,
            "contains_whole_grain": whole, "contains_added_sugar": sugar > 2,
            "contains_artificial_sweetener": False,
            "contains_allergen": False, "processing_level": proc,
        }

    if product_data and st.button("🔬 Analyze", use_container_width=True, type="primary"):
        with st.spinner("Analyzing product..."):
            # Score
            det = score_product(product_data)
            score = det["final_score"]

            # Try ML predictor
            predictor = get_predictor()
            shap_ctx  = None
            if predictor:
                try:
                    ml_result = predictor.predict(product_data)
                    score     = ml_result["predicted_score"]
                    shap_ctx  = ml_result["explanation_context"]
                except Exception:
                    pass

            if shap_ctx is None:
                shap_ctx = {
                    "product_name":    product_data.get("product_name","?"),
                    "category":        product_data.get("category",""),
                    "processing_level":product_data.get("processing_level",""),
                    "predicted_score": score,
                    "base_value":      55.0,
                    "strengths":       det.get("breakdown", {}).get("strengths", []),
                    "concerns":        det.get("breakdown", {}).get("concerns",  []),
                    "shap_top_features": {},
                }

            # Explanation
            explanation = get_explainer().explain_score(shap_ctx)

            # Recommendations
            rec_df = None
            recommender = get_recommender()
            if recommender:
                try:
                    if "product_id" in product_data:
                        rec_df = recommender.recommend(
                            product_id=product_data["product_id"],
                            preference_profile=pref_profile, top_n=3
                        )
                    else:
                        rec_df = recommender.recommend_by_data(
                            current_product=product_data,
                            recommendation_group=product_data.get("recommendation_group","biscuits_cookies"),
                            preference_profile=pref_profile, top_n=3
                        )
                except Exception:
                    rec_df = None

        # ── Display ────────────────────────────────────────────────────────
        st.divider()
        col_gauge, col_info = st.columns([1, 2])

        with col_gauge:
            fig_gauge = make_gauge(score)
            st.plotly_chart(fig_gauge, use_container_width=True)
            tier = score_tier(score)
            color = score_color(score)
            st.markdown(
                f"<div style='text-align:center; color:{color}; "
                f"font-size:1.2rem; font-weight:700;'>{tier}</div>",
                unsafe_allow_html=True
            )

        with col_info:
            pname = product_data.get("product_name", "Product")
            st.subheader(pname)
            cat = product_data.get("category","")
            proc = product_data.get("processing_level","").replace("_"," ").title()
            st.caption(f"📂 {cat}  |  ⚙️ {proc}  |  📊 Score v1.0")

            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Strengths", ", ".join(shap_ctx["strengths"][:2]) or "—")
            c2.metric("⚠️ Concerns",  ", ".join(shap_ctx["concerns"][:2])  or "—")
            c3.metric("🔢 Score",      f"{score:.1f}/100")

            st.markdown(
                f"<div class='explanation-box'>💬 {explanation}</div>",
                unsafe_allow_html=True
            )

        # SHAP chart
        if shap_ctx.get("shap_top_features"):
            shap_fig = make_shap_bar(shap_ctx)
            if shap_fig:
                st.plotly_chart(shap_fig, use_container_width=True)

        # Recommendations
        if rec_df is not None and not rec_df.empty:
            st.divider()
            st.subheader("🔄 Better Alternatives (Same Category)")
            for _, r in rec_df.iterrows():
                rsc = float(r.get("food_score", 0))
                rcol = score_color(rsc)
                st.markdown(
                    f"""<div class='rec-card'>
                    <b style='color:{rcol};'>{r['product_name']}</b>
                    &nbsp;&nbsp;<span style='color:#64748b;font-size:0.85rem;'>Score: {rsc:.0f}/100</span><br>
                    <span style='font-size:0.88rem; color:#94a3b8;'>
                    {r.get('recommendation_reason','')}</span>
                    </div>""",
                    unsafe_allow_html=True
                )

        st.markdown(
            "<div class='disclaimer'>⚠️ Aarogya scores are comparative and for informational purposes only. "
            "Not medical advice. Trained on synthetic data.</div>",
            unsafe_allow_html=True
        )


# ─── COMPARE ───────────────────────────────────────────────────────────────

elif "Compare" in page:
    st.markdown("<h1 class='main-title'>Compare Products</h1>", unsafe_allow_html=True)
    st.divider()

    catalog = load_catalog()
    if catalog.empty:
        st.warning("Dataset not found.")
    else:
        names = catalog["product_name"].tolist()
        col1, col2 = st.columns(2)
        with col1:
            name_a = st.selectbox("Product A", names, key="ca")
        with col2:
            name_b = st.selectbox("Product B", names, index=1, key="cb")

        if st.button("⚖️ Compare", use_container_width=True, type="primary"):
            row_a = catalog[catalog["product_name"] == name_a].iloc[0]
            row_b = catalog[catalog["product_name"] == name_b].iloc[0]
            sa = float(row_a.get("food_score", 0))
            sb = float(row_b.get("food_score", 0))

            col_a, col_b = st.columns(2)
            nutrients = {
                "energy_kcal_100g":     ("⚡ Energy",     "kcal", False),
                "protein_g_100g":       ("💪 Protein",    "g",    True),
                "sugar_g_100g":         ("🍬 Sugar",      "g",    False),
                "fiber_g_100g":         ("🌾 Fiber",      "g",    True),
                "sodium_mg_100g":       ("🧂 Sodium",     "mg",   False),
                "saturated_fat_g_100g": ("🧈 Sat. Fat",  "g",    False),
            }

            with col_a:
                st.markdown(f"### {name_a}")
                fig_a = make_gauge(sa, f"Score: {sa:.0f}")
                st.plotly_chart(fig_a, use_container_width=True)
                for col, (label, unit, higher_is_better) in nutrients.items():
                    va = float(row_a.get(col, 0)) if pd.notna(row_a.get(col)) else 0
                    vb = float(row_b.get(col, 0)) if pd.notna(row_b.get(col)) else 0
                    delta = va - vb
                    is_better = (delta < 0) if not higher_is_better else (delta > 0)
                    arrow = "▲" if delta > 0 else "▼"
                    col_clr = "#4ade80" if is_better else "#f87171"
                    st.markdown(
                        f"**{label}**: {va:.1f} {unit} "
                        f"<span style='color:{col_clr};font-size:0.8rem;'>{arrow} {abs(delta):.1f}</span>",
                        unsafe_allow_html=True
                    )

            with col_b:
                st.markdown(f"### {name_b}")
                fig_b = make_gauge(sb, f"Score: {sb:.0f}")
                st.plotly_chart(fig_b, use_container_width=True)
                for col, (label, unit, higher_is_better) in nutrients.items():
                    va = float(row_a.get(col, 0)) if pd.notna(row_a.get(col)) else 0
                    vb = float(row_b.get(col, 0)) if pd.notna(row_b.get(col)) else 0
                    delta = vb - va
                    is_better = (delta < 0) if not higher_is_better else (delta > 0)
                    arrow = "▲" if delta > 0 else "▼"
                    col_clr = "#4ade80" if is_better else "#f87171"
                    st.markdown(
                        f"**{label}**: {vb:.1f} {unit} "
                        f"<span style='color:{col_clr};font-size:0.8rem;'>{arrow} {abs(delta):.1f}</span>",
                        unsafe_allow_html=True
                    )

            winner_name = name_a if sa >= sb else name_b
            winner_score = max(sa, sb)
            st.success(f"🏆 **{winner_name}** has a higher Aarogya Score ({winner_score:.0f}/100) for your selected preferences.")



# ─── AI CHAT ───────────────────────────────────────────────────────────────

elif "Chat" in page:
    st.markdown("<h1 class='main-title'>💬 AI Product Chat</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Ask anything about a product's nutritional profile</p>",
                unsafe_allow_html=True)
    st.divider()

    exp = get_explainer()
    if not exp.is_ready:
        st.warning(
            "Groq AI is not connected. Add your GROQ_API_KEY to the .env file "
            "to enable natural language chat."
        )

    catalog = load_catalog()

    # Product selector for context
    ctx_product = None
    if not catalog.empty:
        use_ctx = st.checkbox("Chat about a specific product", value=True)
        if use_ctx:
            search = st.text_input("Search product", placeholder="e.g. Muesli")
            if search:
                matches = catalog[catalog["product_name"].str.contains(search, case=False, na=False)]
                if not matches.empty:
                    sel = st.selectbox("Select", matches["product_name"].tolist())
                    row = matches[matches["product_name"] == sel].iloc[0]
                    det = score_product(row.to_dict())
                    ctx_product = {
                        "product_name":    row["product_name"],
                        "category":        row["category"],
                        "predicted_score": float(row.get("food_score", det["final_score"])),
                        "processing_level":row.get("processing_level", ""),
                        "strengths":       det.get("breakdown",{}).get("strengths", []),
                        "concerns":        det.get("breakdown",{}).get("concerns",  []),
                    }
                    st.info(
                        f"📦 **{row['product_name']}** | Score: {ctx_product['predicted_score']:.0f}/100 | "
                        f"{row.get('processing_level','').replace('_',' ').title()}"
                    )

    st.divider()

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask about ingredients, nutrition, alternatives...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = exp.chat(user_input, ctx_product)
            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown(
        "<div class='disclaimer'>⚠️ AI responses are for informational purposes only. "
        "Not medical advice. Powered by Groq Llama 3.1.</div>",
        unsafe_allow_html=True
    )


# ─── PREFERENCES ───────────────────────────────────────────────────────────

elif "Preferences" in page:
    st.markdown("<h1 class='main-title'>Your Preferences</h1>", unsafe_allow_html=True)
    st.divider()

    st.info(
        "Preference weights tell the recommendation engine what you care most about. "
        "They must add up to 1.0. Select a preset or customize below."
    )

    preset = st.selectbox("Load a preset", [
        "balanced","sugar_focused","protein_focused",
        "fiber_focused","sodium_focused"
    ])

    PRESETS = {
        "balanced":       [0.20, 0.20, 0.20, 0.20, 0.10, 0.10],
        "sugar_focused":  [0.40, 0.20, 0.15, 0.10, 0.10, 0.05],
        "protein_focused":[0.10, 0.40, 0.20, 0.10, 0.10, 0.10],
        "fiber_focused":  [0.15, 0.20, 0.35, 0.15, 0.10, 0.05],
        "sodium_focused": [0.15, 0.15, 0.15, 0.40, 0.10, 0.05],
    }
    defaults = PRESETS[preset]
    keys     = ["Sugar", "Protein", "Fiber", "Sodium", "Calories", "Saturated Fat"]

    cols = st.columns(3)
    weights = []
    for i, (key, default) in enumerate(zip(keys, defaults)):
        with cols[i % 3]:
            w = st.slider(f"🎚️ {key}", 0.0, 1.0, default, 0.05, key=f"w_{i}")
            weights.append(w)

    total = sum(weights)
    if abs(total - 1.0) > 0.05:
        st.error(f"Weights sum to {total:.2f}. Must be 1.0. Please adjust sliders.")
    else:
        st.success(f"Total weight: {total:.2f} ✓")

    st.divider()
    st.caption(
        "Note: These weights affect recommendations only. "
        "The Aarogya score is always computed from the same v1.0 methodology."
    )
