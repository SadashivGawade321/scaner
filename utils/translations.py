"""
aarogya/utils/translations.py
=============================
Multilingual dictionary and localization engine supporting 6 languages:
  - English (en)
  - Hindi (hi)
  - Marathi (mr)
  - Gujarati (gu)
  - Tamil (ta)
  - Spanish (es)
"""

LANGUAGES = {
    "en": "🇬🇧 English",
    "hi": "🇮🇳 हिन्दी (Hindi)",
    "mr": "🇮🇳 मराठी (Marathi)",
    "gu": "🇮🇳 ગુજરાતી (Gujarati)",
    "ta": "🇮🇳 தமிழ் (Tamil)",
    "es": "🇪🇸 Español (Spanish)",
}

LANG_CODES = list(LANGUAGES.keys())

TRANSLATIONS = {
    "app_title": {
        "en": "Aarogya Food Intelligence",
        "hi": "आरोग्य फूड इंटेलिजेंस",
        "mr": "आरोग्य अन्न बुद्धिमत्ता",
        "gu": "આરોગ્ય ફૂડ ઇન્ટેલિજન્સ",
        "ta": "ஆரோக்யா உணவு நுண்ணறிவு",
        "es": "Inteligencia Alimentaria Aarogya",
    },
    "app_subtitle": {
        "en": "Instant Food Safety Scoring · AI Nutrition Analysis · Cloud Persistence",
        "hi": "त्वरित खाद्य सुरक्षा स्कोर · एआई पोषण विश्लेषण · क्लाउड डेटाबेस",
        "mr": "त्वरित अन्न सुरक्षा गुण · एआय पोषण विश्लेषण · क्लाउड डेटाबेस",
        "gu": "ત્વરિત ખાદ્ય સુરક્ષા સ્કોર · એઆઈ પોષણ વિશ્લેષણ · ક્લાઉડ ડેટાબેઝ",
        "ta": "உடனடி உணவு பாதுகாப்பு மதிப்பீடு · AI ஊட்டச்சத்து பகுப்பாய்வு",
        "es": "Puntuación Instantánea de Seguridad · Análisis Nutricional IA",
    },
    "tab_camera": {
        "en": "📷 Camera",
        "hi": "📷 कैमरा",
        "mr": "📷 कॅमेरा",
        "gu": "📷 કેમેરા",
        "ta": "📷 கேமரா",
        "es": "📷 Cámara",
    },
    "tab_vision": {
        "en": "🤖 AI Vision",
        "hi": "🤖 एआई विज़न",
        "mr": "🤖 एआय व्हिजन",
        "gu": "🤖 એઆઈ વિઝન",
        "ta": "🤖 AI பார்வை",
        "es": "🤖 Visión IA",
    },
    "tab_upload": {
        "en": "📤 Upload",
        "hi": "📤 अपलोड",
        "mr": "📤 अपलोड",
        "gu": "📤 અપલોડ",
        "ta": "📤 பதிவேற்றவும்",
        "es": "📤 Subir",
    },
    "tab_search": {
        "en": "🔍 Search",
        "hi": "🔍 खोजें",
        "mr": "🔍 शोधा",
        "gu": "🔍 શોધો",
        "ta": "🔍 தேடல்",
        "es": "🔍 Buscar",
    },
    "tab_demo": {
        "en": "🎯 Demo",
        "hi": "🎯 डेमो",
        "mr": "🎯 डेमो",
        "gu": "🎯 ડેમો",
        "ta": "🎯 டெமோ",
        "es": "🎯 Demo",
    },
    "verdict_safe": {
        "en": "SAFE / HEALTHY",
        "hi": "सुरक्षित और स्वस्थ",
        "mr": "सुरक्षित आणि निरोगी",
        "gu": "સુરક્ષિત અને સ્વસ્થ",
        "ta": "பாதுகாப்பானது / ஆரோக்கியமானது",
        "es": "SEGURO / SALUDABLE",
    },
    "verdict_mod": {
        "en": "MODERATE RISK",
        "hi": "मध्यम जोखिम (सीमित खाएं)",
        "mr": "मध्यम प्रमाण (मर्यादित सेवन करा)",
        "gu": "મધ્યમ જોખમ (મર્યાદિત ખાઓ)",
        "ta": "மிதமான அபாயம்",
        "es": "RIESGO MODERADO",
    },
    "verdict_harm": {
        "en": "HARMFUL / ULTRA-PROCESSED",
        "hi": "हानिकारक / अति-प्रसंस्कृत (बचें)",
        "mr": "हानिकारक / अति-प्रक्रिया केलेले (टाळा)",
        "gu": "હાનિકારક / અલ્ટ્રા-પ્રોસેસ્ડ (ટાળો)",
        "ta": "தீங்கு விளைவிக்கும் / தவிர்க்கவும்",
        "es": "DAÑINO / ULTRA-PROCESADO",
    },
    "food_safety_score": {
        "en": "Food Safety Score",
        "hi": "खाद्य सुरक्षा स्कोर",
        "mr": "अन्न सुरक्षा गुण",
        "gu": "ખાદ્ય સુરક્ષા સ્કોર",
        "ta": "உணவு பாதுகாப்பு மதிப்பீடு",
        "es": "Puntuación de Seguridad",
    },
    "calories": {
        "en": "Calories",
        "hi": "कैलोरी",
        "mr": "कॅलरी",
        "gu": "કેલરી",
        "ta": "கலோரிகள்",
        "es": "Calorías",
    },
    "protein": {
        "en": "Protein",
        "hi": "प्रोटीन",
        "mr": "प्रथिने",
        "gu": "પ્રોટીન",
        "ta": "புரதம்",
        "es": "Proteína",
    },
    "carbs": {
        "en": "Carbohydrates",
        "hi": "कार्बोहाइड्रेट",
        "mr": "कर्बोदके",
        "gu": "કાર્બોહાઇડ્રેટ",
        "ta": "கார்போஹைட்ரேட்",
        "es": "Carbohidratos",
    },
    "sugar": {
        "en": "Sugar",
        "hi": "चीनी",
        "mr": "साखर",
        "gu": "ખાંડ",
        "ta": "சர்க்கரை",
        "es": "Azúcar",
    },
    "fat": {
        "en": "Total Fat",
        "hi": "कुल वसा",
        "mr": "एकूण चरबी",
        "gu": "કુલ ચરબી",
        "ta": "மொத்த கொழுப்பு",
        "es": "Grasa Total",
    },
    "sat_fat": {
        "en": "Saturated Fat",
        "hi": "संतृप्त वसा",
        "mr": "संतृप्त चरबी",
        "gu": "સંતૃપ્ત ચરબી",
        "ta": "நிறைவுற்ற கொழுப்பு",
        "es": "Grasa Saturada",
    },
    "fiber": {
        "en": "Dietary Fiber",
        "hi": "आहार फाइबर",
        "mr": "आहार फायबर",
        "gu": "ફાઇબર",
        "ta": "நார்ச்சத்து",
        "es": "Fibra Dietética",
    },
    "sodium": {
        "en": "Sodium",
        "hi": "सोडियम",
        "mr": "सोडियम",
        "gu": "સોડિયમ",
        "ta": "சோடியம்",
        "es": "Sodio",
    },
    "why_score": {
        "en": "💬 Why this score? (AI Nutritionist Explanation)",
        "hi": "💬 यह स्कोर क्यों आया? (एआई पोषण विशेषज्ञ)",
        "mr": "💬 हा स्कोअर का आला? (एआय पोषणतज्ज्ञ विश्लेषण)",
        "gu": "💬 શા માટે આ સ્કોર? (એઆઈ આહાર નિષ્ણાત)",
        "ta": "💬 இந்த மதிப்பெண் ஏன்? (AI விளக்கம்)",
        "es": "💬 ¿Por qué esta puntuación? (Explicación IA)",
    },
    "red_flags": {
        "en": "🚨 Red Flag Ingredients & Concerns",
        "hi": "🚨 खतरनाक तत्व और चिंताएं",
        "mr": "🚨 धोकादायक घटक आणि चिंता",
        "gu": "🚨 લાલ નિશાન ઘટકો અને ચિંતાઓ",
        "ta": "🚨 எச்சரிக்கை பொருட்கள்",
        "es": "🚨 Ingredientes de Alerta",
    },
    "alternatives": {
        "en": "💡 Healthier Alternatives",
        "hi": "💡 स्वास्थ्यवर्धक और बेहतर विकल्प",
        "mr": "💡 अधिक निरोगी आणि उत्तम पर्याय",
        "gu": "💡 વધુ સ્વસ્થ અને શ્રેષ્ઠ વિકલ્પો",
        "ta": "💡 ஆரோக்கியமான சிறந்த மாற்று வழிகள்",
        "es": "💡 Alternativas Más Saludables",
    },
    "scan_history_title": {
        "en": "📊 Recent Scan History (MongoDB Atlas)",
        "hi": "📊 हालिया स्कैन इतिहास (MongoDB Atlas)",
        "mr": "📊 अलीकडील स्कॅन इतिहास (MongoDB Atlas)",
        "gu": "📊 તાજેતરનો સ્કેન ઇતિહાસ (MongoDB Atlas)",
        "ta": "📊 சமீபத்திய ஸ்கேன் வரலாறு (MongoDB)",
        "es": "📊 Historial de Escaneos Recientes (MongoDB)",
    },
    "search_prompt": {
        "en": "Search product database by name",
        "hi": "नाम से उत्पाद खोजें",
        "mr": "नावाने उत्पादन शोधा",
        "gu": "નામ દ્વારા ઉત્પાદન શોધો",
        "ta": "பெயர் மூலம் தயாரிப்பைத் தேடுங்கள்",
        "es": "Buscar producto por nombre",
    },
    "search_placeholder": {
        "en": "e.g. Maggi, Oreo, Parle-G, Lay's, Kurkure...",
        "hi": "उदा. मैगी, ओरियो, पार्ले-जी, लेज, कुरकुरे...",
        "mr": "उदा. मॅगी, ओरिओ, पारले-जी, लेज, कुरकुरे...",
        "gu": "દા.ત. મેગી, ઓરીઓ, પાર્લે-જી, લેઝ...",
        "ta": "எ.கா. மேகி, ஓரியோ, பார்லே-ஜி...",
        "es": "ej. Maggi, Oreo, Parle-G, Papas...",
    },
    "no_products_found": {
        "en": "No products matching '{query}' in local catalog",
        "hi": "स्थानीय सूची में '{query}' से मेल खाता कोई उत्पाद नहीं मिला",
        "mr": "स्थानिक यादीत '{query}' शी जुळणारे उत्पादन सापडले नाही",
        "gu": "સ્થાનિક સૂચિમાં '{query}' સાથે મેળ ખાતું કોઈ ઉત્પાદન નથી",
        "ta": "'{query}' உடன் பொருந்தும் தயாரிப்புகள் எதுவும் இல்லை",
        "es": "No hay productos que coincidan con '{query}'",
    },
    "ai_analyze_btn": {
        "en": "✨ Analyze '{query}' with Groq AI & Save to MongoDB",
        "hi": "✨ Groq AI से '{query}' का तुरंत विश्लेषण करें और MongoDB में सहेजें",
        "mr": "✨ Groq AI द्वारे '{query}' चे त्वरित विश्लेषण करा आणि MongoDB मध्ये जतन करा",
        "gu": "✨ Groq AI થી '{query}' નું વિશ્લેષણ કરો અને MongoDB માં સાચવો",
        "ta": "✨ Groq AI மூலம் '{query}' ஐ பகுப்பாய்வு செய்து சேமிக்கவும்",
        "es": "✨ Analizar '{query}' con IA Groq y Guardar en MongoDB",
    },
    "scan_now": {
        "en": "🔬 Scan Now",
        "hi": "🔬 अभी स्कैन करें",
        "mr": "🔬 आता स्कॅन करा",
        "gu": "🔬 હમણાં સ્કેન કરો",
        "ta": "🔬 இப்போது ஸ்கேன் செய்",
        "es": "🔬 Escanear Ahora",
    },
    "scan_another": {
        "en": "🔄 Scan Another Product",
        "hi": "🔄 दूसरा उत्पाद स्कैन करें",
        "mr": "🔄 दुसरे उत्पादन स्कॅन करा",
        "gu": "🔄 બીજું ઉત્પાદન સ્કેન કરો",
        "ta": "🔄 மற்றொரு தயாரிப்பை ஸ்கேன் செய்",
        "es": "🔄 Escanear Otro Producto",
    },
    "camera_instruction": {
        "en": "Point your camera at a nutrition label, barcode, or product packaging.",
        "hi": "अपने कैमरे को पोषण लेबल, बारकोड या उत्पाद पैकेजिंग पर रखें।",
        "mr": "तुमचा कॅमेरा पोषण लेबल, बारकोड किंवा उत्पादन पॅकिंगसमोर धरा.",
        "gu": "તમારો કેમેરો ન્યુટ્રિશન લેબલ અથવા બારકોડ તરફ રાખો.",
        "ta": "ஊட்டச்சத்து லேபிளை கேமராவில் காட்டவும்.",
        "es": "Apunte su cámara a la etiqueta nutricional o código de barras.",
    },
    "take_photo": {
        "en": "📸 Capture Photo",
        "hi": "📸 फोटो खींचें",
        "mr": "📸 फोटो काढा",
        "gu": "📸 ફોટો લો",
        "ta": "📸 புகைப்படம் எடுக்கவும்",
        "es": "📸 Capturar Foto",
    },
    "select_lang": {
        "en": "🌐 Language",
        "hi": "🌐 भाषा (Language)",
        "mr": "🌐 भाषा (Language)",
        "gu": "🌐 ભાષા (Language)",
        "ta": "🌐 மொழி (Language)",
        "es": "🌐 Idioma (Language)",
    },
    "analyzing_ai": {
        "en": "🤖 AI analyzing product nutrition...",
        "hi": "🤖 एआई पोषण विश्लेषण कर रहा है...",
        "mr": "🤖 एआय पोषण घटकांचे विश्लेषण करत आहे...",
        "gu": "🤖 એઆઈ વિશ્લેષણ કરી રહ્યું છે...",
        "ta": "🤖 AI பகுப்பாய்வு செய்கிறது...",
        "es": "🤖 IA analizando nutrición del producto...",
    },
    "processing_level": {
        "en": "Processing Level",
        "hi": "प्रसंस्करण स्तर",
        "mr": "प्रक्रिया पातळी",
        "gu": "પ્રોસેસિંગ સ્તર",
        "ta": "செயலாக்க நிலை",
        "es": "Nivel de Procesamiento",
    },
    "brand": {
        "en": "Brand",
        "hi": "ब्रांड",
        "mr": "ब्रँड",
        "gu": "બ્રાન્ડ",
        "ta": "பிராண்ட்",
        "es": "Marca",
    },
    "category": {
        "en": "Category",
        "hi": "श्रेणी",
        "mr": "वर्ग / श्रेणी",
        "gu": "શ્રેણી",
        "ta": "வகை",
        "es": "Categoría",
    },
}

def t(key: str, lang: str = "en", **kwargs) -> str:
    """Retrieve translated string for key in specified language."""
    if key not in TRANSLATIONS:
        return key
    mapping = TRANSLATIONS[key]
    val = mapping.get(lang, mapping.get("en", key))
    if kwargs:
        try:
            return val.format(**kwargs)
        except Exception:
            return val
    return val

def get_lang_code(selected_label: str) -> str:
    """Map display label (e.g. '🇮🇳 मराठी (Marathi)') to lang code ('mr')."""
    for code, label in LANGUAGES.items():
        if label == selected_label or code == selected_label:
            return code
    return "en"
