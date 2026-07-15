# ============================================================
# APLIKASI GUI - KLASIFIKASI STATUS GIZI (OBESITAS)
# NIM   : 23071023
# Prodi : Sistem Informasi - Universitas Hang Tuah Pekanbaru
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shap
import os

st.set_page_config(
    page_title="SiGizi — Klasifikasi Status Gizi",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp { background: #ffffff; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8f5e9;
}
[data-testid="stSidebar"] * { color: #1b5e20 !important; }
[data-testid="stSidebar"] .stRadio label {
    color: #2e7d32 !important;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 0.3rem 0;
}
[data-testid="stSidebar"] hr { border-color: #e8f5e9 !important; }

/* ── Logo sidebar ── */
.sb-logo {
    background: linear-gradient(135deg, #2e7d32, #43a047);
    border-radius: 14px;
    padding: 1.2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.sb-logo .ico { font-size: 2.2rem; }
.sb-logo .nm  { font-size: 1.1rem; font-weight: 800; color: white !important; margin-top: 0.3rem; }
.sb-logo .sub { font-size: 0.72rem; color: rgba(255,255,255,0.8) !important; }

.sb-info { font-size: 0.78rem; color: #4caf50 !important; line-height: 2; }
.sb-info b { color: #1b5e20 !important; font-weight: 600; }

/* ── Page header ── */
.page-header {
    border-left: 5px solid #2e7d32;
    padding: 0.5rem 0 0.5rem 1rem;
    margin-bottom: 1.8rem;
}
.page-header h1 {
    font-size: 1.5rem; font-weight: 800;
    color: #1b5e20; margin: 0;
}
.page-header p {
    font-size: 0.85rem; color: #66bb6a; margin: 0.2rem 0 0 0;
}

/* ── Hero banner (beranda) ── */
.hero {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 40%, #43a047 100%);
    border-radius: 18px;
    padding: 2.2rem 2.5rem;
    display: flex; align-items: center; gap: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(46,125,50,0.18);
}
.hero-icon { font-size: 3.5rem; }
.hero h1   { font-size: 1.8rem; font-weight: 800; color: white; margin: 0; }
.hero p    { font-size: 0.9rem; color: rgba(255,255,255,0.85); margin: 0.3rem 0 0 0; max-width: 600px; }

/* ── Stat cards ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.stat {
    flex: 1; background: white;
    border: 1px solid #e8f5e9;
    border-top: 3px solid #2e7d32;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.stat-n { font-size: 1.8rem; font-weight: 800; color: #2e7d32; }
.stat-l { font-size: 0.78rem; color: #757575; font-weight: 500; margin-top: 0.1rem; }

/* ── Content cards ── */
.card {
    background: white;
    border: 1px solid #f1f8e9;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.card-title {
    font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: #2e7d32;
    border-bottom: 2px solid #e8f5e9;
    padding-bottom: 0.5rem; margin-bottom: 1rem;
}

/* ── Input section labels ── */
.inp-label {
    display: flex; align-items: center; gap: 0.5rem;
    background: #f9fbe7;
    border-left: 4px solid #43a047;
    border-radius: 0 8px 8px 0;
    padding: 0.55rem 1rem;
    font-size: 0.82rem; font-weight: 700; color: #33691e;
    margin-bottom: 1rem;
}

/* ── BMI display ── */
.bmi-card {
    border-radius: 12px; padding: 1rem 1.2rem;
    border: 1.5px solid; margin-top: 0.8rem;
}
.bmi-val { font-size: 2.2rem; font-weight: 800; margin: 0.2rem 0; }
.bmi-lbl { font-size: 0.8rem; font-weight: 600; }

/* ── Result boxes ── */
.res-yes {
    background: #fff3e0; border: 2px solid #ffb74d;
    border-radius: 16px; padding: 1.5rem 2rem; text-align: center;
}
.res-no {
    background: #e8f5e9; border: 2px solid #81c784;
    border-radius: 16px; padding: 1.5rem 2rem; text-align: center;
}
.res-icon { font-size: 2.5rem; margin-bottom: 0.4rem; }
.res-title { font-size: 1.25rem; font-weight: 800; }
.res-sub   { font-size: 0.82rem; margin-top: 0.3rem; opacity: 0.85; }

/* ── Prob cards ── */
.prob-card {
    background: white; border: 1px solid #e8f5e9;
    border-radius: 12px; padding: 1rem 1.2rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.prob-n { font-size: 1.7rem; font-weight: 800; }
.prob-l { font-size: 0.78rem; color: #9e9e9e; font-weight: 500; }

/* ── Tables ── */
.gtbl { width:100%; border-collapse:collapse; font-size:0.85rem; }
.gtbl th {
    background: #2e7d32; color: white;
    padding: 0.6rem 0.9rem; text-align:left; font-weight:600;
}
.gtbl td { padding: 0.55rem 0.9rem; border-bottom:1px solid #f1f8e9; color: #424242; }
.gtbl tr:last-child td { border-bottom: none; }
.gtbl tr:hover td { background: #f9fbe7; }
.gtbl tr.best td { background: #e8f5e9; font-weight: 600; }

/* ── Tags ── */
.tag { display:inline-block; padding:0.18rem 0.65rem; border-radius:999px; font-size:0.73rem; font-weight:600; margin:1px; }
.tg  { background:#e8f5e9; color:#1b5e20; }
.tb  { background:#e3f2fd; color:#0d47a1; }
.ty  { background:#fff8e1; color:#e65100; }

/* ── Button ── */
.stButton > button {
    background: #2e7d32 !important;
    color: white !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.7rem 2rem !important; font-size: 0.95rem !important;
    box-shadow: 0 3px 14px rgba(46,125,50,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #1b5e20 !important;
    box-shadow: 0 5px 18px rgba(46,125,50,0.4) !important;
}

/* ── Streamlit overrides ── */
.stSlider [data-baseweb="slider"] { }
div[data-testid="stMetric"] { background: white; border-radius: 12px; padding: 0.8rem; border: 1px solid #e8f5e9; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'output')

@st.cache_resource
def load_artifacts():
    rf    = joblib.load(f'{MODEL_DIR}/random_forest_model.joblib')
    xgb   = joblib.load(f'{MODEL_DIR}/xgboost_optimized_model.joblib')
    sc    = joblib.load(f'{MODEL_DIR}/scaler.joblib')
    feats = joblib.load(f'{MODEL_DIR}/feature_names.joblib')
    encs  = joblib.load(f'{MODEL_DIR}/label_encoders.joblib')
    return rf, xgb, sc, feats, encs

rf_model, xgb_model, scaler, feature_names, label_encoders = load_artifacts()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class='sb-logo'>
        <div class='ico'>🌿</div>
        <div class='nm'>SiGizi</div>
        <div class='sub'>Sistem Klasifikasi Status Gizi</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='sb-info'>
        <b>Program Studi</b><br>Sistem Informasi<br>
        <b>Institusi</b><br>Universitas Hang Tuah Pekanbaru<br>
        <b>Mata Kuliah</b><br>Data Science<br>
        <b>Dosen</b><br>Yuda Irawan, S.Kom, M.TI
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("Menu", [
        "🏠  Beranda",
        "🔍  Prediksi",
        "📊  Evaluasi Model",
        "🤖  Explainable AI"
    ], label_visibility="collapsed")

# ═══════════════════════════════════════════════════════════
# BERANDA
# ═══════════════════════════════════════════════════════════
if "Beranda" in page:

    st.markdown("""
    <div class='hero'>
        <div class='hero-icon'>🌿</div>
        <div>
            <h1>Sistem Klasifikasi Status Gizi</h1>
            <p>Prediksi risiko obesitas secara cerdas menggunakan Machine Learning
            berdasarkan data fisik, kebiasaan makan, dan gaya hidup.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    # Stat cards
    c1,c2,c3,c4 = st.columns(4)
    stats = [("2.111","📊 Total Sampel"),("16","🔢 Jumlah Fitur"),
             ("99.34%","🎯 Akurasi Model"),("2","🤖 Algoritma ML")]
    for col,(num,lbl) in zip([c1,c2,c3,c4],stats):
        with col:
            st.markdown(f"""<div class='stat'>
                <div class='stat-n'>{num}</div>
                <div class='stat-l'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1,1])

    with left:
        st.markdown("""<div class='card'>
        <div class='card-title'>📌 Tentang Sistem</div>
        <p style='font-size:0.87rem;color:#424242;line-height:1.8;margin:0;'>
        <b>SiGizi</b> adalah sistem klasifikasi status gizi berbasis <b>Machine Learning</b>
        yang dirancang untuk memprediksi risiko obesitas seseorang berdasarkan
        data fisik dan kebiasaan hidup sehari-hari.<br><br>
        Dataset bersumber dari <b>Kaggle — ObesityDataSet_raw_and_data_sinthetic</b>
        dengan 2.111 sampel dan 16 fitur input.
        </p></div>""", unsafe_allow_html=True)

        st.markdown("""<div class='card'>
        <div class='card-title'>🔬 Metodologi</div>
        <table class='gtbl'>
        <tr><th>Tahap</th><th>Teknik / Tools</th></tr>
        <tr><td>Preprocessing</td><td>
            <span class='tag tg'>Label Encoding</span>
            <span class='tag tb'>StandardScaler</span>
        </td></tr>
        <tr><td>Balancing Data</td><td><span class='tag ty'>SMOTE</span></td></tr>
        <tr><td>Algoritma 1</td><td><span class='tag tg'>Random Forest</span></td></tr>
        <tr><td>Algoritma 2</td><td><span class='tag tb'>XGBoost</span></td></tr>
        <tr><td>Optimasi</td><td><span class='tag ty'>Optuna — 30 Trials</span></td></tr>
        <tr><td>Explainability</td><td><span class='tag tg'>SHAP TreeExplainer</span></td></tr>
        <tr><td>Deployment</td><td><span class='tag tb'>Streamlit GUI</span></td></tr>
        </table></div>""", unsafe_allow_html=True)

    with right:
        st.markdown("""<div class='card'>
        <div class='card-title'>📋 Deskripsi Fitur Dataset</div>
        <table class='gtbl'>
        <tr><th>Fitur</th><th>Keterangan</th></tr>
        <tr><td><b>Gender</b></td><td>Jenis kelamin</td></tr>
        <tr><td><b>Age</b></td><td>Usia (tahun)</td></tr>
        <tr><td><b>Height</b></td><td>Tinggi badan (meter)</td></tr>
        <tr><td><b>Weight</b></td><td>Berat badan (kg)</td></tr>
        <tr><td><b>family_history</b></td><td>Riwayat keluarga obesitas</td></tr>
        <tr><td><b>FAVC</b></td><td>Sering makan tinggi kalori</td></tr>
        <tr><td><b>FCVC</b></td><td>Frekuensi konsumsi sayur</td></tr>
        <tr><td><b>NCP</b></td><td>Jumlah makan utama per hari</td></tr>
        <tr><td><b>CAEC</b></td><td>Kebiasaan ngemil</td></tr>
        <tr><td><b>SMOKE</b></td><td>Status merokok</td></tr>
        <tr><td><b>CH2O</b></td><td>Konsumsi air harian (liter)</td></tr>
        <tr><td><b>SCC</b></td><td>Memantau kalori harian</td></tr>
        <tr><td><b>FAF</b></td><td>Frekuensi aktivitas fisik (0–3)</td></tr>
        <tr><td><b>TUE</b></td><td>Waktu penggunaan gadget (jam)</td></tr>
        <tr><td><b>CALC</b></td><td>Konsumsi alkohol</td></tr>
        <tr><td><b>MTRANS</b></td><td>Moda transportasi</td></tr>
        </table></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PREDIKSI
# ═══════════════════════════════════════════════════════════
elif "Prediksi" in page:

    st.markdown("""<div class='page-header'>
        <h1>🔍 Prediksi Status Gizi</h1>
        <p>Isi seluruh data di bawah lalu klik tombol Analisis</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='inp-label'>🧍 Data Fisik</div>", unsafe_allow_html=True)
        gender  = st.selectbox("Jenis Kelamin", ["Male","Female"],
                               format_func=lambda x: "Laki-laki" if x=="Male" else "Perempuan")
        age     = st.slider("Usia (tahun)", 14, 65, 25)
        height  = st.slider("Tinggi Badan (m)", 1.40, 2.00, 1.65, step=0.01, format="%.2f")
        weight  = st.slider("Berat Badan (kg)", 35, 180, 70)
        fam     = st.selectbox("Riwayat Keluarga Obesitas", ["yes","no"],
                               format_func=lambda x: "Ada" if x=="yes" else "Tidak Ada")

        bmi = weight / (height**2)
        if   bmi < 18.5: bc,bl = "#1565c0","Kurus"
        elif bmi < 25.0: bc,bl = "#2e7d32","Normal"
        elif bmi < 30.0: bc,bl = "#e65100","Overweight"
        else:            bc,bl = "#b71c1c","Obesitas"

        st.markdown(f"""
        <div class='bmi-card' style='background:#fafafa;border-color:{bc};'>
            <div style='font-size:0.72rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:.05em;color:#757575;'>Indeks Massa Tubuh</div>
            <div class='bmi-val' style='color:{bc};'>{bmi:.1f}</div>
            <div class='bmi-lbl' style='color:{bc};'>{bl}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='inp-label'>🍽️ Kebiasaan Makan</div>", unsafe_allow_html=True)
        favc = st.selectbox("Sering Makan Tinggi Kalori", ["yes","no"],
                            format_func=lambda x: "Ya" if x=="yes" else "Tidak")
        fcvc = st.slider("Frekuensi Konsumsi Sayur (1–3)", 1.0, 3.0, 2.0, step=0.1)
        ncp  = st.slider("Jumlah Makan Utama / Hari", 1.0, 4.0, 3.0, step=0.5)
        caec = st.selectbox("Kebiasaan Ngemil", ["no","Sometimes","Frequently","Always"],
                            format_func=lambda x:
                            {"no":"Tidak","Sometimes":"Kadang-kadang",
                             "Frequently":"Sering","Always":"Selalu"}[x])
        calc = st.selectbox("Konsumsi Alkohol", ["no","Sometimes","Frequently","Always"],
                            format_func=lambda x:
                            {"no":"Tidak","Sometimes":"Kadang-kadang",
                             "Frequently":"Sering","Always":"Selalu"}[x])

    with col3:
        st.markdown("<div class='inp-label'>🏃 Gaya Hidup</div>", unsafe_allow_html=True)
        smoke  = st.selectbox("Merokok", ["yes","no"],
                              format_func=lambda x: "Ya" if x=="yes" else "Tidak")
        ch2o   = st.slider("Konsumsi Air (liter/hari)", 1.0, 3.0, 2.0, step=0.1)
        scc    = st.selectbox("Memantau Kalori Harian", ["yes","no"],
                              format_func=lambda x: "Ya" if x=="yes" else "Tidak")
        faf    = st.slider("Frekuensi Aktivitas Fisik (0–3)", 0.0, 3.0, 1.0, step=0.5)
        tue    = st.slider("Waktu Pemakaian Gadget (jam)", 0.0, 2.0, 1.0, step=0.5)
        mtrans = st.selectbox("Moda Transportasi",
                              ["Automobile","Motorbike","Bike",
                               "Public_Transportation","Walking"],
                              format_func=lambda x:
                              {"Automobile":"Mobil","Motorbike":"Motor","Bike":"Sepeda",
                               "Public_Transportation":"Transportasi Umum",
                               "Walking":"Jalan Kaki"}[x])

    st.markdown("<br>", unsafe_allow_html=True)
    mc = st.radio("Model Prediksi",
                  ["🌲  Random Forest","⚡  XGBoost + Optuna"], horizontal=True)

    if st.button("🔍  Analisis Status Gizi Sekarang"):
        raw = {'Gender':gender,'Age':age,'Height':height,'Weight':weight,
               'family_history_with_overweight':fam,'FAVC':favc,
               'FCVC':fcvc,'NCP':ncp,'CAEC':caec,'SMOKE':smoke,
               'CH2O':ch2o,'SCC':scc,'FAF':faf,'TUE':tue,
               'CALC':calc,'MTRANS':mtrans}
        inp = pd.DataFrame([raw])
        for c in ['Gender','family_history_with_overweight','FAVC','CAEC',
                  'SMOKE','SCC','CALC','MTRANS']:
            inp[c] = label_encoders[c].transform(inp[c])
        inp   = inp[feature_names]
        inp_s = pd.DataFrame(scaler.transform(inp), columns=feature_names)

        model = rf_model if "Random" in mc else xgb_model
        pred  = model.predict(inp_s)[0]
        prob  = model.predict_proba(inp_s)[0]
        risk  = prob[1]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Hasil Analisis")

        r1, r2, r3 = st.columns([2,1,1])
        with r1:
            if pred == 1:
                st.markdown("""<div class='res-yes'>
                    <div class='res-icon'>⚠️</div>
                    <div class='res-title' style='color:#e65100;'>BERISIKO OBESITAS</div>
                    <div class='res-sub' style='color:#bf360c;'>
                        Segera konsultasikan dengan ahli gizi
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='res-no'>
                    <div class='res-icon'>✅</div>
                    <div class='res-title' style='color:#2e7d32;'>STATUS GIZI NORMAL</div>
                    <div class='res-sub' style='color:#388e3c;'>
                        Pertahankan pola makan dan gaya hidup sehat
                    </div>
                </div>""", unsafe_allow_html=True)

        with r2:
            clr = "#e65100" if risk > 0.5 else "#2e7d32"
            st.markdown(f"""<div class='prob-card' style='border-top:3px solid {clr};'>
                <div class='prob-n' style='color:{clr};'>{risk*100:.1f}%</div>
                <div class='prob-l'>Probabilitas Obesitas</div>
            </div>""", unsafe_allow_html=True)

        with r3:
            st.markdown(f"""<div class='prob-card' style='border-top:3px solid #2e7d32;'>
                <div class='prob-n' style='color:#2e7d32;'>{prob[0]*100:.1f}%</div>
                <div class='prob-l'>Probabilitas Normal</div>
            </div>""", unsafe_allow_html=True)

        lv  = "Rendah" if risk<0.3 else ("Sedang" if risk<0.6 else "Tinggi")
        lvc = "🟢" if risk<0.3 else ("🟡" if risk<0.6 else "🔴")
        st.markdown(f"<br><b>Tingkat Risiko: {lvc} {lv}</b>", unsafe_allow_html=True)
        st.progress(float(risk))

        # SHAP
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔬 Penjelasan Prediksi — SHAP")
        expl      = shap.TreeExplainer(model)
        shap_vals = expl.shap_values(inp_s)
        # Handle all SHAP output formats safely
        if isinstance(shap_vals, list):
            raw = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            raw = shap_vals
        shap_arr = np.array(raw).flatten()[:len(feature_names)]

        feat_list = list(feature_names)
        n = len(feat_list)
        pairs = sorted(zip(feat_list, shap_arr.tolist()), key=lambda x: abs(x[1]))
        s_names = [p[0] for p in pairs]
        s_vals  = [float(p[1]) for p in pairs]
        s_clrs  = ['#e53935' if v>0 else '#2e7d32' for v in s_vals]

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#fafafa')
        bars = ax.barh(s_names, s_vals, color=s_clrs,
                       edgecolor='white', height=0.55, linewidth=0.5)
        ax.axvline(0, color='#bdbdbd', linewidth=0.8, zorder=0)
        ax.set_xlabel("SHAP Value", fontsize=9, color='#616161')
        ax.set_title("Kontribusi Setiap Fitur terhadap Hasil Prediksi",
                     fontsize=11, fontweight='bold', color='#212121', pad=14)
        ax.tick_params(colors='#616161', labelsize=8.5)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.grid(axis='x', alpha=0.12, color='#bdbdbd', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("🔴 Merah = meningkatkan risiko obesitas   |   🟢 Hijau = menurunkan risiko")

# ═══════════════════════════════════════════════════════════
# EVALUASI MODEL
# ═══════════════════════════════════════════════════════════
elif "Evaluasi" in page:

    st.markdown("""<div class='page-header'>
        <h1>📊 Evaluasi Model</h1>
        <p>Perbandingan performa Random Forest, XGBoost, dan XGBoost + Optuna</p>
    </div>""", unsafe_allow_html=True)

    mp = f'{MODEL_DIR}/metrics_summary.csv'
    if os.path.exists(mp):
        df  = pd.read_csv(mp)
        best_f1 = df['F1-Score'].max()
        rows = ""
        for _, row in df.iterrows():
            is_best = row['F1-Score'] == best_f1
            cls     = "class='best'" if is_best else ""
            trophy  = " 🏆" if is_best else ""
            rows += f"""<tr {cls}>
                <td><b>{row['Model']}{trophy}</b></td>
                <td>{row['Accuracy']:.4f}</td>
                <td>{row['Precision']:.4f}</td>
                <td>{row['Recall']:.4f}</td>
                <td>{row['F1-Score']:.4f}</td>
                <td>{row['ROC-AUC']:.4f}</td>
            </tr>"""
        st.markdown(f"""<div class='card'>
        <div class='card-title'>📈 Tabel Perbandingan Metrik</div>
        <table class='gtbl'>
        <tr><th>Model</th><th>Accuracy</th><th>Precision</th>
            <th>Recall</th><th>F1-Score</th><th>ROC-AUC</th></tr>
        {rows}
        </table></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, fname, title in [
        (c1,'model_comparison.png','Grafik Perbandingan Metrik'),
        (c2,'roc_curve.png','ROC Curve')]:
        fp = f'{MODEL_DIR}/{fname}'
        if os.path.exists(fp):
            with col:
                st.markdown(f"<div class='card'><div class='card-title'>{title}</div>",
                            unsafe_allow_html=True)
                st.image(fp, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    cm = f'{MODEL_DIR}/confusion_matrices.png'
    if os.path.exists(cm):
        st.markdown("<div class='card'><div class='card-title'>🎯 Confusion Matrix</div>",
                    unsafe_allow_html=True)
        st.image(cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div class='card'>
    <div class='card-title'>📝 Analisis Hasil</div>
    <table class='gtbl'>
    <tr><th>Temuan</th><th>Penjelasan</th></tr>
    <tr><td>✅ Model Terbaik</td>
        <td>XGBoost + Optuna — F1-Score 99.34%, ROC-AUC 99.99%</td></tr>
    <tr><td>✅ SMOTE</td>
        <td>Berhasil menyeimbangkan distribusi kelas 61:39 → 50:50</td></tr>
    <tr><td>✅ Optuna</td>
        <td>Menemukan hyperparameter optimal dalam 30 trials otomatis</td></tr>
    <tr><td>✅ Precision 99.56%</td>
        <td>Hampir tidak ada kesalahan prediksi false positive</td></tr>
    </table></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# EXPLAINABLE AI
# ═══════════════════════════════════════════════════════════
elif "Explainable" in page:

    st.markdown("""<div class='page-header'>
        <h1>🤖 Explainable AI — SHAP</h1>
        <p>Transparansi dan interpretasi keputusan model Machine Learning</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class='card'>
    <div class='card-title'>❓ Apa itu SHAP?</div>
    <p style='font-size:0.87rem;color:#424242;line-height:1.8;margin:0;'>
    <b>SHAP (SHapley Additive exPlanations)</b> adalah metode <i>Explainable AI</i> yang menggunakan
    konsep nilai Shapley dari teori permainan untuk mengukur kontribusi setiap fitur terhadap
    hasil prediksi model. SHAP memberikan penjelasan yang <b>transparan, konsisten,
    dan dapat dipertanggungjawabkan secara matematis</b>.
    </p></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    for col, fname, title, cap in [
        (c1,'shap_summary.png','📊 SHAP Feature Importance (Bar)',
         'Panjang bar menunjukkan besarnya kontribusi fitur secara global'),
        (c2,'shap_beeswarm.png','🐝 SHAP Beeswarm Plot',
         'Merah = nilai tinggi · Biru = nilai rendah · Kanan = pengaruh positif terhadap obesitas')]:
        fp = f'{MODEL_DIR}/{fname}'
        if os.path.exists(fp):
            with col:
                st.markdown(f"<div class='card'><div class='card-title'>{title}</div>",
                            unsafe_allow_html=True)
                st.image(fp, use_container_width=True)
                st.caption(cap)
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div class='card'>
    <div class='card-title'>🔍 Interpretasi Fitur Paling Berpengaruh</div>
    <table class='gtbl'>
    <tr><th>#</th><th>Fitur</th><th>Pengaruh</th><th>Interpretasi Medis</th></tr>
    <tr><td><b>1</b></td><td><b>Weight</b></td>
        <td><span class='tag tg'>Sangat Tinggi</span></td>
        <td>Berat badan adalah penentu utama status gizi dan BMI</td></tr>
    <tr><td><b>2</b></td><td><b>Height</b></td>
        <td><span class='tag tg'>Sangat Tinggi</span></td>
        <td>Bersama Weight, membentuk nilai BMI secara langsung</td></tr>
    <tr><td><b>3</b></td><td><b>FAF (Aktivitas Fisik)</b></td>
        <td><span class='tag tb'>Tinggi</span></td>
        <td>Aktivitas fisik rendah meningkatkan risiko obesitas signifikan</td></tr>
    <tr><td><b>4</b></td><td><b>FAVC (Makanan Kalori Tinggi)</b></td>
        <td><span class='tag ty'>Sedang</span></td>
        <td>Konsumsi makanan tinggi kalori memperburuk status gizi</td></tr>
    <tr><td><b>5</b></td><td><b>family_history</b></td>
        <td><span class='tag ty'>Sedang</span></td>
        <td>Faktor genetik keluarga turut berkontribusi terhadap risiko</td></tr>
    </table></div>""", unsafe_allow_html=True)
