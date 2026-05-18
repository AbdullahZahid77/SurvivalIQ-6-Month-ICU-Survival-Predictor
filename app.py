import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SurvivalIQ · 6-Month ICU Survival Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* font & base */
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

  /* sidebar */
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stSlider > div > div > div { background: #334155; }
  [data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.78rem; }

  /* metric cards */
  .metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    border: 1px solid #f1f5f9;
    text-align: center;
  }
  .metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { font-size: 2.2rem; font-weight: 700; color: #0f172a; margin: 4px 0; }
  .metric-sub   { font-size: 0.85rem; color: #94a3b8; }

  /* risk banner */
  .risk-low    { background:#dcfce7; border-left:5px solid #16a34a; border-radius:8px; padding:14px 20px; }
  .risk-moderate { background:#fef9c3; border-left:5px solid #ca8a04; border-radius:8px; padding:14px 20px; }
  .risk-high   { background:#fee2e2; border-left:5px solid #dc2626; border-radius:8px; padding:14px 20px; }
  .risk-critical { background:#fce7f3; border-left:5px solid #9d174d; border-radius:8px; padding:14px 20px; }
   .risk-low,
   .risk-moderate,
   .risk-high,
   .risk-critical {
    color: #000000;
}

  /* section headings */
  .section-head { font-size:0.65rem; font-weight:700; text-transform:uppercase;
                  letter-spacing:.12em; color:#94a3b8; margin:18px 0 6px; }

  /* disclaimer */
  .disclaimer { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
                padding:12px 16px; font-size:0.75rem; color:#94a3b8; margin-top:20px; color:#000000}
   

  div[data-testid="stExpander"] { border:1px solid #e2e8f0; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model_HistGradientBoosting_untuned.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERR = str(e)

# ── Feature metadata ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    "age", "num.co", "edu", "scoma", "charges", "totmcst", "avtisst", "aps",
    "hday", "meanbp", "wblc", "hrt", "resp", "temp", "pafi", "alb", "bili",
    "crea", "sod", "ph", "glucose", "bun", "urine", "adlsc", "income", "ca",
    "sex", "diabetes", "dementia", "dzclass_COPD/CHF/Cirrhosis",
    "dzclass_Coma", "race_hispanic", "race_other", "race_white",
    "dnr_dnr before sadm",
]

# typical healthy-ish critical patient defaults
DEFAULTS = dict(
    age=65, num_co=2, edu=12, scoma=0, charges=15000, totmcst=8000,
    avtisst=15, aps=60, hday=5, meanbp=80, wblc=10.0, hrt=90,
    resp=18, temp=37.0, pafi=300, alb=3.2, bili=1.0, crea=1.2,
    sod=140, ph=7.4, glucose=120, bun=20, urine=1000, adlsc=6,
)

INCOME_MAP  = {0: "< $11k", 1: "$11–25k", 2: "$25–50k", 3: "> $50k"}
CA_MAP      = {0: "No cancer", 1: "Cancer – not metastatic", 2: "Cancer – metastatic"}
RACE_LABELS = ["White", "Black / other", "Hispanic", "Other (non-Hispanic)"]

def risk_label(p: float):
    if p >= 0.75: return "Low Risk",    "risk-low",      "🟢"
    if p >= 0.50: return "Moderate Risk","risk-moderate", "🟡"
    if p >= 0.25: return "High Risk",   "risk-high",     "🔴"
    return "Critical Risk", "risk-critical", "🚨"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 SurvivalIQ")
    st.markdown("*6-Month ICU Survival Predictor*")
    st.markdown("---")

    # ── Demographics ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">👤 Demographics</div>', unsafe_allow_html=True)
    age    = st.slider("Age (years)", 18, 100, DEFAULTS["age"])
    sex    = st.radio("Sex", ["Female", "Male"], horizontal=True)
    edu    = st.slider("Education (years)", 0, 20, DEFAULTS["edu"])

    race_choice = st.selectbox("Race / Ethnicity", RACE_LABELS)
    race_white    = int(race_choice == "White")
    race_hispanic = int(race_choice == "Hispanic")
    race_other    = int(race_choice in ("Black / other", "Other (non-Hispanic)") and race_choice != "White")

    income = st.select_slider(
        "Household Income",
        options=[0, 1, 2, 3],
        value=1,
        format_func=lambda x: INCOME_MAP[x],
    )

    # ── Diagnoses ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">🏥 Diagnosis & Comorbidities</div>', unsafe_allow_html=True)
    ca       = st.select_slider("Cancer status", options=[0,1,2], value=0, format_func=lambda x: CA_MAP[x])
    num_co   = st.slider("# of Comorbidities", 0, 9, DEFAULTS["num_co"])
    diabetes = st.checkbox("Diabetes")
    dementia = st.checkbox("Dementia / cognitive impairment")

    dz_copd_chf = st.checkbox("Disease class: COPD / CHF / Cirrhosis")
    dz_coma     = st.checkbox("Disease class: Coma")

    dnr = st.checkbox("DNR order before study admission")

    # ── Severity Scores ───────────────────────────────────────────────────────
    st.markdown('<div class="section-head">📊 Severity Scores</div>', unsafe_allow_html=True)
    aps    = st.slider("APACHE III Score (APS)", 0, 299, DEFAULTS["aps"],
                       help="Higher = sicker. Normal ICU patient: 40–80.")
    scoma  = st.slider("SUPPORT Coma Score", 0, 100, DEFAULTS["scoma"],
                       help="0 = alert; higher = deeper coma.")
    avtisst = st.slider("TISS-28 Score (Nursing workload)", 0, 76, DEFAULTS["avtisst"])
    adlsc  = st.slider("ADL Score (Activities of Daily Living)", 0, 7, DEFAULTS["adlsc"],
                       help="Higher = more independent before illness.")

    # ── Vital Signs ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">💓 Vital Signs</div>', unsafe_allow_html=True)
    meanbp = st.slider("Mean Blood Pressure (mmHg)", 20, 200, DEFAULTS["meanbp"])
    hrt    = st.slider("Heart Rate (bpm)", 20, 250, DEFAULTS["hrt"])
    resp   = st.slider("Respiratory Rate (breaths/min)", 0, 80, DEFAULTS["resp"])
    temp   = st.slider("Temperature (°C)", 33.0, 42.0, DEFAULTS["temp"], step=0.1)

    # ── Lab Values ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">🧪 Laboratory Values</div>', unsafe_allow_html=True)
    pafi  = st.slider("PaO₂/FiO₂ Ratio", 0, 600, DEFAULTS["pafi"],
                      help="< 200 = moderate ARDS. > 300 = normal oxygenation.")
    wblc  = st.slider("WBC Count (×10³/mm³)", 0.1, 50.0, DEFAULTS["wblc"], step=0.1)
    alb   = st.slider("Albumin (g/dL)", 0.5, 6.0, DEFAULTS["alb"], step=0.1)
    bili  = st.slider("Bilirubin (mg/dL)", 0.1, 30.0, DEFAULTS["bili"], step=0.1)
    crea  = st.slider("Creatinine (mg/dL)", 0.1, 20.0, DEFAULTS["crea"], step=0.1)
    sod   = st.slider("Sodium (mEq/L)", 110, 175, DEFAULTS["sod"])
    ph    = st.slider("Arterial pH", 6.8, 7.8, DEFAULTS["ph"], step=0.01)
    glucose = st.slider("Glucose (mg/dL)", 30, 600, DEFAULTS["glucose"])
    bun   = st.slider("BUN (mg/dL)", 0, 250, DEFAULTS["bun"])
    urine = st.slider("Urine Output (mL/day)", 0, 9000, DEFAULTS["urine"], step=50)

    # ── Hospital Stay ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">🏨 Hospital Info</div>', unsafe_allow_html=True)
    hday    = st.slider("Hospital Day at Enrollment", 1, 60, DEFAULTS["hday"])
    charges = st.number_input("Hospital Charges ($)", 0, 500_000, DEFAULTS["charges"], step=1000)
    totmcst = st.number_input("Total Costs ($)", 0, 300_000, DEFAULTS["totmcst"], step=500)

# ── Build feature row ─────────────────────────────────────────────────────────
def build_row():
    return pd.DataFrame([[
        age, num_co, edu, scoma, charges, totmcst, avtisst, aps, hday,
        meanbp, wblc, hrt, resp, temp, pafi, alb, bili, crea, sod, ph,
        glucose, bun, urine, adlsc, income, ca,
        int(sex == "Male"),          # sex
        int(diabetes), int(dementia),
        int(dz_copd_chf), int(dz_coma),
        race_hispanic, race_other, race_white,
        int(dnr),
    ]], columns=FEATURE_COLS)

# ── Main layout ───────────────────────────────────────────────────────────────
st.title("🫀 SurvivalIQ — 6-Month ICU Survival Predictor")
st.markdown(
    "Powered by a **HistGradientBoosting** model trained on the "
    "[SUPPORT2 dataset](https://hbiostat.org/data/). "
    "Adjust patient parameters in the sidebar to see the predicted 6-month survival probability."
)
st.divider()

if not MODEL_LOADED:
    st.error(f"❌ Model file not found.\n\n"
             f"Make sure `model_HistGradientBoosting_untuned.joblib` is in the same folder as `app.py`.\n\n"
             f"Error: `{MODEL_ERR}`")
    st.stop()

# ── Predict ──────────────────────────────────────────────────────────────────
row  = build_row()
pred = float(np.clip(model.predict(row)[0], 0.0, 0.948))

label, css_class, emoji = risk_label(pred)
death_risk = round((1 - pred) * 100, 1)
surv_pct   = round(pred * 100, 1)

# ── Top KPI strip ─────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">6-Month Survival</div>
      <div class="metric-value" style="color:{'#16a34a' if pred>=0.5 else '#dc2626'}">{surv_pct}%</div>
      <div class="metric-sub">predicted probability</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Mortality Risk</div>
      <div class="metric-value" style="color:{'#dc2626' if death_risk>=50 else '#64748b'}">{death_risk}%</div>
      <div class="metric-sub">1 – survival probability</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Risk Tier {emoji}</div>
      <div class="metric-value" style="font-size:1.4rem">{label}</div>
      <div class="metric-sub">based on survival cutoffs</div>
    </div>""", unsafe_allow_html=True)
with col4:
    aps_tier = "Severe" if aps > 100 else ("Moderate" if aps > 60 else "Mild")
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">APACHE III Severity</div>
      <div class="metric-value" style="font-size:1.4rem">{aps_tier}</div>
      <div class="metric-sub">APS = {aps}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Gauge + Interpretation ────────────────────────────────────────────────────
left, right = st.columns([1, 1])

with left:
    st.subheader("Survival Gauge")
    needle_angle = pred  # 0 → 1

    gauge_color = (
        "#16a34a" if pred >= 0.75 else
        "#ca8a04" if pred >= 0.50 else
        "#dc2626" if pred >= 0.25 else
        "#9d174d"
    )

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(pred * 100, 1),
        delta={"reference": 50, "suffix": "%", "relative": False},
        number={"suffix": "%", "font": {"size": 44, "color": gauge_color}},
        title={"text": "6-Month Survival Probability", "font": {"size": 14, "color": "#000000"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 11, "color": "black"}},
            "bar": {"color": gauge_color, "thickness": 0.25},
            "steps": [
                {"range": [0, 25],  "color": "#fee2e2"},
                {"range": [25, 50], "color": "#fef9c3"},
                {"range": [50, 75], "color": "#dcfce7"},
                {"range": [75, 100],"color": "#bbf7d0"},
            ],
            "threshold": {
                "line": {"color": "#0f172a", "width": 3},
                "thickness": 0.8,
                "value": round(pred * 100, 1),
            },
        },
    ))
    fig_gauge.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="white",
        font_family="Inter, Segoe UI, sans-serif",
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with right:
    st.subheader("Clinical Interpretation")
    interp_messages = {
        "risk-low": (
            "**This patient profile suggests a favourable prognosis.** "
            "A survival probability ≥ 75% indicates the patient is likely to survive past 6 months. "
            "Continue standard monitoring and supportive care. "
            "Review and optimise modifiable factors such as nutrition (albumin), fluid balance, and infection control."
        ),
        "risk-moderate": (
            "**Moderate risk — close observation recommended.** "
            "A survival probability of 50–75% places the patient in a zone where clinical decisions "
            "can meaningfully shift outcomes. Consider reviewing APACHE score drivers, optimising "
            "organ support, and having early goals-of-care conversations."
        ),
        "risk-high": (
            "**High risk — active intervention or goals-of-care discussion warranted.** "
            "A survival probability of 25–50% signals significant mortality risk. "
            "Evaluate reversible contributors (sepsis, fluid overload, metabolic derangements). "
            "Multidisciplinary team review and palliative care consultation are advisable."
        ),
        "risk-critical": (
            "**Critical prognosis — < 25% survival probability.** "
            "The model estimates a very high likelihood of death within 6 months. "
            "Urgent goals-of-care conversation with patient / family is strongly recommended. "
            "Comfort-focused care should be considered if consistent with patient wishes."
        ),
        
    }
    
    st.markdown(f'<div class="{css_class}">{interp_messages[css_class]}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Key driver summary from inputs
    st.markdown("**Patient snapshot**")
    flags = []
    if aps > 100:  flags.append(("⚠️", "Severe APACHE III score", "red"))
    if scoma > 20: flags.append(("⚠️", f"Coma score elevated ({scoma})", "red"))
    if alb < 2.5:  flags.append(("⚠️", f"Hypoalbuminaemia (alb={alb})", "orange"))
    if pafi < 200: flags.append(("⚠️", f"Impaired oxygenation (P/F={pafi})", "orange"))
    if crea > 3.0: flags.append(("⚠️", f"Renal impairment (creatinine={crea})", "orange"))
    if bili > 5.0: flags.append(("⚠️", f"Elevated bilirubin ({bili})", "orange"))
    if ca == 2:    flags.append(("🔴", "Metastatic cancer", "red"))
    if dnr:        flags.append(("📋", "DNR order in place", "gray"))
    if meanbp < 60:flags.append(("⚠️", f"Hypotension (MAP={meanbp})", "red"))
    if adlsc >= 6: flags.append(("✅", "Good pre-illness function (ADL)", "green"))
    if pred >= 0.75: flags.append(("✅", "Favourable survival estimate", "green"))

    if flags:
        for icon, msg, color in flags:
            color_map = {"red": "#dc2626", "orange": "#ca8a04", "green": "#16a34a", "gray": "#94a3b8"}
            st.markdown(f"<span style='color:{color_map[color]}'>{icon} {msg}</span>", unsafe_allow_html=True)
    else:
        st.markdown("*No major clinical flags detected.*")

st.divider()

# ── Feature Input Summary Table ───────────────────────────────────────────────
with st.expander("📋 Full Patient Input Summary", expanded=False):
    summary = pd.DataFrame({
        "Feature": [
            "Age", "Sex", "Education (yrs)", "Income bracket", "Race/Ethnicity",
            "# Comorbidities", "Cancer status", "Diabetes", "Dementia", "DNR",
            "Disease class COPD/CHF/Cirrhosis", "Disease class Coma",
            "APACHE III (APS)", "Coma score", "TISS-28", "ADL score",
            "Mean BP (mmHg)", "Heart rate", "Resp. rate", "Temperature (°C)",
            "PaO₂/FiO₂", "WBC (×10³)", "Albumin", "Bilirubin", "Creatinine",
            "Sodium", "pH", "Glucose", "BUN", "Urine (mL/day)",
            "Hospital day", "Charges ($)", "Total costs ($)",
        ],
        "Value": [
            age, sex, edu, INCOME_MAP[income], race_choice,
            num_co, CA_MAP[ca], "Yes" if diabetes else "No",
            "Yes" if dementia else "No", "Yes" if dnr else "No",
            "Yes" if dz_copd_chf else "No", "Yes" if dz_coma else "No",
            aps, scoma, avtisst, adlsc,
            meanbp, hrt, resp, temp,
            pafi, wblc, alb, bili, crea,
            sod, ph, glucose, bun, urine,
            hday, f"${charges:,}", f"${totmcst:,}",
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ── What-If Sensitivity Analysis ──────────────────────────────────────────────
st.subheader("🔬 What-If Sensitivity Analysis")
st.markdown("See how survival probability changes when you vary a single parameter, holding everything else fixed.")

sens_col, chart_col = st.columns([1, 2])

SENS_FEATURES = {
    "APACHE III Score": ("aps", list(range(0, 300, 10))),
    "Age": ("age", list(range(20, 101, 5))),
    "Albumin (g/dL)": ("alb", [round(v * 0.2, 1) for v in range(3, 31)]),
    "PaO₂/FiO₂ Ratio": ("pafi", list(range(50, 601, 25))),
    "Creatinine (mg/dL)": ("crea", [round(v * 0.5, 1) for v in range(1, 41)]),
    "Mean Blood Pressure": ("meanbp", list(range(30, 161, 5))),
    "Bilirubin (mg/dL)": ("bili", [round(v * 0.5, 1) for v in range(1, 61)]),
    "BUN (mg/dL)": ("bun", list(range(0, 201, 10))),
    "Coma Score": ("scoma", list(range(0, 101, 5))),
    "TISS-28 Score": ("avtisst", list(range(0, 77, 3))),
}

with sens_col:
    chosen_label = st.selectbox("Choose parameter to vary:", list(SENS_FEATURES.keys()))
    sens_key, sens_range = SENS_FEATURES[chosen_label]

    base_row = build_row().copy()
    sens_preds = []
    for val in sens_range:
        r = base_row.copy()
        r[sens_key] = val
        p = float(np.clip(model.predict(r)[0], 0.0, 0.948))
        sens_preds.append({"x": val, "survival": round(p * 100, 1)})

    current_val = float(base_row[sens_key].values[0])
    st.metric(f"Current {chosen_label}", f"{current_val}")

with chart_col:
    sens_df = pd.DataFrame(sens_preds)
    fig_line = px.line(
        sens_df, x="x", y="survival",
        labels={"x": chosen_label, "survival": "6-Month Survival (%)"},
        color_discrete_sequence=["#3b82f6"],
    )
    # add vertical line for current value
    fig_line.add_vline(
        x=current_val, line_dash="dash", line_color="#f97316",
        annotation_text=f"Current: {current_val}",
        annotation_position="top right",
    )
    fig_line.add_hrect(y0=0, y1=25,  fillcolor="#fee2e2", opacity=0.3, line_width=0)
    fig_line.add_hrect(y0=25, y1=50, fillcolor="#fef9c3", opacity=0.3, line_width=0)
    fig_line.add_hrect(y0=50, y1=75, fillcolor="#dcfce7", opacity=0.3, line_width=0)
    fig_line.add_hrect(y0=75, y1=100,fillcolor="#bbf7d0", opacity=0.3, line_width=0)
    # fig_line.update_layout(
    #     height=280,
    #     margin=dict(l=10, r=10, t=30, b=10),
    #     paper_bgcolor="white",
    #     plot_bgcolor="white",
    #     yaxis=dict(range=[0, 100], gridcolor="#000000"),
    #     xaxis=dict(gridcolor="#000000"),
    #     # font_family="Inter, Segoe UI, sans-serif",
    #     font=dict(
    #     color="black",   # 👈 THIS fixes ALL remaining text
    #     family="Inter, Segoe UI, sans-serif"
    # )
    # )
    fig_line.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            range=[0, 100], 
            gridcolor="#e2e8f0",        # Changed to a subtle gray so it doesn't clash with black text
            showline=True,              # Ensures the axis line is visible
            linecolor="black",          # Makes the Y-axis line black
            tickcolor="black",          # Makes the tick marks black
            tickfont=dict(color="black") # Makes the numbers (scale) black
        ),
        xaxis=dict(
            gridcolor="#e2e8f0",        # Subtle gray grid lines
            showline=True,              # Ensures the axis line is visible
            linecolor="black",          # Makes the X-axis line black
            tickcolor="black",          # Makes the tick marks black
            tickfont=dict(color="black") # Makes the numbers/categories black
        ),
        font=dict(
            color="black",              # Fixes axis titles and global text
            family="Inter, Segoe UI, sans-serif"
        )
    )
    st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# ── Model info ────────────────────────────────────────────────────────────────
with st.expander("🤖 About the Model"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "HistGradientBoosting")
    c2.metric("Training Iterations", "400")
    c3.metric("Learning Rate", "0.05")
    c1.metric("Dataset", "SUPPORT2")
    c2.metric("Input Features", "35")
    c3.metric("Target", "surv6m ∈ [0, 0.948]")
    st.markdown("""
    **SUPPORT2** (Study to Understand Prognoses, Preferences, Outcomes and Risks of Treatment) is a 
    landmark prospective cohort study of seriously ill hospitalised adults.  
    The model was trained to predict the **6-month survival probability** directly as a continuous outcome.

    | Metric | Value |
    |---|---|
    | Baseline (XGBoost, Part 1) RMSE | 0.1086 |
    | Baseline R² | 0.817 |
    | Model type | sklearn Pipeline → ColumnTransformer (passthrough) → HistGradientBoostingRegressor |
    | sklearn version | 1.6.1 |

    Predictions are clipped to `[0.0, 0.948]` matching the training target range.
    """)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚕️ <strong>Clinical Disclaimer:</strong> This tool is for educational and research purposes only.
It is <em>not</em> a validated clinical decision-support tool and must not be used to guide treatment
decisions. Predictions are based on a retrospective dataset and a non-tuned model.
Always defer to qualified clinical judgment.
</div>
""", unsafe_allow_html=True)