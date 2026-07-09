import streamlit as st
import pandas as pd
import joblib

# ---------------- Page Config ---------------- #
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Load Model ---------------- #
model = joblib.load("Logistic_Regression_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ---------------- Custom CSS ---------------- #
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg, #fff5f7 0%, #f0f4f8 100%);
}

/* Hide default streamlit chrome for a cleaner look */
#MainMenu, footer {visibility: hidden;}

.hero{
    text-align:center;
    padding: 1.5rem 1rem 1rem 1rem;
}

.hero h1{
    color:#c2185b;
    font-size:2.4rem;
    margin-bottom:0;
}

.hero p{
    color:#6b7280;
    font-size:1.05rem;
    margin-top:0.3rem;
}

.card{
    background:white;
    padding:1.6rem 1.8rem;
    border-radius:18px;
    box-shadow:0px 8px 24px rgba(0,0,0,0.06);
    margin-bottom:1.2rem;
    border: 1px solid rgba(0,0,0,0.04);
}

.section-title{
    font-size:1.15rem;
    font-weight:700;
    color:#374151;
    margin-bottom:0.8rem;
    border-left: 5px solid #d63384;
    padding-left:10px;
}

.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#dc3545,#c2185b);
    color:white;
    font-size:19px;
    font-weight:600;
    border-radius:14px;
    height:3.2em;
    border:none;
    transition: 0.2s ease-in-out;
}

.stButton>button:hover{
    transform: scale(1.01);
    box-shadow:0px 6px 16px rgba(220,53,69,0.35);
}

.result-card{
    padding:1.8rem;
    border-radius:18px;
    text-align:center;
    color:white;
    font-size:1.3rem;
    font-weight:700;
}

.risk-high{
    background:linear-gradient(135deg,#ff5f6d,#dc3545);
}

.risk-low{
    background:linear-gradient(135deg,#38ef7d,#28a745);
}

.footer-note{
    text-align:center;
    color:#9ca3af;
    font-size:0.85rem;
    margin-top:1.5rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ---------------- #
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.write(
        "This tool estimates the likelihood of heart disease "
        "using a Logistic Regression model trained on clinical data."
    )
    st.markdown("---")
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Library:** Scikit-Learn")
    st.markdown("---")
    st.caption("⚠️ For educational purposes only — not a substitute for professional medical advice.")

# ---------------- Title ---------------- #
st.markdown("""
<div class="hero">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p>Predict the possibility of heart disease using Machine Learning · by Rushikesh</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Input Form ---------------- #
with st.form("prediction_form"):

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧑 Patient Details</div>', unsafe_allow_html=True)

        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Gender", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🫀 Cardiac Metrics</div>', unsafe_allow_html=True)

        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
        oldpeak = st.slider("Old Peak", 0.0, 6.0, 1.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

        st.markdown('</div>', unsafe_allow_html=True)

    submitted = st.form_submit_button("🔍 Predict Heart Disease")

# ---------------- Prediction ---------------- #
if submitted:

    data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "M" else 0,
        "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA": 1 if chest_pain == "TA" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Y" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0,
    }

    input_df = pd.DataFrame([data])

    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[columns]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    st.write("")

    res_col1, res_col2 = st.columns([1.3, 1])

    if prediction[0] == 1:
        risk_pct = probability[0][1] * 100

        with res_col1:
            st.markdown(f"""
            <div class="result-card risk-high">
                ⚠️ High Risk of Heart Disease<br>
                <span style="font-size:1rem;font-weight:400;">
                    Estimated risk probability
                </span>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.metric("Risk Probability", f"{risk_pct:.2f}%")
            st.progress(float(probability[0][1]))

    else:
        healthy_pct = probability[0][0] * 100

        with res_col1:
            st.markdown(f"""
            <div class="result-card risk-low">
                ✅ No Heart Disease Detected<br>
                <span style="font-size:1rem;font-weight:400;">
                    Estimated healthy probability
                </span>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.metric("Healthy Probability", f"{healthy_pct:.2f}%")
            st.progress(float(probability[0][0]))

st.markdown(
    '<div class="footer-note">Developed by Rushikesh Parit · Scikit-Learn · Logistic Regression</div>',
    unsafe_allow_html=True
)