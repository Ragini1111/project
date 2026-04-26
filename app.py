import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fairlearn.metrics import demographic_parity_difference
from sklearn.linear_model import LogisticRegression
import io

st.set_page_config(page_title="FairLens Live", page_icon="⚖️", layout="wide")

# --- Anti-gravity CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main { background-color: #0A0A0F; }
    h1 { color: #fff; text-align: center; animation: float 6s ease-in-out infinite; }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }
    .stButton>button {
        background-color: #2E86C1; color: white; border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(46, 134, 193, 0.3);
    }
    .risk-high {
        background: #E74C3C; padding: 10px; border-radius: 50px; 
        color: white; font-weight: 700; text-align: center;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 rgba(231, 76, 60, 0.7); }
        70% { box-shadow: 0 0 0 15px rgba(231, 76, 60, 0); }
        100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# FairLens: Live AI Bias Detector")
st.markdown("<p style='text-align: center; color: #A0A0B0;'>Upload your loan CSV. Get bias score + fixes in 2 seconds.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload loan_data.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        gender_col = st.selectbox("Select Gender Column", df.columns, index=0)
    with col2:
        age_col = st.selectbox("Select Age Column", df.columns, index=1)
    
    target_col = st.selectbox("Select Approval Column (0/1)", df.columns)
    
    if st.button("Run Bias Scan ⚡"):
        with st.spinner("Scanning for bias..."):
            # --- Run your actual bias logic ---
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            model = LogisticRegression().fit(X, y)
            y_pred = model.predict(X)
            
            gender_bias = demographic_parity_difference(y, y_pred, sensitive_features=df[gender_col])
            age_bias = demographic_parity_difference(y, y_pred, sensitive_features=df[age_col])
            
            # --- Display Results ---
            st.markdown("---")
            st.subheader("Bias Report")
            
            if abs(gender_bias) > 0.1 or abs(age_bias) > 0.1:
                st.markdown('<div class="risk-high">HIGH RISK DETECTED</div>', unsafe_allow_html=True)
            else:
                st.success("LOW RISK: Model passes EU AI Act threshold")
            
            col1, col2 = st.columns(2)
            col1.metric("Gender Bias Score", f"{gender_bias:.3f}", 
                        delta="HIGH" if abs(gender_bias) > 0.1 else "LOW", delta_color="inverse")
            col2.metric("Age Bias Score", f"{age_bias:.3f}", 
                        delta="HIGH" if abs(age_bias) > 0.1 else "LOW", delta_color="inverse")
            
            # --- Plot Chart ---
            fig, ax = plt.subplots()
            ax.bar(['Gender', 'Age'], [abs(gender_bias), abs(age_bias)], color=['#E74C3C', '#2E86C1'])
            ax.axhline(y=0.1, color='red', linestyle='--', label='EU AI Act 10% Threshold')
            ax.set_ylabel('Bias Score')
            ax.set_title('FairLens Bias Detection')
            ax.legend()
            st.pyplot(fig)
            
            # --- How to Solve It ---
            st.markdown("---")
            st.subheader("How to Fix This Bias")
            st.code("""
# Use Fairlearn ExponentiatedGradient to mitigate bias
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(),
    constraints=DemographicParity()
)
mitigator.fit(X, y, sensitive_features=df['gender'])
print("Bias reduced. Re-test with FairLens.")
            """, language='python')
            
            st.info("Next step: Re-train your model with the mitigator above, then re-upload CSV to verify.")
else:
    st.info("👆 Upload a CSV to start. Need test data? Use German Credit Dataset.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #A0A0B0;'>Built for [Hackathon Name] 2026 | <a href='https://github.com/Ragini1111/[repo-name]' style='color:#2E86C1'>GitHub</a></p>", unsafe_allow_html=True)
