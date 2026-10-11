import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import tensorflow as tf

# Set page configuration to wide layout and minimal theme
st.set_page_config(
    page_title="Customer Lifetime Value Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Explicit CSS to guarantee high-contrast visibility on white background
st.markdown("""
    <style>
        /* Force clean white background */
        .stApp {
            background-color: #ffffff !important;
        }
        .main .block-container {
            background-color: #ffffff !important;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Typography and high contrast text */
        h1, h2, h3, h4 {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important;
            color: #0f172a !important;
            font-weight: 600 !important;
        }
        h1 {
            margin-bottom: 0.2rem !important;
        }
        p, span, div {
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #1e293b;
        }

        /* Widget labels visibility */
        .stSelectbox label, .stNumberInput label, .stSlider label, .stFileUploader label {
            color: #0f172a !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
        }

        /* Clean prediction box card */
        .prediction-box {
            padding: 1.5rem;
            border-radius: 6px;
            border: 1px solid #cbd5e1;
            background-color: #ffffff;
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
            margin-top: 0.2rem;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }

        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            border-bottom: 1px solid #e2e8f0;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.95rem;
            font-weight: 600;
            color: #64748b !important;
            padding-bottom: 8px;
        }
        .stTabs [aria-selected="true"] {
            color: #0f172a !important;
            border-bottom: 2px solid #0f172a !important;
        }

        /* Input fields readability */
        div[data-baseweb="input"] {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 4px !important;
        }
        div[data-baseweb="select"] {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 4px !important;
        }
        input {
            color: #0f172a !important;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Cache Model & Preprocessor Artifacts
# -------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    model_path = "final_model.keras"
    scaler_path = "scaler.joblib"
    config_path = "feature_config.json"
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        st.error("Model artifacts not found. Please ensure final_model.keras and scaler.joblib exist.")
        st.stop()
        
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    config = {
        "r2_score": 0.8890,
        "mae_dollar": 1719.61,
        "selected_features": [
            "Response", "Coverage", "Education", "Income", "Monthly Premium Auto",
            "Months Since Last Claim", "Months Since Policy Inception",
            "Number of Policies", "Total Claim Amount", "date"
        ]
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config.update(json.load(f))
        except Exception:
            pass
            
    return model, scaler, config

with st.spinner("Initializing predictive model, please wait..."):
    final_model, scaler, config = load_trained_model()

# Category mappings
COVERAGE_MAP = {
    "Basic": 0,
    "Extended": 1,
    "Premium": 2
}

EDUCATION_MAP = {
    "High School or Below": 0,
    "College": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctor": 4
}

# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def transform_input(response_str, coverage_str, education_str, income_val,
                    monthly_premium_val, months_claim_val, months_inception_val,
                    num_policies_val, total_claim_val, policy_day_val):
    resp_val = 1 if response_str == "Yes" else 0
    cov_val = COVERAGE_MAP[coverage_str]
    edu_val = EDUCATION_MAP[education_str]
    
    resp_transformed = np.log1p(resp_val)
    cov_transformed = np.log1p(cov_val)
    premium_transformed = np.log1p(monthly_premium_val)
    policies_transformed = np.log1p(num_policies_val)
    claim_transformed = np.log1p(total_claim_val)
    
    input_features = np.array([[
        resp_transformed,
        cov_transformed,
        edu_val,
        income_val,
        premium_transformed,
        months_claim_val,
        months_inception_val,
        policies_transformed,
        claim_transformed,
        policy_day_val
    ]], dtype=np.float32)
    
    return scaler.transform(input_features)

def predict_clv(scaled_features):
    pred_log = final_model.predict(scaled_features, verbose=0).flatten()[0]
    return max(0.0, float(np.expm1(pred_log)))

def get_tier_info(clv_val):
    if clv_val >= 15000:
        return (
            "Platinum Tier (High Value)",
            "#1e3a8a",
            "High value client. Strong premium contributions and multiple active policies make this customer a primary candidate for VIP retention and customized loyalty benefits."
        )
    elif clv_val >= 8000:
        return (
            "Gold Tier (Prime Customer)",
            "#b45309",
            "Prime insurance client. Displays consistent policy commitment with solid lifetime valuation. Recommend targeted renewal incentives."
        )
    elif clv_val >= 4000:
        return (
            "Silver Tier (Moderate Value)",
            "#334155",
            "Moderate lifetime valuation. Growth potential exists through cross-selling additional policies or upgrading coverage levels."
        )
    else:
        return (
            "Bronze Tier (Standard Entry)",
            "#64748b",
            "Standard valuation profile with single policy coverage. Focus on customer service engagement and multi-policy bundling to grow lifetime account value."
        )

# -------------------------------------------------------------
# Title and Metrics Banner
# -------------------------------------------------------------
st.title("Customer Lifetime Value Predictor")
st.markdown("Enter customer demographic, policy, and claims metrics below to evaluate projected Customer Lifetime Value (CLV).")

st.markdown("---")
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown('<div class="metric-label">Model Type</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-value">Deep Neural Network (TensorFlow)</div>', unsafe_allow_html=True)
with col_m2:
    st.markdown('<div class="metric-label">Validation R² Score</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{config["r2_score"]:.4f}</div>', unsafe_allow_html=True)
with col_m3:
    st.markdown('<div class="metric-label">Validation MAE (USD)</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">${config["mae_dollar"]:,.2f}</div>', unsafe_allow_html=True)
st.markdown("---")

# -------------------------------------------------------------
# Tabs Interface
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Analysis", "Model Architecture and Stats"])

# =============================================================
# TAB 1: Single Prediction
# =============================================================
with tab1:
    st.subheader("Customer Demographics and Policy Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        education_input = st.selectbox("Education Level", options=list(EDUCATION_MAP.keys()), index=1)
    with c2:
        income_input = st.number_input("Annual Income (USD)", min_value=0.0, max_value=250000.0, value=37600.0, step=1000.0)
    with c3:
        coverage_input = st.selectbox("Coverage Level", options=list(COVERAGE_MAP.keys()), index=0)
    with c4:
        policies_input = st.number_input("Number of Policies", min_value=1, max_value=9, value=2, step=1)

    st.subheader("Premium, Claims, and Account History")
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        premium_input = st.number_input("Monthly Premium Auto (USD)", min_value=10.0, max_value=1000.0, value=93.0, step=5.0)
    with c6:
        claim_input = st.number_input("Total Claim Amount (USD)", min_value=0.0, max_value=10000.0, value=434.0, step=25.0)
    with c7:
        inception_input = st.slider("Months Since Policy Inception", min_value=0, max_value=120, value=48)
    with c8:
        last_claim_input = st.slider("Months Since Last Claim", min_value=0, max_value=36, value=15)

    c9, c10 = st.columns(2)
    with c9:
        response_input = st.selectbox("Response to Renewal / Marketing Offer", options=["No", "Yes"], index=0)
    with c10:
        day_input = st.slider("Policy Effective Day of Month", min_value=1, max_value=31, value=15)

    st.markdown("---")
    
    evaluate_clicked = st.button("Evaluate Customer Lifetime Value", type="primary", use_container_width=True)
    
    # Store in session state so results persist across interactions
    if evaluate_clicked:
        st.session_state['has_evaluated'] = True

    if st.session_state.get('has_evaluated', False):
        scaled_x = transform_input(
            response_input, coverage_input, education_input, income_input,
            premium_input, last_claim_input, inception_input,
            policies_input, claim_input, day_input
        )
        predicted_clv = predict_clv(scaled_x)
        tier_label, tier_color, context_note = get_tier_info(predicted_clv)
        
        st.subheader("Evaluation Results")
        res1, res2 = st.columns(2)
        with res1:
            st.markdown(f'''
                <div class="prediction-box">
                    <div class="metric-label">Estimated Customer Lifetime Value</div>
                    <div class="metric-value">${predicted_clv:,.2f}</div>
                </div>
            ''', unsafe_allow_html=True)
            
        with res2:
            st.markdown(f'''
                <div class="prediction-box">
                    <div class="metric-label">Customer Value Tier</div>
                    <div class="metric-value" style="color:{tier_color};">{tier_label}</div>
                </div>
            ''', unsafe_allow_html=True)
            
        st.subheader("Model Decision Context")
        st.info(context_note)
        
        # What-If Scenario Analysis
        st.subheader("What-If Scenario Analysis")
        st.markdown("Evaluate how the projected lifetime value changes if the customer acquires additional active policies:")
        sim_policies = st.slider("Simulate Total Policies Held", min_value=1, max_value=9, value=policies_input)
        
        sim_scaled = transform_input(
            response_input, coverage_input, education_input, income_input,
            premium_input, last_claim_input, inception_input,
            sim_policies, claim_input, day_input
        )
        sim_clv = predict_clv(sim_scaled)
        delta_val = sim_clv - predicted_clv
        
        sim1, sim2 = st.columns(2)
        with sim1:
            st.markdown(f'''
                <div class="prediction-box">
                    <div class="metric-label">Simulated Customer Lifetime Value</div>
                    <div class="metric-value">${sim_clv:,.2f}</div>
                </div>
            ''', unsafe_allow_html=True)
        with sim2:
            delta_color = "#16a34a" if delta_val >= 0 else "#dc2626"
            delta_sign = "+" if delta_val >= 0 else ""
            st.markdown(f'''
                <div class="prediction-box">
                    <div class="metric-label">Projected Value Difference</div>
                    <div class="metric-value" style="color:{delta_color};">{delta_sign}${delta_val:,.2f}</div>
                </div>
            ''', unsafe_allow_html=True)

# =============================================================
# TAB 2: Batch Analysis (CSV Upload)
# =============================================================
with tab2:
    st.subheader("Batch Customer Lifetime Value Prediction")
    st.markdown("Upload a CSV file containing multiple customer records to calculate projected lifetime values in bulk.")
    
    sample_template = pd.DataFrame([{
        "Response": "No",
        "Coverage": "Basic",
        "Education": "College",
        "Income": 45000,
        "Monthly Premium Auto": 85.0,
        "Months Since Last Claim": 12,
        "Months Since Policy Inception": 36,
        "Number of Policies": 2,
        "Total Claim Amount": 380.0,
        "date": 14
    }])
    
    st.download_button(
        label="Download Sample CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="clv_batch_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} customer rows.")
            
            required_cols = [
                'Response', 'Coverage', 'Education', 'Income', 'Monthly Premium Auto',
                'Months Since Last Claim', 'Months Since Policy Inception',
                'Number of Policies', 'Total Claim Amount', 'date'
            ]
            
            missing_cols = [c for c in required_cols if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                predictions = []
                tiers = []
                for _, row in batch_df.iterrows():
                    scaled_row = transform_input(
                        str(row['Response']), str(row['Coverage']), str(row['Education']), float(row['Income']),
                        float(row['Monthly Premium Auto']), float(row['Months Since Last Claim']),
                        float(row['Months Since Policy Inception']), int(row['Number of Policies']),
                        float(row['Total Claim Amount']), int(row['date'])
                    )
                    clv_pred = predict_clv(scaled_row)
                    predictions.append(clv_pred)
                    tiers.append(get_tier_info(clv_pred)[0])
                
                batch_df['Predicted_CLV_USD'] = np.round(predictions, 2)
                batch_df['Customer_Tier'] = tiers
                
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.markdown(f'''
                        <div class="prediction-box">
                            <div class="metric-label">Total Portfolio Projected Value</div>
                            <div class="metric-value">${sum(predictions):,.2f}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                with b2:
                    st.markdown(f'''
                        <div class="prediction-box">
                            <div class="metric-label">Mean Customer Lifetime Value</div>
                            <div class="metric-value">${np.mean(predictions):,.2f}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                with b3:
                    st.markdown(f'''
                        <div class="prediction-box">
                            <div class="metric-label">Median Customer Lifetime Value</div>
                            <div class="metric-value">${np.median(predictions):,.2f}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                
                st.dataframe(batch_df, use_container_width=True)
                
                st.download_button(
                    label="Download Predictions as CSV",
                    data=batch_df.to_csv(index=False),
                    file_name="clv_predictions_output.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# =============================================================
# TAB 3: Model Architecture and Stats
# =============================================================
with tab3:
    st.subheader("TensorFlow Neural Network Model Specifications")
    
    st.markdown("#### Neural Network Architecture")
    st.code("""
Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.25),
    Dense(32, activation='relu'),
    Dense(1) # Linear output for log(CLV)
])
    """, language="python")
    
    st.markdown("#### Top 10 Selected Features")
    st.markdown("Selected through Mutual Information and SelectKBest feature ranking:")
    st.json(config["selected_features"])
    
    st.markdown("#### Training and Validation Performance")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f'''
            <div class="prediction-box">
                <div class="metric-label">Log-Scale R² Score</div>
                <div class="metric-value">{config["r2_score"]:.4f}</div>
            </div>
        ''', unsafe_allow_html=True)
    with s2:
        st.markdown(f'''
            <div class="prediction-box">
                <div class="metric-label">Validation MAE (USD)</div>
                <div class="metric-value">${config["mae_dollar"]:,.2f}</div>
            </div>
        ''', unsafe_allow_html=True)
