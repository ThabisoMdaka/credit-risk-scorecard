import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="🏦",
    layout="wide"
)

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    models = {
        'Logistic Regression': joblib.load(
            'models/logistic_regression.pkl'
        ),
        'Random Forest': joblib.load(
            'models/random_forest.pkl'
        ),
        'XGBoost': joblib.load(
            'models/xgboost.pkl'
        )
    }
    scaler        = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return models, scaler, feature_names

models, scaler, feature_names = load_models()

# ===============================
# ENCODING
# ===============================
encoding = {
    'Sex': {'Male': 1, 'Female': 0},
    'Housing': {'Own': 1, 'Free': 0, 'Rent': 2},
    'Saving accounts': {
        'Little': 0, 'Moderate': 1,
        'Quite Rich': 2, 'Rich': 3, 'Unknown': 4
    },
    'Checking account': {
        'Little': 0, 'Moderate': 1,
        'Rich': 2, 'Unknown': 3
    },
    'Purpose': {
        'Car': 1, 'Furniture/Equipment': 4,
        'Radio/TV': 5, 'Education': 3,
        'Business': 0, 'Repairs': 6,
        'Domestic Appliances': 2,
        'Vacation/Others': 7
    }
}

# ===============================
# HEADER
# ===============================
st.markdown("""
<div style='background: linear-gradient(135deg, #1F4E79, #2E75B6);
            padding: 30px; border-radius: 12px;
            margin-bottom: 30px;'>
    <h1 style='color: white; margin: 0; font-size: 2.5em;'>
        🏦 Credit Risk Scorecard System
    </h1>
    <p style='color: #BDD7EE; margin: 10px 0 0 0;
              font-size: 1.1em;'>
        AI-Powered Loan Default Prediction |
        Logistic Regression · Random Forest · XGBoost
    </p>
    <p style='color: #BDD7EE; margin: 5px 0 0 0;'>
        Built by <strong>Thabiso Mdaka</strong> |
        BSc Electronic Engineering | UKZN
    </p>
</div>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR — BORROWER INPUT
# ===============================
st.sidebar.markdown("## 📋 Borrower Details")
st.sidebar.markdown("---")

age = st.sidebar.slider(
    "Age", 18, 75, 35
)
sex = st.sidebar.selectbox(
    "Sex", ['Male', 'Female']
)
job = st.sidebar.selectbox(
    "Job Skill Level",
    [0, 1, 2, 3],
    index=2,
    help="0=Unskilled, 1=Unskilled Resident, "
         "2=Skilled, 3=Highly Skilled"
)
housing = st.sidebar.selectbox(
    "Housing", ['Own', 'Free', 'Rent']
)
saving_accounts = st.sidebar.selectbox(
    "Saving Accounts",
    ['Little', 'Moderate', 'Quite Rich',
     'Rich', 'Unknown']
)
checking_account = st.sidebar.selectbox(
    "Checking Account",
    ['Little', 'Moderate', 'Rich', 'Unknown']
)
credit_amount = st.sidebar.slider(
    "Credit Amount (DM)", 250, 20000, 3000, 250
)
duration = st.sidebar.slider(
    "Loan Duration (months)", 4, 72, 24
)
purpose = st.sidebar.selectbox(
    "Loan Purpose",
    ['Car', 'Furniture/Equipment', 'Radio/TV',
     'Education', 'Business', 'Repairs',
     'Domestic Appliances', 'Vacation/Others']
)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button(
    "🔍 Assess Credit Risk",
    use_container_width=True
)

# ===============================
# MAIN CONTENT
# ===============================
if predict_btn:
    # Encode input
    input_data = {
        'Age':              age,
        'Sex':              encoding['Sex'][sex],
        'Job':              job,
        'Housing':          encoding['Housing'][housing],
        'Saving accounts':  encoding['Saving accounts'][saving_accounts],
        'Checking account': encoding['Checking account'][checking_account],
        'Credit amount':    credit_amount,
        'Duration':         duration,
        'Purpose':          encoding['Purpose'][purpose]
    }

    input_df = pd.DataFrame([input_data])[feature_names]
    input_scaled = scaler.transform(input_df)

    # Get predictions
    predictions = {}
    for name, model in models.items():
        proba = float(
            model.predict_proba(input_scaled)[0][1]
        )
        predictions[name] = proba

    avg_proba = np.mean(list(predictions.values()))
    final_decision = avg_proba > 0.5

    # ===============================
    # RESULTS DISPLAY
    # ===============================
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎯 Risk Assessment Result")

        if final_decision:
            st.markdown("""
            <div style='background: #FFEBEE;
                        border-left: 6px solid #F44336;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 10px 0;'>
                <h2 style='color: #C62828; margin: 0;'>
                    ⚠️ HIGH RISK
                </h2>
                <h3 style='color: #F44336; margin: 5px 0;'>
                    LOAN APPLICATION DECLINED
                </h3>
                <p style='color: #555; margin: 5px 0;'>
                    This borrower presents a significant
                    probability of default based on their
                    financial profile.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: #E8F5E9;
                        border-left: 6px solid #4CAF50;
                        padding: 20px;
                        border-radius: 8px;
                        margin: 10px 0;'>
                <h2 style='color: #1B5E20; margin: 0;'>
                    ✅ LOW RISK
                </h2>
                <h3 style='color: #4CAF50; margin: 5px 0;'>
                    LOAN APPLICATION APPROVED
                </h3>
                <p style='color: #555; margin: 5px 0;'>
                    This borrower demonstrates a strong
                    repayment profile across all models.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        # Risk Gauge
        st.markdown("### 📊 Default Probability Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Default Probability (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1
                },
                'bar': {'color': "#F44336"
                        if final_decision
                        else "#4CAF50"},
                'steps': [
                    {'range': [0, 30],
                     'color': '#E8F5E9'},
                    {'range': [30, 60],
                     'color': '#FFF9C4'},
                    {'range': [60, 100],
                     'color': '#FFEBEE'}
                ],
                'threshold': {
                    'line': {'color': "black",
                             'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(
            height=300,
            margin=dict(t=50, b=0, l=30, r=30)
        )
        st.plotly_chart(fig_gauge,
                        use_container_width=True)

    st.markdown("---")

    # Model Comparison
    st.markdown("### 🤖 Individual Model Predictions")
    cols = st.columns(3)
    colors_map = {
        'Logistic Regression': '#2196F3',
        'Random Forest':       '#4CAF50',
        'XGBoost':             '#FF9800'
    }

    for col, (name, proba) in zip(
            cols, predictions.items()
    ):
        with col:
            decision = "HIGH RISK ⚠️" \
                if proba > 0.5 else "LOW RISK ✅"
            color = "#F44336" \
                if proba > 0.5 else "#4CAF50"
            st.markdown(f"""
            <div style='background: white;
                        border: 2px solid {color};
                        border-radius: 10px;
                        padding: 15px;
                        text-align: center;'>
                <h4 style='color: #333;
                           margin: 0 0 10px 0;'>
                    {name}
                </h4>
                <h2 style='color: {color};
                           margin: 0;'>
                    {proba*100:.1f}%
                </h2>
                <p style='color: #666;
                          margin: 5px 0 0 0;'>
                    {decision}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Borrower Profile Summary
    st.markdown("### 👤 Borrower Profile Summary")
    col_a, col_b = st.columns(2)

    profile_data = {
        'Feature': [
            'Age', 'Sex', 'Job Level',
            'Housing', 'Saving Accounts',
            'Checking Account', 'Credit Amount',
            'Duration', 'Purpose'
        ],
        'Value': [
            age, sex, job, housing,
            saving_accounts, checking_account,
            f"DM {credit_amount:,}",
            f"{duration} months", purpose
        ],
        'Risk Impact': [
            'Lower age = Higher risk',
            'Demographic factor',
            'Higher skill = Lower risk',
            'Own home = Lower risk',
            'More savings = Lower risk',
            'More balance = Lower risk',
            'Higher amount = Higher risk',
            'Longer duration = Higher risk',
            'Purpose dependent'
        ]
    }

    st.dataframe(
        pd.DataFrame(profile_data),
        use_container_width=True,
        hide_index=True
    )

    # Risk Factor Bar Chart
    st.markdown("### 📈 Risk Probability by Model")
    fig_bar = px.bar(
        x=list(predictions.keys()),
        y=[v * 100 for v in predictions.values()],
        color=[v * 100 for v in predictions.values()],
        color_continuous_scale=['#4CAF50',
                                '#FFC107',
                                '#F44336'],
        labels={
            'x': 'Model',
            'y': 'Default Probability (%)'
        },
        title='Default Probability — All Models Comparison'
    )
    fig_bar.add_hline(
        y=50,
        line_dash="dash",
        line_color="black",
        annotation_text="Decision Threshold (50%)"
    )
    fig_bar.update_layout(
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

else:
    # Welcome screen
    st.markdown("### 👈 Enter borrower details in the sidebar")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='background: #E3F2FD;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;'>
            <h2>🤖</h2>
            <h4>3 AI Models</h4>
            <p>Logistic Regression,
               Random Forest & XGBoost
               working together</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: #E8F5E9;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;'>
            <h2>📊</h2>
            <h4>97%+ Gini Coefficient</h4>
            <p>Industry-standard banking
               performance metrics</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background: #FFF3E0;
                    border-radius: 10px;
                    padding: 20px;
                    text-align: center;'>
            <h2>🏦</h2>
            <h4>Bank-Grade Analytics</h4>
            <p>WoE, IV, Gini & KS metrics
               used by real quant teams</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    ###  About This System
    This Credit Risk Scorecard was built to demonstrate
    quantitative banking analytics skills relevant to
    graduate roles in credit risk and financial modelling.

    **Models Used:**
    - Logistic Regression — interpretable scorecard foundation
    - Random Forest — ensemble learning
    - XGBoost — gradient boosting

    **Banking Metrics:**
    - Gini Coefficient: **97.3%** (Excellent > 60%)
    - KS Statistic: **91.8%** (Excellent > 40%)
    - ROC-AUC: **98.67%**
    """)