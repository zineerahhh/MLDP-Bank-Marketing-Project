import streamlit as st
import pandas as pd
import joblib

# ================================
# Page Configuration
# ================================
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide"
)

# ================================
# Custom CSS Styling
# ================================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 0 10px rgba(0,0,0,0.4);
}

.result-yes {
    background: linear-gradient(135deg, #1f7a1f, #2ecc71);
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-size: 22px;
    text-align: center;
}

.result-no {
    background: linear-gradient(135deg, #7a1f1f, #e74c3c);
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-size: 22px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ================================
# Load ML Components
# ================================
model = joblib.load("best_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
threshold = joblib.load("best_threshold.pkl")

# ================================
# Title Section
# ================================
st.markdown(
    """
    <h1 style="text-align:center;">🏦 Bank Term Deposit Subscription Predictor</h1>
    <p style="text-align:center; color: gray;">
    Predict whether a customer is likely to subscribe to a term deposit
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ================================
# Input Layout
# ================================
col1, col2 = st.columns(2)

# -------- Customer Profile --------
with col1:
    st.markdown("<div class='card'><h3>👤 Customer Profile</h3>", unsafe_allow_html=True)

    age = st.slider("Age", 18, 95, 35)

    job = st.selectbox("Job", [
        'admin.', 'blue-collar', 'technician', 'services',
        'management', 'retired', 'student', 'unemployed', 'self-employed'
    ])

    marital = st.selectbox("Marital Status", ['single', 'married', 'divorced'])

    education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])

    st.markdown("</div>", unsafe_allow_html=True)

# -------- Financial Status --------
with col2:
    st.markdown("<div class='card'><h3>💳 Financial Status</h3>", unsafe_allow_html=True)

    default = st.selectbox("Credit Default?", ['yes', 'no'])
    housing = st.selectbox("Housing Loan?", ['yes', 'no'])
    loan = st.selectbox("Personal Loan?", ['yes', 'no'])

    st.markdown("</div>", unsafe_allow_html=True)

# -------- Campaign + Economy --------
col3, col4 = st.columns(2)

with col3:
    st.markdown("<div class='card'><h3>📞 Campaign Details</h3>", unsafe_allow_html=True)

    contact = st.selectbox("Contact Type", ['cellular', 'telephone'])

    month = st.selectbox("Contact Month", [
        'jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'
    ])

    day_of_week = st.selectbox("Day of Week", ['mon','tue','wed','thu','fri'])

    campaign = st.slider("Campaign Contacts", 1, 50, 2)
    pdays = st.slider("Days Since Last Contact", 0, 999, 999)
    previous = st.slider("Previous Contacts", 0, 10, 0)

    poutcome = st.selectbox("Previous Outcome", ['success', 'failure', 'nonexistent'])

    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='card'><h3>📊 Economic Indicators</h3>", unsafe_allow_html=True)

    emp_var_rate = st.number_input("Employment Variation Rate", value=1.1)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.5)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)
    euribor3m = st.number_input("Euribor 3 Month Rate", value=4.5)
    nr_employed = st.number_input("Number of Employees", value=5200.0)

    st.markdown("</div>", unsafe_allow_html=True)

# ================================
# Prediction Button
# ================================
if st.button("🔮 Predict Subscription"):

    input_df = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }])

    processed_input = preprocessor.transform(input_df)

    prob = model.predict_proba(processed_input)[0][1]
    prediction = 1 if prob >= threshold else 0

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            f"<div class='result-yes'>✅ Likely to Subscribe<br><b>{prob:.2%} Probability</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-no'>❌ Unlikely to Subscribe<br><b>{prob:.2%} Probability</b></div>",
            unsafe_allow_html=True
        )

# ================================
# Footer
# ================================
st.markdown(
    "<hr><p style='text-align:center; color: gray;'>MLDP Project • Bank Marketing Prediction</p>",
    unsafe_allow_html=True
)


