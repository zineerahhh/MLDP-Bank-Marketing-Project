import streamlit as st
import pandas as pd
import joblib

# ================================
# Page Configuration
# ================================
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Custom CSS Styling - Modern Dark Theme
# ================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 40px;
    border-radius: 20px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    color: white;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.main-header p {
    color: rgba(255,255,255,0.9);
    font-size: 1.1rem;
    font-weight: 300;
}

.card {
    background: linear-gradient(145deg, #1e1e30 0%, #2d2d44 100%);
    padding: 25px;
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.card-title {
    color: #667eea;
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 15px;
}

.result-yes {
    background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    padding: 40px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0, 176, 155, 0.4);
    margin: 20px 0;
}

.result-no {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    padding: 40px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 20px 60px rgba(235, 51, 73, 0.4);
    margin: 20px 0;
}

.result-icon { font-size: 4rem; margin-bottom: 15px; }
.result-title { font-size: 2rem; font-weight: 700; margin-bottom: 10px; }
.result-prob { font-size: 3rem; font-weight: 300; }

.prob-container {
    background: rgba(255,255,255,0.1);
    border-radius: 50px;
    padding: 5px;
    margin: 20px 0;
}

.prob-bar {
    height: 30px;
    border-radius: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 600;
}

.info-box {
    background: linear-gradient(145deg, #1e3a5f 0%, #1a2d47 100%);
    padding: 20px;
    border-radius: 12px;
    border-left: 4px solid #667eea;
    margin: 15px 0;
}

.info-box p { color: rgba(255,255,255,0.9); margin: 0; }

.factor-positive {
    background: linear-gradient(145deg, #1a3d2e 0%, #0d2818 100%);
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #00b09b;
    margin: 10px 0;
    color: white;
}

.factor-negative {
    background: linear-gradient(145deg, #3d1a1a 0%, #280d0d 100%);
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #eb3349;
    margin: 10px 0;
    color: white;
}

.factor-neutral {
    background: linear-gradient(145deg, #2d2d44 0%, #1e1e30 100%);
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 10px 0;
    color: white;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 15px 60px;
    font-size: 1.2rem;
    font-weight: 600;
    border-radius: 50px;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
}

.footer {
    text-align: center;
    padding: 30px;
    margin-top: 50px;
    color: rgba(255,255,255,0.5);
    border-top: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)

# ================================
# Load ML Components
# ================================
@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    threshold = joblib.load("best_threshold.pkl")
    return model, preprocessor, threshold

try:
    model, preprocessor, threshold = load_model()
    model_loaded = True
except:
    model_loaded = False

# ================================
# Header
# ================================
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Term Deposit Predictor</h1>
    <p>AI-Powered Customer Subscription Prediction System</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found. Please ensure best_model.pkl, preprocessor.pkl, and best_threshold.pkl are in the same directory.")
    st.stop()

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.markdown("## 📊 Model Dashboard")
    st.markdown("---")
    st.markdown("### 🎯 Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "83.4%")
        st.metric("Recall", "62.9%")
    with col2:
        st.metric("Precision", "36.3%")
        st.metric("F1 Score", "0.46")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("Predicts term deposit subscription based on customer demographics, financial status, campaign history & economic indicators.")
    
    st.markdown("---")
    st.markdown("### 💡 High Conversion Signals")
    st.success("✅ Students & Retired\n✅ Previous success\n✅ Cellular contact")

# ================================
# Main Content
# ================================
st.markdown("### 📝 Enter Customer Information")

tab1, tab2, tab3 = st.tabs(["👤 Customer Profile", "📞 Campaign Details", "📊 Economic & Predict"])

# -------- Tab 1: Customer Profile --------
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card"><div class="card-title">👤 Personal Information</div></div>', unsafe_allow_html=True)
        age = st.slider("🎂 Age", 18, 95, 35)
        job = st.selectbox("💼 Occupation", ['admin.', 'blue-collar', 'technician', 'services', 'management', 'retired', 'student', 'unemployed', 'self-employed', 'entrepreneur', 'housemaid', 'unknown'])
        marital = st.selectbox("💑 Marital Status", ['single', 'married', 'divorced', 'unknown'])
        education = st.selectbox("🎓 Education", ['university.degree', 'high.school', 'basic.9y', 'professional.course', 'basic.4y', 'basic.6y', 'illiterate', 'unknown'])
    
    with col2:
        st.markdown('<div class="card"><div class="card-title">💳 Financial Status</div></div>', unsafe_allow_html=True)
        default = st.selectbox("⚠️ Credit Default?", ['no', 'yes', 'unknown'])
        housing = st.selectbox("🏠 Housing Loan?", ['no', 'yes', 'unknown'])
        loan = st.selectbox("💰 Personal Loan?", ['no', 'yes', 'unknown'])
        
        score = 3 - (1 if default=='yes' else 0) - (0.5 if housing=='yes' else 0) - (0.5 if loan=='yes' else 0)
        health = '🟢 Good' if score >= 2.5 else '🟡 Moderate' if score >= 1.5 else '🔴 At Risk'
        st.markdown(f'<div class="info-box"><p>📈 <strong>Financial Health:</strong> {health}</p></div>', unsafe_allow_html=True)
    
    st.info("👉 Continue to **Campaign Details** tab")

# -------- Tab 2: Campaign Details --------
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card"><div class="card-title">📱 Contact Info</div></div>', unsafe_allow_html=True)
        contact = st.selectbox("📞 Contact Type", ['cellular', 'telephone'])
        month = st.selectbox("📅 Month", ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
        day_of_week = st.selectbox("📆 Day", ['mon','tue','wed','thu','fri'])
    
    with col2:
        st.markdown('<div class="card"><div class="card-title">📊 Campaign History</div></div>', unsafe_allow_html=True)
        campaign = st.slider("🔄 Contacts This Campaign", 1, 50, 2)
        pdays = st.slider("⏰ Days Since Last Contact", 0, 999, 999)
        previous = st.slider("📈 Previous Contacts", 0, 10, 0)
        poutcome = st.selectbox("✅ Previous Outcome", ['nonexistent', 'failure', 'success'])
        
        if poutcome == 'success':
            st.success("🎯 High potential!")
        elif poutcome == 'failure':
            st.warning("⚠️ Previous failed.")
    
    st.info("👉 Continue to **Economic & Predict** tab")

# -------- Tab 3: Economic & Predict --------
with tab3:
    st.markdown('<div class="info-box"><p>📊 Macroeconomic factors influence investment decisions.</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        emp_var_rate = st.number_input("📉 Employment Variation Rate", value=1.1, step=0.1)
        cons_price_idx = st.number_input("💹 Consumer Price Index", value=93.5, step=0.1)
        cons_conf_idx = st.number_input("😊 Consumer Confidence Index", value=-40.0, step=0.1)
    with col2:
        euribor3m = st.number_input("🏛️ Euribor 3 Month Rate", value=4.5, step=0.1)
        nr_employed = st.number_input("👥 Number of Employees (k)", value=5200.0, step=10.0)

    # ================================
    # PREDICT BUTTON - ONLY HERE
    # ================================
    st.markdown("---")
    st.markdown("### 🔮 Ready to Predict?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🚀 Predict Subscription Likelihood", use_container_width=True)

    if predict_button:
        input_df = pd.DataFrame([{
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'housing': housing, 'loan': loan,
            'contact': contact, 'month': month, 'day_of_week': day_of_week,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
            'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
            'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
        }])
        
        with st.spinner("🔄 Analyzing..."):
            import time
            time.sleep(0.5)
            processed_input = preprocessor.transform(input_df)
            prob = model.predict_proba(processed_input)[0][1]
            prediction = 1 if prob >= threshold else 0
        
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        
        if prediction == 1:
            st.markdown(f'<div class="result-yes"><div class="result-icon">✅</div><div class="result-title">Likely to Subscribe!</div><div class="result-prob">{prob:.1%}</div></div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f'<div class="result-no"><div class="result-icon">❌</div><div class="result-title">Unlikely to Subscribe</div><div class="result-prob">{prob:.1%}</div></div>', unsafe_allow_html=True)
        
        bar_color = "#00b09b" if prob >= 0.5 else "#f45c43" if prob < 0.3 else "#ffc107"
        st.markdown(f'<div class="prob-container"><div class="prob-bar" style="width:{prob*100}%;background:linear-gradient(90deg,{bar_color},#667eea);">{prob:.1%}</div></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔑 Key Factors")
            if poutcome == 'success':
                st.markdown('<div class="factor-positive">✅ Previous campaign successful!</div>', unsafe_allow_html=True)
            if job in ['student', 'retired']:
                st.markdown('<div class="factor-positive">✅ Favorable job type</div>', unsafe_allow_html=True)
            if contact == 'cellular':
                st.markdown('<div class="factor-positive">✅ Cellular contact</div>', unsafe_allow_html=True)
            if default == 'yes':
                st.markdown('<div class="factor-negative">❌ Credit default risk</div>', unsafe_allow_html=True)
            if campaign > 5:
                st.markdown('<div class="factor-negative">❌ High contact frequency</div>', unsafe_allow_html=True)
            if poutcome == 'failure':
                st.markdown('<div class="factor-negative">❌ Previous campaign failed</div>', unsafe_allow_html=True)
            if poutcome == 'nonexistent' and job not in ['student', 'retired']:
                st.markdown('<div class="factor-neutral">➖ No previous history</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 💡 Recommendations")
            if prediction == 1:
                st.success("**Actions:**\n- 📞 Prioritize follow-up\n- 💬 Personalized communication\n- 🎯 Highlight benefits")
            else:
                st.warning("**Approach:**\n- 📧 Send info materials\n- ⏳ Wait before contact\n- 🔄 Consider other products")
            
            st.info(f"**Details:**\n- Probability: {prob:.4f}\n- Threshold: {threshold:.2f}")

# ================================
# Footer
# ================================
st.markdown('<div class="footer"><p>🏦 Bank Term Deposit Prediction System</p><p style="font-size:0.8rem;">Built with Streamlit • ML Project</p></div>', unsafe_allow_html=True)