import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

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
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Header */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 40px;
    border-radius: 20px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    animation: fadeInDown 0.8s ease-out;
}

.main-header h1 {
    color: white;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.main-header p {
    color: rgba(255,255,255,0.9);
    font-size: 1.2rem;
    font-weight: 300;
}

/* Card Styling */
.card {
    background: linear-gradient(145deg, #1e1e30 0%, #2d2d44 100%);
    padding: 25px;
    border-radius: 16px;
    margin-bottom: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
}

.card-header {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 2px solid rgba(102, 126, 234, 0.3);
}

.card-icon {
    font-size: 2rem;
    margin-right: 15px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card-title {
    color: #fff;
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0;
}

/* Result Cards */
.result-container {
    margin-top: 30px;
    animation: fadeInUp 0.6s ease-out;
}

.result-yes {
    background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    padding: 40px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 20px 60px rgba(0, 176, 155, 0.4);
    animation: pulse 2s infinite;
}

.result-no {
    background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    padding: 40px;
    border-radius: 20px;
    color: white;
    text-align: center;
    box-shadow: 0 20px 60px rgba(235, 51, 73, 0.4);
}

.result-icon {
    font-size: 4rem;
    margin-bottom: 15px;
}

.result-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.result-prob {
    font-size: 3rem;
    font-weight: 300;
}

/* Probability Meter */
.prob-meter {
    background: rgba(255,255,255,0.1);
    border-radius: 50px;
    padding: 5px;
    margin: 20px 0;
}

.prob-fill {
    height: 20px;
    border-radius: 50px;
    transition: width 1s ease-out;
}

/* Stats Cards */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 20px;
}

.stat-card {
    background: linear-gradient(145deg, #252540 0%, #1a1a2e 100%);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(102, 126, 234, 0.2);
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    color: rgba(255,255,255,0.6);
    font-size: 0.9rem;
    margin-top: 5px;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 15px 60px;
    font-size: 1.2rem;
    font-weight: 600;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
}

/* Selectbox and Slider Styling */
.stSelectbox > div > div {
    background-color: #252540;
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 10px;
}

.stSlider > div > div > div {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Sidebar Styling */
.css-1d391kg {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

/* Info Box */
.info-box {
    background: linear-gradient(145deg, #1e3a5f 0%, #1a2d47 100%);
    padding: 20px;
    border-radius: 12px;
    border-left: 4px solid #667eea;
    margin: 20px 0;
}

.info-box h4 {
    color: #667eea;
    margin-bottom: 10px;
}

.info-box p {
    color: rgba(255,255,255,0.8);
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Feature Importance */
.feature-item {
    display: flex;
    align-items: center;
    margin: 10px 0;
}

.feature-bar {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    margin-left: 10px;
}

/* Animations */
@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        box-shadow: 0 20px 60px rgba(0, 176, 155, 0.4);
    }
    50% {
        box-shadow: 0 25px 70px rgba(0, 176, 155, 0.6);
    }
}

/* Footer */
.footer {
    text-align: center;
    padding: 30px;
    margin-top: 50px;
    color: rgba(255,255,255,0.5);
    border-top: 1px solid rgba(255,255,255,0.1);
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #252540;
    border-radius: 10px;
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
    st.error("⚠️ Model files not found. Please ensure best_model.pkl, preprocessor.pkl, and best_threshold.pkl are in the same directory.")

# ================================
# Header Section
# ================================
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Term Deposit Predictor</h1>
    <p>AI-Powered Customer Subscription Prediction System</p>
</div>
""", unsafe_allow_html=True)

# ================================
# Sidebar - Model Info
# ================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: #667eea;">📊 Model Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Performance Metrics
    st.markdown("### 🎯 Model Performance")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "83.4%", "↑ 5%")
        st.metric("Recall", "62.9%", "↑ 12%")
    with col2:
        st.metric("Precision", "36.3%", "↑ 8%")
        st.metric("F1 Score", "0.46", "↑ 0.15")
    
    st.markdown("---")
    
    # About Section
    st.markdown("### ℹ️ About")
    st.info("""
    This model predicts whether a bank customer will subscribe to a term deposit based on:
    - Customer demographics
    - Financial status
    - Campaign history
    - Economic indicators
    """)
    
    st.markdown("---")
    
    # Tips
    st.markdown("### 💡 Tips")
    st.success("""
    **High subscription likelihood:**
    - Students & Retired customers
    - Previous campaign success
    - Cellular contact method
    - Lower economic uncertainty
    """)

# ================================
# Main Content - Input Form
# ================================
st.markdown("### 📝 Enter Customer Information")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["👤 Customer Profile", "📞 Campaign Details", "📊 Economic Indicators"])

# -------- Tab 1: Customer Profile --------
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <span class="card-icon">👤</span>
                <h3 class="card-title">Personal Information</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        age = st.slider("🎂 Age", 18, 95, 35, help="Customer's age in years")
        
        job = st.selectbox("💼 Occupation", [
            'admin.', 'blue-collar', 'technician', 'services',
            'management', 'retired', 'student', 'unemployed', 
            'self-employed', 'entrepreneur', 'housemaid', 'unknown'
        ], help="Type of job")
        
        marital = st.selectbox("💑 Marital Status", 
            ['single', 'married', 'divorced', 'unknown'],
            help="Customer's marital status")
        
        education = st.selectbox("🎓 Education Level", 
            ['university.degree', 'high.school', 'basic.9y', 'professional.course',
             'basic.4y', 'basic.6y', 'illiterate', 'unknown'],
            help="Highest education level achieved")
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <span class="card-icon">💳</span>
                <h3 class="card-title">Financial Status</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        default = st.selectbox("⚠️ Has Credit Default?", 
            ['no', 'yes', 'unknown'],
            help="Has credit in default?")
        
        housing = st.selectbox("🏠 Has Housing Loan?", 
            ['no', 'yes', 'unknown'],
            help="Has housing loan?")
        
        loan = st.selectbox("💰 Has Personal Loan?", 
            ['no', 'yes', 'unknown'],
            help="Has personal loan?")
        
        # Visual indicator for financial health
        financial_score = 3
        if default == 'yes': financial_score -= 1
        if housing == 'yes': financial_score -= 0.5
        if loan == 'yes': financial_score -= 0.5
        
        st.markdown(f"""
        <div class="info-box">
            <h4>📈 Financial Health Score</h4>
            <p>Based on loan status: <strong>{'🟢 Good' if financial_score >= 2.5 else '🟡 Moderate' if financial_score >= 1.5 else '🔴 At Risk'}</strong></p>
        </div>
        """, unsafe_allow_html=True)

# -------- Tab 2: Campaign Details --------
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <span class="card-icon">📱</span>
                <h3 class="card-title">Contact Information</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        contact = st.selectbox("📞 Contact Type", 
            ['cellular', 'telephone'],
            help="Type of communication")
        
        month = st.selectbox("📅 Last Contact Month", 
            ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
            help="Last contact month of year")
        
        day_of_week = st.selectbox("📆 Last Contact Day", 
            ['mon','tue','wed','thu','fri'],
            help="Last contact day of the week")
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <span class="card-icon">📊</span>
                <h3 class="card-title">Campaign History</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        campaign = st.slider("🔄 Contacts This Campaign", 1, 50, 2,
            help="Number of contacts during this campaign")
        
        pdays = st.slider("⏰ Days Since Previous Contact", 0, 999, 999,
            help="Days passed since last contact (999 = never contacted)")
        
        previous = st.slider("📈 Previous Campaign Contacts", 0, 10, 0,
            help="Number of contacts before this campaign")
        
        poutcome = st.selectbox("✅ Previous Campaign Outcome", 
            ['nonexistent', 'failure', 'success'],
            help="Outcome of the previous marketing campaign")
        
        # Show insight based on previous outcome
        if poutcome == 'success':
            st.success("🎯 High conversion potential! Previous campaign was successful.")
        elif poutcome == 'failure':
            st.warning("⚠️ Previous campaign failed. Consider different approach.")

# -------- Tab 3: Economic Indicators --------
with tab3:
    st.markdown("""
    <div class="info-box">
        <h4>📊 About Economic Indicators</h4>
        <p>These macroeconomic factors can significantly influence a customer's decision to invest in term deposits. 
        During economic uncertainty, customers may be more conservative with investments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        emp_var_rate = st.number_input("📉 Employment Variation Rate", 
            value=1.1, min_value=-5.0, max_value=5.0, step=0.1,
            help="Quarterly employment variation rate")
        
        cons_price_idx = st.number_input("💹 Consumer Price Index", 
            value=93.5, min_value=90.0, max_value=100.0, step=0.1,
            help="Monthly consumer price index")
        
        cons_conf_idx = st.number_input("😊 Consumer Confidence Index", 
            value=-40.0, min_value=-60.0, max_value=0.0, step=0.1,
            help="Monthly consumer confidence index")
    
    with col2:
        euribor3m = st.number_input("🏛️ Euribor 3 Month Rate", 
            value=4.5, min_value=0.0, max_value=6.0, step=0.1,
            help="Euribor 3 month rate - daily indicator")
        
        nr_employed = st.number_input("👥 Number of Employees (thousands)", 
            value=5200.0, min_value=4900.0, max_value=5300.0, step=10.0,
            help="Number of employees - quarterly indicator")

# ================================
# Prediction Section
# ================================
st.markdown("---")
st.markdown("### 🔮 Make Prediction")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button("🚀 Predict Subscription Likelihood", use_container_width=True)

if predict_button and model_loaded:
    
    # Create input dataframe
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
    
    # Make prediction
    with st.spinner("🔄 Analyzing customer profile..."):
        import time
        time.sleep(1)  # Add slight delay for effect
        
        processed_input = preprocessor.transform(input_df)
        prob = model.predict_proba(processed_input)[0][1]
        prediction = 1 if prob >= threshold else 0
    
    # Display Result
    st.markdown("---")
    
    if prediction == 1:
        st.markdown(f"""
        <div class="result-container">
            <div class="result-yes">
                <div class="result-icon">✅</div>
                <div class="result-title">Likely to Subscribe!</div>
                <div class="result-prob">{prob:.1%}</div>
                <p style="margin-top: 15px; opacity: 0.9;">Probability of subscription</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.balloons()
        
    else:
        st.markdown(f"""
        <div class="result-container">
            <div class="result-no">
                <div class="result-icon">❌</div>
                <div class="result-title">Unlikely to Subscribe</div>
                <div class="result-prob">{prob:.1%}</div>
                <p style="margin-top: 15px; opacity: 0.9;">Probability of subscription</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability Gauge Chart
    st.markdown("### 📈 Probability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Subscription Probability", 'font': {'size': 20, 'color': 'white'}},
            delta = {'reference': 50, 'increasing': {'color': "#00b09b"}, 'decreasing': {'color': "#eb3349"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#667eea"},
                'bgcolor': "rgba(255,255,255,0.1)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.3)",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(235, 51, 73, 0.3)'},
                    {'range': [30, 60], 'color': 'rgba(255, 193, 7, 0.3)'},
                    {'range': [60, 100], 'color': 'rgba(0, 176, 155, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Key factors display
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <span class="card-icon">🔑</span>
                <h3 class="card-title">Key Influencing Factors</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        factors = []
        
        if poutcome == 'success':
            factors.append(("Previous Success", "positive", 90))
        if job in ['student', 'retired']:
            factors.append(("Favorable Job Type", "positive", 75))
        if contact == 'cellular':
            factors.append(("Cellular Contact", "positive", 60))
        if default == 'yes':
            factors.append(("Credit Default", "negative", 80))
        if campaign > 5:
            factors.append(("High Contact Frequency", "negative", 65))
        
        if not factors:
            factors.append(("Standard Profile", "neutral", 50))
        
        for factor, sentiment, strength in factors[:5]:
            color = "#00b09b" if sentiment == "positive" else "#eb3349" if sentiment == "negative" else "#667eea"
            icon = "✅" if sentiment == "positive" else "❌" if sentiment == "negative" else "➖"
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin: 10px 0; padding: 10px; 
                        background: rgba(255,255,255,0.05); border-radius: 8px;">
                <span style="font-size: 1.5rem; margin-right: 10px;">{icon}</span>
                <div style="flex-grow: 1;">
                    <div style="color: white; font-weight: 500;">{factor}</div>
                    <div style="background: rgba(255,255,255,0.1); border-radius: 4px; height: 6px; margin-top: 5px;">
                        <div style="background: {color}; width: {strength}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.success("""
            **Recommended Actions:**
            - 📞 Prioritize this customer for follow-up
            - 💬 Use personalized communication
            - 🎯 Highlight term deposit benefits
            - ⏰ Contact during optimal hours
            """)
        else:
            st.warning("""
            **Suggested Approach:**
            - 📧 Send informational materials first
            - ⏳ Wait before direct contact
            - 🔄 Consider different product offerings
            - 📊 Monitor for profile changes
            """)
    
    with col2:
        st.info(f"""
        **Prediction Details:**
        - Model: Logistic Regression (SMOTE)
        - Threshold: {threshold:.2f}
        - Probability: {prob:.4f}
        - Decision: {'Subscribe' if prediction == 1 else 'No Subscribe'}
        """)

# ================================
# Footer
# ================================
st.markdown("""
<div class="footer">
    <p>🏦 Bank Term Deposit Prediction System</p>
    <p style="font-size: 0.8rem;">Built with Streamlit • Machine Learning for Developers Project</p>
</div>
""", unsafe_allow_html=True)

