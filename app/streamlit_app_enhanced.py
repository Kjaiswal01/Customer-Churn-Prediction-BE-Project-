# app/streamlit_app_enhanced.py
# Enhanced Customer Churn Prediction with Simple Relationship Features
import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction - Enhanced",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Simple & Clean CSS
# ---------------------------
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
    }
    .simple-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .relationship-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
    }
    .action-item {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: #f8f9fa;
        border-left: 4px solid #28a745;
    }
    .urgent-action {
        border-left-color: #dc3545;
        background: #fff5f5;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

FEATURE_LIST_FILE = os.path.join(MODELS_DIR, "feature_list.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODERS_FILE = os.path.join(MODELS_DIR, "label_encoders.pkl")
MODEL_FILE = os.path.join(MODELS_DIR, "churn_model.pkl")
TUNED_MODEL_FILE = os.path.join(MODELS_DIR, "churn_model_tuned.pkl")
DATA_FILE = os.path.join(DATA_DIR, "customer_churn.csv")

if os.path.exists(TUNED_MODEL_FILE):
    MODEL_FILE = TUNED_MODEL_FILE

@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, FEATURE_LIST_FILE, LABEL_ENCODERS_FILE]):
        st.error("⚠️ Model files missing! Please run model_training.py first.")
        st.stop()
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_cols = joblib.load(FEATURE_LIST_FILE)
    label_encoders = joblib.load(LABEL_ENCODERS_FILE)
    return model, scaler, feature_cols, label_encoders

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return None

try:
    model, scaler, feature_cols, label_encoders = load_artifacts()
    df_data = load_data()
except:
    st.error("⚠️ Error loading models. Please ensure model files exist.")
    st.stop()

# ---------------------------
# Helper Functions
# ---------------------------
def get_sentiment(feedback):
    try:
        return round(TextBlob(str(feedback)).sentiment.polarity, 3)
    except:
        return 0.0

def encode_input(value, col_name):
    le = label_encoders.get(col_name)
    if le:
        try:
            return le.transform([value])[0]
        except:
            return 0
    return value

def calculate_relationship_score(churn_prob, tenure, last_interaction, sentiment, subscription):
    """Simple Relationship Score (0-100) - Easy to Understand"""
    # Base score from churn probability (inverse)
    base_score = (1 - churn_prob) * 50
    
    # Tenure bonus (longer = better relationship)
    tenure_bonus = min(tenure / 24 * 20, 20)
    
    # Engagement bonus (recent interaction = better)
    engagement_bonus = max(0, (30 - last_interaction) / 30 * 15)
    
    # Sentiment bonus (positive feedback = better)
    sentiment_bonus = max(0, (sentiment + 1) / 2 * 10)
    
    # Subscription bonus (premium = better relationship)
    subscription_bonus = {"Basic": 0, "Standard": 2, "Premium": 5}.get(subscription, 0)
    
    total_score = min(100, base_score + tenure_bonus + engagement_bonus + sentiment_bonus + subscription_bonus)
    return round(total_score, 0)

def get_relationship_status(score):
    """Simple Status - Easy to Understand"""
    if score >= 80:
        return "🟢 Excellent", "आपका ग्राहक बहुत खुश है!"
    elif score >= 60:
        return "🟡 Good", "ग्राहक संबंध अच्छा है, लेकिन सुधार की जरूरत है"
    elif score >= 40:
        return "🟠 Fair", "ग्राहक संबंध औसत है, ध्यान देने की जरूरत"
    else:
        return "🔴 Poor", "तुरंत कार्रवाई करें - ग्राहक खोने का खतरा!"

def get_simple_actions(churn_prob, relationship_score, last_interaction, tenure, sentiment):
    """Simple Action Items - Easy to Understand"""
    actions = []
    
    if relationship_score < 40:
        actions.append({
            "title": "🚨 तुरंत कॉल करें",
            "description": "ग्राहक को तुरंत फोन करके पूछें कि क्या कोई समस्या है",
            "priority": "urgent"
        })
    
    if last_interaction > 30:
        actions.append({
            "title": "📧 Personalized Email भेजें",
            "description": f"ग्राहक को {last_interaction} दिन से संपर्क नहीं हुआ। एक व्यक्तिगत ईमेल भेजें",
            "priority": "high"
        })
    
    if sentiment < -0.2:
        actions.append({
            "title": "😟 समस्या का समाधान करें",
            "description": "ग्राहक की feedback नकारात्मक है। समस्या को तुरंत हल करें",
            "priority": "urgent"
        })
    
    if tenure < 6:
        actions.append({
            "title": "👋 Welcome Call करें",
            "description": "नया ग्राहक है। उनका स्वागत करें और मदद करें",
            "priority": "medium"
        })
    
    if relationship_score >= 80:
        actions.append({
            "title": "🎁 Loyalty Reward दें",
            "description": "ग्राहक बहुत खुश है! उन्हें एक loyalty reward या discount offer दें",
            "priority": "low"
        })
    
    if not actions:
        actions.append({
            "title": "✅ सामान्य Engagement जारी रखें",
            "description": "ग्राहक संबंध अच्छा है। नियमित संपर्क बनाए रखें",
            "priority": "low"
        })
    
    return actions

def predict_churn(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df[feature_cols])
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0, 1]
    return pred, prob

# ---------------------------
# Main App
# ---------------------------
st.sidebar.title("💼 Customer Churn Prediction")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "मेनू चुनें (Choose Menu)",
    ["🏠 Dashboard", "🔮 Predict Churn", "📊 Analytics", "💡 Customer Relationship"]
)

# ---------------------------
# Dashboard
# ---------------------------
if page == "🏠 Dashboard":
    st.markdown("""
        <div class='main-header'>
            <h1>💼 Customer Churn Prediction Platform</h1>
            <p>ग्राहक संबंध मजबूत बनाएं | Strong Customer Relationships</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if df_data is not None:
        total_customers = len(df_data)
        churned = df_data['Churn'].map(lambda x: 1 if str(x).strip().lower() in ["yes","1","y","true","t"] else 0).sum()
        churn_rate = (churned / total_customers) * 100 if total_customers > 0 else 0
        avg_tenure = df_data['Tenure'].mean()
        avg_charges = df_data['Monthly_Charges'].mean()
    else:
        total_customers = churned = churn_rate = avg_tenure = avg_charges = 0
    
    with col1:
        st.metric("कुल ग्राहक", f"{total_customers:,}", help="Total Customers")
    with col2:
        st.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"-{churn_rate:.1f}%")
    with col3:
        st.metric("औसत Tenure", f"{avg_tenure:.1f} महीने", help="Average Tenure in months")
    with col4:
        st.metric("औसत Charges", f"₹{avg_charges:.0f}", help="Average Monthly Charges")
    
    st.markdown("---")
    
    # Simple Feature Cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='simple-card'>
            <h3>🎯 मुख्य सुविधाएं (Key Features)</h3>
            <ul>
                <li>✅ ग्राहक Churn Prediction</li>
                <li>✅ Relationship Score (0-100)</li>
                <li>✅ Simple Action Items</li>
                <li>✅ Customer Analytics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='simple-card'>
            <h3>💡 ग्राहक संबंध बनाने के तरीके</h3>
            <ul>
                <li>📞 Regular Follow-up Calls</li>
                <li>📧 Personalized Emails</li>
                <li>🎁 Special Offers & Rewards</li>
                <li>😊 Quick Problem Resolution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Chart
    if df_data is not None:
        st.markdown("### 📊 Customer Overview")
        df_data['Churn_Binary'] = df_data['Churn'].map(lambda x: 1 if str(x).strip().lower() in ["yes","1","y","true","t"] else 0)
        churn_counts = df_data['Churn_Binary'].value_counts()
        fig = px.pie(values=churn_counts.values, names=['Retained', 'Churned'], 
                     title="Customer Churn Distribution",
                     color_discrete_sequence=['#28a745', '#dc3545'])
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Predict Churn
# ---------------------------
elif page == "🔮 Predict Churn":
    st.markdown("""
        <div class='main-header'>
            <h1>🔮 Churn Prediction</h1>
            <p>ग्राहक का Churn Risk जानें और Relationship Score देखें</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("prediction_form"):
            st.markdown("### ग्राहक की जानकारी दर्ज करें (Enter Customer Details)")
            
            col_a, col_b = st.columns(2)
            with col_a:
                customer_id = st.text_input("Customer ID", value=f"C{datetime.now().strftime('%Y%m%d')}-001")
                gender = st.selectbox("Gender", ["Male", "Female"])
                age = st.number_input("Age (उम्र)", min_value=18, max_value=100, value=35)
                tenure = st.number_input("Tenure (महीने)", min_value=1, max_value=60, value=12, 
                                        help="ग्राहक कितने महीने से है")
            
            with col_b:
                subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
                monthly_charges = st.number_input("Monthly Charges (₹)", min_value=100, max_value=2000, value=499)
                last_interaction_days = st.number_input("Last Interaction (दिन)", min_value=0, max_value=365, value=7,
                                                        help="आखिरी बार कब बात हुई (days ago)")
                feedback = st.text_area("Customer Feedback", "Very satisfied with service", 
                                       help="ग्राहक की feedback")
            
            submit = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Tips")
        st.info("""
        **अच्छा Relationship Score:**
        - High Tenure (>12 months)
        - Recent Interaction (<7 days)
        - Positive Feedback
        - Premium Subscription
        """)
    
    if submit:
        total_spend = monthly_charges * tenure
        sentiment = get_sentiment(feedback)
        
        input_dict = {
            "Gender": encode_input(gender, "Gender"),
            "Age": age,
            "Tenure": tenure,
            "Subscription_Type": encode_input(subscription, "Subscription_Type"),
            "Monthly_Charges": monthly_charges,
            "Total_Spend": total_spend,
            "Last_Interaction_Days": last_interaction_days,
            "Sentiment": sentiment
        }
        
        pred, prob = predict_churn(input_dict)
        relationship_score = calculate_relationship_score(prob, tenure, last_interaction_days, sentiment, subscription)
        status_icon, status_text = get_relationship_status(relationship_score)
        
        st.markdown("---")
        st.markdown("## 📊 Results (परिणाम)")
        
        # Relationship Score - Big and Clear
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div class='simple-card'>
                <h3>Relationship Score</h3>
                <div class='relationship-score' style='color: {'#28a745' if relationship_score >= 60 else '#ffc107' if relationship_score >= 40 else '#dc3545'};'>
                    {relationship_score}/100
                </div>
                <p style='text-align: center; font-size: 1.2rem;'>{status_icon} {status_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            churn_prob_percent = prob * 100
            st.markdown(f"""
            <div class='simple-card'>
                <h3>Churn Probability</h3>
                <div class='relationship-score' style='color: {'#dc3545' if churn_prob_percent > 50 else '#28a745'};'>
                    {churn_prob_percent:.1f}%
                </div>
                <p style='text-align: center; font-size: 1.2rem;'>
                    {'🔴 High Risk' if churn_prob_percent > 50 else '🟢 Low Risk'}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Simple Action Items
        st.markdown("---")
        st.markdown("## 💡 Action Items (कार्रवाई)")
        
        actions = get_simple_actions(prob, relationship_score, last_interaction_days, tenure, sentiment)
        
        for action in actions:
            priority_class = "urgent-action" if action["priority"] == "urgent" else "action-item"
            st.markdown(f"""
            <div class='{priority_class}'>
                <h4>{action["title"]}</h4>
                <p>{action["description"]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Simple Chart
        st.markdown("---")
        st.markdown("### 📈 Visual Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = relationship_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Relationship Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Stay', 'Churn'],
                y=[(1-prob)*100, prob*100],
                marker_color=['green', 'red'],
                text=[f'{(1-prob)*100:.1f}%', f'{prob*100:.1f}%'],
                textposition='auto',
            ))
            fig.update_layout(
                title="Churn Probability",
                yaxis_title="Probability (%)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Analytics
# ---------------------------
elif page == "📊 Analytics":
    st.markdown("""
        <div class='main-header'>
            <h1>📊 Analytics</h1>
            <p>ग्राहक Analytics और Insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    if df_data is None:
        st.warning("No data available.")
    else:
        df_analytics = df_data.copy()
        df_analytics['Churn_Binary'] = df_analytics['Churn'].map(
            lambda x: 1 if str(x).strip().lower() in ["yes","1","y","true","t"] else 0
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn by subscription
            churn_by_sub = df_analytics.groupby('Subscription_Type')['Churn_Binary'].agg(['sum', 'count'])
            churn_by_sub['rate'] = (churn_by_sub['sum'] / churn_by_sub['count']) * 100
            fig = px.bar(x=churn_by_sub.index, y=churn_by_sub['rate'],
                        title="Churn Rate by Subscription Type",
                        labels={'x': 'Subscription Type', 'y': 'Churn Rate (%)'},
                        color=churn_by_sub['rate'],
                        color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Churn by gender
            churn_by_gender = df_analytics.groupby('Gender')['Churn_Binary'].agg(['sum', 'count'])
            churn_by_gender['rate'] = (churn_by_gender['sum'] / churn_by_gender['count']) * 100
            fig = px.pie(values=churn_by_gender['rate'], names=churn_by_gender.index,
                        title="Churn Rate by Gender")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Customer Relationship
# ---------------------------
elif page == "💡 Customer Relationship":
    st.markdown("""
        <div class='main-header'>
            <h1>💡 Customer Relationship Tips</h1>
            <p>ग्राहक संबंध मजबूत बनाने के तरीके</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='simple-card'>
        <h3>🤝 Strong Customer Relationship बनाने के 5 तरीके</h3>
        <ol>
            <li><strong>Regular Communication:</strong> हर 15-30 दिन में एक बार ग्राहक से बात करें</li>
            <li><strong>Quick Problem Solving:</strong> समस्याओं को तुरंत हल करें (24-48 hours)</li>
            <li><strong>Personalized Offers:</strong> ग्राहक के अनुसार special offers दें</li>
            <li><strong>Feedback लें:</strong> नियमित रूप से feedback लें और सुधार करें</li>
            <li><strong>Loyalty Rewards:</strong> लंबे समय के ग्राहकों को rewards दें</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='simple-card'>
            <h4>📞 Communication Tips</h4>
            <ul>
                <li>Friendly tone में बात करें</li>
                <li>Customer का नाम लें</li>
                <li>Their problems को सुनें</li>
                <li>Quick response दें</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='simple-card'>
            <h4>🎁 Retention Strategies</h4>
            <ul>
                <li>Special discounts for loyal customers</li>
                <li>Early access to new features</li>
                <li>Birthday/anniversary wishes</li>
                <li>Referral bonuses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>💼 Customer Churn Prediction Platform | Built with ❤️</p>", unsafe_allow_html=True)
