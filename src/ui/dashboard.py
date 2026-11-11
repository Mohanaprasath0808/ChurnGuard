import streamlit as st
import requests
import json
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="ChurnGuard - 3D Intelligence Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced CSS: Glassmorphism, 3D Card Shadows, and Gradients
st.markdown("""
    <style>
    /* Global Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #090d16 100%);
        color: #f8fafc;
    }
    
    /* 3D Elevated Card Styling */
    .css-1r15c4s, .stMetric, div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
    }

    /* Custom Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
        text-align: center;
    }

    /* Buttons Styling */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        height: 3.2em;
        border: none;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.title("🛡️ ChurnGuard: Enterprise Risk Intelligence")
st.markdown("### Next-Generation MLOps Decision Support System powered by a Calibrated ANN (MLP)")
st.markdown("---")

# Sidebar for Input Features
st.sidebar.header("⚙️ Customer Profile Parameters")
st.sidebar.markdown("Customize profile attributes to simulate churn vulnerability.")

def user_input_features():
    credit_score = st.sidebar.slider("Credit Score", 350, 850, 650)
    geography = st.sidebar.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.slider("Age", 18, 92, 38)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    balance = st.sidebar.number_input("Account Balance ($)", 0.0, 250000.0, 60000.0, step=1000.0)
    num_of_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0, step=1000.0)

    data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary
    }
    return data

input_df = user_input_features()

# Main Layout Split
col1, col2 = st.columns([1.2, 1.8], gap="large")

with col1:
    st.subheader("📋 Selected Profile JSON")
    st.json(input_df)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🚀 Run Neural Risk Assessment", use_container_width=True)

with col2:
    st.subheader("📊 Diagnostic & 3D Risk Visualizer")
    
    if predict_button:
        with st.spinner("Analyzing neural activations & querying FastAPI endpoint..."):
            try:
                response = requests.post("http://127.0.0.1:8000/predict", json=input_df)
                
                if response.status_code == 200:
                    result = response.json()
                    churn_probability = result.get("churn_probability", 0.0)
                    prediction = result.get("prediction", 0)
                    
                    # Metrics row with 3D card layout
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style='color: #94a3b8; margin:0;'>Churn Probability</h4>
                            <h2 style='color: #38bdf8; margin:5px 0;'>{churn_probability * 100:.2f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        status_color = "#ef4444" if (prediction == 1 or churn_probability >= 0.5) else "#22c55e"
                        status_text = "HIGH RISK" if prediction == 1 or churn_probability >= 0.5 else "STABLE"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style='color: #94a3b8; margin:0;'>Classification</h4>
                            <h2 style='color: {status_color}; margin:5px 0;'>{status_text}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Render Plotly 3D-style Gauge Chart if available
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = churn_probability * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Customer Churn Risk Meter", 'font': {'color': '#f8fafc', 'size': 18}},
                            number = {'font': {'color': '#f8fafc', 'size': 26}},
                            gauge = {
                                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': "#3b82f6"},
                                'bgcolor': "rgba(30, 41, 59, 0.8)",
                                'borderwidth': 2,
                                'bordercolor': "rgba(255, 255, 255, 0.1)",
                                'steps': [
                                    {'range': [0, 40], 'color': 'rgba(34, 197, 94, 0.3)'},
                                    {'range': [40, 70], 'color': 'rgba(234, 179, 8, 0.3)'},
                                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': "white"},
                            height=280,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.progress(float(churn_probability))

                    # Actionable Insights Box
                    if prediction == 1 or churn_probability >= 0.5:
                        st.error("⚠️ **Retention Intervention Required:** High risk score detected by the MLP decision boundary.")
                        st.markdown("""
                        - **Immediate Action:** Trigger automated promotional discount campaign.
                        - **Relationship Management:** Schedule a customer success check-in call within 48 hours.
                        """)
                    else:
                        st.success("✅ **Customer Status Stable:** Low churn vulnerability.")
                        st.markdown("""
                        - **Strategy:** Maintain standard nurturing workflow.
                        - **Cross-Sell Opportunity:** Offer supplementary premium accounts or investment products.
                        """)
                        
                else:
                    st.error(f"API Error: Received status code {response.status_code}.")
                    
            except requests.exceptions.ConnectionError:
                st.error("Connection Refused: FastAPI backend is not reachable. Verify your terminal is running `python -m uvicorn src.api.app:app --reload --port 8000`.")
    else:
        st.info("👈 Use the sidebar to tune customer attributes and click **'Run Neural Risk Assessment'** to initiate live prediction.")
        
        # Architectural Showcase Card
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
            <h4 style='color: #818cf8; margin-top:0;'>🧠 System Architecture Specs</h4>
            <ul style='color: #cbd5e1; margin-bottom:0;'>
                <li><b>Backbone:</b> Artificial Neural Network (MLP 64 → 32 hidden layers)</li>
                <li><b>Optimization:</b> Threshold-tuned for optimal Precision-Recall balance</li>
                <li><b>API Validation:</b> Pydantic strict schemas over FastAPI</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)