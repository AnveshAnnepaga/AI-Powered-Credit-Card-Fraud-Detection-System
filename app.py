import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from data_loader import load_dataset
from models import FraudMLPipeline

# 1. Page Configuration & Custom Premium Styling
st.set_page_config(
    page_title="Guardian | Intelligent Fraud Analytics & Risk Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Composed Premium Dark UI Styling
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global overrides */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b0f19 !important;
    font-family: 'Inter', sans-serif !important;
    color: #cbd5e1 !important; /* Slate-300 */
}

[data-testid="stHeader"] {
    background-color: rgba(11, 15, 25, 0.8) !important;
    backdrop-filter: blur(8px) !important;
}

/* Header typography */
[data-testid="stAppViewContainer"] h1, 
[data-testid="stAppViewContainer"] h2, 
[data-testid="stAppViewContainer"] h3, 
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6 {
    color: #f8fafc !important; /* Slate-50 */
    font-family: 'Outfit', sans-serif !important;
}

/* Form inputs & widget labels */
label, 
.stWidgetLabel, 
[data-testid="stWidgetLabel"] p, 
[data-testid="stWidgetLabel"] label, 
[data-testid="stWidgetLabel"] span,
div[data-testid="stWidgetLabel"] {
    color: #f1f5f9 !important; /* Slate-100 */
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0f172a !important; /* Dark Slate */
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] code,
[data-testid="stSidebar"] .stMarkdown {
    color: #e2e8f0 !important; /* Slate-200 */
    font-family: 'Inter', sans-serif !important;
}

.main-title {
    font-size: 2.8rem;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: #38bdf8 !important; /* Sky-Blue */
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Premium Solid Slate Cards */
.glass-card {
    background: #1e293b !important;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
    margin-bottom: 20px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(56, 189, 248, 0.35);
    transform: translateY(-1px);
}

/* Custom KPI accented cards */
.kpi-card-blue {
    background: #1e293b !important;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #38bdf8 !important;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}
.kpi-card-indigo {
    background: #1e293b !important;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #6366f1 !important;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}
.kpi-card-orange {
    background: #1e293b !important;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #f97316 !important;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}
.kpi-card-red {
    background: #1e293b !important;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-top: 4px solid #ef4444 !important;
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    margin-bottom: 20px;
}

/* Status Badges */
.status-badge-green {
    background: rgba(16, 185, 129, 0.08) !important;
    border: 2px solid #10b981;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.status-badge-red {
    background: rgba(239, 68, 68, 0.08) !important;
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.badge-text-green {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #10b981 !important;
    letter-spacing: 0.05em;
}

.badge-text-red {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ef4444 !important;
    letter-spacing: 0.05em;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 2px;
}

.metric-label {
    font-size: 0.85rem;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Standard Widget Overrides */
div[data-baseweb="select"] {
    background-color: #0f172a !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* Dropdown list overrides */
div[data-testid="stVirtualDropdown"] {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
}

div[role="option"] {
    color: #cbd5e1 !important;
}

div[role="option"]:hover {
    background-color: #0284c7 !important;
    color: #ffffff !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(56, 189, 248, 0.2) !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.3) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: rgba(30, 41, 59, 0.5);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"] {
    height: 45px;
    background-color: transparent;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
    font-family: 'Outfit', sans-serif;
    border: none !important;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* Styled Table for Confusion Matrix */
.matrix-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
    font-family: 'Inter', sans-serif;
}

.matrix-table th, .matrix-table td {
    padding: 12px 15px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.matrix-table th {
    background-color: #0f172a;
    color: #94a3b8;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
}

.matrix-cell-active {
    background-color: rgba(56, 189, 248, 0.15);
    color: #ffffff;
    font-weight: 700;
}

.matrix-cell-error {
    background-color: rgba(239, 68, 68, 0.12);
    color: #fca5a5;
    font-weight: 700;
}
</style>
<script>
// Prevent visual scale jumping
window.scrollTo(0, 0);
</script>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 2. Session State Initialization
if 'active_transaction' not in st.session_state:
    # Initialize with default normal transaction
    st.session_state.active_transaction = {
        'Amount': 28.50,
        **{f'V{i}': 0.0 for i in range(1, 29)}
    }

# 3. Data Loading & Model Training Caching
@st.cache_data
def get_dataset():
    """Loads dataset and outputs status parameters"""
    return load_dataset()

df, is_synthetic, filepath = get_dataset()

@st.cache_resource
def get_trained_pipeline(_df, version=2):
    """Trains scikit-learn model models and returns trained pipeline wrapper"""
    pipeline = FraudMLPipeline(random_state=42)
    pipeline.fit(_df)
    return pipeline

with st.spinner("⚔️ Shield Up! Initializing Guardian AI Core Systems..."):
    pipeline = get_trained_pipeline(df, version=2)

# 4. Sidebar System Telemetry
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 20px;'>", unsafe_allow_html=True)
    st.markdown("🌐 **GUARDIAN AI COMMAND**", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 📡 Database Status")
    if is_synthetic:
        st.markdown(
            f"""
            <div style='background-color:rgba(249, 115, 22, 0.1); border:1px solid #f97316; border-radius:8px; padding:12px; margin-bottom:15px;'>
                <strong style='color:#f97316;'>Mode: High-Fidelity Simulator</strong><br>
                <span style='font-size:0.8rem; color:#94a3b8;'>Real Kaggle CSV not found locally or offline. Loaded 25,000 highly calibrated samples.</span>
            </div>
            """, 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style='background-color:rgba(16, 185, 129, 0.1); border:1px solid #10b981; border-radius:8px; padding:12px; margin-bottom:15px;'>
                <strong style='color:#10b981;'>Mode: Real Kaggle Dataset</strong><br>
                <span style='font-size:0.8rem; color:#94a3b8;'>Successfully connected to raw dataset: <code>{os.path.basename(filepath)}</code></span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    st.markdown("### 📊 Database Summary")
    st.metric(label="Total Transactions Logged", value=f"{len(df):,}")
    
    fraud_count = df['Class'].sum()
    fraud_rate = (fraud_count / len(df)) * 100
    st.metric(label="Fraud Cases Identified", value=f"{int(fraud_count)}")
    st.metric(label="Calibrated Skew (Fraud %)", value=f"{fraud_rate:.3f}%")
    
    st.markdown("---")
    st.markdown("### 📥 Active Database Export")
    csv_data = df.sample(min(5000, len(df))).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sub-Sample (CSV)",
        data=csv_data,
        file_name="creditcard_sub_sample.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("<br><p style='text-align:center; font-size:0.75rem; color:#64748b;'>Guardian Fraud Platform v1.2.0</p>", unsafe_allow_html=True)

# 5. Header Section
st.markdown("<h1 class='main-title'>🛡️ GUARDIAN</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>State-of-the-Art Credit Card Fraud Detection & Risk Analytics Hub</p>", unsafe_allow_html=True)

# 6. Tab Definitions
tab_skew, tab_eval, tab_arena, tab_pca = st.tabs([
    "📊 Imbalance Analytics", 
    "⚙️ Interactive Scanner", 
    "📈 Model Arena & Operational Cost", 
    "🌌 3D PCA Space Mapping"
])

# ==============================================================================
# TAB 1: SKEWNESS & IMBALANCE ANALYTICS
# ==============================================================================
with tab_skew:
    st.markdown("### The Challenge of Highly Imbalanced Datasets")
    st.write(
        "Real-world banking AI systems face an extreme classification challenge: "
        "only a tiny fraction of card transactions are fraudulent. Predicting 'normal' on everything "
        "yields a deceptively high accuracy of 99.8%, but fails entirely to secure the cardholders."
    )
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(
            f"""
            <div class='kpi-card-blue'>
                <div class='metric-label'>Regular Transactions</div>
                <div class='metric-value'>{len(df) - int(fraud_count):,}</div>
                <div style='font-size:0.8rem; color:#10b981; font-weight:600;'>99.65% to 99.83% of total</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col_kpi2:
        st.markdown(
            f"""
            <div class='kpi-card-red'>
                <div class='metric-label'>Anomalous/Fraud Cases</div>
                <div class='metric-value' style='color:#ef4444;'>{int(fraud_count)}</div>
                <div style='font-size:0.8rem; color:#ef4444; font-weight:600;'>Extreme Class Skew</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col_kpi3:
        st.markdown(
            f"""
            <div class='kpi-card-indigo'>
                <div class='metric-label'>Naïve Model Accuracy</div>
                <div class='metric-value'>{(1.0 - (fraud_count/len(df)))*100:.3f}%</div>
                <div style='font-size:0.8rem; color:#cbd5e1;'>Achieved by predicting 0 on all!</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col_kpi4:
        st.markdown(
            f"""
            <div class='kpi-card-orange'>
                <div class='metric-label'>Average Transaction Amount</div>
                <div class='metric-value'>${df['Amount'].mean():.2f}</div>
                <div style='font-size:0.8rem; color:#cbd5e1;'>Skewed Log-Normal distribution</div>
            </div>
            """, unsafe_allow_html=True
        )
        
    plot_c1, plot_c2 = st.columns(2)
    
    with plot_c1:
        st.markdown("#### Class Proportion Breakdown (Extreme Class Skew)")
        labels = ['Normal Transactions', 'Fraud Transactions']
        values = [len(df) - int(fraud_count), int(fraud_count)]
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=.6,
            marker_colors=['#0ea5e9', '#ef4444'],
            textinfo='percent+label',
            insidetextorientation='radial'
        )])
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            showlegend=False,
            height=320,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with plot_c2:
        st.markdown("#### Log-Normal Distribution of Transaction Amounts")
        # Sub-sample to keep charts blazing fast
        chart_df = df.sample(min(5000, len(df))).copy()
        chart_df['Class Label'] = chart_df['Class'].map({1: 'Fraudulent', 0: 'Normal'})
        
        # Add a tiny constant to amount to prevent log(0)
        chart_df['Log Amount'] = np.log10(chart_df['Amount'] + 0.01)
        
        fig_hist = px.histogram(
            chart_df,
            x='Log Amount',
            color='Class Label',
            barmode='overlay',
            color_discrete_map={'Normal': '#0ea5e9', 'Fraudulent': '#ef4444'},
            labels={'Log Amount': 'Log10 Transaction Amount ($)', 'count': 'Transaction Count'},
            opacity=0.7
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_hist.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig_hist.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig_hist, use_container_width=True)

# ==============================================================================
# TAB 2: INTERACTIVE TRANSACTION EVALUATOR
# ==============================================================================
with tab_eval:
    st.markdown("### Live Transaction Risk Assessor")
    st.write(
        "Load preset credit card templates or custom construct transaction attributes below. "
        "Guardian AI runs real-time supervised classification alongside unsupervised anomaly scores."
    )
    
    # Preset Templates
    st.markdown("#### 🚀 Speed Presets (Select to load instantly)")
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("✅ Standard Coffee Shop Purchase", use_container_width=True):
            st.session_state.active_transaction = {
                'Amount': 4.75,
                **{f'V{i}': np.random.normal(0, 0.4) for i in range(1, 29)}
            }
            st.toast("Grocery transaction profile loaded!")
            
    with col_btn2:
        if st.button("⚠️ High-Amount Foreign Hotel Booking", use_container_width=True):
            # Moderate shifts
            hotel_profile = {f'V{i}': np.random.normal(0, 0.8) for i in range(1, 29)}
            hotel_profile['V3'] = -1.2
            hotel_profile['V4'] = 1.1
            hotel_profile['V14'] = -1.8
            hotel_profile['Amount'] = 980.00
            st.session_state.active_transaction = hotel_profile
            st.toast("High-Value transaction profile loaded!")
            
    with col_btn3:
        if st.button("🚨 Confirmed Phishing & Card Cloning", use_container_width=True):
            # Massive shifts in highly predictive components
            fraud_profile = {f'V{i}': np.random.normal(0, 0.9) for i in range(1, 29)}
            fraud_profile['V3'] = -5.8
            fraud_profile['V4'] = 4.2
            fraud_profile['V10'] = -4.5
            fraud_profile['V11'] = 3.6
            fraud_profile['V12'] = -6.2
            fraud_profile['V14'] = -7.9
            fraud_profile['V17'] = -8.5
            fraud_profile['V18'] = -4.0
            fraud_profile['Amount'] = 450.00
            st.session_state.active_transaction = fraud_profile
            st.toast("🚨 Phishing transaction profile injected!")

    # Grid Form of sliders
    with st.form("custom_transaction_form"):
        st.markdown("#### 🛠️ Transaction Parameters")
        
        col_amt, col_main1, col_main2, col_main3 = st.columns(4)
        with col_amt:
            tx_amount = st.number_input(
                "Transaction Amount ($)", 
                min_value=0.01, 
                max_value=25000.0, 
                value=float(st.session_state.active_transaction['Amount']), 
                step=10.0
            )
        with col_main1:
            tx_v14 = st.slider("V14 (Highly predictive Fraud Shift)", -15.0, 10.0, float(st.session_state.active_transaction['V14']), step=0.1)
        with col_main2:
            tx_v17 = st.slider("V17 (Highly predictive Fraud Shift)", -15.0, 10.0, float(st.session_state.active_transaction['V17']), step=0.1)
        with col_main3:
            tx_v3 = st.slider("V3 (Transaction Security Component)", -15.0, 10.0, float(st.session_state.active_transaction['V3']), step=0.1)
            
        with st.expander("🔬 Advanced Latent Space Components (V1-V28 PCA Projections)"):
            st.write("These variables represent anonymized transaction attributes derived using Principal Component Analysis (PCA).")
            
            # Group rest of the V elements in columns
            cols_v = st.columns(4)
            v_values = {}
            for i in range(1, 29):
                # Skip the 3 primary ones which are already on the outer page
                if i in [14, 17, 3]:
                    v_values[f'V{i}'] = float(st.session_state.active_transaction[f'V{i}'])
                    continue
                    
                col_idx = (i - 1) % 4
                with cols_v[col_idx]:
                    v_values[f'V{i}'] = st.slider(f"V{i}", -10.0, 10.0, float(st.session_state.active_transaction[f'V{i}']), step=0.1)
                    
        submit_tx = st.form_submit_button("Launch Guardian Inspection Engine", type="primary")
        
    if submit_tx:
        # Save custom inputs into session state
        st.session_state.active_transaction = {
            'Amount': tx_amount,
            'V14': tx_v14,
            'V17': tx_v17,
            'V3': tx_v3,
            **{f'V{i}': v_values[f'V{i}'] for i in range(1, 29) if i not in [14, 17, 3]}
        }
        st.toast("Analyzing transaction features...")

    # Run Prediction
    pred_res = pipeline.predict_transaction(st.session_state.active_transaction)
    
    # Visual Output
    c_res1, c_res2 = st.columns([1, 1])
    
    with c_res1:
        st.markdown("#### 🛡️ AI Model Consensus Results")
        
        # Max of standard model probabilities determines high threat
        max_prob = max(pred_res['lr_prob'], pred_res['rf_prob'])
        
        if max_prob < 0.20:
            st.markdown(
                f"""
                <div class='status-badge-green'>
                    <div class='badge-text-green'>APPROVED</div>
                    <p style='color:#e2e8f0; margin-top:15px; font-size:1.05rem;'>
                        Consensus: Safe and Compliant.<br>
                        <strong>Risk Probability: {max_prob*100:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
        elif max_prob < 0.65:
            st.markdown(
                f"""
                <div class='status-badge-red' style='background:rgba(249, 115, 22, 0.08); border-color:#f97316;'>
                    <div class='badge-text-red' style='color:#f97316;'>REVIEW REQUIRED</div>
                    <p style='color:#e2e8f0; margin-top:15px; font-size:1.05rem;'>
                        Consensus: Suspicious Pattern Identified.<br>
                        <strong>Risk Probability: {max_prob*100:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class='status-badge-red'>
                    <div class='badge-text-red'>BLOCKED</div>
                    <p style='color:#e2e8f0; margin-top:15px; font-size:1.05rem;'>
                        Consensus: High Probability of Fraud / Card Theft.<br>
                        <strong>Risk Probability: {max_prob*100:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True
            )
            
        # 3 Speedometers/Gauges for the models
        st.markdown("##### 🚀 Multi-Algorithm Diagnostic Breakdown")
        
        col_g1, col_g2, col_g3 = st.columns(3)
        
        with col_g1:
            # Logistic Regression
            lr_pct = pred_res['lr_prob'] * 100
            fig_lr = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = lr_pct,
                title = {'text': "Logistic Regression", 'font': {'size': 13, 'color': '#cbd5e1', 'family': 'Outfit'}},
                number = {'font': {'size': 20, 'color': '#ffffff', 'family': 'Outfit'}, 'suffix': '%'},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': '#6366f1'},
                    'bgcolor': "rgba(30, 41, 59, 0.4)",
                    'borderwidth': 1
                }
            ))
            fig_lr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#cbd5e1', height=130, margin=dict(l=10, r=10, t=35, b=0))
            st.plotly_chart(fig_lr, use_container_width=True)
            
        with col_g2:
            # Random Forest
            rf_pct = pred_res['rf_prob'] * 100
            fig_rf = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = rf_pct,
                title = {'text': "Random Forest", 'font': {'size': 13, 'color': '#cbd5e1', 'family': 'Outfit'}},
                number = {'font': {'size': 20, 'color': '#ffffff', 'family': 'Outfit'}, 'suffix': '%'},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#38bdf8'},
                    'bgcolor': "rgba(30, 41, 59, 0.4)",
                    'borderwidth': 1
                }
            ))
            fig_rf.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#cbd5e1', height=130, margin=dict(l=10, r=10, t=35, b=0))
            st.plotly_chart(fig_rf, use_container_width=True)
            
        with col_g3:
            # Isolation Forest Anomaly
            if_pct = pred_res['if_prob'] * 100
            fig_if = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = if_pct,
                title = {'text': "Isolation Forest", 'font': {'size': 13, 'color': '#cbd5e1', 'family': 'Outfit'}},
                number = {'font': {'size': 20, 'color': '#ffffff', 'family': 'Outfit'}, 'suffix': '%'},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#f97316'},
                    'bgcolor': "rgba(30, 41, 59, 0.4)",
                    'borderwidth': 1
                }
            ))
            fig_if.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#cbd5e1', height=130, margin=dict(l=10, r=10, t=35, b=0))
            st.plotly_chart(fig_if, use_container_width=True)
            
    with c_res2:
        st.markdown("#### 🔍 Real-Time Feature Influence Profiler")
        st.write("Highlights exactly which transaction parameters push the decision toward high risk (red) or protect it (green).")
        
        # Risk factors vs protective factors from Logistic Regression coefficients
        r_factors = pred_res['top_risk_factors']
        p_factors = pred_res['top_protective_factors']
        
        drivers_df = pd.DataFrame(r_factors + p_factors, columns=['Feature', 'Coefficient Impact'])
        drivers_df = drivers_df.sort_values(by='Coefficient Impact', ascending=True)
        
        # Color coding
        drivers_df['Influence Type'] = np.where(drivers_df['Coefficient Impact'] > 0, 'Fraud Risk Factor (+)', 'Protective Component (-)')
        
        fig_drivers = px.bar(
            drivers_df,
            y='Feature',
            x='Coefficient Impact',
            color='Influence Type',
            color_discrete_map={'Fraud Risk Factor (+)': '#ef4444', 'Protective Component (-)': '#10b981'},
            orientation='h',
            labels={'Coefficient Impact': 'Log-Odds Strength', 'Feature': 'PCA Latent Feature / Amount'}
        )
        fig_drivers.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_drivers.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig_drivers.update_yaxes(showgrid=False)
        st.plotly_chart(fig_drivers, use_container_width=True)
        
        st.markdown("<div class='glass-card' style='padding: 18px;'>", unsafe_allow_html=True)
        st.markdown("<h5 style='color:#38bdf8; font-family:Outfit; margin-top:0px; margin-bottom:10px;'>📊 Guardian Security Assessment</h5>", unsafe_allow_html=True)
        
        # Custom logic explanation
        if max_prob >= 0.65:
            st.markdown(
                "🚨 **System Action Taken**: Immediate hold placed. This transaction displays an extreme signature typical "
                "of fraudulent card cloning or high-volume automated phishing scams. The primary driver is "
                f"**{r_factors[0][0]}** (shift of {st.session_state.active_transaction[r_factors[0][0]]:.2f})."
            )
        elif max_prob >= 0.20:
            st.markdown(
                "⚠️ **System Action Taken**: Diverted to manual compliance review queue. The transaction contains slight anomaly markers "
                f"on **{r_factors[0][0]}**, but is balanced by the protective nature of **{p_factors[0][0] if len(p_factors) > 0 else 'None'}**."
            )
        else:
            st.markdown(
                "✅ **System Action Taken**: Auto-cleared successfully. No risk indicators detected. The transaction is well within the "
                "safe statistical boundaries of normal credit card transactions."
            )
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 3: MODEL ARENA & FINANCIAL COST SIMULATOR
# ==============================================================================
with tab_arena:
    st.markdown("### Model Comparison & Financial Risk Simulator")
    st.write(
        "Adjust the decision threshold slider below to witness the real-world operational cost trade-offs. "
        "In banking AI, setting the threshold too high lets fraud pass (costing chargebacks), while setting it too low "
        "triggers false alarms, blocking honest cardholders (costing customer friction and churn)."
    )
    
    # Dynamic Plot Row
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.markdown("#### Precision-Recall Curve (Crucial for Imbalanced Data)")
        
        fig_pr = go.Figure()
        # LR
        lr_pr = pipeline.lr_metrics['pr_curve']
        fig_pr.add_trace(go.Scatter(x=lr_pr[1], y=lr_pr[0], name=f"Logistic Regression (AP = {pipeline.lr_metrics['pr_auc']:.3f})", line=dict(color='#6366f1', width=2.5)))
        # RF
        rf_pr = pipeline.rf_metrics['pr_curve']
        fig_pr.add_trace(go.Scatter(x=rf_pr[1], y=rf_pr[0], name=f"Random Forest (AP = {pipeline.rf_metrics['pr_auc']:.3f})", line=dict(color='#38bdf8', width=2.5)))
        # IF
        if_pr = pipeline.if_metrics['pr_curve']
        fig_pr.add_trace(go.Scatter(x=if_pr[1], y=if_pr[0], name=f"Isolation Forest (AP = {pipeline.if_metrics['pr_auc']:.3f})", line=dict(color='#f97316', width=2.5)))
        
        fig_pr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#cbd5e1',
            xaxis_title="Recall (Sensitivity)", yaxis_title="Precision (Reliability)",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_pr.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig_pr.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig_pr, use_container_width=True)
        
    with col_plot2:
        st.markdown("#### ROC Curve Comparison")
        fig_roc = go.Figure()
        
        # LR
        lr_roc = pipeline.lr_metrics['roc_curve']
        fig_roc.add_trace(go.Scatter(x=lr_roc[0], y=lr_roc[1], name=f"Logistic Regression (AUC = {pipeline.lr_metrics['roc_auc']:.3f})", line=dict(color='#6366f1', width=2.5)))
        # RF
        rf_roc = pipeline.rf_metrics['roc_curve']
        fig_roc.add_trace(go.Scatter(x=rf_roc[0], y=rf_roc[1], name=f"Random Forest (AUC = {pipeline.rf_metrics['roc_auc']:.3f})", line=dict(color='#38bdf8', width=2.5)))
        # IF
        if_roc = pipeline.if_metrics['roc_curve']
        fig_roc.add_trace(go.Scatter(x=if_roc[0], y=if_roc[1], name=f"Isolation Forest (AUC = {pipeline.if_metrics['roc_auc']:.3f})", line=dict(color='#f97316', width=2.5)))
        
        # Diagonal random line
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], showlegend=False, line=dict(color='rgba(255,255,255,0.1)', dash='dash')))
        
        fig_roc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#cbd5e1',
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig_roc.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        fig_roc.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig_roc, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 🎛️ Operational Control & Financial Impact Simulator")
    
    # Inputs for Simulator
    col_sim_in1, col_sim_in2 = st.columns([1, 2])
    
    with col_sim_in1:
        st.markdown("<div class='glass-card' style='padding: 20px;'>", unsafe_allow_html=True)
        st.markdown("<h5 style='color:#38bdf8; margin-top:0px;'>1. Configure Model & Threshold</h5>", unsafe_allow_html=True)
        
        selected_model = st.selectbox(
            "Select Target Algorithm",
            options=["Logistic Regression", "Random Forest", "Isolation Forest"]
        )
        
        threshold = st.slider(
            "Decision Threshold", 
            min_value=0.01, 
            max_value=0.99, 
            value=0.50, 
            step=0.01,
            help="Adjust the cut-off boundary. Lower thresholds block more transactions (increasing Recall, lowering Precision). Higher thresholds let more pass."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='glass-card' style='padding: 20px; margin-top: 15px;'>", unsafe_allow_html=True)
        st.markdown("<h5 style='color:#ef4444; margin-top:0px;'>2. Bank Operational Unit Costs</h5>", unsafe_allow_html=True)
        cost_fn = st.number_input(
            "Cost of Missed Fraud (False Neg.)", 
            min_value=10, 
            max_value=10000, 
            value=1200, 
            step=100,
            help="The financial loss of paying for a chargeback, refunding the customer, plus banking audit fines."
        )
        cost_fp = st.number_input(
            "Cost of False Alarm (False Pos.)", 
            min_value=1, 
            max_value=500, 
            value=45, 
            step=5,
            help="The cost of blocking an honest user: card replacement cost, call center load, and customer relationship friction."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_sim_in2:
        # Calculate dynamic metrics based on selected threshold
        eval_metrics = pipeline.evaluate_threshold(selected_model, threshold)
        
        # Layout metrics side-by-side
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric(label="Dynamic Precision (Reliability)", value=f"{eval_metrics['precision']*100:.2f}%")
        with col_m2:
            st.metric(label="Dynamic Recall (Coverage)", value=f"{eval_metrics['recall']*100:.2f}%")
        with col_m3:
            st.metric(label="F1-Score (Harmonic Balance)", value=f"{eval_metrics['f1']:.4f}")
            
        # Draw dynamic Confusion Matrix
        st.markdown("##### Dynamic Confusion Matrix (Test Set)")
        
        matrix_html = f"""
        <table class='matrix-table'>
            <thead>
                <tr>
                    <th rowspan="2">Actual Class</th>
                    <th colspan="2">Predicted Class</th>
                </tr>
                <tr>
                    <th>Predicted Normal (y = 0)</th>
                    <th>Predicted Fraud (y = 1)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style='font-weight:600; background-color:#0f172a;'>Actual Normal (y = 0)</td>
                    <td class='matrix-cell-active'>True Negatives (TN)<br><span style='font-size:1.3rem;'>{eval_metrics['tn']:,}</span></td>
                    <td class='matrix-cell-error'>False Positives (FP) - False Alarms<br><span style='font-size:1.3rem;'>{eval_metrics['fp']:,}</span></td>
                </tr>
                <tr>
                    <td style='font-weight:600; background-color:#0f172a;'>Actual Fraud (y = 1)</td>
                    <td class='matrix-cell-error'>False Negatives (FN) - Missed Fraud<br><span style='font-size:1.3rem;'>{eval_metrics['fn']:,}</span></td>
                    <td class='matrix-cell-active' style='background-color:rgba(16, 185, 129, 0.15);'>True Positives (TP) - Stopped Fraud<br><span style='font-size:1.3rem;'>{eval_metrics['tp']:,}</span></td>
                </tr>
            </tbody>
        </table>
        """
        st.markdown(matrix_html, unsafe_allow_html=True)
        
        # Calculate dynamic dollar financial impact
        total_missed_fraud_cost = eval_metrics['fn'] * cost_fn
        total_false_alarm_cost = eval_metrics['fp'] * cost_fp
        total_cost = total_missed_fraud_cost + total_false_alarm_cost
        
        # Calculate baseline cost if threshold was 1.0 (Naïve - let all fraud pass, no false alarms)
        baseline_missed = eval_metrics['fn'] + eval_metrics['tp']
        baseline_cost = baseline_missed * cost_fn
        savings = baseline_cost - total_cost
        
        st.markdown("##### 💵 Financial Loss & Savings Analysis")
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            st.markdown(
                f"""
                <div style='background-color:rgba(239, 68, 68, 0.08); border:1px solid rgba(239, 68, 68, 0.2); border-radius:8px; padding:15px; text-align:center;'>
                    <span style='font-size:0.8rem; color:#cbd5e1; text-transform:uppercase;'>Missed Fraud Losses</span><br>
                    <strong style='font-size:1.4rem; color:#ef4444;'>${total_missed_fraud_cost:,}</strong><br>
                    <span style='font-size:0.75rem; color:#94a3b8;'>{eval_metrics['fn']} cases missed</span>
                </div>
                """, unsafe_allow_html=True
            )
            
        with col_c2:
            st.markdown(
                f"""
                <div style='background-color:rgba(249, 115, 22, 0.08); border:1px solid rgba(249, 115, 22, 0.2); border-radius:8px; padding:15px; text-align:center;'>
                    <span style='font-size:0.8rem; color:#cbd5e1; text-transform:uppercase;'>False Alarm Cost</span><br>
                    <strong style='font-size:1.4rem; color:#f97316;'>${total_false_alarm_cost:,}</strong><br>
                    <span style='font-size:0.75rem; color:#94a3b8;'>{eval_metrics['fp']} honest cards blocked</span>
                </div>
                """, unsafe_allow_html=True
            )
            
        with col_c3:
            if savings > 0:
                st.markdown(
                    f"""
                    <div style='background-color:rgba(16, 185, 129, 0.08); border:1px solid #10b981; border-radius:8px; padding:15px; text-align:center;'>
                        <span style='font-size:0.8rem; color:#cbd5e1; text-transform:uppercase;'>Net Stopped Losses (Savings)</span><br>
                        <strong style='font-size:1.4rem; color:#10b981;'>+ ${savings:,}</strong><br>
                        <span style='font-size:0.75rem; color:#94a3b8;'>vs. doing nothing</span>
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='background-color:rgba(239, 68, 68, 0.08); border:1px solid #ef4444; border-radius:8px; padding:15px; text-align:center;'>
                        <span style='font-size:0.8rem; color:#cbd5e1; text-transform:uppercase;'>Net Operational Loss</span><br>
                        <strong style='font-size:1.4rem; color:#ef4444;'>- ${abs(savings):,}</strong><br>
                        <span style='font-size:0.75rem; color:#94a3b8;'>Worse than baseline</span>
                    </div>
                    """, unsafe_allow_html=True
                )

# ==============================================================================
# TAB 4: 3D PCA SPACE MAPPING
# ==============================================================================
with tab_pca:
    st.markdown("### 3D PCA Latent Space Mapping")
    st.write(
        "Dimensionality reduction helps security engineers visualize high-dimensional financial space. "
        "Select different principal components below to see how fraudulent/anomalous transactions form distinct "
        "clusters separated from the normal background transaction volume."
    )
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        pc_x = st.selectbox("Select X-Axis Component", options=[f'V{i}' for i in range(1, 29)], index=13) # V14
    with col_p2:
        pc_y = st.selectbox("Select Y-Axis Component", options=[f'V{i}' for i in range(1, 29)], index=16) # V17
    with col_p3:
        pc_z = st.selectbox("Select Z-Axis Component", options=[f'V{i}' for i in range(1, 29)], index=2)  # V3
        
    # Sub-sample to keep 3D plotting responsive (e.g. max 3500 points)
    # Make sure to include all fraud cases in the sub-sample so they are highly visible!
    normal_subset = df[df['Class'] == 0].sample(min(3000, len(df[df['Class'] == 0])))
    fraud_subset = df[df['Class'] == 1]
    plot_3d_df = pd.concat([normal_subset, fraud_subset]).copy()
    plot_3d_df['Class Label'] = plot_3d_df['Class'].map({1: 'Fraudulent Transaction', 0: 'Normal Transaction'})
    
    st.markdown("##### Interactive 3D PCA Scatter Plot")
    fig_3d = px.scatter_3d(
        plot_3d_df,
        x=pc_x,
        y=pc_y,
        z=pc_z,
        color='Class Label',
        color_discrete_map={'Normal Transaction': 'rgba(14, 165, 233, 0.4)', 'Fraudulent Transaction': 'rgba(239, 68, 68, 0.95)'},
        opacity=0.8,
        size=plot_3d_df['Class'].map({1: 8, 0: 4}),  # Make fraud dots larger and highly visible!
        labels={pc_x: f'{pc_x} Component', pc_y: f'{pc_y} Component', pc_z: f'{pc_z} Component'},
        hover_data=['Amount']
    )
    
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1',
        height=550,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.92, xanchor="right", x=1)
    )
    # Force dark scene layout
    fig_3d.update_scenes(
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', backgroundcolor='rgba(11, 15, 25, 0.8)', showbackground=True),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', backgroundcolor='rgba(11, 15, 25, 0.8)', showbackground=True),
        zaxis=dict(gridcolor='rgba(255,255,255,0.06)', backgroundcolor='rgba(11, 15, 25, 0.8)', showbackground=True)
    )
    st.plotly_chart(fig_3d, use_container_width=True)
