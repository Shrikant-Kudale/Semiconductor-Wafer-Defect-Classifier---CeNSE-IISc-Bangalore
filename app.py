
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="Semiconductor AI", layout="wide", page_icon="⚡")

# Load models
xgb = joblib.load("xgb_defect_model.joblib")
scaler = joblib.load("feature_scaler.joblib")
le = joblib.load("label_encoder.joblib")

# Custom CSS: Strict 30% Purple, White, and Black Theme
st.markdown("""
    <style>
    :root {--purple: #6f42c1; --white: #ffffff; --black: #000000;}
    .stApp {background-color: var(--white);}
    h1, h2, h3, p, label {color: var(--black) !important;}
    .stMetric {background-color: #f8f9fa; border-left: 5px solid var(--purple); padding: 15px; border-radius: 5px;}
    .css-1r6slp0 {background-color: var(--white);}
    </style>
    """, unsafe_allow_html=True)

st.title("⚡ Semiconductor Defect Intelligence Dashboard")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("🔧 Metrology Inputs")
col1, col2 = st.sidebar.columns(2)

with col1:
    defect_count = st.slider("Defect Count", 0, 500, 50)
    defect_density = st.slider("Defect Density", 0.0, 1.0, 0.05)
    mean_rad = st.slider("Mean Radial Pos", 0.0, 1.0, 0.5)
    std_rad = st.slider("Std Radial Pos", 0.0, 0.5, 0.1)
    edge_ratio = st.slider("Edge Ratio", 0.0, 1.0, 0.2)
    center_ratio = st.slider("Center Ratio", 0.0, 1.0, 0.2)
    cluster_count = st.slider("Cluster Count", 0, 20, 2)
    max_cluster = st.slider("Max Cluster Size", 0, 100, 10)

with col2:
    oxide = st.slider("Oxide Thick (nm)", 80.0, 120.0, 100.0)
    crit_dim = st.slider("Crit Dimension (nm)", 30.0, 60.0, 45.0)
    overlay = st.slider("Overlay Error (nm)", 0.0, 10.0, 3.0)
    roughness = st.slider("Edge Roughness (nm)", 0.0, 5.0, 2.0)
    etch = st.slider("Etch Uniformity (%)", 80.0, 100.0, 96.0)
    particle = st.slider("Particle Index", 0.0, 20.0, 5.0)
    yield_pct = st.slider("Wafer Yield (%)", 0.0, 100.0, 90.0)

# Prediction Logic
feature_values = [defect_count, defect_density, mean_rad, std_rad, edge_ratio, center_ratio, 
                  cluster_count, max_cluster, oxide, crit_dim, overlay, roughness, etch, particle, yield_pct]
x = np.array(feature_values).reshape(1, -1)
pred_idx = xgb.predict(x)[0]
pred_class = le.inverse_transform([pred_idx])[0]
proba = xgb.predict_proba(x)[0]

# Main Dashboard Layout
m1, m2, m3 = st.columns(3)
m1.metric("Predicted Pattern", pred_class)
m2.metric("Confidence", f"{np.max(proba)*100:.1f}%")
m3.metric("Status", "CRITICAL" if pred_class != "Normal" else "STABLE")

# Visualizations with Purple Accent
st.markdown("### 📊 Classification Probability Distribution")
fig = go.Figure(data=[go.Bar(
    x=le.classes_, 
    y=proba,
    marker_color='#6f42c1' # Applied Purple Accent
)])
fig.update_layout(height=400, template="plotly_white", plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)

# Feature Summary Table
st.markdown("### ⚙️ Input Feature Profile")
df_display = pd.DataFrame([feature_values], columns=["Count", "Density", "MeanRad", "StdRad", "Edge", "Center", 
                                                    "Clusters", "MaxClust", "Oxide", "CD", "Overlay", "Rough", 
                                                    "Etch", "PartIdx", "Yield"])
st.table(df_display)
