import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta

# ====== PAGE CONFIG ======
st.set_page_config(page_title="Investlink AI Trader", layout="wide")

# ====== CUSTOM CSS ======
st.markdown("""
    <style>
    /* Glass effect background */
    .stApp {
        background: radial-gradient(circle at 30% 10%, #0a0a0a, #050505);
        color: #e6e6e6;
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #ffffffcc;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 25px rgba(255,255,255,0.08);
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 25px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 0 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease-in-out;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 0 35px rgba(0, 0, 0, 0.5);
    }

    /* Metric labels */
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #cccccc !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Plotly chart container */
    .plotly-chart {
        border-radius: 20px;
        overflow: hidden;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ====== HEADER ======
st.markdown("<h1 class='main-title'>📊 Investlink AI Dashboard</h1>", unsafe_allow_html=True)

# ====== MOCK DATA ======
# Example portfolio data
data = {
    "symbol": ["AAPL", "TSLA", "MSFT", "NVDA", "GOOG"],
    "price": [235.5, 258.7, 412.3, 894.2, 167.9],
    "change": [1.2, -0.5, 0.8, 2.1, -1.1],
    "allocation": [25, 20, 20, 20, 15]
}
df = pd.DataFrame(data)

# ====== TOP ROW: Metrics ======
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Portfolio Value", "$126,450", "+1.4%")
with col2:
    st.metric("Today's P/L", "+$1,125", "+0.9%")
with col3:
    st.metric("Open Positions", "12")
with col4:
    st.metric("Cash Balance", "$14,820")

st.markdown("")

# ====== MIDDLE ROW: Portfolio Table ======
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📁 Current Holdings")
    st.dataframe(df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ====== BOTTOM ROW: Plotly Chart ======
# Generate fake price data
dates = [datetime.now() - timedelta(days=i) for i in range(60)]
prices = [400 + i * 0.5 + (i % 5) * 2 for i in range(60)]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates, y=prices, mode='lines', fill='tozeroy',
    line=dict(color='#00CC96', width=3),
    hovertemplate="%{x|%b %d}: $%{y:.2f}"
))
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
)

st.markdown("<div class='glass-card plotly-chart'>", unsafe_allow_html=True)
st.subheader("📈 Portfolio Value Over Time")
st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ====== FOOTER ======
st.markdown("<br><center style='color:#777;'>© 2025 Investlink | Powered by Alpaca & Streamlit</center>", unsafe_allow_html=True)
