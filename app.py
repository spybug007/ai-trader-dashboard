import json, requests, streamlit as st
from datetime import datetime

# GitHub raw URL настроен под твой репозиторий
GITHUB_USER = "spybug007"
REPO = "ai-trader-frontend"
BRANCH = "main"

RAW_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO}/{BRANCH}/data.json"

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("📈 AI Trading Dashboard")

@st.cache_data(ttl=60)
def fetch_data():
    r = requests.get(RAW_URL, timeout=10)
    r.raise_for_status()
    return r.json()

def fmt_usd(x):
    return f"${x:,.2f}" if isinstance(x, (int,float)) and x is not None else "-"

try:
    data = fetch_data()
    updated = data.get("updated_at_utc", "")
    account = data.get("account", {})
    positions = data.get("positions", [])
    orders = data.get("orders", [])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", account.get("status", "-"))
    col2.metric("Cash", fmt_usd(account.get("cash")))
    col3.metric("Buying Power", fmt_usd(account.get("buying_power")))
    col4.metric("Portfolio Value", fmt_usd(account.get("portfolio_value")))

    st.caption(f"Last update (UTC): {updated}")

    st.subheader("Open Positions")
    if positions:
        st.dataframe(positions, use_container_width=True)
    else:
        st.info("No open positions")

    st.subheader("Recent Orders")
    if orders:
        st.dataframe(orders, use_container_width=True)
    else:
        st.info("No recent orders")

except Exception as e:
    st.error(f"Failed to load data.json from GitHub: {e}")
    st.write("Make sure GitHub Actions has generated and committed data.json.")
