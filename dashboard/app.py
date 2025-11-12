# dashboard/app.py
# AI Trader Dashboard — Alpaca live data + Total P/L + color-coded metrics + Equity chart

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="AI Trader Dashboard", layout="wide")

# ---------- Optional auto-refresh ----------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=15_000, key="autorfr")  # refresh every 15s
except Exception:
    pass

# ---------- Optional timezone ----------
try:
    import pytz  # noqa
except Exception:
    pass


# ===============================
# Helpers
# ===============================
def _get_secret(key: str, default: str | None = None) -> str | None:
    """Try Streamlit secrets first, then env vars."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)


def _load_json(path: str | Path) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _to_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


# ===============================
# Live fetch (Alpaca) or fallback
# ===============================
def load_portfolio_data() -> Dict[str, Any]:
    api_key = _get_secret("ALPACA_API_KEY")
    api_secret = _get_secret("ALPACA_SECRET_KEY")
    base_url = _get_secret("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

    api = None
    account: Dict[str, Any] = {}
    positions: List[Any] = []

    if api_key and api_secret:
        try:
            import alpaca_trade_api as tradeapi  # type: ignore
            api = tradeapi.REST(api_key, api_secret, base_url, api_version="v2")
            account = api.get_account()._raw
            positions = api.list_positions()
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch live data: {e}")

    # Fallback if no live data
    if not account:
        sample = _load_json("sample_portfolio.json")
        account = sample.get("account", {})
        positions = []

    return {"account": account, "positions": positions, "api": api}


def load_equity_history(api, period: str, timeframe: str) -> pd.DataFrame:
    """
    period: '1D','1W','1M','3M','1Y','all'
    timeframe: '1Min','5Min','15Min','1H','1D'
    """
    # Live from Alpaca
    if api is not None:
        try:
            hist = api.get_portfolio_history(period=period, timeframe=timeframe)
            ts = hist.timestamp if hasattr(hist, "timestamp") else hist["timestamp"]
            eq = hist.equity if hasattr(hist, "equity") else hist["equity"]
            df = pd.DataFrame({"time": pd.to_datetime(ts, unit="s"),
                               "equity": pd.to_numeric(eq, errors="coerce")}).dropna()
            return df
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch equity history ({period}/{timeframe}): {e}")

    # Fallback from local sample
    sample = _load_json("sample_portfolio_history.json")
    if "timestamp" in sample and "equity" in sample:
        df = pd.DataFrame({
            "time": pd.to_datetime(sample["timestamp"], unit="s"),
            "equity": pd.to_numeric(sample["equity"], errors="coerce")
        }).dropna()
        return df

    # Synthetic minimal fallback
    base = 25_000.0
    rng = pd.date_range(end=pd.Timestamp.utcnow().floor("T"), periods=120, freq="T")
    drift = base * (1 + 0.00015 * np.arange(len(rng)))
    noise = np.random.normal(0, base * 0.0008, len(rng))
    series = pd.Series(drift + noise, index=rng).rolling(3, min_periods=1).mean()
    return pd.DataFrame({"time": series.index, "equity": series.values})


# ===============================
# Load & Process Data
# ===============================
data = load_portfolio_data()
account: Dict[str, Any] = data.get("account", {})
positions: List[Any] = data.get("positions", [])
api = data.get("api")

cash = _to_float(account.get("cash", 0))
portfolio_value = _to_float(account.get("portfolio_value", 0))
buying_power = _to_float(account.get("buying_power", 0))
equity_now = _to_float(account.get("equity", portfolio_value))
last_equity = _to_float(account.get("last_equity", equity_now))
day_pl = equity_now - last_equity

# Total P/L: при наличии своего трекера подставь сюда значение initial_investment
initial_investment = _to_float(account.get("initial_margin_requirement", 0)) or (portfolio_value - 5000)
total_pl = equity_now - initial_investment


# ===============================
# UI — Portfolio Summary
# ===============================
st.subheader("📊 Portfolio summary")

def color_text(value: float) -> str:
    color = "green" if value > 0 else "red" if value < 0 else "gray"
    sign = "+" if value > 0 else ""
    return f":{color}[{sign}{value:,.2f}]"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💵 Cash", f"${cash:,.2f}")
c2.metric("📈 Portfolio value", f"${portfolio_value:,.2f}")
c3.metric("⚙️ Buying power", f"${buying_power:,.2f}")
c4.metric("📅 Day P/L", color_text(day_pl))
c5.metric("💰 Total P/L", color_text(total_pl))


# ===============================
# Equity Chart (selectable range)
# ===============================
st.subheader("📉 Equity chart")

left, right = st.columns([2, 1])
with left:
    period_label = st.selectbox(
        "Range",
        ["1D", "1W", "1M", "3M", "1Y", "All"],
        index=0,
        help="Portfolio equity over selected range"
    )
with right:
    timeframe_label = st.selectbox(
        "Timeframe",
        ["Auto", "1Min", "5Min", "15Min", "1H", "1D"],
        index=0,
        help="Data granularity"
    )

def default_timeframe(period: str) -> str:
    if period == "1D":
        return "5Min"
    if period == "1W":
        return "15Min"
    if period in ("1M", "3M"):
        return "1H"
    return "1D"

mapped_timeframe = default_timeframe(period_label) if timeframe_label == "Auto" else timeframe_label
period_key = period_label.lower() if period_label != "All" else "all"

eq_df = load_equity_history(api, period=period_key, timeframe=mapped_timeframe)

if not eq_df.empty:
    eq_df = eq_df.sort_values("time").set_index("time")
    if period_label == "1D" and len(eq_df) >= 2:
        day_pl_chart = float(eq_df["equity"].iloc[-1]) - float(eq_df["equity"].iloc[0])
        st.caption(f"Session P/L (from chart): {color_text(day_pl_chart)}")
    st.line_chart(eq_df["equity"], height=280)
else:
    st.info("No equity history available for the selected range.")


# ===============================
# Positions Table (color-coded)
# ===============================
if positions:
    st.subheader("📋 Open positions")

    rows = []
    for p in positions:
        try:
            rows.append({
                "Symbol": getattr(p, "symbol", ""),
                "Qty": _to_float(getattr(p, "qty", 0)),
                "Market value": _to_float(getattr(p, "market_value", 0)),
                "Unrealized P/L": _to_float(getattr(p, "unrealized_pl", 0)),
                "Change today": _to_float(getattr(p, "unrealized_intraday_pl", 0)),
            })
        except Exception:
            pass

    if rows:
        df = pd.DataFrame(rows)

        def style_pl(v):
            if pd.isna(v):
                return ""
            if v > 0:
                return "color: green"
            if v < 0:
                return "color: red"
            return "color: gray"

        st.dataframe(
            df.style.format({
                "Qty": "{:,.0f}",
                "Market value": "${:,.2f}",
                "Unrealized P/L": "${:,.2f}",
                "Change today": "${:,.2f}",
            }).applymap(style_pl, subset=["Unrealized P/L", "Change today"]),
            use_container_width=True,
            height=360
        )
    else:
        st.info("No open positions found.")
else:
    st.info("No open positions found.")
