# dashboard/app.py
# AI Trader Dashboard — live Alpaca + Total P/L from first equity + $/% P&L + color-coded table & chart
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


# ===============================
# Helpers
# ===============================
def _get_secret(key: str, default: str | None = None) -> str | None:
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

def _to_float(x, default: float = 0.0) -> float:
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

    if not account:
        sample = _load_json("sample_portfolio.json")
        account = sample.get("account", {})
        positions = []

    return {"account": account, "positions": positions, "api": api}

def load_equity_history(api, period: str, timeframe: str) -> pd.DataFrame:
    if api is not None:
        try:
            hist = api.get_portfolio_history(period=period, timeframe=timeframe)
            ts = hist.timestamp if hasattr(hist, "timestamp") else hist["timestamp"]
            eq = hist.equity if hasattr(hist, "equity") else hist["equity"]
            df = pd.DataFrame({
                "time": pd.to_datetime(ts, unit="s"),
                "equity": pd.to_numeric(eq, errors="coerce")
            }).dropna()
            return df
        except Exception as e:
            st.warning(f"⚠️ Failed to fetch equity history ({period}/{timeframe}): {e}")

    sample = _load_json("sample_portfolio_history.json")
    if "timestamp" in sample and "equity" in sample:
        df = pd.DataFrame({
            "time": pd.to_datetime(sample["timestamp"], unit="s"),
            "equity": pd.to_numeric(sample["equity"], errors="coerce")
        }).dropna()
        return df

    base = 25_000.0
    rng = pd.date_range(end=pd.Timestamp.utcnow().floor("T"), periods=120, freq="T")
    drift = base * (1 + 0.00015 * np.arange(len(rng)))
    noise = np.random.normal(0, base * 0.0008, len(rng))
    series = pd.Series(drift + noise, index=rng).rolling(3, min_periods=1).mean()
    return pd.DataFrame({"time": series.index, "equity": series.values})

def get_initial_equity(api, fallback_equity_now: float) -> float:
    if api is not None:
        try:
            hist = api.get_portfolio_history(period="all", timeframe="1D")
            eq = hist.equity if hasattr(hist, "equity") else hist["equity"]
            if eq and len(eq) > 0:
                first = float(pd.to_numeric(eq, errors="coerce")[0])
                if first > 0:
                    return first
        except Exception:
            pass
    sample = _load_json("sample_portfolio_history.json")
    if "equity" in sample and len(sample["equity"]) > 0:
        first = float(pd.to_numeric(sample["equity"], errors="coerce")[0])
        if first > 0:
            return first
    return float(fallback_equity_now)


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

initial_equity = get_initial_equity(api, equity_now)
total_pl = equity_now - initial_equity


# ===============================
# UI — Portfolio Summary
# ===============================
st.subheader("📊 Portfolio summary")

def pl_text(amount: float, base: float) -> str:
    pct = (amount / base * 100.0) if base else 0.0
    color = "green" if amount > 0 else "red" if amount < 0 else "gray"
    sign_amt = "+" if amount > 0 else ""
    sign_pct = "+" if pct > 0 else ""
    return f":{color}[{sign_amt}{amount:,.2f} ({sign_pct}{pct:.2f}%)]"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💵 Cash", f"${cash:,.2f}")
c2.metric("📈 Portfolio value", f"${portfolio_value:,.2f}")
c3.metric("⚙️ Buying power", f"${buying_power:,.2f}")
c4.metric("📅 Day P/L", pl_text(day_pl, last_equity if last_equity else equity_now))
c5.metric("💰 Total P/L", pl_text(total_pl, initial_equity if initial_equity else equity_now))


# ===============================
# Equity Chart (selectable range)
# ===============================
st.subheader("📉 Equity chart")

left, right = st.columns([2, 1])
with left:
    period_label = st.selectbox(
        "Range", ["1D", "1W", "1M", "3M", "1Y", "All"], index=0,
        help="Portfolio equity over selected range"
    )
with right:
    timeframe_label = st.selectbox(
        "Timeframe", ["Auto", "1Min", "5Min", "15Min", "1H", "1D"], index=0,
        help="Data granularity"
    )

def default_timeframe(period: str) -> str:
    if period == "1D": return "5Min"
    if period == "1W": return "15Min"
    if period in ("1M", "3M"): return "1H"
    return "1D"

mapped_timeframe = default_timeframe(period_label) if timeframe_label == "Auto" else timeframe_label
period_key = period_label.lower() if period_label != "All" else "all"

eq_df = load_equity_history(api, period=period_key, timeframe=mapped_timeframe)

if not eq_df.empty:
    eq_df = eq_df.sort_values("time").set_index("time")
    if period_label == "1D" and len(eq_df) >= 2:
        day_pl_chart = float(eq_df["equity"].iloc[-1]) - float(eq_df["equity"].iloc[0])
        st.caption(f"Session P/L (from chart): {pl_text(day_pl_chart, float(eq_df['equity'].iloc[0]))}")
    st.line_chart(eq_df["equity"], height=280)
else:
    st.info("No equity history available for the selected range.")


# ===============================
# Open Positions — FULL columns with total change & share of equity
# ===============================
def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"

def build_positions_df(positions: List[Any], equity_now_val: float) -> pd.DataFrame:
    rows = []
    for p in positions:
        try:
            symbol = getattr(p, "symbol", "")
            qty = _to_float(getattr(p, "qty", 0))
            avg_entry = _to_float(getattr(p, "avg_entry_price", 0))
            current_price = _to_float(getattr(p, "current_price", 0))
            cost_basis = _to_float(getattr(p, "cost_basis", 0))
            market_value = _to_float(getattr(p, "market_value", 0))
            unrealized_pl = _to_float(getattr(p, "unrealized_pl", 0))
            unrealized_plpc = _to_float(getattr(p, "unrealized_plpc", 0)) * 100.0  # %
            intraday_pl = _to_float(getattr(p, "unrealized_intraday_pl", 0))
            intraday_plpc = _to_float(getattr(p, "unrealized_intraday_plpc", 0)) * 100.0  # %
            change_today = _to_float(getattr(p, "change_today", 0)) * 100.0  # %

            # New: share of equity & P/L contribution
            share_of_equity = (market_value / equity_now_val * 100.0) if equity_now_val else 0.0
            pl_contribution_pct = (unrealized_pl / equity_now_val * 100.0) if equity_now_val else 0.0

            rows.append({
                "Symbol": symbol,
                "Qty": qty,
                "Avg entry": avg_entry,
                "Current price": current_price,
                "Cost basis": cost_basis,
                "Market value": market_value,

                # Total change (a.k.a. Unrealized P/L)
                "Total change ($)": unrealized_pl,
                "Total change (%)": unrealized_plpc,

                # Intraday change
                "Intraday P/L ($)": intraday_pl,
                "Intraday P/L (%)": intraday_plpc,
                "Change today (%)": change_today,

                # New analytics
                "Share of equity (%)": share_of_equity,
                "P/L contribution (%)": pl_contribution_pct,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

if positions:
    st.subheader("📋 Open positions")

    df = build_positions_df(positions, equity_now)
    if not df.empty:
        def color_posneg(v):
            if pd.isna(v): return ""
            if v > 0: return "color: green"
            if v < 0: return "color: red"
            return "color: gray"

        money_cols = ["Market value", "Total change ($)", "Intraday P/L ($)"]
        pct_cols = [
            "Total change (%)", "Intraday P/L (%)", "Change today (%)",
            "Share of equity (%)", "P/L contribution (%)"
        ]

        st.dataframe(
            df.style
              .format({
                  "Qty": "{:,.0f}",
                  "Avg entry": "${:,.2f}",
                  "Current price": "${:,.2f}",
                  "Cost basis": "${:,.2f}",
                  "Market value": "${:,.2f}",
                  "Total change ($)": "${:,.2f}",
                  "Intraday P/L ($)": "${:,.2f}",
                  "Total change (%)": fmt_pct,
                  "Intraday P/L (%)": fmt_pct,
                  "Change today (%)": fmt_pct,
                  "Share of equity (%)": fmt_pct,
                  "P/L contribution (%)": fmt_pct,
              })
              .applymap(color_posneg, subset=money_cols + pct_cols),
            use_container_width=True,
            height=460
        )
    else:
        st.info("No open positions found.")
else:
    st.info("No open positions found.")
