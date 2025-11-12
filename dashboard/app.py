# dashboard/app.py
# AI Trader Dashboard — live Alpaca + Total P/L from first equity + $/% P&L + color-coded table & chart + Open Orders
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

def _fmt_money(x: float) -> str:
    return f"${x:,.2f}"

def _pl_html(amount: float, base: float) -> str:
    """Возвращает HTML-строку с окрашенным P/L: '+1,234.56 (+2.34%)'."""
    pct = (amount / base * 100.0) if base else 0.0
    color = "rgb(16,185,129)" if amount > 0 else "rgb(239,68,68)" if amount < 0 else "#9CA3AF"
    text = f"{amount:+,.2f} ({pct:+.2f}%)"
    return f"<span style='color:{color};font-weight:600'>{text}</span>"


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

# KPI блок
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("💵 Cash", _fmt_money(cash))
with c2:
    st.metric("📈 Portfolio value", _fmt_money(portfolio_value))
with c3:
    st.metric("⚙️ Buying power", _fmt_money(buying_power))

# P/L с устойчивой подсветкой в отдельной строке
p1, p2 = st.columns(2)
with p1:
    st.markdown("**📅 Day P/L**")
    st.markdown(_pl_html(day_pl, last_equity if last_equity else equity_now), unsafe_allow_html=True)
with p2:
    st.markdown("**💰 Total P/L**")
    st.markdown(_pl_html(total_pl, initial_equity if initial_equity else equity_now), unsafe_allow_html=True)


# ===============================
# Equity Chart (selectable range) — dynamic axis
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

# тумблер: начинать ось Y с нуля
start_at_zero = st.checkbox("Start Y-axis at zero", value=False)

if not eq_df.empty:
    eq_df = eq_df.sort_values("time")
    # Быстрый P/L по сессии из графика
    if period_label == "1D" and len(eq_df) >= 2:
        day_pl_chart = float(eq_df["equity"].iloc[-1]) - float(eq_df["equity"].iloc[0])
        base_eq = float(eq_df["equity"].iloc[0])
        st.markdown(
            f"<div style='font-size:0.9rem;opacity:0.8'>Session P/L (from chart): {_pl_html(day_pl_chart, base_eq)}</div>",
            unsafe_allow_html=True
        )

    # Altair с аккуратной шкалой
    try:
        import altair as alt
        y_min = float(eq_df["equity"].min())
        y_max = float(eq_df["equity"].max())
        span = max(y_max - y_min, 1e-6)
        pad = span * 0.06
        domain = [0, y_max + pad] if start_at_zero else [y_min - pad, y_max + pad]

        chart = (
            alt.Chart(eq_df)
            .mark_line()
            .encode(
                x=alt.X("time:T", title="Time"),
                y=alt.Y("equity:Q", title="Equity", scale=alt.Scale(domain=domain, nice=False)),
                tooltip=[
                    alt.Tooltip("time:T", title="Time"),
                    alt.Tooltip("equity:Q", title="Equity", format="$.2f"),
                ],
            )
            .properties(height=280)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.warning("Altair is unavailable, falling back to a basic line chart.")
        st.line_chart(eq_df.set_index("time")["equity"], height=280, use_container_width=True)
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

                # Total change (Unrealized P/L)
                "Total change ($)": unrealized_pl,
                "Total change (%)": unrealized_plpc,

                # Intraday change
                "Intraday P/L ($)": intraday_pl,
                "Intraday P/L (%)": intraday_plpc,
                "Change today (%)": change_today,

                # Analytics
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


# ===============================
# Open Orders — separate from Positions
# ===============================
def load_open_orders(api):
    """Возвращает список открытых ордеров Alpaca. Фоллбэк — пусто."""
    if api is None:
        return []
    try:
        orders = api.list_orders(
            status="open",         # new/accepted/partially_filled/open
            nested=False,
            direction="desc",
            limit=200
        )
        return orders
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch open orders: {e}")
        return []

def build_orders_df(orders: List[Any]) -> pd.DataFrame:
    rows = []
    for o in orders:
        try:
            rows.append({
                "Submitted": str(getattr(o, "submitted_at", "")),
                "Symbol": getattr(o, "symbol", ""),
                "Side": getattr(o, "side", ""),
                "Type": getattr(o, "type", ""),
                "Qty": _to_float(getattr(o, "qty", 0)),
                "Limit price": _to_float(getattr(o, "limit_price", 0)),
                "Stop price": _to_float(getattr(o, "stop_price", 0)),
                "Trail": _to_float(getattr(o, "trail_price", 0)) or _to_float(getattr(o, "trail_percent", 0)),
                "Time in force": getattr(o, "time_in_force", ""),
                "Status": getattr(o, "status", ""),
                "Filled qty": _to_float(getattr(o, "filled_qty", 0)),
                "Avg fill": _to_float(getattr(o, "filled_avg_price", 0)),
                "ID": getattr(o, "id", ""),
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

orders = load_open_orders(api)
if orders:
    st.subheader("🧾 Open orders")
    odf = build_orders_df(orders)
    if not odf.empty:
        def color_side(v):
            if pd.isna(v): return ""
            if str(v).lower() == "buy": return "color: green"
            if str(v).lower() == "sell": return "color: red"
            return ""
        def color_status(v):
            if pd.isna(v): return ""
            s = str(v).lower()
            if s in ("new","accepted","open","partially_filled"): return "color: #2563EB"  # blue
            if s in ("rejected","canceled","stopped","expired"): return "color: #EF4444"   # red
            return "color: #6B7280"  # gray

        st.dataframe(
            odf.style
               .format({
                   "Qty": "{:,.0f}",
                   "Limit price": "${:,.2f}",
                   "Stop price": "${:,.2f}",
                   "Trail": "{:,.2f}",
                   "Filled qty": "{:,.0f}",
                   "Avg fill": "${:,.2f}",
               })
               .applymap(color_side, subset=["Side"])
               .applymap(color_status, subset=["Status"]),
            use_container_width=True,
            height=360
        )
else:
    st.caption("No open orders.")
