# dashboard/app.py
# AI Trader Dashboard — live (Alpaca) or local JSON fallback + Refresh Now & Charts
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(page_title="AI Trader Dashboard", layout="wide")

# Optional: lightweight auto-refresh (install streamlit-autorefresh to enable)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=15_000, key="autorfr")  # refresh every 15s
except Exception:
    pass

# ---------- Optional timezone formatting ----------
try:
    import pytz
except Exception:
    pytz = None


# ===============================
# LIVE-FETCH FROM ALPACA (auto if secrets present)
# ===============================
def _get_secret(key: str, default: str | None = None) -> str | None:
    """
    Try Streamlit secrets first (on Streamlit Cloud), then environment variables.
    """
    try:
        v = st.secrets.get(key)  # type: ignore[attr-defined]
        if v is not None:
            return str(v)
    except Exception:
        pass
    return os.getenv(key, default)

ALPACA_API_KEY = _get_secret("ALPACA_API_KEY")
ALPACA_SECRET_KEY = _get_secret("ALPACA_SECRET_KEY")
ALPACA_PAPER = (_get_secret("ALPACA_PAPER", "true") or "true").lower() == "true"
USE_ALPACA = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)

if USE_ALPACA:
    try:
        from alpaca.trading.client import TradingClient
    except Exception as e:
        USE_ALPACA = False
        st.warning(f"alpaca-py import failed ({e}). Falling back to JSON files.")

def fetch_from_alpaca() -> tuple[dict, list[dict], list[dict]]:
    """
    Pull account, positions, and recent orders from Alpaca.
    Returns (account_dict, positions_list[dict], orders_list[dict]).
    """
    client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)

    # ---- Account
    acc = client.get_account()
    try:
        day_pl = float(acc.equity or 0) - float(acc.last_equity or 0)
    except Exception:
        day_pl = 0.0

    account = {
        "cash": float(getattr(acc, "cash", 0) or 0),
        "portfolio_value": float(getattr(acc, "portfolio_value", 0) or 0),
        "buying_power": float(getattr(acc, "buying_power", 0) or 0),
        "day_pl": day_pl,
        "paper": ALPACA_PAPER,
        "updated_at": getattr(acc, "updated_at", None) or getattr(acc, "last_update_at", None),
    }

    # ---- Positions
    positions: List[dict] = []
    try:
        for p in client.get_all_positions():
            positions.append({
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "current_price": getattr(p, "current_price", None),
                "market_value": p.market_value,
                "unrealized_pl": p.unrealized_pl,
                "unrealized_plpc": p.unrealized_plpc,
                "change_today": getattr(p, "change_today", None),                 # NEW
                "unrealized_intraday_pl": getattr(p, "unrealized_intraday_pl", None),  # NEW
            })
    except Exception:
        positions = []

    # ---- Orders (latest 50)
    orders: List[dict] = []
    try:
        for o in client.get_orders(status="all", limit=50):
            orders.append({
                "id": o.id,
                "symbol": getattr(o, "symbol", None),
                "side": getattr(o, "side", None),
                "type": getattr(o, "type", None),
                "qty": getattr(o, "qty", None),
                "notional": getattr(o, "notional", None),
                "limit_price": getattr(o, "limit_price", None),
                "submitted_at": getattr(o, "submitted_at", None),
                "filled_at": getattr(o, "filled_at", None),
                "filled_qty": getattr(o, "filled_qty", None),
                "status": getattr(o, "status", None),
            })
    except Exception:
        orders = []

    return account, positions, orders


# ===============================
# JSON fallback (file IO helpers)
# ===============================
CANDIDATE_DIRS = [
    Path.cwd(),
    Path.cwd() / "dashboard",
    Path.cwd() / "dashboard" / "data",
    Path.home() / "mcp" / "ai-trader-frontend" / "dashboard",
    Path.home() / "mcp" / "ai-trader-frontend" / "dashboard" / "data",
]
ACCOUNT_FILES = ["account.json", "data.json"]
POSITIONS_FILES = ["positions.json"]
ORDERS_FILES = ["orders.json"]

def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def load_json_from_candidates(names: List[str]) -> tuple[Optional[Dict[str, Any]], Optional[Path]]:
    paths: List[Path] = []
    for d in CANDIDATE_DIRS:
        for name in names:
            paths.append(d / name)
    found = _find_first_existing(paths)
    if not found:
        return None, None
    try:
        with open(found, "r") as f:
            return json.load(f), found
    except Exception:
        return None, found


# ===============================
# Normalizers & utils
# ===============================
def safe_to_datetime(series: pd.Series, utc: bool = True) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", utc=utc, infer_datetime_format=True)
    except Exception:
        return pd.to_datetime(pd.Series([None] * len(series)), errors="coerce", utc=utc)

def normalize_account(raw: Dict[str, Any] | None) -> Dict[str, Any]:
    if not raw:
        return {}
    acc = raw.get("account", raw) if isinstance(raw, dict) else raw

    def fget(key: str, default=None):
        v = acc.get(key, default)
        if isinstance(v, str):
            try:
                if v.isdigit():
                    return int(v)
                return float(v.replace(",", ""))
            except Exception:
                return v
        return v

    return {
        "cash": fget("cash", 0.0),
        "portfolio_value": fget("portfolio_value", fget("portfolioValue", 0.0)),
        "buying_power": fget("buying_power", fget("buyingPower", 0.0)),
        "day_pl": fget("day_pl", 0.0),
        "paper": bool(acc.get("paper", acc.get("is_paper", True))),
        "updated_at": acc.get("updated_at") or acc.get("timestamp") or acc.get("last_update_at"),
    }

def normalize_positions(raw: Any) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame(columns=[
            "symbol", "qty", "avg_entry", "market_price", "market_value",
            "unrealized_pl", "unrealized_plpc", "change_today", "unrealized_intraday_pl"
        ])

    if isinstance(raw, dict) and isinstance(raw.get("positions"), list):
        items = raw["positions"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    if not items:
        return pd.DataFrame(columns=[
            "symbol", "qty", "avg_entry", "market_price", "market_value",
            "unrealized_pl", "unrealized_plpc", "change_today", "unrealized_intraday_pl"
        ])

    df = pd.json_normalize(items)

    mappings = {
        "symbol": ["symbol", "asset_symbol", "asset.symbol"],
        "qty": ["qty", "quantity"],
        "avg_entry": ["avg_entry_price", "avg_entry", "avg_price"],
        "market_price": ["current_price", "market_price", "asset_current_price"],
        "market_value": ["market_value", "market_val"],
        "unrealized_pl": ["unrealized_pl", "unrealizedProfitLoss"],
        "unrealized_plpc": ["unrealized_plpc", "unrealizedProfitLossPct"],
        "change_today": ["change_today"],                        # NEW
        "unrealized_intraday_pl": ["unrealized_intraday_pl"],    # NEW
    }

    out: Dict[str, pd.Series] = {}
    for col, candidates in mappings.items():
        for c in candidates:
            if c in df.columns:
                if col == "symbol":
                    out[col] = df[c].astype(str)
                else:
                    out[col] = pd.to_numeric(df[c], errors="coerce")
                break
        if col not in out:
            out[col] = pd.Series([None] * len(df))

    ndf = pd.DataFrame(out)
    ndf = ndf.sort_values("market_value", ascending=False, na_position="last")
    return ndf

def normalize_orders(raw: Any) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame(columns=[
            "id", "symbol", "side", "type", "qty", "notional",
            "limit_price", "submitted_at", "filled_at", "filled_qty", "status"
        ])

    if isinstance(raw, dict) and isinstance(raw.get("orders"), list):
        items = raw["orders"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    if not items:
        return pd.DataFrame(columns=[
            "id", "symbol", "side", "type", "qty", "notional",
            "limit_price", "submitted_at", "filled_at", "filled_qty", "status"
        ])

    df = pd.json_normalize(items)

    def pick(*cols):
        for c in cols:
            if c in df.columns:
                return df[c]
        return pd.Series([None] * len(df))

    out = pd.DataFrame({
        "id": pick("id", "order_id"),
        "symbol": pick("symbol", "asset_symbol"),
        "side": pick("side"),
        "type": pick("type", "order_type"),
        "qty": pd.to_numeric(pick("qty", "quantity"), errors="coerce"),
        "notional": pd.to_numeric(pick("notional"), errors="coerce"),
        "limit_price": pd.to_numeric(pick("limit_price", "limitPrice"), errors="coerce"),
        "submitted_at": pick("submitted_at", "submittedAt", "created_at", "createdAt"),
        "filled_at": pick("filled_at", "filledAt"),
        "filled_qty": pd.to_numeric(pick("filled_qty", "filledQty"), errors="coerce"),
        "status": pick("status"),
    })

    out["submitted_at"] = safe_to_datetime(out["submitted_at"])
    out["filled_at"] = safe_to_datetime(out["filled_at"])

    # IMPORTANT: pandas uses 'ascending', not 'descending'
    # Newest first by default
    out = out.sort_values("submitted_at", ascending=False, na_position="last")
    return out

def to_dubai_str(ts: str | pd.Timestamp | None) -> Optional[str]:
    if not ts:
        return None
    try:
        if isinstance(ts, str):
            dt = pd.to_datetime(ts, utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        if pytz:
            dt = dt.tz_convert("Asia/Dubai")
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return None


# ===============================
# Load data (live if keys present; else JSON files)
# ===============================
if USE_ALPACA:
    try:
        account, positions_raw, orders_raw = fetch_from_alpaca()
    except Exception as e:
        st.error(f"Failed to fetch from Alpaca ({e}). Falling back to JSON.")
        USE_ALPACA = False

if not USE_ALPACA:
    account_raw, account_path = load_json_from_candidates(ACCOUNT_FILES)
    positions_raw, positions_path = load_json_from_candidates(POSITIONS_FILES)
    orders_raw, orders_path = load_json_from_candidates(ORDERS_FILES)

    account = normalize_account(account_raw)
    df_pos = normalize_positions(positions_raw if positions_raw else (account_raw.get("positions") if account_raw else None))
    df_ord = normalize_orders(orders_raw if orders_raw else (account_raw.get("orders") if account_raw else None))
else:
    df_pos = normalize_positions(positions_raw)
    df_ord = normalize_orders(orders_raw)

# Updated timestamp
updated_at = account.get("updated_at") if isinstance(account, dict) else None
if not updated_at:
    if not df_ord.empty and df_ord["submitted_at"].notna().any():
        updated_at = df_ord["submitted_at"].max()

paper_flag = bool(account.get("paper", True)) if isinstance(account, dict) else True


# ===============================
# Enhance positions with weights & returns (incl. cash)
# ===============================
import numpy as np

equity = float(account.get("portfolio_value") or 0.0)
cash = float(account.get("cash") or 0.0)

if not df_pos.empty:
    # ensure numeric types
    for col in ["market_value", "unrealized_pl", "unrealized_plpc", "change_today", "unrealized_intraday_pl"]:
        if col in df_pos.columns:
            df_pos[col] = pd.to_numeric(df_pos[col], errors="coerce")

    # weight from total equity (includes cash)
    if equity > 0 and "market_value" in df_pos.columns:
        df_pos["weight_pct"] = (df_pos["market_value"] / equity) * 100.0
    else:
        df_pos["weight_pct"] = 0.0

    # day P/L ($) and (%)
    df_pos["pl_day_$"] = df_pos["unrealized_intraday_pl"] if "unrealized_intraday_pl" in df_pos.columns else np.nan
    if "change_today" in df_pos.columns:
        df_pos["pl_day_%"] = df_pos["change_today"] * 100.0  # 0.0123 -> 1.23%
    else:
        df_pos["pl_day_%"] = np.nan

    # total P/L ($) and (%)
    df_pos["pl_total_$"] = df_pos["unrealized_pl"] if "unrealized_pl" in df_pos.columns else np.nan
    if "unrealized_plpc" in df_pos.columns:
        df_pos["pl_total_%"] = df_pos["unrealized_plpc"] * 100.0
    else:
        df_pos["pl_total_%"] = np.nan

    # sort by weight
    df_pos = df_pos.sort_values("weight_pct", ascending=False, na_position="last").reset_index(drop=True)

    # add CASH row for clarity
    if equity > 0 and cash >= 0:
        cash_weight = (cash / equity) * 100.0
        cash_row = {
            "symbol": "CASH",
            "qty": np.nan,
            "avg_entry": np.nan,
            "market_price": np.nan,
            "market_value": cash,
            "unrealized_pl": np.nan,
            "unrealized_plpc": np.nan,
            "change_today": np.nan,
            "unrealized_intraday_pl": np.nan,
            "weight_pct": cash_weight,
            "pl_day_$": np.nan,
            "pl_day_%": np.nan,
            "pl_total_$": np.nan,
            "pl_total_%": np.nan,
        }
        df_pos = pd.concat([df_pos, pd.DataFrame([cash_row])], ignore_index=True)


# ===============================
# UI — Header with Refresh Now
# ===============================
header_left, header_right = st.columns([1, 1])
with header_left:
    st.title("AI Trader Dashboard")
with header_right:
    st.markdown("<div style='text-align:right;'>", unsafe_allow_html=True)
    if st.button("🔄 Обновить сейчас", use_container_width=True):
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

caption = f"Updated at: {to_dubai_str(updated_at) or 'n/a'} • Paper: {paper_flag}"
st.caption(caption)

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cash", f"${float(account.get('cash', 0) or 0):,.2f}")
c2.metric("Portfolio value", f"${float(account.get('portfolio_value', 0) or 0):,.2f}")
c3.metric("Buying power", f"${float(account.get('buying_power', 0) or 0):,.2f}")
day_pl = float(account.get("day_pl", 0) or 0)
pl_prefix = "🟢" if day_pl > 0 else ("🔴" if day_pl < 0 else "⚪️")
c4.metric("Day P/L", f"{pl_prefix} ${day_pl:,.2f}")

# Data sources (debug info)
with st.expander("Data sources / mode", expanded=False):
    st.write("Mode:", "Alpaca LIVE" if USE_ALPACA else "Local JSON")
    if not USE_ALPACA:
        st.write("Account file path:", str(locals().get("account_path")) if locals().get("account_path") else "not found")
        st.write("Positions file path:", str(locals().get("positions_path")) if locals().get("positions_path") else "not found")
        st.write("Orders file path:", str(locals().get("orders_path")) if locals().get("orders_path") else "not found")


# ===============================
# Positions Table
# ===============================
st.subheader("Positions")
if df_pos.empty:
    st.info("No positions to display.")
else:
    fmt_pos = df_pos.copy()

    money_cols = ["avg_entry", "market_price", "market_value", "pl_day_$", "pl_total_$"]
    pct_cols = ["unrealized_plpc", "pl_day_%", "pl_total_%", "weight_pct"]

    if "qty" in fmt_pos.columns:
        fmt_pos["qty"] = fmt_pos["qty"].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip(".") if pd.notna(x) else "")

    for col in money_cols:
        if col in fmt_pos.columns:
            fmt_pos[col] = fmt_pos[col].map(lambda x: ("" if pd.isna(x) else f"${x:,.2f}"))

    for col in pct_cols:
        if col in fmt_pos.columns:
            fmt_pos[col] = fmt_pos[col].map(lambda x: ("" if pd.isna(x) else f"{x:.2f}%"))

    rename_map = {
        "avg_entry": "Avg price",
        "market_price": "Last price",
        "market_value": "Market value, $",
        "weight_pct": "Weight, %",
        "pl_day_$": "P/L day, $",
        "pl_day_%": "P/L day, %",
        "pl_total_$": "P/L total, $",
        "pl_total_%": "P/L total, %",
        "unrealized_plpc": "Unrealized PL, %",
    }
    fmt_pos = fmt_pos.rename(columns=rename_map)

    st.dataframe(fmt_pos[
        [c for c in ["symbol","qty","Avg price","Last price","Market value, $","Weight, %",
                     "P/L day, $","P/L day, %","P/L total, $","P/L total, %"]
         if c in fmt_pos.columns]
    ], use_container_width=True, hide_index=True)


# ===============================
# Orders Table
# ===============================
st.subheader("Recent orders")
if df_ord.empty:
    st.info("No orders to display.")
else:
    fmt_ord = df_ord.copy()
    # Format numbers
    for col in ["notional", "limit_price"]:
        if col in fmt_ord.columns:
            fmt_ord[col] = fmt_ord[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    for col in ["qty", "filled_qty"]:
        if col in fmt_ord.columns:
            fmt_ord[col] = fmt_ord[col].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip(".") if pd.notna(x) else "")
    # Friendly datetime strings
    for col in ["submitted_at", "filled_at"]:
        if col in fmt_ord.columns and pd.api.types.is_datetime64_any_dtype(df_ord[col]):
            try:
                if pytz:
                    fmt_ord[col] = fmt_ord[col].dt.tz_convert("Asia/Dubai").dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    fmt_ord[col] = fmt_ord[col].dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                fmt_ord[col] = fmt_ord[col].astype(str)
    st.dataframe(fmt_ord, use_container_width=True, hide_index=True)


# ===============================
# Charts
# ===============================
st.subheader("Charts")

chart_col1, chart_col2 = st.columns(2)

# --- Chart 1: Portfolio allocation by symbol (bar) ---
with chart_col1:
    st.markdown("**Portfolio allocation (by market value)**")
    if df_pos.empty or "symbol" not in df_pos.columns or "market_value" not in df_pos.columns:
        st.info("No positions to chart.")
    else:
        alloc = df_pos[["symbol", "market_value"]].dropna()
        alloc = alloc.groupby("symbol", as_index=False)["market_value"].sum().sort_values("market_value", ascending=False)
        alloc = alloc.set_index("symbol")
        st.bar_chart(alloc)  # Streamlit native chart

# --- Chart 2: Filled notional by day (line) ---
with chart_col2:
    st.markdown("**Filled notional by day**")
    if df_ord.empty:
        st.info("No orders to chart.")
    else:
        # Use filled_at if present, else submitted_at
        ts = df_ord["filled_at"].copy()
        if ts.isna().all():
            ts = df_ord["submitted_at"].copy()
        ts = pd.to_datetime(ts, errors="coerce", utc=True)

        # Compute notional fallback if NaN: filled_qty * limit_price
        notional = pd.to_numeric(df_ord.get("notional"), errors="coerce")
        if notional.isna().all():
            filled_qty = pd.to_numeric(df_ord.get("filled_qty"), errors="coerce")
            limit_price = pd.to_numeric(df_ord.get("limit_price"), errors="coerce")
            notional = filled_qty * limit_price

        df_line = pd.DataFrame({"ts": ts, "notional": notional}).dropna()
        if df_line.empty:
            st.info("Not enough filled data to chart.")
        else:
            df_line["date"] = df_line["ts"].dt.tz_convert("Asia/Dubai").dt.date if pytz else df_line["ts"].dt.date
            daily = df_line.groupby("date", as_index=False)["notional"].sum().sort_values("date")
            daily = daily.set_index("date")
            st.line_chart(daily)
