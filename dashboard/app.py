# dashboard/app.py
# Streamlit dashboard for AI Trader (Paper/Live)
# Requires: streamlit, pandas
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="AI Trader Dashboard", layout="wide")

# Where to look for data files produced by your jobs
CANDIDATE_DIRS = [
    Path.cwd(),                             # current dir
    Path.cwd() / "dashboard",               # ./dashboard
    Path.cwd() / "dashboard" / "data",      # ./dashboard/data
    Path.home() / "mcp" / "ai-trader-frontend" / "dashboard",       # ~/mcp/ai-trader-frontend/dashboard
    Path.home() / "mcp" / "ai-trader-frontend" / "dashboard" / "data",
]

ACCOUNT_FILES = ["account.json", "data.json"]  # some projects save as data.json
POSITIONS_FILES = ["positions.json"]
ORDERS_FILES = ["orders.json"]


# -----------------------------
# IO helpers
# -----------------------------
def find_first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_json_from_candidates(names: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    paths: List[Path] = []
    for d in CANDIDATE_DIRS:
        for name in names:
            paths.append(d / name)
    found = find_first_existing(paths)
    if not found:
        return None, None
    try:
        with open(found, "r") as f:
            return json.load(f), found
    except Exception:
        return None, found


# -----------------------------
# Normalizers
# -----------------------------
def safe_to_datetime(series: pd.Series, utc: bool = True) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", utc=utc, infer_datetime_format=True)
    except Exception:
        return pd.to_datetime(pd.Series([None] * len(series)), errors="coerce", utc=utc)


def normalize_account(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports two shapes:
    - Alpaca-like 'account' dict
    - A custom 'data.json' with { 'account': {...}, 'positions': [...], 'orders': [...] }
    """
    if raw is None:
        return {}

    # unwrap if it's a container
    if "account" in raw and isinstance(raw["account"], dict):
        acc = raw["account"]
    else:
        acc = raw

    # Normalize numeric fields if possible
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
        "day_pl": fget("day_pl", fget("equity", 0.0)) if "day_pl" in acc else acc.get("day_pl", 0.0),
        "paper": bool(acc.get("paper", acc.get("is_paper", True))),
        "updated_at": acc.get("updated_at") or acc.get("timestamp") or acc.get("last_update_at"),
    }


def normalize_positions(raw: Any) -> pd.DataFrame:
    """
    Accepts:
      - list of Alpaca positions (dicts)
      - dict with 'positions': [...]
    """
    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, dict) and "positions" in raw and isinstance(raw["positions"], list):
        items = raw["positions"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    if not items:
        return pd.DataFrame(columns=[
            "symbol", "qty", "avg_entry", "market_price", "market_value",
            "unrealized_pl", "unrealized_plpc"
        ])

    df = pd.json_normalize(items)
    # Try to map common fields
    mappings = {
        "symbol": ["symbol", "asset_symbol", "asset.symbol"],
        "qty": ["qty", "quantity"],
        "avg_entry": ["avg_entry_price", "avg_entry", "avg_price"],
        "market_price": ["current_price", "market_price", "asset_current_price"],
        "market_value": ["market_value", "market_val"],
        "unrealized_pl": ["unrealized_pl", "unrealizedProfitLoss"],
        "unrealized_plpc": ["unrealized_plpc", "unrealizedProfitLossPct"],
    }

    out = {}
    for col, candidates in mappings.items():
        for c in candidates:
            if c in df.columns:
                out[col] = pd.to_numeric(df[c], errors="coerce")
                break
        if col not in out:
            out[col] = pd.Series([None] * len(df))

    # symbol as string
    if "symbol" in df.columns:
        out["symbol"] = df["symbol"].astype(str)
    elif "asset_symbol" in df.columns:
        out["symbol"] = df["asset_symbol"].astype(str)
    elif "asset.symbol" in df.columns:
        out["symbol"] = df["asset.symbol"].astype(str)

    ndf = pd.DataFrame(out)
    # Sort by market value desc by default
    ndf = ndf.sort_values("market_value", ascending=False, na_position="last")
    return ndf


def normalize_orders(raw: Any) -> pd.DataFrame:
    """
    Accepts:
      - list of Alpaca orders (dicts)
      - dict with 'orders': [...]
    """
    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, dict) and "orders" in raw and isinstance(raw["orders"], list):
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

    # Columns mapping
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

    # Parse datetimes
    out["submitted_at"] = safe_to_datetime(out["submitted_at"])
    out["filled_at"] = safe_to_datetime(out["filled_at"])

    # IMPORTANT: pandas uses 'ascending', not 'descending'
    # Show newest first in the UI by default
    out = out.sort_values("submitted_at", ascending=False, na_position="last")
    return out


# -----------------------------
# Load data
# -----------------------------
account_raw, account_path = load_json_from_candidates(ACCOUNT_FILES)
positions_raw, positions_path = load_json_from_candidates(POSITIONS_FILES)
orders_raw, orders_path = load_json_from_candidates(ORDERS_FILES)

account = normalize_account(account_raw)
df_pos = normalize_positions(positions_raw if positions_raw else (account_raw.get("positions") if account_raw else None))
df_ord = normalize_orders(orders_raw if orders_raw else (account_raw.get("orders") if account_raw else None))

# Updated time
updated_at = account.get("updated_at")
if pd.isna(updated_at) or not updated_at:
    # fall back to newest timestamp among orders
    if not df_ord.empty and df_ord["submitted_at"].notna().any():
        updated_at = df_ord["submitted_at"].max().isoformat()
    else:
        updated_at = None

paper_flag = account.get("paper", True)

# -----------------------------
# UI
# -----------------------------
st.title("AI Trader Dashboard")
subtitle = f"Updated at: {updated_at}" if updated_at else "Updated at: n/a"
subtitle += f" • Paper: {str(bool(paper_flag))}"
st.caption(subtitle)

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cash", f"${account.get('cash', 0):,.2f}")
c2.metric("Portfolio value", f"${account.get('portfolio_value', 0):,.2f}")
c3.metric("Buying power", f"${account.get('buying_power', 0):,.2f}")
c4.metric("Day P/L", f"${account.get('day_pl', 0):,.2f}")

# Helpful debug about where data was loaded from
with st.expander("Data sources", expanded=False):
    st.write("Account file:", str(account_path) if account_path else "not found")
    st.write("Positions file:", str(positions_path) if positions_path else "not found")
    st.write("Orders file:", str(orders_path) if orders_path else "not found")

# Positions
st.subheader("Positions")
if df_pos.empty:
    st.info("No positions to display.")
else:
    fmt_pos = df_pos.copy()
    money_cols = ["avg_entry", "market_price", "market_value", "unrealized_pl"]
    pct_cols = ["unrealized_plpc"]
    for col in money_cols:
        if col in fmt_pos.columns:
            fmt_pos[col] = fmt_pos[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    for col in pct_cols:
        if col in fmt_pos.columns:
            fmt_pos[col] = fmt_pos[col].map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "")
    if "qty" in fmt_pos.columns:
        fmt_pos["qty"] = fmt_pos["qty"].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip(".") if pd.notna(x) else "")
    st.dataframe(fmt_pos, use_container_width=True, hide_index=True)

# Recent orders
st.subheader("Recent orders")
if df_ord.empty:
    st.info("No orders to display.")
else:
    # Let users toggle sort direction client-side with Streamlit table, but default newest first
    fmt_ord = df_ord.copy()
    money_cols = ["notional", "limit_price"]
    qty_cols = ["qty", "filled_qty"]
    for col in money_cols:
        if col in fmt_ord.columns:
            fmt_ord[col] = fmt_ord[col].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
    for col in qty_cols:
        if col in fmt_ord.columns:
            fmt_ord[col] = fmt_ord[col].map(lambda x: f"{x:,.4f}".rstrip("0").rstrip(".") if pd.notna(x) else "")
    # Friendly datetime strings
    for col in ["submitted_at", "filled_at"]:
        if col in fmt_ord.columns:
            fmt_ord[col] = fmt_ord[col].dt.tz_convert(None).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    st.dataframe(fmt_ord, use_container_width=True, hide_index=True)
