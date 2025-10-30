import os, json, datetime as dt
from typing import Any, Dict, List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
PAPER = True

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def fetch() -> Dict[str, Any]:
    trading = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=PAPER)
    data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    account = trading.get_account()
    positions = trading.get_all_positions()

    # последние 25 ордеров (open + closed за последние дни)
    orders_req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=25)
    orders = trading.get_orders(filter=orders_req)

    account_dict = {
        "id": str(account.id),
        "status": str(account.status),
        "cash": to_float(account.cash),
        "buying_power": to_float(account.buying_power),
        "portfolio_value": to_float(getattr(account, "portfolio_value", None)),
        "last_equity": to_float(getattr(account, "last_equity", None)),
    }

    positions_list: List[Dict[str, Any]] = []
    for p in positions:
        positions_list.append({
            "symbol": p.symbol,
            "qty": to_float(p.qty),
            "avg_entry_price": to_float(p.avg_entry_price),
            "market_value": to_float(p.market_value),
            "unrealized_pl": to_float(getattr(p, "unrealized_pl", None)),
            "unrealized_plpc": to_float(getattr(p, "unrealized_plpc", None)),
        })

    orders_list: List[Dict[str, Any]] = []
    for o in orders:
        orders_list.append({
            "id": str(o.id),
            "symbol": o.symbol,
            "side": o.side.value if getattr(o, "side", None) else None,
            "qty": to_float(getattr(o, "qty", None)),
            "type": o.type.value if getattr(o, "type", None) else None,
            "status": o.status,
            "submitted_at": str(getattr(o, "submitted_at", "")),
            "filled_at": str(getattr(o, "filled_at", "")),
        })

    payload = {
        "updated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "account": account_dict,
        "positions": positions_list,
        "orders": orders_list,
    }
    return payload

if __name__ == "__main__":
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY/ALPACA_SECRET_KEY are not set")
    data = fetch()
    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Wrote data.json")
