import os
import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest

# В новых версиях Alpaca используется QueryOrderStatus
try:
    from alpaca.trading.enums import QueryOrderStatus
except Exception:
    QueryOrderStatus = None

try:
    from pydantic import BaseModel  # для model_dump()
except Exception:
    BaseModel = None  # fallback

def getenv(name: str, default: str | None = None) -> str:
    val = os.environ.get(name, default)
    if val is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return val

def to_serializable(obj: Any) -> Any:
    # Pydantic v2: model_dump, v1: dict
    if BaseModel is not None and isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        # default=str — сериализует UUID, Decimal, datetime и т.д.
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def fetch_orders(client: TradingClient) -> List[Dict[str, Any]]:
    orders = []
    if QueryOrderStatus is not None:
        for status in (QueryOrderStatus.OPEN, QueryOrderStatus.CLOSED):
            req = GetOrdersRequest(status=status, limit=200)
            orders.extend(client.get_orders(req))
    else:
        req = GetOrdersRequest(limit=200)
        orders.extend(client.get_orders(req))
    def to_dict(o):
        o = to_serializable(o)
        return o
    od = [to_dict(o) for o in orders]
    # сортируем по времени, если поле есть
    od.sort(key=lambda x: (x.get("submitted_at") or x.get("created_at") or ""), reverse=True)
    return od

def main() -> None:
    key = getenv("ALPACA_API_KEY")
    secret = getenv("ALPACA_SECRET_KEY")
    paper = getenv("ALPACA_PAPER", "true").lower() in ("1", "true", "yes", "y")

    client = TradingClient(api_key=key, secret_key=secret, paper=paper)

    account = to_serializable(client.get_account())
    positions = [to_serializable(p) for p in client.get_all_positions()]
    orders = fetch_orders(client)

    now_utc = dt.datetime.now(dt.UTC)
    ts = now_utc.strftime("%Y%m%dT%H%M%SZ")

    root = Path(__file__).resolve().parents[1]  # dashboard/
    data_dir = root / "data"
    snap_dir = data_dir / "snapshots" / ts
    ensure_dir(snap_dir)

    payload: Dict[str, Any] = {
        "updated_at_utc": now_utc.isoformat(),
        "paper": paper,
        "account": account,
        "positions": positions,
        "orders": orders,
    }

    atomic_write_json(snap_dir / "snapshot.json", payload)
    atomic_write_json(data_dir / "latest.json", payload)

    print(f"Wrote snapshot to {snap_dir} and latest.json")

if __name__ == "__main__":
    main()
