from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

import pandas as pd

from .models import Tick


STANDARD_COLUMNS = ["DateTime", "Open", "High", "Low", "Close", "Volume", "TradeValue"]


def _as_tick(raw: Tick | dict[str, Any]) -> Tick:
    if isinstance(raw, Tick):
        return raw
    timestamp = raw.get("timestamp") or raw.get("DateTime") or raw.get("datetime") or raw.get("time")
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp).to_pydatetime()
    if not isinstance(timestamp, datetime):
        raise ValueError("tick timestamp is required")
    price = float(raw.get("price", raw.get("Close", raw.get("현재가", 0))))
    volume = float(raw.get("volume", raw.get("Volume", raw.get("거래량", 0))))
    trade_value = raw.get("trade_value", raw.get("TradeValue", raw.get("거래대금")))
    if trade_value is None:
        trade_value = price * volume
    return Tick(code=str(raw.get("code", raw.get("종목코드", ""))), timestamp=timestamp, price=price, volume=volume, trade_value=float(trade_value))


def ticks_to_ohlcv(ticks: Iterable[Tick | dict[str, Any]], interval_minutes: int = 1) -> pd.DataFrame:
    """Build standard OHLCV candles from Kiwoom ticks.

    The returned frame has a DateTime index and also keeps a DateTime column so it
    can be consumed by both pandas indicator code and export/QA code.
    """

    parsed = [_as_tick(tick) for tick in ticks]
    if not parsed:
        return pd.DataFrame(columns=STANDARD_COLUMNS).set_index(pd.DatetimeIndex([], name="DateTime"), drop=False)

    frame = pd.DataFrame(
        {
            "DateTime": [tick.timestamp for tick in parsed],
            "Price": [tick.price for tick in parsed],
            "Volume": [tick.volume for tick in parsed],
            "TradeValue": [tick.trade_value if tick.trade_value is not None else tick.price * tick.volume for tick in parsed],
        }
    ).sort_values("DateTime")
    frame = frame.set_index(pd.to_datetime(frame["DateTime"]))
    rule = f"{int(interval_minutes)}min"
    out = frame.resample(rule).agg({"Price": ["first", "max", "min", "last"], "Volume": "sum", "TradeValue": "sum"})
    out.columns = ["Open", "High", "Low", "Close", "Volume", "TradeValue"]
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out.index.name = "DateTime"
    out.insert(0, "DateTime", out.index)
    return out[STANDARD_COLUMNS]


def normalize_ohlcv_frame(rows: Iterable[dict[str, Any]], datetime_column: str = "DateTime") -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS).set_index(pd.DatetimeIndex([], name=datetime_column), drop=False)

    rename_map = {
        "date": "Date",
        "datetime": "DateTime",
        "timestamp": "DateTime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "price": "Close",
        "volume": "Volume",
        "trade_value": "TradeValue",
        "현재가": "Close",
        "시가": "Open",
        "고가": "High",
        "저가": "Low",
        "거래량": "Volume",
        "거래대금": "TradeValue",
    }
    frame = frame.rename(columns={col: rename_map.get(str(col), col) for col in frame.columns})
    if "DateTime" not in frame.columns and "Date" in frame.columns:
        frame["DateTime"] = pd.to_datetime(frame["Date"])
    if "DateTime" not in frame.columns:
        raise ValueError("DateTime or Date column is required")
    frame["DateTime"] = pd.to_datetime(frame["DateTime"])
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in frame.columns:
            raise ValueError(f"{column} column is required")
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "TradeValue" not in frame.columns:
        frame["TradeValue"] = frame["Close"] * frame["Volume"]
    frame["TradeValue"] = pd.to_numeric(frame["TradeValue"], errors="coerce")
    frame = frame.dropna(subset=["DateTime", "Open", "High", "Low", "Close"]).sort_values("DateTime")
    frame = frame.set_index("DateTime", drop=False)
    return frame[STANDARD_COLUMNS]

