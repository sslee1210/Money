from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from .candles import normalize_ohlcv_frame, ticks_to_ohlcv
from .client import KiwoomBridgeClient, KiwoomClientProtocol
from .models import Quote, Tick


class KiwoomDataError(RuntimeError):
    """Raised when Kiwoom analysis data is missing or invalid."""


class KiwoomDataProvider:
    """Read-only Kiwoom data provider for chart analysis."""

    def __init__(self, client: KiwoomClientProtocol | None = None):
        self.client = client or KiwoomBridgeClient.from_env()

    def get_quote(self, code: str) -> Quote:
        raw = self.client.get_quote(code)
        price = _number(raw.get("price", raw.get("현재가", raw.get("Close"))))
        if price is None or price <= 0:
            raise KiwoomDataError("키움 현재가 수집 실패")
        timestamp = raw.get("timestamp") or raw.get("DateTime") or raw.get("time")
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp).to_pydatetime()
        if timestamp is not None and not isinstance(timestamp, datetime):
            timestamp = None
        return Quote(
            code=code,
            name=raw.get("name") or raw.get("종목명"),
            price=price,
            open=_number(raw.get("open", raw.get("시가"))),
            high=_number(raw.get("high", raw.get("고가"))),
            low=_number(raw.get("low", raw.get("저가"))),
            volume=_number(raw.get("volume", raw.get("거래량"))),
            trade_value=_number(raw.get("trade_value", raw.get("거래대금"))),
            timestamp=timestamp,
        )

    def get_ticks(self, code: str, limit: int = 600) -> list[Tick]:
        rows = self.client.get_ticks(code, limit=limit)
        ticks: list[Tick] = []
        for row in rows:
            timestamp = row.get("timestamp") or row.get("DateTime") or row.get("datetime") or row.get("time")
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp).to_pydatetime()
            price = _number(row.get("price", row.get("현재가", row.get("Close"))))
            volume = _number(row.get("volume", row.get("거래량", row.get("Volume"))))
            if not isinstance(timestamp, datetime) or price is None or price <= 0 or volume is None:
                continue
            trade_value = _number(row.get("trade_value", row.get("거래대금", row.get("TradeValue"))))
            ticks.append(Tick(code=code, timestamp=timestamp, price=price, volume=volume, trade_value=trade_value))
        if len(ticks) < 5:
            raise KiwoomDataError("키움 체결 데이터 부족")
        return ticks

    def get_intraday_ohlcv(self, code: str, interval_minutes: int = 1, limit: int = 600) -> pd.DataFrame:
        try:
            rows = self.client.get_minute_candles(code, interval=interval_minutes, limit=limit)
            frame = normalize_ohlcv_frame(rows)
        except Exception:
            frame = pd.DataFrame()
        if frame.empty:
            frame = ticks_to_ohlcv(self.get_ticks(code, limit=limit), interval_minutes=interval_minutes)
        if frame.empty:
            raise KiwoomDataError("분봉 생성 실패")
        return frame

    def get_daily_ohlcv(self, code: str, limit: int = 400) -> pd.DataFrame:
        rows = self.client.get_daily_candles(code, limit=limit)
        if not rows:
            raise KiwoomDataError("키움 일봉 데이터 수집 실패")
        return normalize_ohlcv_frame(rows)


def _number(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return abs(float(str(value).replace(",", "")))
    except Exception:
        return None

