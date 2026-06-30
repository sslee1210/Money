from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class Quote:
    code: str
    name: str | None
    price: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    trade_value: float | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True)
class Tick:
    code: str
    timestamp: datetime
    price: float
    volume: float
    trade_value: float | None = None


@dataclass(frozen=True)
class Candle:
    code: str
    timestamp: datetime | date
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_value: float

