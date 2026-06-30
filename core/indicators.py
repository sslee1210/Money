from __future__ import annotations

import pandas as pd

from analyze_stock import add_indicators


DAILY_REQUIRED_INDICATORS = [
    "MA5",
    "MA10",
    "MA20",
    "MA60",
    "MA120",
    "MA240",
    "BB상단",
    "BB중심",
    "BB하단",
    "전환선",
    "기준선",
    "선행스팬1",
    "선행스팬2",
    "거래량20평균",
    "거래량비율20",
    "ATR14",
]

INTRADAY_REQUIRED_INDICATORS = [
    "MA5",
    "MA10",
    "MA20",
    "BB상단",
    "BB중심",
    "BB하단",
    "거래량20평균",
    "거래량비율20",
]

# Backward-compatible alias for the original daily validation behavior.
REQUIRED_INDICATORS = DAILY_REQUIRED_INDICATORS


def calculate_standard_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators from a standard OHLCV frame."""

    frame = ohlcv.copy()
    if "DateTime" in frame.columns:
        frame = frame.set_index(pd.to_datetime(frame["DateTime"]), drop=False)
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in frame.columns:
            raise ValueError(f"{column} column is required")
    out = add_indicators(frame, [5, 10, 20, 60, 120, 240])
    out["Bollinger Upper"] = out["BB상단"]
    out["Bollinger Mid"] = out["BB중심"]
    out["Bollinger Lower"] = out["BB하단"]
    out["Ichimoku Conversion"] = out["전환선"]
    out["Ichimoku Base"] = out["기준선"]
    out["Ichimoku Span1"] = out["선행스팬1"]
    out["Ichimoku Span2"] = out["선행스팬2"]
    return out


def indicators_valid(indicators: pd.DataFrame, required: list[str] | tuple[str, ...] | None = None) -> bool:
    if indicators.empty:
        return False
    row = indicators.iloc[-1]
    required_columns = required or DAILY_REQUIRED_INDICATORS
    return all(column in indicators.columns and pd.notna(row.get(column)) for column in required_columns)


def daily_indicators_valid(indicators: pd.DataFrame) -> bool:
    return indicators_valid(indicators, DAILY_REQUIRED_INDICATORS)


def intraday_indicators_valid(indicators: pd.DataFrame) -> bool:
    return indicators_valid(indicators, INTRADAY_REQUIRED_INDICATORS)
