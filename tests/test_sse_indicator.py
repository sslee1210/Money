from __future__ import annotations

import numpy as np
import pandas as pd

from core.sse_indicator import (
    SSELevels,
    add_sse_columns,
    calculate_sse_indicator,
    classify_sse_verdict,
    validate_sse_levels,
)
from command_chart_analyzer import render_sse_section


def _frame(rows: int = 260) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    close = pd.Series(np.linspace(10000, 13000, rows), dtype=float)
    return pd.DataFrame(
        {
            "DateTime": dates,
            "Open": close - 20,
            "High": close + 100,
            "Low": close - 100,
            "Close": close,
            "Volume": 100000 + close,
            "TradeValue": close * (100000 + close),
        }
    )


def _valid_levels(**overrides) -> SSELevels:
    values = dict(
        base=100.0,
        upper=118.0,
        lower=82.0,
        pressure=0.5,
        entry=103.0,
        stop=92.0,
        target1=118.0,
        target2=130.0,
        no_chase=125.0,
        rr1=1.36,
        rr2=2.45,
    )
    values.update(overrides)
    return SSELevels(**values)


def test_sse_base_uses_recombined_ma_and_mid_formula():
    out = add_sse_columns(_frame())
    row = out.iloc[-1]
    expected = (
        0.35 * row["SSE_MA20"]
        + 0.20 * row["SSE_MA60"]
        + 0.20 * row["SSE_MID26"]
        + 0.15 * row["SSE_MID52"]
        + 0.10 * row["SSE_MID9"]
    )
    assert row["SSE_BASE"] == expected


def test_sse_volatility_uses_std_mid_gap_and_ma_gap():
    out = add_sse_columns(_frame())
    row = out.iloc[-1]
    expected = (
        0.50 * row["SSE_STD20"]
        + 0.25 * abs(row["SSE_MID26"] - row["SSE_MID52"])
        + 0.25 * abs(row["SSE_MA20"] - row["SSE_MA60"])
    )
    assert row["SSE_VOLATILITY"] == expected


def test_sse_pressure_uses_close_base_and_volatility():
    out = add_sse_columns(_frame())
    row = out.iloc[-1]
    expected = (row["Close"] - row["SSE_BASE"]) / row["SSE_VOLATILITY"]
    assert row["SSE_PRESSURE"] == expected


def test_no_chase_blocks_conditional_buy():
    levels = _valid_levels()
    verdict = classify_sse_verdict(levels, levels.no_chase + 1, is_intraday=True)
    assert verdict == "사지 마라"


def test_rr1_below_threshold_blocks_buy():
    levels = _valid_levels(rr1=1.0)
    assert classify_sse_verdict(levels, 110, is_intraday=False) == "사지 마라"


def test_invalid_stop_entry_relationship_is_blocking():
    levels = _valid_levels(stop=104)
    assert any("SSE_STOP >= SSE_ENTRY" in error for error in validate_sse_levels(levels))


def test_invalid_target1_entry_relationship_is_blocking():
    levels = _valid_levels(target1=102)
    assert any("SSE_TARGET1 <= SSE_ENTRY" in error for error in validate_sse_levels(levels))


def test_pressure_overheated_blocks_chase():
    levels = _valid_levels(pressure=1.6)
    assert classify_sse_verdict(levels, 110, is_intraday=False) == "사지 마라"


def test_pressure_weak_break_blocks_new_buy():
    levels = _valid_levels(pressure=-1.2)
    assert classify_sse_verdict(levels, 99, is_intraday=False) == "사지 마라"


def test_intraday_sse_report_does_not_use_confirmed_breakout_wording():
    result = calculate_sse_indicator(_frame(), _frame(80), _frame(80), current_price=12800, is_intraday=True)
    section = render_sse_section(result)
    assert "확정 돌파" not in section
    assert "일봉 돌파 확정" not in section
    assert "돌파 확인 완료" not in section
