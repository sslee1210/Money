from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.decision_engine import DecisionResult, PriceEvidence
from core.sse_indicator import (
    SSELevels,
    SSEResult,
    add_sse_columns,
    calculate_sse_indicator,
    classify_sse_verdict,
    validate_sse_levels,
)
from command_chart_analyzer import apply_sse_safety_filter, render_sse_section


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


def test_sse_pressure_recalculates_from_current_price_when_supplied():
    out = add_sse_columns(_frame())
    row = out.iloc[-1]
    current_price = float(row["Close"] + row["SSE_VOLATILITY"] * 2)
    result = calculate_sse_indicator(out, current_price=current_price, is_intraday=False)
    expected = (current_price - row["SSE_BASE"]) / row["SSE_VOLATILITY"]

    assert result.levels.pressure == pytest.approx(expected)
    assert result.levels.pressure != pytest.approx(row["SSE_PRESSURE"])


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


def test_target1_reached_does_not_return_conditional_buy():
    levels = _valid_levels(pressure=0.8, target1=110.0, target2=130.0, no_chase=125.0)
    verdict = classify_sse_verdict(levels, 110.0, is_intraday=False)

    assert verdict == "보유하라"
    assert verdict != "조건부로 사라"


def test_intraday_close_below_entry_blocks_immediate_buy_verdict():
    daily = _frame()
    daily.loc[daily.index[-40], "High"] = 13500
    daily.loc[daily.index[-1], "High"] = 13120
    sse_frame = add_sse_columns(daily)
    row = sse_frame.iloc[-1]
    current_price = float(row["SSE_BASE"] + 1.3 * row["SSE_VOLATILITY"])
    entry = float(row["SSE_ENTRY"])
    minute3 = _frame(80)
    minute5 = _frame(80)
    minute3.loc[minute3.index[-1], "Close"] = entry - 1
    minute5.loc[minute5.index[-1], "Close"] = entry - 1
    control = calculate_sse_indicator(daily, current_price=current_price, is_intraday=False)

    result = calculate_sse_indicator(
        daily,
        minute3_ind=minute3,
        minute5_ind=minute5,
        current_price=current_price,
        is_intraday=True,
    )

    assert control.verdict == "조건부로 사라"
    assert result.verdict == "기다려라"
    assert result.verdict != "조건부로 사라"
    assert any("진입 조건 미충족" in warning for warning in result.warnings)


def test_sse_safety_filter_overrides_to_more_conservative_decision():
    evidence = PriceEvidence("매수 확인선", 103, ("테스트 근거",))
    decision = DecisionResult(
        verdict="조건부로 사라",
        headline="기존 판단",
        actions=("기존 조건부 매수",),
        buy_conditions=("지지 매수: 100원 지지 후 103원 회복 시",),
        no_buy_conditions=("125원 이상에서는 추격 매수하지 마라.",),
        sell_conditions=("92원 이탈 시 팔아라.",),
        holder_conditions=("100원 이탈 전까지 보유하라.",),
        price_evidence=(evidence,),
        final_action_state="WATCH_INTRADAY_BREAKOUT",
    )
    sse_result = SSEResult(
        verdict="사지 마라",
        levels=_valid_levels(no_chase=125.0, rr1=1.36),
        evidence=(),
        warnings=(),
        blocking_errors=(),
    )

    filtered = apply_sse_safety_filter(decision, sse_result, current_price=110, is_intraday=True)

    assert filtered.verdict == "사지 마라"
    assert "SSE 안전 필터 적용" in filtered.headline


def test_intraday_sse_report_does_not_use_confirmed_breakout_wording():
    result = calculate_sse_indicator(_frame(), _frame(80), _frame(80), current_price=12800, is_intraday=True)
    section = render_sse_section(result)
    assert "확정 돌파" not in section
    assert "일봉 돌파 확정" not in section
    assert "돌파 확인 완료" not in section
