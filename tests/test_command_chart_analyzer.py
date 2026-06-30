from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import command_chart_analyzer
from core.decision_engine import DecisionContext, DecisionLevels, PriceEvidence, evaluate_decision
from core.indicators import calculate_standard_indicators, indicators_valid
from kiwoom.candles import ticks_to_ohlcv
from kiwoom.models import Quote, Tick
from kiwoom.provider import KiwoomDataError, KiwoomDataProvider


def _daily_frame(rows: int = 260) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=rows, freq="B")
    base = pd.Series(range(rows), dtype=float) * 20 + 48000
    return pd.DataFrame(
        {
            "DateTime": dates,
            "Open": base + 20,
            "High": base + 500,
            "Low": base - 500,
            "Close": base + 100,
            "Volume": 100000 + base,
            "TradeValue": (base + 100) * (100000 + base),
        }
    ).set_index("DateTime", drop=False)


def _minute_frame(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2026-06-30 09:00", periods=rows, freq="3min")
    base = pd.Series(range(rows), dtype=float) * 5 + 53000
    return pd.DataFrame(
        {
            "DateTime": dates,
            "Open": base,
            "High": base + 80,
            "Low": base - 80,
            "Close": base + 20,
            "Volume": 1000 + base,
            "TradeValue": (base + 20) * (1000 + base),
        }
    ).set_index("DateTime", drop=False)


class MockProvider(KiwoomDataProvider):
    def __init__(self, fail: bool = False):
        self.fail = fail

    def get_quote(self, code: str) -> Quote:
        if self.fail:
            raise KiwoomDataError("mock quote failure")
        return Quote(code=code, name="삼성전자", price=54500, volume=123456, trade_value=6700000000)

    def get_intraday_ohlcv(self, code: str, interval_minutes: int = 1, limit: int = 600) -> pd.DataFrame:
        if self.fail:
            raise KiwoomDataError("mock minute failure")
        frame = _minute_frame()
        if interval_minutes == 5:
            frame = frame.resample("5min").agg({"DateTime": "first", "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "TradeValue": "sum"}).dropna()
            frame["DateTime"] = frame.index
        return frame


def test_command_chart_analyzer_imports():
    assert callable(command_chart_analyzer.analyze_command_chart)


def test_ticks_to_ohlcv_builds_standard_frame():
    start = datetime(2026, 6, 30, 9, 0)
    ticks = [Tick("005930", start + timedelta(seconds=i * 20), 50000 + i, 10 + i, None) for i in range(9)]
    frame = ticks_to_ohlcv(ticks, interval_minutes=1)
    assert {"DateTime", "Open", "High", "Low", "Close", "Volume", "TradeValue"} <= set(frame.columns)
    assert len(frame) == 3
    assert frame.iloc[0]["Open"] == 50000


def test_indicator_calculation_has_required_values():
    ind = calculate_standard_indicators(_daily_frame())
    assert indicators_valid(ind)
    assert "Bollinger Upper" in ind.columns
    assert "Ichimoku Base" in ind.columns


def test_kiwoom_provider_mock_returns_quote_and_minutes():
    provider = MockProvider()
    quote = provider.get_quote("005930")
    minutes = provider.get_intraday_ohlcv("005930", 3)
    assert quote.price == 54500
    assert not minutes.empty


def test_invalid_data_writes_only_qa_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name: command_chart_analyzer.DailyData("삼성전자", "KOSPI", ".KS", _daily_frame(), "높음", "mock", False))
    output = command_chart_analyzer.analyze_command_chart("005930", "삼성전자", provider=MockProvider(fail=True))
    out_dir = tmp_path / "삼성전자_005930"
    assert "분석 중단" in output
    assert (out_dir / "삼성전자_005930_보고서_QA실패.md").exists()
    assert not (out_dir / "삼성전자_005930_조건부명령형_차트분석.md").exists()
    assert not (out_dir / "삼성전자_005930_조건부명령형_차트분석.html").exists()


def test_intraday_breakout_never_uses_confirmed_word():
    levels = DecisionLevels(
        support=PriceEvidence("핵심 지지선", 49000, ("20일선",)),
        confirmation=PriceEvidence("매수 확인선", 49300, ("3분봉 20이평선",)),
        breakout=PriceEvidence("돌파선", 52000, ("최근 20일 고점",)),
        stop=PriceEvidence("손절/방어선", 48500, ("최근 20일 저점",)),
        no_chase=PriceEvidence("추격 금지선", 53200, ("볼린저밴드 상단",)),
    )
    decision = evaluate_decision(DecisionContext(current_price=52500, levels=levels, is_intraday=True, risk_reward=1.5))
    rendered = "\n".join(decision.actions) + decision.headline
    assert decision.verdict == "조건부로 사라"
    assert "확정" not in rendered
    assert "3분봉 또는 5분봉 종가" in rendered


def test_price_without_evidence_stops_analysis():
    levels = DecisionLevels(
        support=PriceEvidence("핵심 지지선", 49000, ()),
        confirmation=PriceEvidence("매수 확인선", 49300, ("3분봉 20이평선",)),
        breakout=None,
        stop=PriceEvidence("손절/방어선", 48500, ("최근 20일 저점",)),
        no_chase=PriceEvidence("추격 금지선", 53200, ("볼린저밴드 상단",)),
    )
    decision = evaluate_decision(DecisionContext(current_price=50000, levels=levels, is_intraday=True))
    assert decision.verdict == "분석 중단"
    assert decision.blocking_errors
