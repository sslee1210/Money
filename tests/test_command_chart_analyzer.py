from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import command_chart_analyzer
from core.decision_engine import DecisionContext, DecisionLevels, PriceEvidence, evaluate_decision
from core.indicators import calculate_standard_indicators, indicators_valid, intraday_indicators_valid
from core.qa import validate_command_report
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
        return Quote(
            code=code,
            name="삼성전자",
            price=54500,
            prev_close=54300,
            high=55000,
            low=54000,
            volume=123456,
            trade_value=6700000000,
            timestamp=datetime(2026, 6, 30, 10, 0),
            source_label="키움현재가TR",
            is_current_tr=True,
        )

    def get_intraday_ohlcv(self, code: str, interval_minutes: int = 1, limit: int = 600) -> pd.DataFrame:
        if self.fail:
            raise KiwoomDataError("mock minute failure")
        frame = _minute_frame()
        if interval_minutes == 5:
            frame = frame.resample("5min").agg({"DateTime": "first", "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "TradeValue": "sum"}).dropna()
            frame["DateTime"] = frame.index
        return frame

    def get_daily_ohlcv(self, code: str, limit: int = 400) -> pd.DataFrame:
        if self.fail:
            raise KiwoomDataError("mock daily failure")
        return _daily_frame(limit)


class BadQuoteProvider(MockProvider):
    def get_quote(self, code: str) -> Quote:
        return Quote(
            code=code,
            name="삼성전자",
            price=54500,
            prev_close=58000,
            high=55000,
            low=54000,
            timestamp=datetime(2026, 6, 30, 10, 0),
        )


class MissingTimestampProvider(MockProvider):
    def get_quote(self, code: str) -> Quote:
        return Quote(
            code=code,
            name="삼성전자",
            price=54500,
            prev_close=54300,
            high=55000,
            low=54000,
            timestamp=None,
        )


class ShortMinuteProvider(MockProvider):
    def get_intraday_ohlcv(self, code: str, interval_minutes: int = 1, limit: int = 600) -> pd.DataFrame:
        return _minute_frame(5)


def _report_path(base: Path, name: str = "삼성전자", code: str = "005930") -> Path:
    return base / f"{name}_{code}" / f"[{name}, {code}] 분석 보고서.md"


def _html_report_path(base: Path, name: str = "삼성전자", code: str = "005930") -> Path:
    return base / f"{name}_{code}" / f"[{name}, {code}] 분석 보고서.html"


def _qa_report_path(base: Path, name: str = "삼성전자", code: str = "005930") -> Path:
    return base / f"{name}_{code}" / f"[{name}, {code}] 분석 실패 보고서.md"


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


def test_intraday_indicator_validation_does_not_require_long_daily_windows():
    ind = calculate_standard_indicators(_minute_frame(80))
    assert intraday_indicators_valid(ind)
    assert pd.isna(ind.iloc[-1].get("MA120"))


def test_kiwoom_provider_mock_returns_quote_and_minutes():
    provider = MockProvider()
    quote = provider.get_quote("005930")
    minutes = provider.get_intraday_ohlcv("005930", 3)
    assert quote.price == 54500
    assert not minutes.empty


def test_invalid_data_writes_only_qa_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: command_chart_analyzer.DailyData("삼성전자", "KOSPI", ".KS", _daily_frame(), "높음", "mock", False, 54300))
    output = command_chart_analyzer.analyze_command_chart("005930", "삼성전자", provider=MockProvider(fail=True))
    assert "분석 중단" in output
    assert (_qa_report_path(tmp_path)).exists()
    assert not (_report_path(tmp_path)).exists()
    assert not (_html_report_path(tmp_path)).exists()


def test_success_report_includes_sse_indicator_section(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "is_korea_regular_session", lambda now=None: False)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: command_chart_analyzer.DailyData("삼성전자", "KOSPI", ".KS", _daily_frame(), "높음", "mock", False, 54300))
    output = command_chart_analyzer.analyze_command_chart("005930", "삼성전자", provider=MockProvider())
    report = _report_path(tmp_path)
    assert "분석 완료" in output
    text = report.read_text(encoding="utf-8")
    assert "# [삼성전자, 005930] 분석 보고서" in text
    assert "장마감 기준가:" in text
    assert "가격 기준:" in text
    assert "## SSE Indicator 분석" in text
    assert "SSE 기준선" in text
    assert "SSE 최종 판정" in text
    assert "산출 근거:" in text


def test_integrated_public_ok_kiwoom_ok_saves_layered_report(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "is_korea_regular_session", lambda now=None: False)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: command_chart_analyzer.DailyData("삼성전자", "KOSPI", ".KS", _daily_frame(), "높음", "mock", False, 54300))

    output = command_chart_analyzer.analyze_integrated_chart("005930", "삼성전자", provider=MockProvider())
    report = _report_path(tmp_path)

    assert "분석 완료" in output
    text = report.read_text(encoding="utf-8")
    assert "## 분석 레이어 상태" in text
    assert "키움 장마감 TR 보정:" in text
    assert "실시간 매수 제한 여부: 아니오" in text


def test_integrated_public_ok_kiwoom_fail_stops_without_normal_report(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "is_korea_regular_session", lambda now=None: True)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: command_chart_analyzer.DailyData("삼성전자", "KOSPI", ".KS", _daily_frame(), "높음", "mock", False, 54300))

    output = command_chart_analyzer.analyze_integrated_chart("005930", "삼성전자", provider=MockProvider(fail=True))
    report = _report_path(tmp_path)
    qa_report = _qa_report_path(tmp_path)

    assert "분석 중단" in output
    assert not report.exists()
    assert qa_report.exists()
    assert "키움 현재가 수집 실패" in qa_report.read_text(encoding="utf-8")


def test_integrated_public_fail_kiwoom_ok_blocks_kiwoom_only_report(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: (_ for _ in ()).throw(RuntimeError("public failure")))

    output = command_chart_analyzer.analyze_integrated_chart("005930", "삼성전자", provider=MockProvider())

    assert "분석 중단" in output
    assert (_qa_report_path(tmp_path)).exists()


def test_integrated_public_fail_kiwoom_fail_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: (_ for _ in ()).throw(RuntimeError("public failure")))

    output = command_chart_analyzer.analyze_integrated_chart("005930", "삼성전자", provider=MockProvider(fail=True))

    assert "분석 중단" in output
    assert (_qa_report_path(tmp_path)).exists()


def test_integrated_quote_missing_timestamp_stops_without_normal_report(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: command_chart_analyzer.DailyData("삼성전자", "KOSPI", ".KS", _daily_frame(), "높음", "mock", False, 54300))

    output = command_chart_analyzer.analyze_integrated_chart("005930", "삼성전자", provider=MissingTimestampProvider())
    report = _report_path(tmp_path)
    qa_report = _qa_report_path(tmp_path)

    assert "분석 중단" in output
    assert not report.exists()
    assert qa_report.exists()
    assert "timestamp" in qa_report.read_text(encoding="utf-8")


def test_integrated_short_minutes_stops_without_normal_report(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: command_chart_analyzer.DailyData("삼성전자", "KOSPI", ".KS", _daily_frame(), "높음", "mock", False, 54300))

    output = command_chart_analyzer.analyze_integrated_chart("005930", "삼성전자", provider=ShortMinuteProvider())
    report = _report_path(tmp_path)
    qa_report = _qa_report_path(tmp_path)

    assert "분석 중단" in output
    assert not report.exists()
    assert qa_report.exists()
    assert "키움 체결 데이터 부족 또는 분봉 생성 실패" in qa_report.read_text(encoding="utf-8")


def test_realtime_limited_positive_buy_is_qa_failure():
    levels = DecisionLevels(
        support=PriceEvidence("핵심 지지선", 49000, ("20일선",)),
        confirmation=PriceEvidence("매수 확인선", 49300, ("3분봉 20이평선",)),
        breakout=PriceEvidence("돌파선", 52000, ("최근 20일 고점",)),
        stop=PriceEvidence("손절/방어선", 48500, ("최근 20일 저점",)),
        no_chase=PriceEvidence("추격 금지선", 53200, ("볼린저밴드 상단",)),
    )
    decision = evaluate_decision(DecisionContext(current_price=50000, levels=levels, is_intraday=True, risk_reward=2.0))
    report = "## 내부 검증\n내부 검증: 통과\n"

    errors = validate_command_report(
        report,
        decision,
        levels,
        is_intraday=True,
        data_valid=True,
        current_price=50000,
        realtime_limited=True,
    )

    assert any("실시간 보정 실패" in error for error in errors)


def test_quote_prev_close_mismatch_stops_normal_report(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: command_chart_analyzer.DailyData("삼성전자", "KOSPI", ".KS", _daily_frame(), "높음", "mock", False, 54300))
    output = command_chart_analyzer.analyze_command_chart("005930", "삼성전자", provider=BadQuoteProvider())
    assert "분석 중단" in output
    assert (_qa_report_path(tmp_path)).exists()


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
    assert "장중 돌파 시도" in rendered
    assert "오늘 종가 확인 필요" in rendered


def test_no_chase_line_blocks_buy_even_with_good_rr():
    levels = DecisionLevels(
        support=PriceEvidence("핵심 지지선", 49000, ("20일선",)),
        confirmation=PriceEvidence("매수 확인선", 49300, ("3분봉 20이평선",)),
        breakout=PriceEvidence("돌파선", 52000, ("최근 20일 고점",)),
        stop=PriceEvidence("손절/방어선", 48500, ("최근 20일 저점",)),
        no_chase=PriceEvidence("추격 금지선", 53200, ("볼린저밴드 상단",)),
    )
    decision = evaluate_decision(DecisionContext(current_price=53500, levels=levels, is_intraday=True, risk_reward=2.0))
    assert decision.verdict == "사지 마라"
    assert "추격 금지선 이상" in decision.headline


def test_intraday_stale_quote_timestamp_stops_analysis():
    quote = Quote(
        code="005930",
        name="삼성전자",
        price=54500,
        prev_close=54300,
        high=55000,
        low=54000,
        timestamp=datetime(2026, 6, 30, 9, 0),
    )
    errors, warnings = command_chart_analyzer._validate_quote(
        quote,
        public_reference_close=54300,
        current_price=54500,
        is_intraday=True,
        now=datetime(2026, 6, 30, 10, 0),
    )
    assert not warnings
    assert any("30분" in error for error in errors)


def test_quote_price_outside_intraday_range_stops_analysis():
    quote = Quote(
        code="005930",
        name="삼성전자",
        price=56000,
        prev_close=54300,
        high=55000,
        low=54000,
        timestamp=datetime(2026, 6, 30, 10, 0),
    )
    errors, _warnings = command_chart_analyzer._validate_quote(
        quote,
        public_reference_close=54300,
        current_price=56000,
        is_intraday=True,
        now=datetime(2026, 6, 30, 10, 1),
    )
    assert any("범위 밖" in error for error in errors)


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


def test_collect_daily_data_prefers_validated_kiwoom_daily(monkeypatch):
    public = _daily_frame()
    kiwoom = public.copy()
    kiwoom["Close"] = kiwoom["Close"] + 50
    class KiwoomDailyProvider(MockProvider):
        def get_daily_ohlcv(self, code: str, limit: int = 400) -> pd.DataFrame:
            return kiwoom

    monkeypatch.setattr(command_chart_analyzer, "detect_name_market", lambda code, name, end: ("삼성전자", "KOSPI", ".KS"))
    monkeypatch.setattr(command_chart_analyzer, "load_pykrx", lambda code, start, end: command_chart_analyzer.SourceFrame("pykrx", public))
    monkeypatch.setattr(command_chart_analyzer, "load_fdr", lambda code, start, end: command_chart_analyzer.SourceFrame("FinanceDataReader", public))
    monkeypatch.setattr(command_chart_analyzer, "load_yfinance", lambda ticker, start, end, name="yfinance": command_chart_analyzer.SourceFrame(name, public))
    daily = command_chart_analyzer.collect_daily_data("005930", "삼성전자", KiwoomDailyProvider())
    assert daily.frame.iloc[-1]["Close"] == kiwoom.iloc[-1]["Close"]
    assert not daily.stop_precision


def test_collect_daily_data_excludes_intraday_incomplete_daily_candle(monkeypatch):
    public = _daily_frame()
    kiwoom = pd.concat(
        [
            public,
            pd.DataFrame(
                {
                    "DateTime": [pd.Timestamp("2026-06-30")],
                    "Open": [90000.0],
                    "High": [91000.0],
                    "Low": [89000.0],
                    "Close": [90500.0],
                    "Volume": [999999.0],
                    "TradeValue": [90499909500.0],
                }
            ).set_index("DateTime", drop=False),
        ]
    )

    class KiwoomDailyProvider(MockProvider):
        def get_daily_ohlcv(self, code: str, limit: int = 400) -> pd.DataFrame:
            return kiwoom

    monkeypatch.setattr(command_chart_analyzer, "today_kst", lambda: datetime(2026, 6, 30, 10, 0))
    monkeypatch.setattr(command_chart_analyzer, "detect_name_market", lambda code, name, end: ("삼성전자", "KOSPI", ".KS"))
    monkeypatch.setattr(command_chart_analyzer, "load_pykrx", lambda code, start, end: command_chart_analyzer.SourceFrame("pykrx", public))
    monkeypatch.setattr(command_chart_analyzer, "load_fdr", lambda code, start, end: command_chart_analyzer.SourceFrame("FinanceDataReader", public))
    monkeypatch.setattr(command_chart_analyzer, "load_yfinance", lambda ticker, start, end, name="yfinance": command_chart_analyzer.SourceFrame(name, public))
    daily = command_chart_analyzer.collect_daily_data("005930", "삼성전자", KiwoomDailyProvider())
    assert pd.Timestamp("2026-06-30") not in set(pd.to_datetime(daily.frame["DateTime"]))
