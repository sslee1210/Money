from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import command_chart_analyzer
from core.decision_engine import DecisionContext, DecisionLevels, PriceEvidence, evaluate_decision
from core.indicators import calculate_standard_indicators, indicators_valid, intraday_indicators_valid
from core.qa import validate_command_report
from core.sse_indicator import SSELevels, SSEOpportunity, SSEResult, SSEEvidence
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


class TransientShortMinuteProvider(MockProvider):
    def __init__(self):
        super().__init__(fail=False)
        self.calls: dict[int, int] = {}
        self.limits: list[int] = []

    def get_intraday_ohlcv(self, code: str, interval_minutes: int = 1, limit: int = 600) -> pd.DataFrame:
        self.limits.append(limit)
        count = self.calls.get(interval_minutes, 0)
        self.calls[interval_minutes] = count + 1
        if count == 0:
            return _minute_frame(5)
        return super().get_intraday_ohlcv(code, interval_minutes, limit)


class MissingMinuteEndpointClient:
    def get_quote(self, code: str):
        return {}

    def get_ticks(self, code: str, limit: int = 600):
        raise KiwoomDataError("키움 브릿지 endpoint 미지원: /ticks")

    def get_minute_candles(self, code: str, interval: int = 1, limit: int = 240):
        raise KiwoomDataError("키움 브릿지 endpoint 미지원: /candles/minute")

    def get_daily_candles(self, code: str, limit: int = 400):
        return []


class StringBoolQuoteClient:
    def get_quote(self, code: str):
        return {
            "code": code,
            "name": "삼성전자",
            "price": 311000,
            "timestamp": "2026-06-30T10:00:00+09:00",
            "source": "kiwoom-realtime-fid-stock-trade",
            "sourceLabel": "실시간 FID",
            "isRealtime": "true",
            "isCurrentTr": "false",
            "time": "100000",
        }

    def get_ticks(self, code: str, limit: int = 600):
        return []

    def get_minute_candles(self, code: str, interval: int = 1, limit: int = 240):
        return []

    def get_daily_candles(self, code: str, limit: int = 400):
        return []


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


def test_kiwoom_provider_parses_string_boolean_flags():
    provider = KiwoomDataProvider(client=StringBoolQuoteClient())
    quote = provider.get_quote("005930")

    assert quote.is_realtime is True
    assert quote.is_current_tr is False
    assert quote.source_label == "실시간 FID"
    assert quote.quote_time == "100000"


def test_kiwoom_provider_preserves_minute_endpoint_failure_reason():
    provider = KiwoomDataProvider(client=MissingMinuteEndpointClient())

    try:
        provider.get_intraday_ohlcv("005930", 5)
    except KiwoomDataError as exc:
        message = str(exc)
        assert "/candles/minute" in message
        assert "endpoint" in message
    else:
        raise AssertionError("expected minute endpoint failure")


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
    monkeypatch.setattr(command_chart_analyzer, "INTRADAY_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(command_chart_analyzer, "INTRADAY_RETRY_DELAY_SEC", 0)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: command_chart_analyzer.DailyData("삼성전자", "KOSPI", ".KS", _daily_frame(), "높음", "mock", False, 54300))

    output = command_chart_analyzer.analyze_integrated_chart("005930", "삼성전자", provider=ShortMinuteProvider())
    report = _report_path(tmp_path)
    qa_report = _qa_report_path(tmp_path)

    assert "분석 중단" in output
    assert not report.exists()
    assert qa_report.exists()
    assert "키움 체결 데이터 부족 또는 분봉 생성 실패" in qa_report.read_text(encoding="utf-8")


def test_integrated_transient_short_minutes_retries_and_saves_report(tmp_path, monkeypatch):
    monkeypatch.setattr(command_chart_analyzer, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(command_chart_analyzer, "INTRADAY_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(command_chart_analyzer, "INTRADAY_RETRY_DELAY_SEC", 0)
    monkeypatch.setattr(command_chart_analyzer, "is_korea_regular_session", lambda now=None: False)
    monkeypatch.setattr(command_chart_analyzer, "collect_daily_data", lambda code, name, provider=None: command_chart_analyzer.DailyData("삼성전자", "KOSPI", ".KS", _daily_frame(), "높음", "mock", False, 54300))

    provider = TransientShortMinuteProvider()
    output = command_chart_analyzer.analyze_integrated_chart("005930", "삼성전자", provider=provider)
    report = _report_path(tmp_path)

    assert "분석 완료" in output
    assert report.exists()
    assert not _qa_report_path(tmp_path).exists()
    assert provider.calls[3] == 2
    assert provider.calls[5] == 2
    assert set(provider.limits) == {command_chart_analyzer.INTRADAY_REQUEST_LIMIT}


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


def test_holder_condition_merges_same_support_and_stop_price():
    levels = DecisionLevels(
        support=PriceEvidence("핵심 지지선", 49000, ("볼린저밴드 하단",)),
        confirmation=PriceEvidence("매수 확인선", 49300, ("3분봉 20이평선",)),
        breakout=PriceEvidence("돌파선", 52000, ("최근 20일 고점",)),
        stop=PriceEvidence("손절/방어선", 49000, ("볼린저밴드 하단",)),
        no_chase=PriceEvidence("추격 금지선", 53200, ("볼린저밴드 상단",)),
    )
    decision = evaluate_decision(DecisionContext(current_price=49100, levels=levels, is_intraday=True))

    assert decision.holder_conditions == ("보유자는 49,000원 이탈 시 추가매수 보류 및 방어/손절하라.",)


def test_recovery_line_above_current_does_not_render_as_support_buy():
    levels = DecisionLevels(
        support=PriceEvidence("핵심 지지선", 322000, ("일목 전환선",)),
        confirmation=PriceEvidence("매수 확인선", 320500, ("직전 분봉 반등 고점",)),
        breakout=PriceEvidence("돌파선", 343000, ("최근 5일 고점",)),
        stop=PriceEvidence("손절/방어선", 282500, ("최근 20일 저점",)),
        no_chase=PriceEvidence("추격 금지선", 373000, ("볼린저밴드 상단",)),
    )
    decision = evaluate_decision(DecisionContext(current_price=316000, levels=levels, is_intraday=True, risk_reward=0.57))

    assert decision.verdict == "사지 마라"
    assert decision.buy_conditions[0] == "회복 매수: 320,500원 회복 후 322,000원 이상에서 3분봉 또는 5분봉 종가 유지 시"
    assert "322,000원 지지 후 320,500원 회복" not in decision.buy_conditions[0]
    assert decision.no_buy_conditions[0] == "322,000원 회복 전이거나 이 가격 아래에서 5분봉 종가가 마감되면 사지 마라."
    assert decision.holder_conditions[0] == "보유자는 322,000원 회복 실패 구간에서는 추가매수 보류, 282,500원 이탈 시 방어/손절하라."


def test_price_evidence_above_current_is_labeled_recovery_line():
    evidence = PriceEvidence("핵심 지지선", 322000, ("일목 전환선",))

    summary = command_chart_analyzer._price_evidence_summary(evidence, current_price=316000)

    assert summary == "회복/안착 기준선 322,000원: 일목 전환선 근거"


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
        public_previous_close=None,
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
        public_previous_close=None,
        current_price=56000,
        is_intraday=True,
        now=datetime(2026, 6, 30, 10, 1),
    )
    assert any("범위 밖" in error for error in errors)


def test_quote_price_range_unit_mismatch_stops_analysis():
    quote = Quote(
        code="005930",
        name="삼성전자",
        price=311000,
        high=4340,
        low=3155,
        timestamp=datetime(2026, 6, 30, 10, 0),
    )
    errors, _warnings = command_chart_analyzer._validate_quote(
        quote,
        public_reference_close=311000,
        public_previous_close=None,
        current_price=311000,
        is_intraday=True,
        now=datetime(2026, 6, 30, 10, 1),
    )
    assert any("단위" in error for error in errors)


def test_after_close_prev_close_compares_to_previous_completed_close():
    quote = Quote(
        code="005930",
        name="삼성전자",
        price=309500,
        prev_close=286000,
        high=313000,
        low=283500,
        timestamp=datetime(2026, 7, 4, 0, 20),
    )
    errors, warnings = command_chart_analyzer._validate_quote(
        quote,
        public_reference_close=309500,
        public_previous_close=286000,
        current_price=309500,
        is_intraday=False,
        now=datetime(2026, 7, 4, 0, 20),
    )

    assert not errors
    assert not warnings


def test_price_source_infers_realtime_fid_without_quote_time():
    quote = Quote(
        code="005930",
        name="삼성전자",
        price=311000,
        timestamp=datetime(2026, 6, 30, 10, 0),
        source="kiwoom-realtime-fid-stock-trade",
        source_label="실시간 FID",
        is_realtime=False,
        quote_time=None,
    )
    price_source = command_chart_analyzer._price_source_info(quote, is_intraday=True)
    assert price_source.label == "실시간 현재가"
    assert price_source.status_name == "키움 실시간 체결 보정"
    assert price_source.is_realtime
    assert "TR 기준가" not in price_source.note


def test_render_sse_section_prints_opportunity_block_without_confirmed_breakout_words():
    result = SSEResult(
        verdict="기다려라",
        levels=SSELevels(
            base=100.0,
            upper=118.0,
            lower=82.0,
            pressure=0.52,
            entry=103.0,
            stop=92.0,
            target1=118.0,
            target2=130.0,
            no_chase=125.0,
            rr1=1.36,
            rr2=2.45,
        ),
        evidence=(
            SSEEvidence("SSE 기준선", 100.0, "formula", "reason"),
            SSEEvidence("SSE 기회 점수", 72.0, "old formula", "old mixed reason"),
            SSEEvidence("SSE 기회 경고", 0.0, "old warning", "old mixed warning"),
        ),
        warnings=(),
        blocking_errors=(),
        opportunity=SSEOpportunity(
            score=72.0,
            grade="B급 후보",
            setup="BASE_RECLAIM",
            reasons=(
                "현재가가 SSE 예상 진입가 이상으로 회복",
                "SSE 압력값 0.52: 초기 회복 우수 구간",
            ),
            warnings=(),
        ),
    )

    section = command_chart_analyzer.render_sse_section(result)

    assert "SSE 기회 점수: 72.0점" in section
    assert "SSE 기회 등급: B급 후보" in section
    assert "SSE 셋업: BASE_RECLAIM" in section
    assert "SSE 기회 사유:" in section
    assert "* 현재가가 SSE 예상 진입가 이상으로 회복" in section
    assert "SSE 기회 경고:" in section
    assert "* 경고 없음" in section
    assert "* SSE 기회 점수 72.00:" not in section
    assert "* SSE 기회 경고 0.00:" not in section
    assert "확정 돌파" not in section
    assert "일봉 돌파 확정" not in section
    assert "돌파 확인 완료" not in section


def test_render_sse_section_handles_missing_opportunity():
    result = SSEResult(
        verdict="분석 중단",
        levels=SSELevels(
            base=float("nan"),
            upper=float("nan"),
            lower=float("nan"),
            pressure=float("nan"),
            entry=float("nan"),
            stop=float("nan"),
            target1=float("nan"),
            target2=float("nan"),
            no_chase=float("nan"),
            rr1=float("nan"),
            rr2=float("nan"),
        ),
        evidence=(),
        warnings=(),
        blocking_errors=("SSE 계산 실패",),
        opportunity=None,
    )

    section = command_chart_analyzer.render_sse_section(result)

    assert "SSE 기회 점수: 계산 불가" in section
    assert "SSE 기회 등급: 계산 불가" in section
    assert "SSE 셋업: 계산 불가" in section


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


def test_collect_daily_data_excludes_short_kiwoom_daily_sample(monkeypatch):
    public = _daily_frame()
    kiwoom = public.tail(239).copy()
    kiwoom["Close"] = kiwoom["Close"] + 50

    class ShortKiwoomDailyProvider(MockProvider):
        def get_daily_ohlcv(self, code: str, limit: int = 400) -> pd.DataFrame:
            return kiwoom

    monkeypatch.setattr(command_chart_analyzer, "detect_name_market", lambda code, name, end: ("삼성전자", "KOSPI", ".KS"))
    monkeypatch.setattr(command_chart_analyzer, "load_pykrx", lambda code, start, end: command_chart_analyzer.SourceFrame("pykrx", public))
    monkeypatch.setattr(command_chart_analyzer, "load_fdr", lambda code, start, end: command_chart_analyzer.SourceFrame("FinanceDataReader", public))
    monkeypatch.setattr(command_chart_analyzer, "load_yfinance", lambda ticker, start, end, name="yfinance": command_chart_analyzer.SourceFrame(name, public))

    daily = command_chart_analyzer.collect_daily_data("005930", "삼성전자", ShortKiwoomDailyProvider())

    assert len(daily.frame) == len(public)
    assert daily.frame.iloc[-1]["Close"] == public.iloc[-1]["Close"]
    assert "키움 일봉 표본 부족" in daily.validation_note


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
