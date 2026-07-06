from __future__ import annotations

"""Kiwoom-backed conditional command chart analyzer.

This entrypoint generates analysis-only reports. It never sends orders and does
not expose automated trading functions.
"""

import argparse
import sys
import time as time_module
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from analyze_stock import (
    REPORTS_DIR,
    SourceFrame,
    detect_name_market,
    html_from_markdown,
    latest_completed_candidate,
    load_fdr,
    load_pykrx,
    load_stooq,
    load_yfinance,
    money,
    round_to_tick,
    sanitize_filename,
    source_validation,
    today_kst,
    validation_labels,
)
from core.analysis_pipeline import IntegratedAnalysisResult, LayerStatus, run_integrated_analysis_pipeline
from core.decision_engine import DecisionContext, DecisionLevels, DecisionResult, PriceEvidence, evaluate_decision, format_price
from core.indicators import calculate_standard_indicators, daily_indicators_valid, intraday_indicators_valid
from core.qa import validate_command_report
from core.sse_indicator import SSEResult, calculate_sse_indicator
from kiwoom.provider import KiwoomDataError, KiwoomDataProvider


KST = ZoneInfo("Asia/Seoul")
DAILY_MIN_COMPLETED_ROWS = 240
INTRADAY_MIN_ROWS = 20
INTRADAY_REQUEST_LIMIT = 120
INTRADAY_RETRY_ATTEMPTS = 4
INTRADAY_RETRY_DELAY_SEC = 1.0


@dataclass(frozen=True)
class DailyData:
    stock_name: str
    market: str
    yf_suffix: str
    frame: pd.DataFrame
    reliability: str
    validation_note: str
    stop_precision: bool
    public_reference_close: float | None = None


@dataclass(frozen=True)
class PriceSourceInfo:
    label: str
    status_name: str
    note: str
    is_realtime: bool


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if np.isfinite(out) else None


def last_valid(row: pd.Series, key: str) -> float | None:
    return finite(row.get(key)) if key in row.index else None


def price_text(value: Any) -> str:
    v = finite(value)
    if v is None or v <= 0:
        return "해당 없음"
    return money(round_to_tick(v, "nearest"))


def safe_round(value: Any, direction: str = "nearest") -> int | None:
    v = finite(value)
    if v is None or v <= 0:
        return None
    return int(round_to_tick(v, direction))


def is_korea_regular_session(now: datetime | None = None) -> bool:
    current = now.astimezone(KST) if now else datetime.now(KST)
    return current.weekday() < 5 and time(9, 0) <= current.time() <= time(15, 30)


def collect_daily_data(code: str, fallback_name: str | None, provider: KiwoomDataProvider | None = None) -> DailyData:
    now = today_kst()
    end_limit = latest_completed_candidate(now)
    start_daily = end_limit - timedelta(days=365 * 6)
    start_validation = end_limit - timedelta(days=365 * 2)
    stock_name, market, yf_suffix = detect_name_market(code, fallback_name, end_limit)

    if market == "US":
        src_primary = load_yfinance(code, start_validation, end_limit, "yfinance")
        src_secondary = load_stooq(code, start_daily, end_limit)
        public_sources = [src_primary, src_secondary]
        kiwoom_source = SourceFrame("Kiwoom", pd.DataFrame(), "미국 주식은 키움 일봉 후보에서 제외")
    else:
        kiwoom_source = _load_kiwoom_daily_source(provider, code)
        src_pykrx = load_pykrx(code, start_daily, end_limit)
        src_fdr = load_fdr(code, start_daily, end_limit)
        src_yf = load_yfinance(code + yf_suffix, start_validation, end_limit, "yfinance")
        public_sources = [src_pykrx, src_fdr, src_yf]

    validation, reliability, stop_precision = source_validation(public_sources, end_limit)
    _price_label, _volume_label, validation_note = validation_labels(validation)
    if stop_precision:
        validation_note = f"{validation_note}; 대표가격 산정 불가"

    public_close = _public_reference_close(public_sources, end_limit)
    kiwoom_eligible = False
    if market != "US" and kiwoom_source.data is not None and not kiwoom_source.data.empty:
        kiwoom_close = _latest_close_from_frame(kiwoom_source.data, end_limit)
        kiwoom_completed_rows = _completed_row_count(kiwoom_source.data, end_limit)
        if public_close is None:
            stop_precision = True
            validation_note = f"{validation_note}; 키움 일봉 단독 사용 금지: pykrx/FDR 기준 종가 없음"
        elif kiwoom_close is None:
            validation_note = f"{validation_note}; 키움 일봉 최신 완료 종가 산정 불가"
        elif kiwoom_completed_rows < DAILY_MIN_COMPLETED_ROWS:
            validation_note = f"{validation_note}; 키움 일봉 표본 부족({kiwoom_completed_rows}개, MA240 계산 불가)으로 대표 후보 제외"
        else:
            diff_pct = _pct_diff(kiwoom_close, public_close)
            if diff_pct > 3.0:
                stop_precision = True
                validation_note = f"{validation_note}; 키움 일봉과 pykrx/FDR 종가 차이 {diff_pct:.2f}%로 분석 중단"
            else:
                kiwoom_eligible = True
                if diff_pct > 1.0:
                    validation_note = f"{validation_note}; 키움 일봉과 pykrx/FDR 종가 차이 {diff_pct:.2f}% 경고"
                else:
                    validation_note = f"{validation_note}; 키움 일봉 교차검증 통과({diff_pct:.2f}%)"
    elif market != "US":
        validation_note = f"{validation_note}; 키움 일봉 후보 미사용(비차단): {kiwoom_source.note}"

    selected = _select_daily_source(
        [kiwoom_source if kiwoom_eligible else None, *public_sources],
        validation,
        end_limit,
    )
    if selected is None:
        raise RuntimeError("대표 일봉 데이터를 수집하지 못했습니다.")

    frame = selected.data[selected.data.index.date <= end_limit].copy()
    if frame.empty:
        raise RuntimeError("대표 일봉 데이터가 비어 있습니다.")
    frame = _standardize_daily_frame(frame)
    return DailyData(stock_name, market, yf_suffix, frame, reliability, validation_note, stop_precision, public_close)


def _load_kiwoom_daily_source(provider: KiwoomDataProvider | None, code: str) -> SourceFrame:
    if provider is None:
        return SourceFrame("Kiwoom", pd.DataFrame(), "키움 Provider 없음")
    try:
        frame = provider.get_daily_ohlcv(code, limit=500)
        return SourceFrame("Kiwoom", _standardize_daily_frame(frame))
    except Exception as exc:
        return SourceFrame("Kiwoom", pd.DataFrame(), f"{type(exc).__name__}: {exc}")


def _latest_close_from_frame(frame: pd.DataFrame, end_limit: date) -> float | None:
    if frame is None or frame.empty:
        return None
    data = frame[frame.index.date <= end_limit]
    if data.empty:
        return None
    return finite(data.iloc[-1].get("Close"))


def _previous_close_from_frame(frame: pd.DataFrame) -> float | None:
    if frame is None or frame.empty or len(frame) < 2:
        return None
    return finite(frame.iloc[-2].get("Close"))


def _completed_row_count(frame: pd.DataFrame, end_limit: date) -> int:
    if frame is None or frame.empty:
        return 0
    return int(len(frame[frame.index.date <= end_limit]))


def _public_reference_close(sources: list[SourceFrame], end_limit: date) -> float | None:
    closes: list[tuple[date, float]] = []
    for src in sources:
        if src.name not in {"pykrx", "FinanceDataReader"} or src.data is None or src.data.empty:
            continue
        data = src.data[src.data.index.date <= end_limit]
        if data.empty:
            continue
        close = finite(data.iloc[-1].get("Close"))
        if close is not None and close > 0:
            closes.append((data.index[-1].date(), close))
    if not closes:
        return None
    latest = max(item[0] for item in closes)
    latest_closes = [close for row_date, close in closes if row_date == latest]
    return sum(latest_closes) / len(latest_closes) if latest_closes else None


def _pct_diff(left: float, right: float) -> float:
    if right <= 0:
        return float("inf")
    return abs(left - right) / right * 100


def _select_daily_source(sources: list[SourceFrame | None], validation: pd.DataFrame, end_limit: date) -> SourceFrame | None:
    representative = ""
    if "대표가격사용" in validation.columns:
        reps = validation[validation["대표가격사용"] == "예"]
        if not reps.empty:
            representative = str(reps.iloc[0]["소스"])
    priority_names = ["Kiwoom", representative, "pykrx", "FinanceDataReader", "yfinance"]
    for name in priority_names:
        if not name:
            continue
        for src in sources:
            if src is None or src.name != name or src.data is None or src.data.empty:
                continue
            frame = src.data[src.data.index.date <= end_limit]
            if not frame.empty:
                return src
    return None


def _get_intraday_ohlcv_with_retry(
    provider: KiwoomDataProvider,
    code: str,
    interval_minutes: int,
    *,
    min_rows: int = INTRADAY_MIN_ROWS,
) -> pd.DataFrame:
    last_error: Exception | None = None
    last_rows = 0
    attempts = max(1, int(INTRADAY_RETRY_ATTEMPTS))
    for attempt in range(attempts):
        try:
            frame = provider.get_intraday_ohlcv(code, interval_minutes=interval_minutes, limit=INTRADAY_REQUEST_LIMIT)
            last_rows = 0 if frame is None else len(frame)
            if frame is not None and last_rows >= min_rows:
                return frame
        except Exception as exc:
            last_error = exc
        if attempt < attempts - 1 and INTRADAY_RETRY_DELAY_SEC > 0:
            time_module.sleep(INTRADAY_RETRY_DELAY_SEC)

    detail = f"{interval_minutes}분봉 {last_rows}개"
    if last_error is not None:
        detail = f"{detail}, 마지막 오류: {last_error}"
    raise KiwoomDataError(f"키움 체결 데이터 부족 또는 분봉 생성 실패: {detail}")


def analyze_command_chart(
    code: str,
    fallback_name: str | None = None,
    provider: KiwoomDataProvider | None = None,
    now: datetime | None = None,
    *,
    require_kiwoom: bool = True,
    sse_required: bool = True,
) -> str:
    code = code.strip()
    if not (code.isdigit() and len(code) == 6):
        code = code.upper()

    provider = provider or KiwoomDataProvider()
    public_errors: list[str] = []
    public_warnings: list[str] = []
    kiwoom_errors: list[str] = []
    kiwoom_warnings: list[str] = []
    sse_errors: list[str] = []
    sse_warnings: list[str] = []
    daily_data: DailyData | None = None
    quote = None
    minute3 = pd.DataFrame()
    minute5 = pd.DataFrame()
    safe_name = sanitize_filename(fallback_name or code)

    try:
        daily_data = collect_daily_data(code, fallback_name, provider)
        safe_name = sanitize_filename(daily_data.stock_name)
    except Exception as exc:
        public_errors.append(f"대표가격 산정 불가: {exc}")

    try:
        quote = provider.get_quote(code)
    except Exception as exc:
        kiwoom_errors.append(f"키움 현재가 수집 실패: {exc}")

    try:
        minute3 = _get_intraday_ohlcv_with_retry(provider, code, interval_minutes=3)
        minute5 = _get_intraday_ohlcv_with_retry(provider, code, interval_minutes=5)
        if len(minute3) < INTRADAY_MIN_ROWS or len(minute5) < INTRADAY_MIN_ROWS:
            kiwoom_errors.append("키움 체결 데이터 부족 또는 분봉 생성 실패")
    except Exception as exc:
        kiwoom_errors.append(f"분봉 생성 실패: {exc}")

    session_intraday = is_korea_regular_session(now)
    if daily_data is None:
        return _stop_and_write_failure(safe_name, code, public_errors or ["대표가격 산정 불가"])
    if quote is None and require_kiwoom:
        return _stop_and_write_failure(safe_name, code, kiwoom_errors or ["키움 현재가 수집 실패"])

    out_dir = REPORTS_DIR / f"{safe_name}_{code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    public_close = finite(daily_data.frame.iloc[-1].get("Close")) if not daily_data.frame.empty else None
    public_previous_close = _previous_close_from_frame(daily_data.frame)
    current_price = safe_round(getattr(quote, "price", None)) or safe_round(public_close) or 0
    if quote is not None:
        quote_errors, quote_warnings = _validate_quote(
            quote,
            daily_data.public_reference_close,
            public_previous_close,
            current_price,
            is_intraday=session_intraday,
            now=now,
        )
        kiwoom_errors.extend(quote_errors)
        kiwoom_warnings.extend(quote_warnings)
        price_source = _price_source_info(quote, session_intraday)
        if price_source.note:
            kiwoom_warnings.append(price_source.note)
        if quote_warnings:
            daily_data = replace(daily_data, validation_note=f"{daily_data.validation_note}; {'; '.join(quote_warnings)}")
    elif not require_kiwoom:
        price_source = PriceSourceInfo("공개 데이터 기준가", "키움 가격 보정", "키움 가격 데이터 없음", False)
        kiwoom_warnings.append("키움 실시간 데이터 미확인으로 장중 매수 지시는 제한합니다. 공개 데이터 기준 큰 그림만 참고하십시오.")
    else:
        price_source = PriceSourceInfo("가격 기준", "키움 가격 보정", "키움 가격 데이터 없음", False)

    try:
        daily_ind = calculate_standard_indicators(daily_data.frame)
    except Exception as exc:
        return _stop_and_write_failure(safe_name, code, public_errors + [f"필수 지표 계산 실패: {exc}"])

    if not daily_indicators_valid(daily_ind):
        public_errors.append("일봉 필수 지표 계산 실패")
    if daily_data.reliability == "낮음" or daily_data.stop_precision:
        public_errors.append(f"데이터 신뢰도 낮음: {daily_data.validation_note}")

    if minute3.empty or minute5.empty:
        minute3 = _daily_frame_as_minute_fallback(daily_data.frame, "3min")
        minute5 = _daily_frame_as_minute_fallback(daily_data.frame, "5min")
    try:
        minute3_ind = calculate_standard_indicators(minute3)
        minute5_ind = calculate_standard_indicators(minute5)
    except Exception as exc:
        minute3_ind = calculate_standard_indicators(_daily_frame_as_minute_fallback(daily_data.frame, "3min"))
        minute5_ind = calculate_standard_indicators(_daily_frame_as_minute_fallback(daily_data.frame, "5min"))
        kiwoom_errors.append(f"분봉 필수 지표 계산 실패: {exc}")

    if quote is not None and (not intraday_indicators_valid(minute3_ind) or not intraday_indicators_valid(minute5_ind)):
        kiwoom_errors.append("분봉 필수 지표 계산 실패")

    daily_close = finite(daily_ind.iloc[-1].get("Close"))
    if daily_close and current_price and abs(current_price - daily_close) / daily_close > 0.30:
        kiwoom_errors.append("키움 현재가와 일봉 대표 가격의 비정상 불일치")

    sse_result = calculate_sse_indicator(
        daily_ind,
        minute3_ind,
        minute5_ind,
        current_price=current_price,
        is_intraday=session_intraday,
    )
    if sse_result.blocking_errors:
        sse_errors.extend(sse_result.blocking_errors)
    sse_warnings.extend(sse_result.warnings)

    if require_kiwoom and kiwoom_errors:
        return _stop_and_write_failure(safe_name, code, kiwoom_errors)
    if public_errors:
        return _stop_and_write_failure(safe_name, code, public_errors)
    if sse_required and sse_errors:
        return _stop_and_write_failure(safe_name, code, sse_errors)

    public_status = LayerStatus("공개 데이터 분석", ok=not public_errors, warnings=tuple(public_warnings), blocking_errors=tuple(public_errors))
    kiwoom_status = LayerStatus(price_source.status_name, ok=not kiwoom_errors and quote is not None, warnings=tuple(kiwoom_warnings), blocking_errors=tuple(kiwoom_errors))
    sse_status = LayerStatus("SSE Indicator", ok=not sse_errors, warnings=tuple(sse_warnings), blocking_errors=tuple(sse_errors))

    levels = build_decision_levels(current_price, daily_ind, minute3_ind, minute5_ind)
    decision = evaluate_decision(
        DecisionContext(
            current_price=current_price,
            levels=levels,
            is_intraday=session_intraday,
            data_valid=True,
            invalid_reasons=(),
            volume_ratio20=last_valid(daily_ind.iloc[-1], "거래량비율20"),
            rsi14=last_valid(daily_ind.iloc[-1], "RSI14"),
            bollinger_upper=last_valid(daily_ind.iloc[-1], "BB상단"),
            risk_reward=_risk_reward(levels),
        )
    )
    pipeline = run_integrated_analysis_pipeline(
        public_status,
        kiwoom_status,
        sse_status,
        decision.verdict,
        sse_verdict=sse_result.verdict if sse_status.ok else None,
        sse_required=sse_required,
    )
    if not decision.stopped and sse_status.ok:
        decision = apply_sse_safety_filter(decision, sse_result, current_price, session_intraday)
    if pipeline.realtime_limited and decision.verdict in {"사라", "조건부로 사라"}:
        decision = apply_realtime_limit(decision)
    if decision.verdict != pipeline.final_verdict and pipeline.final_verdict in {"분석 중단", "팔아라", "사지 마라", "기다려라", "보유하라"}:
        decision = replace(decision, verdict=pipeline.final_verdict, headline=f"{decision.headline}; {pipeline.final_reason}")
    decision = _guard_conservative_decision_text(decision, sse_result if sse_status.ok else None)

    if decision.stopped:
        return _stop_and_write_failure(safe_name, code, list(decision.blocking_errors))

    report = render_report(safe_name, code, current_price, session_intraday, daily_data, decision, sse_result if sse_status.ok else None, pipeline, price_source)
    qa_errors = validate_command_report(
        report,
        decision,
        levels,
        is_intraday=session_intraday,
        data_valid=True,
        current_price=current_price,
        sse_result=sse_result if sse_status.ok else None,
        realtime_limited=pipeline.realtime_limited,
    )
    if qa_errors:
        return _stop_and_write_failure(safe_name, code, qa_errors)

    md_path, html_path, qa_path = _report_paths(out_dir, safe_name, code)
    _remove_legacy_reports(out_dir, safe_name, code)
    qa_path.unlink(missing_ok=True)
    md_path.write_text(report, encoding="utf-8-sig")
    html_path.write_text(html_from_markdown(report, f"[{safe_name}, {code}] 분석 보고서"), encoding="utf-8-sig")

    return _console_summary(safe_name, code, current_price, decision, md_path, price_source)


def build_decision_levels(current_price: int, daily_ind: pd.DataFrame, minute3_ind: pd.DataFrame, minute5_ind: pd.DataFrame) -> DecisionLevels:
    daily = daily_ind.iloc[-1]
    m3 = minute3_ind.iloc[-1]
    m5 = minute5_ind.iloc[-1]
    support = _cluster_evidence(
        "핵심 지지선",
        [
            ("20일선", last_valid(daily, "MA20")),
            ("60일선", last_valid(daily, "MA60")),
            ("볼린저밴드 중심선", last_valid(daily, "BB중심")),
            ("볼린저밴드 하단", last_valid(daily, "BB하단")),
            ("일목 전환선", last_valid(daily, "전환선")),
            ("일목 기준선", last_valid(daily, "기준선")),
            ("일목 구름 상단", _cloud_high(daily)),
            ("일목 구름 하단", _cloud_low(daily)),
            ("최근 5일 저점", finite(daily_ind["Low"].tail(5).min())),
            ("최근 20일 저점", finite(daily_ind["Low"].tail(20).min())),
        ],
        current_price,
        prefer="below",
    )
    confirmation_floor = support.price if support else current_price
    confirmation = _cluster_evidence(
        "매수 확인선",
        [
            ("3분봉 20이평선", last_valid(m3, "MA20")),
            ("3분봉 볼린저밴드 중심선", last_valid(m3, "BB중심")),
            ("직전 분봉 반등 고점", finite(minute3_ind["High"].tail(8).max())),
            ("5분봉 20이평선", last_valid(m5, "MA20")),
            ("5분봉 볼린저밴드 중심선", last_valid(m5, "BB중심")),
            ("직전 분봉 반등 고점", finite(minute5_ind["High"].tail(8).max())),
        ],
        confirmation_floor,
        prefer="above",
    )
    breakout = _cluster_evidence(
        "돌파선",
        [
            ("최근 5일 고점", finite(daily_ind["High"].tail(5).max())),
            ("최근 20일 고점", finite(daily_ind["High"].tail(20).max())),
            ("볼린저밴드 상단", last_valid(daily, "BB상단")),
            ("일목 구름 상단", _cloud_high(daily)),
        ],
        current_price,
        prefer="above",
    )
    stop = _cluster_evidence(
        "손절/방어선",
        [
            ("최근 20일 저점", finite(daily_ind["Low"].tail(20).min())),
            ("일목 구름 하단", _cloud_low(daily)),
            ("60일선", last_valid(daily, "MA60")),
            ("볼린저밴드 하단", last_valid(daily, "BB하단")),
        ],
        support.price if support else current_price,
        prefer="below",
    )
    no_chase = _cluster_evidence(
        "추격 금지선",
        [
            ("볼린저밴드 상단", last_valid(daily, "BB상단")),
            ("최근 20일 고점", finite(daily_ind["High"].tail(20).max())),
        ],
        current_price,
        prefer="above",
    )
    target1 = breakout
    target2 = _cluster_evidence(
        "신규매수 기준 2차 목표",
        [("최근 20일 고점", finite(daily_ind["High"].tail(20).max())), ("볼린저밴드 상단", last_valid(daily, "BB상단"))],
        (breakout.price if breakout else current_price) * 1.03,
        prefer="above",
    )
    return DecisionLevels(support=support, confirmation=confirmation, breakout=breakout, stop=stop, no_chase=no_chase, target1=target1, target2=target2)


SSE_SAFETY_PRIORITY = {
    "사라": 0,
    "조건부로 사라": 1,
    "보유하라": 2,
    "기다려라": 3,
    "사지 마라": 4,
    "팔아라": 5,
    "분석 중단": 6,
}


def apply_sse_safety_filter(decision: DecisionResult, sse_result: SSEResult, current_price: int, is_intraday: bool) -> DecisionResult:
    if sse_result.verdict == "분석 중단":
        return replace(
            decision,
            verdict="분석 중단",
            blocking_errors=tuple(decision.blocking_errors) + tuple(sse_result.blocking_errors or ("SSE 분석 중단",)),
            final_action_state="NO_BUY_DATA_INVALID",
        )
    if SSE_SAFETY_PRIORITY.get(sse_result.verdict, 0) <= SSE_SAFETY_PRIORITY.get(decision.verdict, 0):
        return decision

    levels = sse_result.levels
    sse_evidence = (
        PriceEvidence("SSE 예상 진입가", safe_round(levels.entry) or int(levels.entry), ("SSE 기준선+통합 변동성",)),
        PriceEvidence("SSE 예상 손절가", safe_round(levels.stop) or int(levels.stop), ("SSE 손절 공식",)),
        PriceEvidence("SSE 추격 금지선", safe_round(levels.no_chase) or int(levels.no_chase), ("SSE 기준선+1.50*통합 변동성",)),
    )
    action, final_state = _sse_action_text(sse_result, current_price, is_intraday)
    return replace(
        decision,
        verdict=sse_result.verdict,
        headline=f"{decision.headline}; SSE 안전 필터 적용: {sse_result.verdict}",
        actions=(action,) + _demote_positive_buy_lines(decision.actions, sse_result),
        buy_conditions=_demote_positive_buy_lines(decision.buy_conditions, sse_result),
        no_buy_conditions=decision.no_buy_conditions + (_sse_no_buy_text(sse_result),),
        sell_conditions=decision.sell_conditions + (f"SSE 예상 손절가 {format_price(levels.stop)} 이탈 시 팔아라 또는 비중 축소하라.",),
        holder_conditions=decision.holder_conditions + (f"SSE 압력값 {levels.pressure:.2f} 기준으로 보유자는 {format_price(levels.target1)} 접근 시 1차 익절을 관리하라.",),
        price_evidence=decision.price_evidence + sse_evidence,
        final_action_state=final_state,
    )


def _sse_action_text(sse_result: SSEResult, current_price: int, is_intraday: bool) -> tuple[str, str]:
    levels = sse_result.levels
    if sse_result.verdict == "팔아라":
        return (f"SSE 예상 손절가 {format_price(levels.stop)} 이탈 구조이므로 팔아라 또는 비중 축소하라.", "DEFENSE_REQUIRED")
    if sse_result.verdict == "사지 마라":
        return (f"SSE 기준 신규매수 금지. {format_price(levels.no_chase)} 이상 추격 금지 또는 손익비 부족 구간이다.", "NO_BUY_OVERHEATED_BAD_RR")
    if sse_result.verdict == "기다려라":
        return (f"SSE 예상 진입가 {format_price(levels.entry)} 회복 전까지 기다려라.", "WAIT_RECOVERY_CLOSE")
    if sse_result.verdict == "보유하라":
        return (f"SSE 1차 익절가 {format_price(levels.target1)} 전까지 보유하되 압력값을 관리하라.", "HOLD_AND_TRAIL")
    suffix = "3분봉 또는 5분봉 종가 유지 시에만 1차 진입하라." if is_intraday else "종가 유지 확인 후에만 1차 진입하라."
    return (f"SSE 예상 진입가 {format_price(levels.entry)} 이상에서 {suffix}", "WATCH_INTRADAY_BREAKOUT" if is_intraday else "HOLD_AND_TRAIL")


def _sse_no_buy_text(sse_result: SSEResult) -> str:
    levels = sse_result.levels
    return f"SSE 추격 금지선 {format_price(levels.no_chase)} 이상 또는 RR1 {levels.rr1:.2f}배 미만이면 사지 마라."


BUY_POSITIVE_PHRASES = (
    "1차 매수하라",
    "1차 매수를 검토하라",
    "1차 분할 매수",
    "분할 매수 가능",
    "매수 가능",
    "1차 진입하라",
)


def _demote_positive_buy_lines(lines: tuple[str, ...], sse_result: SSEResult | None) -> tuple[str, ...]:
    if sse_result is not None and sse_result.verdict in {"사라", "조건부로 사라"}:
        return lines
    return tuple(_demote_positive_buy_line(line, sse_result) for line in lines)


def _guard_conservative_decision_text(decision: DecisionResult, sse_result: SSEResult | None) -> DecisionResult:
    if decision.verdict in {"사라", "조건부로 사라", "분석 중단"}:
        return decision
    guarded_actions = _demote_positive_buy_lines(decision.actions, sse_result)
    guarded_buy_conditions = _demote_positive_buy_lines(decision.buy_conditions, sse_result)
    return replace(decision, actions=guarded_actions, buy_conditions=guarded_buy_conditions)


def _demote_positive_buy_line(line: str, sse_result: SSEResult | None) -> str:
    if not any(phrase in line for phrase in BUY_POSITIVE_PHRASES):
        if line.startswith("지지 매수:"):
            return line.replace("지지 매수:", "지지 관찰:", 1)
        if line.startswith("회복 매수:"):
            return line.replace("회복 매수:", "회복 관찰:", 1)
        if line.startswith("돌파 매수:"):
            return line.replace("돌파 매수:", "돌파 관찰:", 1)
        return line
    entry_text = format_price(sse_result.levels.entry) if sse_result is not None else ""
    wait_text = f"SSE 예상 진입가 {entry_text} 회복 전까지" if sse_result is not None else "최종 보수 판정이 해제되기 전까지"
    if line.startswith("돌파 매수:"):
        prefix = line.replace("돌파 매수:", "돌파 관찰:", 1).split(" 시", 1)[0]
        return f"{prefix} 조건은 관심 신호로만 관찰하고, {wait_text} 신규매수하지 마라."
    if line.startswith("지지 매수:") or line.startswith("회복 매수:"):
        prefix = line.replace("지지 매수:", "지지 관찰:", 1).replace("회복 매수:", "회복 관찰:", 1).split(" 시", 1)[0]
        return f"{prefix} 조건은 관심 신호로만 관찰하고, {wait_text} 신규매수하지 마라."
    if " 이상" in line:
        prefix = line.split(" 이상", 1)[0]
        return f"{prefix} 이상 돌파 시도는 관심 신호로만 관찰하고, {wait_text} 신규매수하지 마라."
    return f"기존 매수 조건은 관심 신호로만 관찰하고, {wait_text} 신규매수하지 마라."


def apply_realtime_limit(decision: DecisionResult) -> DecisionResult:
    limit_text = "키움 실시간 데이터 미확인으로 장중 매수 지시는 제한합니다. 공개 데이터 기준 큰 그림만 참고하십시오."
    return replace(
        decision,
        verdict="기다려라",
        headline=f"{decision.headline}; {limit_text}",
        actions=(limit_text,) + decision.actions,
        no_buy_conditions=decision.no_buy_conditions + (limit_text,),
        final_action_state="WAIT_RECOVERY_CLOSE",
    )


def render_sse_section(sse_result: SSEResult) -> str:
    levels = sse_result.levels
    evidence_lines = "\n".join(
        f"* {item.label} {format_sse_value(item.value)}: {item.formula} - {item.reason}"
        for item in sse_result.evidence
        if item.label not in {"SSE 기회 점수", "SSE 기회 경고"}
    )
    warning_lines = "\n".join(f"* {warning}" for warning in sse_result.warnings) if sse_result.warnings else "* 경고 없음"
    opportunity_block = render_sse_opportunity_block(sse_result)
    return f"""## SSE Indicator 분석

SSE 기준선: {format_sse_price(levels.base)}
SSE 상단선: {format_sse_price(levels.upper)}
SSE 하단선: {format_sse_price(levels.lower)}
SSE 압력값: {format_sse_number(levels.pressure)}

예상 진입가: {format_sse_price(levels.entry)}
예상 손절가: {format_sse_price(levels.stop)}
1차 익절가: {format_sse_price(levels.target1)}
2차 익절가: {format_sse_price(levels.target2)}
추격 금지선: {format_sse_price(levels.no_chase)}

1차 목표 기준 손익비: {format_sse_number(levels.rr1)}배
2차 목표 기준 손익비: {format_sse_number(levels.rr2)}배

SSE 최종 판정: {sse_result.verdict}

{opportunity_block}

산출 근거:

{evidence_lines}

SSE 경고:

{warning_lines}
"""


def render_sse_opportunity_block(sse_result: SSEResult) -> str:
    opportunity = sse_result.opportunity
    if opportunity is None:
        return """SSE 기회 점수: 계산 불가
SSE 기회 등급: 계산 불가
SSE 셋업: 계산 불가

SSE 기회 사유:
* 계산 불가

SSE 기회 경고:
* 경고 없음"""

    reason_lines = "\n".join(f"* {reason}" for reason in opportunity.reasons) if opportunity.reasons else "* 우수 셋업 근거 부족"
    warning_lines = "\n".join(f"* {warning}" for warning in opportunity.warnings) if opportunity.warnings else "* 경고 없음"
    return f"""SSE 기회 점수: {opportunity.score:.1f}점
SSE 기회 등급: {opportunity.grade}
SSE 셋업: {opportunity.setup}

SSE 기회 사유:
{reason_lines}

SSE 기회 경고:
{warning_lines}"""


def format_sse_number(value: float | None) -> str:
    v = finite(value)
    return "계산 불가" if v is None else f"{v:.2f}"


def format_sse_price(value: float | None) -> str:
    v = finite(value)
    return "계산 불가" if v is None else format_price(v)


def format_sse_value(value: float | None) -> str:
    v = finite(value)
    if v is None:
        return "계산 불가"
    if abs(v) >= 100:
        return format_price(v)
    return f"{v:.2f}"


def render_report(
    stock_name: str,
    code: str,
    current_price: int,
    is_intraday: bool,
    daily_data: DailyData,
    decision: DecisionResult,
    sse_result: SSEResult | None = None,
    pipeline: IntegratedAnalysisResult | None = None,
    price_source: PriceSourceInfo | None = None,
) -> str:
    session_text = "장중" if is_intraday else "장마감 이후"
    price_source = price_source or PriceSourceInfo("현재가", "키움 가격 보정", "", False)
    evidence = "\n".join(f"* {_price_evidence_summary(item, current_price)}" for item in decision.price_evidence)
    actions = "\n".join(f"* {line}" for line in decision.actions)
    buy = "\n".join(f"* {line}" for line in decision.buy_conditions)
    no_buy = "\n".join(f"* {line}" for line in decision.no_buy_conditions)
    sell = "\n".join(f"* {line}" for line in decision.sell_conditions)
    holder = "\n".join(f"* {line}" for line in decision.holder_conditions)
    sse_section = render_sse_section(sse_result) if sse_result is not None else ""
    pipeline_section = render_pipeline_section(pipeline) if pipeline is not None else ""
    internal_validation = render_internal_validation_section(daily_data, pipeline, sse_result)
    return f"""# [{stock_name}, {code}] 분석 보고서

[최종 매매 지시]

최종 판정: {decision.verdict}
{price_source.label}: {format_price(current_price)}
분석 구분: {session_text}
가격 기준: {price_source.note or price_source.status_name}

지금 할 행동:

{actions}

매수 조건:

{buy}

매수 금지:

{no_buy}

매도/방어:

{sell}

보유자 대응:

{holder}

가격 근거:

{evidence}

## 판단 메모

{decision.headline}

데이터 검증 메모: {daily_data.validation_note}

{sse_section}

{pipeline_section}

## 내부 검증

{internal_validation}
"""


def _price_evidence_summary(item: PriceEvidence, current_price: int) -> str:
    label = item.label
    if item.label == "핵심 지지선" and item.price > current_price:
        label = "회복/안착 기준선"
    reasons = ", ".join(item.reasons)
    return f"{label} {format_price(item.price)}: {reasons} 근거"


def render_internal_validation_section(
    daily_data: DailyData,
    pipeline: IntegratedAnalysisResult | None,
    sse_result: SSEResult | None,
) -> str:
    warnings: list[str] = []
    if "후보 미사용" in daily_data.validation_note:
        warnings.append("키움 일봉 후보 미사용")
    if "경고" in daily_data.validation_note:
        warnings.append("데이터 검증 경고 포함")
    if pipeline is not None:
        for status in (pipeline.public_status, pipeline.kiwoom_status, pipeline.sse_status):
            warnings.extend(status.warnings)
            warnings.extend(status.blocking_errors)
    if sse_result is not None:
        warnings.extend(sse_result.warnings)
        warnings.extend(sse_result.blocking_errors)

    normalized_warnings = tuple(dict.fromkeys(item for item in warnings if item))
    status = "통과" if not normalized_warnings else "통과(경고 있음)"
    cross_validation = daily_data.reliability if not normalized_warnings else "중간"
    sse_reliability = "높음" if sse_result is not None and not sse_result.blocking_errors else "제한"
    warning_text = "없음" if not normalized_warnings else "; ".join(normalized_warnings)
    return f"""내부 검증: {status}
가격 신뢰도: {daily_data.reliability}
거래량 신뢰도: {daily_data.reliability}
지표 신뢰도: 높음
SSE 신뢰도: {sse_reliability}
교차검증 완전성: {cross_validation}
수급 신뢰도: 낮음
해석 완전성: 높음
경고 요약: {warning_text}"""


def render_pipeline_section(pipeline: IntegratedAnalysisResult) -> str:
    return f"""## 분석 레이어 상태

공개 데이터 분석:
- 상태: {_status_text(pipeline.public_status)}
- 경고: {_join_status_items(pipeline.public_status.warnings)}
- 차단 오류: {_join_status_items(pipeline.public_status.blocking_errors)}

{pipeline.kiwoom_status.name}:
- 상태: {_status_text(pipeline.kiwoom_status)}
- 경고: {_join_status_items(pipeline.kiwoom_status.warnings)}
- 차단 오류: {_join_status_items(pipeline.kiwoom_status.blocking_errors)}

SSE Indicator:
- 상태: {_status_text(pipeline.sse_status)}
- 경고: {_join_status_items(pipeline.sse_status.warnings)}
- 차단 오류: {_join_status_items(pipeline.sse_status.blocking_errors)}

최종 통합 판단:
- 최종 판정: {pipeline.final_verdict}
- 실시간 매수 제한 여부: {'예' if pipeline.realtime_limited else '아니오'}
- 통합 사유: {pipeline.final_reason}
"""


def _status_text(status: LayerStatus) -> str:
    return "정상" if status.ok else "제한"


def _join_status_items(items: tuple[str, ...]) -> str:
    return "없음" if not items else "; ".join(items)


def _stop_and_write_failure(stock_name: str, code: str, reasons: list[str]) -> str:
    safe_name = sanitize_filename(stock_name or code)
    out_dir = REPORTS_DIR / f"{safe_name}_{code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, html_path, qa_path = _report_paths(out_dir, safe_name, code)
    _remove_legacy_reports(out_dir, safe_name, code)
    md_path.unlink(missing_ok=True)
    html_path.unlink(missing_ok=True)
    body = "\n".join(f"- {reason}" for reason in reasons) if reasons else "- 원인 미상"
    qa_path.write_text(f"# [{safe_name}, {code}] 분석 실패 보고서\n\n최종판정: 분석 중단\n\n## 실패 사유\n\n{body}\n", encoding="utf-8-sig")
    return f"""[조건부 명령형 차트 분석 중단]

종목: {safe_name} {code}
최종 판정: 분석 중단
QA 실패 파일: {qa_path}"""


def _console_summary(stock_name: str, code: str, current_price: int, decision: DecisionResult, md_path: Any, price_source: PriceSourceInfo | None = None) -> str:
    price_source = price_source or PriceSourceInfo("현재가", "키움 가격 보정", "", False)
    return f"""[조건부 명령형 차트 분석 완료]

종목: {stock_name} {code}
{price_source.label}: {format_price(current_price)}
최종 판정: {decision.verdict}
지금 할 행동: {' '.join(decision.actions)}
보고서 경로: {md_path}"""


def _report_paths(out_dir: Any, safe_name: str, code: str) -> tuple[Any, Any, Any]:
    return (
        out_dir / f"[{safe_name}, {code}] 분석 보고서.md",
        out_dir / f"[{safe_name}, {code}] 분석 보고서.html",
        out_dir / f"[{safe_name}, {code}] 분석 실패 보고서.md",
    )


def _remove_legacy_reports(out_dir: Any, safe_name: str, code: str) -> None:
    for path in (
        out_dir / f"{safe_name}_{code}_조건부명령형_차트분석.md",
        out_dir / f"{safe_name}_{code}_조건부명령형_차트분석.html",
        out_dir / f"{safe_name}_{code}_보고서_QA실패.md",
    ):
        path.unlink(missing_ok=True)


def _price_source_info(quote: Any, is_intraday: bool) -> PriceSourceInfo:
    source_label = getattr(quote, "source_label", None) or getattr(quote, "source", None) or "키움"
    quote_time = getattr(quote, "quote_time", None)
    source_text = f"{getattr(quote, 'source', '')} {source_label}".lower()
    is_realtime = bool(getattr(quote, "is_realtime", False)) or "fid" in source_text or "realtime" in source_text or "실시간" in source_text
    is_current_tr = bool(getattr(quote, "is_current_tr", False))
    if is_realtime:
        time_note = f", 체결시간 {quote_time}" if quote_time else ""
        return PriceSourceInfo("실시간 현재가", "키움 실시간 체결 보정", f"키움 실시간 체결 FID 기준({source_label}{time_note})", True)
    if is_intraday:
        return PriceSourceInfo("키움 TR 기준가", "키움 장중 TR 보정", f"실시간 체결 FID가 아니라 키움 TR 기준가입니다({source_label}). 장중 실시간 호가/체결 화면과 차이가 날 수 있습니다.", False)
    if is_current_tr:
        return PriceSourceInfo("장마감 기준가", "키움 장마감 TR 보정", f"장마감 이후 키움 TR 가격 기준입니다({source_label}). 완료 일봉 대표값은 공개 데이터와 별도 교차검증합니다. 시간외·관심화면 가격과 다를 수 있습니다.", False)
    return PriceSourceInfo("키움 기준가", "키움 가격 보정", f"키움 가격 데이터 기준입니다({source_label}).", False)


def analyze_integrated_chart(
    code: str,
    fallback_name: str | None = None,
    provider: KiwoomDataProvider | None = None,
    now: datetime | None = None,
) -> str:
    return analyze_command_chart(code, fallback_name, provider=provider, now=now, require_kiwoom=True, sse_required=True)


def _standardize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "DateTime" not in out.columns:
        out.insert(0, "DateTime", pd.to_datetime(out.index))
    if "TradeValue" not in out.columns:
        out["TradeValue"] = out["Close"] * out["Volume"]
    return out[["DateTime", "Open", "High", "Low", "Close", "Volume", "TradeValue"]]


def _daily_frame_as_minute_fallback(frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    out = _standardize_daily_frame(frame).tail(80).copy()
    end = datetime.now(KST).replace(second=0, microsecond=0)
    out["DateTime"] = pd.date_range(end=end, periods=len(out), freq=freq)
    out = out.set_index(pd.to_datetime(out["DateTime"]), drop=False)
    return out


def _validate_quote(
    quote: Any,
    public_reference_close: float | None,
    public_previous_close: float | None,
    current_price: int,
    *,
    is_intraday: bool,
    now: datetime | None = None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if current_price <= 0:
        errors.append("키움 현재가가 0 이하입니다")
    if quote.timestamp is None:
        errors.append("키움 현재가 timestamp가 없어 데이터 신뢰도를 확인할 수 없습니다")
    elif is_intraday:
        quote_time = _as_kst(quote.timestamp)
        current_time = (_as_kst(now) if now else datetime.now(KST))
        age = current_time - quote_time
        if age.total_seconds() < -300:
            errors.append("키움 현재가 timestamp가 현재 시각보다 미래입니다")
        elif age > timedelta(minutes=30):
            errors.append("키움 현재가 timestamp가 장중 기준 30분을 초과해 지연되었습니다")

    prev_close = finite(getattr(quote, "prev_close", None))
    if prev_close is not None and public_reference_close is not None:
        prev_reference = public_reference_close
        prev_reference_label = "pykrx/FDR 최신 완료 종가"
        quote_price = finite(getattr(quote, "price", None)) or float(current_price or 0)
        if (
            not is_intraday
            and public_previous_close is not None
            and quote_price > 0
            and _pct_diff(quote_price, public_reference_close) <= 3.0
        ):
            prev_reference = public_previous_close
            prev_reference_label = "공개 기준 직전 완료 종가"
        diff_pct = _pct_diff(prev_close, prev_reference)
        if diff_pct > 3.0:
            errors.append(f"키움 전일종가와 {prev_reference_label} 차이 {diff_pct:.2f}%로 분석 중단")
        elif diff_pct > 1.0:
            warnings.append(f"키움 전일종가와 {prev_reference_label} 차이 {diff_pct:.2f}% 경고")

    high = finite(getattr(quote, "high", None))
    low = finite(getattr(quote, "low", None))
    if high is not None and low is not None:
        if high < low:
            errors.append("키움 장중 고가/저가 범위가 비정상입니다")
        elif current_price and (high < current_price * 0.8 or low > current_price * 1.2):
            errors.append("키움 현재가와 장중 고가/저가 단위가 비정상적으로 불일치합니다")
        elif current_price and not (low <= current_price <= high):
            errors.append("키움 현재가가 장중 고가/저가 범위 밖입니다")
    return errors, warnings


def _as_kst(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=KST)
    return value.astimezone(KST)


def _cloud_low(row: pd.Series) -> float | None:
    values = [last_valid(row, "선행스팬1"), last_valid(row, "선행스팬2")]
    values = [v for v in values if v is not None]
    return min(values) if values else None


def _cloud_high(row: pd.Series) -> float | None:
    values = [last_valid(row, "선행스팬1"), last_valid(row, "선행스팬2")]
    values = [v for v in values if v is not None]
    return max(values) if values else None


def _cluster_evidence(label: str, candidates: list[tuple[str, float | None]], anchor: float, prefer: str) -> PriceEvidence | None:
    valid = [(reason, price) for reason, price in candidates if price is not None and price > 0]
    if not valid:
        return None
    if prefer == "below":
        directional = [(r, p) for r, p in valid if p <= anchor * 1.02] or valid
        center_reason, center = min(directional, key=lambda item: abs(item[1] - anchor))
        direction = "down"
    else:
        directional = [(r, p) for r, p in valid if p >= anchor * 0.98] or valid
        center_reason, center = min(directional, key=lambda item: abs(item[1] - anchor))
        direction = "up"
    band = max(anchor * 0.008, 1)
    reasons = tuple(dict.fromkeys([reason for reason, price in valid if abs(price - center) <= band] or [center_reason]))
    return PriceEvidence(label=label, price=safe_round(center, direction) or int(center), reasons=reasons)


def _risk_reward(levels: DecisionLevels) -> float | None:
    if not levels.target1 or not levels.confirmation or not levels.stop:
        return None
    reward = levels.target1.price - levels.confirmation.price
    risk = levels.confirmation.price - levels.stop.price
    if reward <= 0 or risk <= 0:
        return None
    return reward / risk


def main() -> int:
    parser = argparse.ArgumentParser(description="키움 API 기반 조건부 명령형 차트 분석기")
    parser.add_argument("code", help="종목코드")
    parser.add_argument("name", nargs="?", default=None, help="종목명")
    args = parser.parse_args()
    try:
        print(analyze_command_chart(args.code, args.name))
        return 0
    except Exception as exc:
        print(f"조건부 명령형 차트 분석 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
