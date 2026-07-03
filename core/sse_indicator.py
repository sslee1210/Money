from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SSE_BAND_MULTIPLIER = 1.8
SSE_BUY_RR_MIN = 1.2

SSE_VERDICT_PRIORITY = {
    "조건부로 사라": 1,
    "보유하라": 2,
    "기다려라": 3,
    "사지 마라": 4,
    "팔아라": 5,
    "분석 중단": 6,
}

REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume", "TradeValue")


@dataclass(frozen=True)
class SSEEvidence:
    label: str
    value: float
    formula: str
    reason: str


@dataclass(frozen=True)
class SSELevels:
    base: float
    upper: float
    lower: float
    pressure: float
    entry: float
    stop: float
    target1: float
    target2: float
    no_chase: float
    rr1: float
    rr2: float


@dataclass(frozen=True)
class SSEOpportunity:
    score: float
    grade: str
    setup: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SSEResult:
    verdict: str
    levels: SSELevels
    evidence: tuple[SSEEvidence, ...]
    warnings: tuple[str, ...]
    blocking_errors: tuple[str, ...]
    opportunity: SSEOpportunity | None = None


def add_sse_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add SSE raw math columns from standard OHLCV inputs.

    This function intentionally recalculates SSE columns from the original
    OHLCV inputs every time, so callers may pass either a raw indicator frame
    or a frame that already contains older SSE columns.
    """

    _validate_ohlcv_frame(frame)
    out = frame.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")
    high = pd.to_numeric(out["High"], errors="coerce")
    low = pd.to_numeric(out["Low"], errors="coerce")
    volume = pd.to_numeric(out["Volume"], errors="coerce")
    trade_value = pd.to_numeric(out["TradeValue"], errors="coerce")

    out["SSE_MA20"] = close.rolling(20).mean()
    out["SSE_MA60"] = close.rolling(60).mean()
    out["SSE_MA120"] = close.rolling(120).mean()
    out["SSE_MA240"] = close.rolling(240).mean()
    out["SSE_MEAN20"] = close.rolling(20).mean()
    out["SSE_STD20"] = close.rolling(20).std()
    out["SSE_MID9"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    out["SSE_MID26"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    out["SSE_MID52"] = (high.rolling(52).max() + low.rolling(52).min()) / 2

    out["SSE_MA_GAP"] = (out["SSE_MA20"] - out["SSE_MA60"]).abs()
    out["SSE_BALANCE_GAP"] = (out["SSE_MID26"] - out["SSE_MID52"]).abs()
    out["SSE_CLOUD_SPAN1_RAW"] = (out["SSE_MID9"] + out["SSE_MID26"]) / 2
    out["SSE_CLOUD_SPAN2_RAW"] = out["SSE_MID52"]
    out["SSE_CLOUD_THICKNESS"] = (out["SSE_CLOUD_SPAN1_RAW"] - out["SSE_CLOUD_SPAN2_RAW"]).abs()

    out["SSE_BASE"] = (
        0.35 * out["SSE_MA20"]
        + 0.20 * out["SSE_MA60"]
        + 0.20 * out["SSE_MID26"]
        + 0.15 * out["SSE_MID52"]
        + 0.10 * out["SSE_MID9"]
    )
    out["SSE_VOLATILITY"] = (
        0.50 * out["SSE_STD20"]
        + 0.25 * out["SSE_BALANCE_GAP"]
        + 0.25 * out["SSE_MA_GAP"]
    )
    out["SSE_UPPER"] = out["SSE_BASE"] + SSE_BAND_MULTIPLIER * out["SSE_VOLATILITY"]
    out["SSE_LOWER"] = out["SSE_BASE"] - SSE_BAND_MULTIPLIER * out["SSE_VOLATILITY"]
    out["SSE_PRESSURE"] = (close - out["SSE_BASE"]) / out["SSE_VOLATILITY"].replace(0, np.nan)
    out["SSE_ENTRY"] = out["SSE_BASE"] + 0.25 * out["SSE_VOLATILITY"]
    out["SSE_STOP_RAW"] = out["SSE_BASE"] - 0.75 * out["SSE_VOLATILITY"]
    out["SSE_RECENT5_HIGH"] = high.rolling(5).max()
    out["SSE_RECENT20_HIGH"] = high.rolling(20).max()
    out["SSE_RECENT60_HIGH"] = high.rolling(60).max()
    out["SSE_RECENT20_LOW"] = low.rolling(20).min()
    out["SSE_STOP"] = out[["SSE_STOP_RAW", "SSE_MID26", "SSE_MID52", "SSE_RECENT20_LOW"]].min(axis=1)
    out["SSE_TARGET1_RAW"] = out["SSE_BASE"] + 1.25 * out["SSE_VOLATILITY"]
    out["SSE_TARGET2_RAW"] = out["SSE_BASE"] + 1.80 * out["SSE_VOLATILITY"]
    out["SSE_NO_CHASE"] = out["SSE_BASE"] + 1.50 * out["SSE_VOLATILITY"]

    out["SSE_VOLUME_RATIO20"] = volume / volume.rolling(20).mean().replace(0, np.nan)
    out["SSE_TRADE_VALUE_RATIO20"] = trade_value / trade_value.rolling(20).mean().replace(0, np.nan)
    out["SSE_VOLATILITY_AVG60"] = out["SSE_VOLATILITY"].rolling(60).mean()
    out["SSE_VOLATILITY_PERCENTILE120"] = out["SSE_VOLATILITY"].rolling(120).apply(_percentile_rank_latest, raw=False)
    return out


def latest_sse_levels(frame: pd.DataFrame) -> SSELevels:
    sse_frame = add_sse_columns(frame)
    if sse_frame.empty:
        return _empty_levels()
    row = sse_frame.iloc[-1]
    return _levels_from_frame(sse_frame, _finite(row.get("Close")))


def validate_sse_levels(levels: SSELevels) -> list[str]:
    errors: list[str] = []
    required = {
        "SSE_BASE": levels.base,
        "SSE_VOLATILITY": _volatility_from_levels(levels),
        "SSE_ENTRY": levels.entry,
        "SSE_STOP": levels.stop,
        "SSE_TARGET1": levels.target1,
        "SSE_TARGET2": levels.target2,
        "SSE_NO_CHASE": levels.no_chase,
        "SSE_PRESSURE": levels.pressure,
    }
    for label, value in required.items():
        if not _is_finite(value):
            errors.append(f"{label} NaN")
    volatility = _volatility_from_levels(levels)
    if _is_finite(volatility) and volatility <= 0:
        errors.append("SSE_VOLATILITY <= 0")
    if _is_finite(levels.stop) and _is_finite(levels.entry) and levels.stop >= levels.entry:
        errors.append("SSE_STOP >= SSE_ENTRY")
    if _is_finite(levels.target1) and _is_finite(levels.entry) and levels.target1 <= levels.entry:
        errors.append("SSE_TARGET1 <= SSE_ENTRY")
    if _is_finite(levels.target2) and _is_finite(levels.target1) and levels.target2 <= levels.target1:
        errors.append("SSE_TARGET2 <= SSE_TARGET1")
    risk = levels.entry - levels.stop if _is_finite(levels.entry) and _is_finite(levels.stop) else np.nan
    if _is_finite(risk) and risk <= 0:
        errors.append("SSE_RISK <= 0")
    return errors


def classify_sse_verdict(levels: SSELevels, current_price: float, is_intraday: bool) -> str:
    errors = validate_sse_levels(levels)
    if errors or not _is_finite(current_price) or current_price <= 0:
        return "분석 중단"
    if current_price < levels.stop:
        return "팔아라"
    if current_price >= levels.no_chase:
        return "사지 마라"
    if levels.pressure >= 1.5:
        return "사지 마라"
    if levels.pressure < -1.0:
        return "사지 마라"
    if current_price >= levels.target1:
        return "보유하라"
    if current_price < levels.base:
        return "기다려라"
    if current_price < levels.entry:
        return "기다려라"
    if levels.rr1 < SSE_BUY_RR_MIN:
        return "사지 마라"
    if 0.3 <= levels.pressure < 1.5 and current_price < levels.no_chase:
        return "조건부로 사라"
    return "기다려라"


def calculate_sse_indicator(
    daily_ind: pd.DataFrame,
    minute3_ind: pd.DataFrame | None = None,
    minute5_ind: pd.DataFrame | None = None,
    current_price: float | None = None,
    is_intraday: bool = False,
) -> SSEResult:
    warnings: list[str] = []
    try:
        sse_frame = add_sse_columns(daily_ind)
        latest_close = _finite(sse_frame.iloc[-1].get("Close")) if not sse_frame.empty else np.nan
        effective_price = _finite(current_price) if current_price is not None else latest_close
        levels = _levels_from_frame(sse_frame, effective_price)
        errors = validate_sse_levels(levels)
        if not _is_finite(effective_price) or effective_price <= 0:
            errors.append("현재가 산정 불가")

        intraday_entry_confirmed = True
        if is_intraday:
            intraday_entry_confirmed = _has_intraday_close_condition(minute3_ind, minute5_ind, levels.entry)
        if is_intraday and not intraday_entry_confirmed:
            warnings.append("장중 SSE 진입은 3분봉 또는 5분봉 종가가 SSE_ENTRY 이상에서 유지될 때만 유효")
        if _is_finite(levels.rr1) and levels.rr1 < SSE_BUY_RR_MIN:
            warnings.append(f"SSE_RR1 {levels.rr1:.2f}배로 신규매수 기준 {SSE_BUY_RR_MIN:.2f}배 미만")
        if _is_finite(levels.rr2) and _is_finite(levels.rr1) and levels.rr2 <= levels.rr1:
            warnings.append("SSE_RR2 <= SSE_RR1")

        opportunity = calculate_sse_opportunity(
            sse_frame,
            levels,
            effective_price,
            is_intraday=is_intraday,
            intraday_entry_confirmed=intraday_entry_confirmed,
            blocking_errors=tuple(errors),
        )

        verdict = "분석 중단" if errors else classify_sse_verdict(levels, effective_price, is_intraday)
        if verdict in {"사라", "조건부로 사라"} and is_intraday and not intraday_entry_confirmed:
            verdict = "기다려라"
            warnings.append("장중 3분봉/5분봉 종가가 SSE_ENTRY 이상에서 유지되지 않아 아직 진입 조건 미충족")
        evidence = _build_evidence(sse_frame, levels, opportunity)
        return SSEResult(verdict, levels, evidence, tuple(warnings), tuple(errors), opportunity)
    except Exception as exc:
        return SSEResult("분석 중단", _empty_levels(), (), (), (f"SSE 계산 실패: {type(exc).__name__}: {exc}",), _empty_opportunity("DATA_BLOCKED"))

def calculate_sse_opportunity(
    frame: pd.DataFrame,
    levels: SSELevels,
    current_price: float,
    *,
    is_intraday: bool = False,
    intraday_entry_confirmed: bool = True,
    blocking_errors: tuple[str, ...] = (),
) -> SSEOpportunity:
    """Soft-score good setups without weakening the hard SSE safety filter."""

    if blocking_errors or frame.empty or not _is_finite(current_price):
        return _empty_opportunity("DATA_BLOCKED", warnings=("SSE 데이터/가격 검증 실패로 기회 점수 산정 제한",))

    row = frame.iloc[-1]
    score = 0.0
    reasons: list[str] = []
    soft_warnings: list[str] = []

    for add_score, reason, warning in (
        _score_pressure(levels.pressure),
        _score_volume(row),
        _score_rr(levels),
    ):
        score += add_score
        if reason:
            reasons.append(reason)
        if warning:
            soft_warnings.append(warning)

    reclaim_score, reclaim_reason = _score_reclaim(levels, current_price)
    score += reclaim_score
    if reclaim_reason:
        reasons.append(reclaim_reason)

    compression_score, compression_reason = _score_compression(row, current_price, levels)
    score += compression_score
    if compression_reason:
        reasons.append(compression_reason)

    pullback_reclaim = _is_pullback_reclaim(frame, levels, current_price)
    if pullback_reclaim:
        score += 8.0
        reasons.append("최근 눌림 후 SSE 기준선/진입가 회복 구조 감지")

    if is_intraday:
        if intraday_entry_confirmed:
            score += 5.0
            reasons.append("장중 3분봉 또는 5분봉 종가가 SSE 진입가 이상")
        else:
            soft_warnings.append("장중 진입 확인 미충족: 관심 후보로만 관리")

    setup = _classify_opportunity_setup(frame, levels, current_price, pullback_reclaim)
    if setup in {"OVERHEATED_HOLD_ONLY", "WEAK_BREAKDOWN"}:
        score = min(score, 49.0)
    if setup == "DATA_BLOCKED":
        score = 0.0

    score = max(0.0, min(100.0, score))
    grade = _opportunity_grade(score, setup)
    if not reasons:
        reasons.append("우수 셋업 근거 부족")

    return SSEOpportunity(
        round(score, 1),
        grade,
        setup,
        tuple(dict.fromkeys(reasons)),
        tuple(dict.fromkeys(soft_warnings)),
    )


def _score_pressure(pressure: float) -> tuple[float, str, str]:
    if not _is_finite(pressure):
        return 0.0, "", "SSE 압력값 계산 불가"
    if 0.3 <= pressure <= 0.8:
        return 25.0, f"SSE 압력값 {pressure:.2f}: 초기 회복 우수 구간", ""
    if 0.8 < pressure <= 1.2:
        return 18.0, f"SSE 압력값 {pressure:.2f}: 상승 우위이나 손익비 확인 필요", ""
    if -0.3 <= pressure < 0.3:
        return 10.0, f"SSE 압력값 {pressure:.2f}: 기준선 근처 방향 확인 구간", ""
    if -1.0 <= pressure < -0.3:
        return 6.0, f"SSE 압력값 {pressure:.2f}: 하단권 반등 감시 구간", ""
    if 1.2 < pressure < 1.5:
        return 6.0, f"SSE 압력값 {pressure:.2f}: 늦은 진입 가능성", "과열 접근 구간"
    if pressure >= 1.5:
        return 0.0, "", "SSE 압력값 과열권"
    return 0.0, "", "SSE 압력값 약세 이탈"


def _score_reclaim(levels: SSELevels, current_price: float) -> tuple[float, str]:
    if not _is_finite(current_price):
        return 0.0, ""
    if _is_finite(levels.entry) and current_price >= levels.entry:
        return 20.0, "현재가가 SSE 예상 진입가 이상으로 회복"
    if _is_finite(levels.base) and current_price >= levels.base:
        return 12.0, "현재가가 SSE 기준선 위로 회복"
    if _is_finite(levels.entry) and levels.entry > 0 and abs(current_price - levels.entry) / levels.entry <= 0.015:
        return 8.0, "SSE 예상 진입가 근처 접근"
    return 0.0, ""


def _score_compression(row: pd.Series, current_price: float, levels: SSELevels) -> tuple[float, str]:
    percentile = _finite(row.get("SSE_VOLATILITY_PERCENTILE120"))
    avg60 = _finite(row.get("SSE_VOLATILITY_AVG60"))
    volatility = _volatility_from_levels(levels)
    if _is_finite(percentile) and percentile <= 40 and _is_finite(levels.entry) and current_price >= levels.entry:
        return 20.0, f"SSE 변동성 백분위 {percentile:.0f}%: 수축 후 회복 후보"
    if _is_finite(avg60) and _is_finite(volatility) and volatility < avg60 and _is_finite(levels.base) and current_price >= levels.base:
        return 12.0, "SSE 변동성이 60일 평균보다 낮고 기준선 회복"
    return 0.0, ""


def _score_volume(row: pd.Series) -> tuple[float, str, str]:
    ratio = _max_finite(_finite(row.get("SSE_VOLUME_RATIO20")), _finite(row.get("SSE_TRADE_VALUE_RATIO20")))
    if not _is_finite(ratio):
        return 0.0, "", "거래량/거래대금 비율 계산 불가"
    if ratio >= 2.0:
        return 20.0, f"거래량/거래대금 20일 평균 대비 {ratio:.2f}배로 강한 유입", ""
    if ratio >= 1.3:
        return 14.0, f"거래량/거래대금 20일 평균 대비 {ratio:.2f}배로 유입 개선", ""
    if ratio >= 0.8:
        return 6.0, f"거래량/거래대금 20일 평균 대비 {ratio:.2f}배로 중립", ""
    return 0.0, "", f"거래량/거래대금 20일 평균 대비 {ratio:.2f}배로 약함"


def _score_rr(levels: SSELevels) -> tuple[float, str, str]:
    if not _is_finite(levels.rr1):
        return 0.0, "", "SSE RR1 계산 불가"
    if levels.rr1 < SSE_BUY_RR_MIN:
        return 0.0, "", f"SSE RR1 {levels.rr1:.2f}배로 신규매수 기준 미달"
    score = 15.0 if levels.rr1 >= 1.8 else 10.0
    reason = f"SSE RR1 {levels.rr1:.2f}배로 {'손익비 우수' if levels.rr1 >= 1.8 else '신규 진입 최소 손익비 충족'}"
    if _is_finite(levels.rr2) and levels.rr2 > levels.rr1:
        score += 3.0
        reason += f", RR2 {levels.rr2:.2f}배로 2차 여유 존재"
    return score, reason, ""


def _is_pullback_reclaim(frame: pd.DataFrame, levels: SSELevels, current_price: float) -> bool:
    if frame.empty or not _is_finite(current_price):
        return False
    recent = frame.tail(20)
    recent_low = _finite(recent["Low"].min()) if "Low" in recent.columns else np.nan
    touched_base = _is_finite(recent_low) and _is_finite(levels.base) and recent_low <= levels.base * 1.01
    touched_lower = _is_finite(recent_low) and _is_finite(levels.lower) and recent_low <= levels.lower * 1.03
    reclaimed = _is_finite(levels.entry) and current_price >= levels.entry
    return bool((touched_base or touched_lower) and reclaimed)


def _classify_opportunity_setup(frame: pd.DataFrame, levels: SSELevels, current_price: float, pullback_reclaim: bool) -> str:
    if not _is_finite(current_price) or validate_sse_levels(levels):
        return "DATA_BLOCKED"
    if current_price >= levels.no_chase or levels.pressure >= 1.5:
        return "OVERHEATED_HOLD_ONLY"
    if current_price < levels.stop or levels.pressure < -1.0:
        return "WEAK_BREAKDOWN"
    if pullback_reclaim:
        return "PULLBACK_RECLAIM"
    row = frame.iloc[-1] if not frame.empty else pd.Series(dtype=float)
    percentile = _finite(row.get("SSE_VOLATILITY_PERCENTILE120"))
    if _is_finite(percentile) and percentile <= 40 and _is_finite(levels.entry) and current_price >= levels.entry:
        return "COMPRESSION_BREAKOUT"
    if _is_finite(levels.entry) and current_price >= levels.entry:
        return "BASE_RECLAIM"
    if _is_finite(levels.entry) and levels.entry > 0 and abs(current_price - levels.entry) / levels.entry <= 0.015:
        return "WATCH_ENTRY"
    return "NO_SETUP"


def _opportunity_grade(score: float, setup: str) -> str:
    if setup in {"DATA_BLOCKED", "OVERHEATED_HOLD_ONLY", "WEAK_BREAKDOWN"}:
        return "제외"
    if score >= 80:
        return "A급 후보"
    if score >= 65:
        return "B급 후보"
    if score >= 50:
        return "C급 관찰"
    return "대기"


def _empty_opportunity(setup: str = "NO_SETUP", warnings: tuple[str, ...] = ()) -> SSEOpportunity:
    grade = "제외" if setup in {"DATA_BLOCKED", "OVERHEATED_HOLD_ONLY", "WEAK_BREAKDOWN"} else "대기"
    return SSEOpportunity(0.0, grade, setup, ("우수 셋업 근거 부족",), warnings)


def _percentile_rank_latest(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return float("nan")
    return float((series <= float(series.iloc[-1])).mean() * 100)



def _levels_from_frame(frame: pd.DataFrame, current_price: float | None) -> SSELevels:
    if frame.empty:
        return _empty_levels()
    row = frame.iloc[-1]
    base = _finite(row.get("SSE_BASE"))
    volatility = _finite(row.get("SSE_VOLATILITY"))
    entry = _finite(row.get("SSE_ENTRY"))
    stop = _finite(row.get("SSE_STOP"))
    no_chase = _finite(row.get("SSE_NO_CHASE"))
    target1 = _select_target1(frame, current_price, entry)
    target2 = _select_target2(frame, current_price, target1)
    row_pressure = _finite(row.get("SSE_PRESSURE"))
    pressure = (
        (_finite(current_price) - base) / volatility
        if _is_finite(current_price) and _is_finite(base) and _is_finite(volatility) and volatility > 0
        else row_pressure
    )
    risk = entry - stop if _is_finite(entry) and _is_finite(stop) else np.nan
    rr1 = (target1 - entry) / risk if _is_finite(target1) and _is_finite(entry) and _is_finite(risk) and risk > 0 else np.nan
    rr2 = (target2 - entry) / risk if _is_finite(target2) and _is_finite(entry) and _is_finite(risk) and risk > 0 else np.nan
    return SSELevels(
        base=base,
        upper=_finite(row.get("SSE_UPPER")),
        lower=_finite(row.get("SSE_LOWER")),
        pressure=pressure,
        entry=entry,
        stop=stop,
        target1=target1,
        target2=target2,
        no_chase=no_chase,
        rr1=rr1,
        rr2=rr2,
    )


def _select_target1(frame: pd.DataFrame, current_price: float | None, entry: float) -> float:
    row = frame.iloc[-1]
    floor = _max_finite(_finite(current_price), entry)
    volatility = _finite(row.get("SSE_VOLATILITY"))
    candidates = [
        _finite(row.get("SSE_TARGET1_RAW")),
        _finite(row.get("SSE_RECENT5_HIGH")),
        _finite(row.get("SSE_RECENT20_HIGH")),
        _finite(row.get("BB상단")),
        _finite(row.get("SSE_MA120")),
        _finite(row.get("SSE_MA240")),
    ]
    fallback = _finite(row.get("SSE_TARGET1_RAW"))
    if _is_finite(floor) and (not _is_finite(fallback) or fallback <= floor):
        fallback = floor + volatility if _is_finite(volatility) and volatility > 0 else floor * 1.01
    return _nearest_conservative_above(candidates, floor, fallback)


def _select_target2(frame: pd.DataFrame, current_price: float | None, target1: float) -> float:
    row = frame.iloc[-1]
    floor = _max_finite(_finite(current_price), target1)
    mid52 = _finite(row.get("SSE_MID52"))
    volatility = _finite(row.get("SSE_VOLATILITY"))
    candidates = [
        _finite(row.get("SSE_TARGET2_RAW")),
        _finite(row.get("SSE_RECENT20_HIGH")),
        _finite(row.get("SSE_RECENT60_HIGH")),
        _finite(row.get("SSE_MA120")),
        _finite(row.get("SSE_MA240")),
        mid52 + volatility if _is_finite(mid52) and _is_finite(volatility) else np.nan,
    ]
    fallback = _finite(row.get("SSE_TARGET2_RAW"))
    if _is_finite(floor) and (not _is_finite(fallback) or fallback <= floor):
        fallback = floor + volatility if _is_finite(volatility) and volatility > 0 else floor * 1.01
    return _nearest_conservative_above(candidates, floor, fallback)


def _nearest_conservative_above(candidates: list[float], floor: float, fallback: float) -> float:
    valid = [v for v in candidates if _is_finite(v) and (not _is_finite(floor) or v > floor)]
    if valid:
        return min(valid)
    return fallback


def _build_evidence(frame: pd.DataFrame, levels: SSELevels, opportunity: SSEOpportunity | None = None) -> tuple[SSEEvidence, ...]:
    if frame.empty:
        return ()
    row = frame.iloc[-1]
    evidence = (
        SSEEvidence("SSE 기준선", levels.base, "0.35*MA20 + 0.20*MA60 + 0.20*MID26 + 0.15*MID52 + 0.10*MID9", "종가 평균 원리와 고저 중간값 원리를 재조합"),
        SSEEvidence("SSE 평균 이격", _finite(row.get("SSE_MA_GAP")), "abs(MA20-MA60)", "단기/중기 종가 평균 이격 구조"),
        SSEEvidence("SSE 고저 균형 이격", _finite(row.get("SSE_BALANCE_GAP")), "abs(MID26-MID52)", "중기/장기 고저 균형 이격 구조"),
        SSEEvidence("SSE 구름 두께", _finite(row.get("SSE_CLOUD_THICKNESS")), "abs(((MID9+MID26)/2)-MID52)", "일목 시간 구조를 원천값으로 재구성한 구름 두께"),
        SSEEvidence("SSE 통합 변동성", _finite(row.get("SSE_VOLATILITY")), "0.50*STD20 + 0.25*abs(MID26-MID52) + 0.25*abs(MA20-MA60)", "표준편차, 고저 균형 간격, 평균선 간격을 통합"),
        SSEEvidence("SSE 상단선", levels.upper, "SSE_BASE + 1.8*SSE_VOLATILITY", "통합 변동성 기준 상단 범위"),
        SSEEvidence("SSE 하단선", levels.lower, "SSE_BASE - 1.8*SSE_VOLATILITY", "통합 변동성 기준 하단 범위"),
        SSEEvidence("SSE 압력값", levels.pressure, "(현재가 또는 Close - SSE_BASE) / SSE_VOLATILITY", "현재 가격의 기준선 대비 압력"),
        SSEEvidence("예상 진입가", levels.entry, "SSE_BASE + 0.25*SSE_VOLATILITY", "기준선 회복 후 조건부 진입 기준"),
        SSEEvidence("예상 손절가", levels.stop, "min(SSE_BASE - 0.75*SSE_VOLATILITY, MID26, MID52, 최근20일저점)", "보수적 방어 기준"),
        SSEEvidence("1차 익절가", levels.target1, "SSE_TARGET1 후보군 중 현재가/진입가 위의 보수적 저항", "가까운 상단 저항 우선"),
        SSEEvidence("2차 익절가", levels.target2, "SSE_TARGET2 후보군 중 1차 목표 위의 보수적 저항", "2차 목표 구조 검증"),
        SSEEvidence("추격 금지선", levels.no_chase, "SSE_BASE + 1.50*SSE_VOLATILITY", "과열 추격 금지 기준"),
    )
    return evidence



def _has_intraday_close_condition(minute3_ind: pd.DataFrame | None, minute5_ind: pd.DataFrame | None, entry: float) -> bool:
    if not _is_finite(entry):
        return False
    for frame in (minute3_ind, minute5_ind):
        if frame is not None and not frame.empty and "Close" in frame.columns:
            close = _finite(frame.iloc[-1].get("Close"))
            if _is_finite(close) and close >= entry:
                return True
    return False


def _validate_ohlcv_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]
    if "DateTime" not in frame.columns and "Date" not in frame.columns:
        missing.append("DateTime")
    if missing:
        raise ValueError(f"필수 OHLCV 컬럼 누락: {', '.join(missing)}")


def _empty_levels() -> SSELevels:
    nan = float("nan")
    return SSELevels(nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)


def _volatility_from_levels(levels: SSELevels) -> float:
    if _is_finite(levels.upper) and _is_finite(levels.base):
        return (levels.upper - levels.base) / SSE_BAND_MULTIPLIER
    if _is_finite(levels.base) and _is_finite(levels.lower):
        return (levels.base - levels.lower) / SSE_BAND_MULTIPLIER
    return float("nan")


def _max_finite(*values: float) -> float:
    finite_values = [value for value in values if _is_finite(value)]
    return max(finite_values) if finite_values else float("nan")


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False
