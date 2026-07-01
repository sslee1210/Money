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
class SSEResult:
    verdict: str
    levels: SSELevels
    evidence: tuple[SSEEvidence, ...]
    warnings: tuple[str, ...]
    blocking_errors: tuple[str, ...]


def add_sse_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add SSE raw math columns from standard OHLCV inputs."""

    _validate_ohlcv_frame(frame)
    out = frame.copy()
    close = pd.to_numeric(out["Close"], errors="coerce")
    high = pd.to_numeric(out["High"], errors="coerce")
    low = pd.to_numeric(out["Low"], errors="coerce")

    out["SSE_MA20"] = close.rolling(20).mean()
    out["SSE_MA60"] = close.rolling(60).mean()
    out["SSE_MA120"] = close.rolling(120).mean()
    out["SSE_MA240"] = close.rolling(240).mean()
    out["SSE_STD20"] = close.rolling(20).std()
    out["SSE_MID9"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    out["SSE_MID26"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    out["SSE_MID52"] = (high.rolling(52).max() + low.rolling(52).min()) / 2

    out["SSE_BASE"] = (
        0.35 * out["SSE_MA20"]
        + 0.20 * out["SSE_MA60"]
        + 0.20 * out["SSE_MID26"]
        + 0.15 * out["SSE_MID52"]
        + 0.10 * out["SSE_MID9"]
    )
    out["SSE_VOLATILITY"] = (
        0.50 * out["SSE_STD20"]
        + 0.25 * (out["SSE_MID26"] - out["SSE_MID52"]).abs()
        + 0.25 * (out["SSE_MA20"] - out["SSE_MA60"]).abs()
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
    return out


def latest_sse_levels(frame: pd.DataFrame) -> SSELevels:
    sse_frame = frame if "SSE_BASE" in frame.columns else add_sse_columns(frame)
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
    if current_price < levels.base:
        return "기다려라"
    if current_price < levels.entry:
        return "기다려라"
    if levels.rr1 < SSE_BUY_RR_MIN:
        return "사지 마라"
    if 0.3 <= levels.pressure < 1.5 and current_price < levels.no_chase:
        return "조건부로 사라"
    if current_price >= levels.target1:
        return "보유하라"
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
        if is_intraday and not _has_intraday_close_condition(minute3_ind, minute5_ind, levels.entry):
            warnings.append("장중 SSE 진입은 3분봉 또는 5분봉 종가가 SSE_ENTRY 이상에서 유지될 때만 유효")
        if _is_finite(levels.rr2) and _is_finite(levels.rr1) and levels.rr2 <= levels.rr1:
            warnings.append("SSE_RR2 <= SSE_RR1")
        verdict = "분석 중단" if errors else classify_sse_verdict(levels, effective_price, is_intraday)
        return SSEResult(verdict, levels, _build_evidence(sse_frame, levels), tuple(warnings), tuple(errors))
    except Exception as exc:
        return SSEResult("분석 중단", _empty_levels(), (), (), (f"SSE 계산 실패: {type(exc).__name__}: {exc}",))


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
    risk = entry - stop if _is_finite(entry) and _is_finite(stop) else np.nan
    rr1 = (target1 - entry) / risk if _is_finite(target1) and _is_finite(entry) and _is_finite(risk) and risk > 0 else np.nan
    rr2 = (target2 - entry) / risk if _is_finite(target2) and _is_finite(entry) and _is_finite(risk) and risk > 0 else np.nan
    return SSELevels(
        base=base,
        upper=_finite(row.get("SSE_UPPER")),
        lower=_finite(row.get("SSE_LOWER")),
        pressure=_finite(row.get("SSE_PRESSURE")),
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
    floor = max(v for v in [_finite(current_price), entry] if _is_finite(v)) if any(_is_finite(v) for v in [_finite(current_price), entry]) else np.nan
    candidates = [
        _finite(row.get("SSE_TARGET1_RAW")),
        _finite(row.get("SSE_RECENT5_HIGH")),
        _finite(row.get("SSE_RECENT20_HIGH")),
        _finite(row.get("BB상단")),
        _finite(row.get("SSE_MA120")),
        _finite(row.get("SSE_MA240")),
    ]
    return _nearest_conservative_above(candidates, floor, _finite(row.get("SSE_TARGET1_RAW")))


def _select_target2(frame: pd.DataFrame, current_price: float | None, target1: float) -> float:
    row = frame.iloc[-1]
    floor_values = [_finite(current_price), target1]
    floor = max(v for v in floor_values if _is_finite(v)) if any(_is_finite(v) for v in floor_values) else np.nan
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
    return _nearest_conservative_above(candidates, floor, _finite(row.get("SSE_TARGET2_RAW")))


def _nearest_conservative_above(candidates: list[float], floor: float, fallback: float) -> float:
    valid = [v for v in candidates if _is_finite(v) and (_is_finite(floor) is False or v > floor)]
    if valid:
        return min(valid)
    return fallback


def _build_evidence(frame: pd.DataFrame, levels: SSELevels) -> tuple[SSEEvidence, ...]:
    if frame.empty:
        return ()
    row = frame.iloc[-1]
    return (
        SSEEvidence("SSE 기준선", levels.base, "0.35*MA20 + 0.20*MA60 + 0.20*MID26 + 0.15*MID52 + 0.10*MID9", "종가 평균 원리와 고저 중간값 원리를 재조합"),
        SSEEvidence("SSE 통합 변동성", _finite(row.get("SSE_VOLATILITY")), "0.50*STD20 + 0.25*abs(MID26-MID52) + 0.25*abs(MA20-MA60)", "표준편차, 고저 균형 간격, 평균선 간격을 통합"),
        SSEEvidence("SSE 상단선", levels.upper, "SSE_BASE + 1.8*SSE_VOLATILITY", "통합 변동성 기준 상단 범위"),
        SSEEvidence("SSE 하단선", levels.lower, "SSE_BASE - 1.8*SSE_VOLATILITY", "통합 변동성 기준 하단 범위"),
        SSEEvidence("SSE 압력값", levels.pressure, "(Close - SSE_BASE) / SSE_VOLATILITY", "현재 가격의 기준선 대비 압력"),
        SSEEvidence("예상 진입가", levels.entry, "SSE_BASE + 0.25*SSE_VOLATILITY", "기준선 회복 후 조건부 진입 기준"),
        SSEEvidence("예상 손절가", levels.stop, "min(SSE_BASE - 0.75*SSE_VOLATILITY, MID26, MID52, 최근20일저점)", "보수적 방어 기준"),
        SSEEvidence("1차 익절가", levels.target1, "SSE_TARGET1 후보군 중 현재가/진입가 위의 보수적 저항", "가까운 상단 저항 우선"),
        SSEEvidence("2차 익절가", levels.target2, "SSE_TARGET2 후보군 중 1차 목표 위의 보수적 저항", "2차 목표 구조 검증"),
        SSEEvidence("추격 금지선", levels.no_chase, "SSE_BASE + 1.50*SSE_VOLATILITY", "과열 추격 금지 기준"),
    )


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
    required = ["Open", "High", "Low", "Close", "Volume", "TradeValue"]
    missing = [column for column in required if column not in frame.columns]
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
