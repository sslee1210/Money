from __future__ import annotations

FINALIZED_STOCK_AGENT_VERSION = "2026-06-17-final"

import argparse
import contextlib
import csv
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parent
_reports_dir_env = os.getenv("MONEY_REPORTS_DIR")
REPORTS_DIR = Path(_reports_dir_env) if _reports_dir_env else ROOT / "reports"
if not REPORTS_DIR.is_absolute():
    REPORTS_DIR = ROOT / REPORTS_DIR


@contextlib.contextmanager
def suppress_external_output():
    """Silence noisy third-party libraries that write below Python stdout."""

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                yield
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


@dataclass
class SourceFrame:
    name: str
    data: pd.DataFrame
    note: str = ""


@dataclass
class TradeState:
    price_position_state: str
    trend_state: str
    volume_state: str
    candle_state: str
    momentum_state: str
    macd_state: str
    rsi_state: str
    risk_reward_state: str
    price_data_state: str
    volume_data_state: str
    supply_state: str
    cross_validation_state: str
    final_action_state: str
    new_buyer_action: str
    holder_action: str
    add_buyer_action: str
    stop_loss_action: str
    blocking_errors: list[str]
    warnings: list[str]

    @property
    def qa_blocking_errors(self) -> list[str]:
        return self.blocking_errors

    @property
    def qa_warnings(self) -> list[str]:
        return self.warnings


PRICE_POSITION_STATES = {
    "BELOW_PULLBACK",
    "IN_PULLBACK",
    "ABOVE_PULLBACK_BELOW_RECOVERY",
    "RECOVERY_TEST",
    "ABOVE_RECOVERY_BELOW_BREAKOUT",
    "BREAKOUT_INTRADAY_UNCONFIRMED",
    "BREAKOUT_CONFIRMED",
    "NEAR_TARGET",
    "BELOW_DEFENSE",
}

VOLUME_STATES = {
    "LOW_VOLUME",
    "NORMAL_VOLUME",
    "STRONG_VOLUME_UP",
    "STRONG_VOLUME_DOWN",
    "HIGH_VOLUME_BEARISH_REVERSAL",
    "VOLUME_SPIKE_WARNING",
}

MACD_STATES = {
    "MACD_POSITIVE_MOMENTUM",
    "MACD_NEGATIVE_RECOVERY",
    "MACD_WEAKENING",
    "MACD_BEARISH",
    "MACD_INDETERMINATE",
}

RSI_STATES = {
    "RSI_OVERHEATED",
    "RSI_STRONG",
    "RSI_POSITIVE",
    "RSI_NEUTRAL_RECOVERY",
    "RSI_WEAK_REBOUND",
    "RSI_BEARISH",
}

RISK_REWARD_STATES = {
    "RR_GOOD",
    "RR_ACCEPTABLE_INTRADAY_ONLY",
    "RR_WEAK",
    "RR_BAD",
    "RR_STRATEGY_INVALID",
}

DATA_STATES = {
    "DATA_OK",
    "DATA_PARTIAL_SUPPLY_MISSING",
    "DATA_STALE_SECONDARY_SOURCE",
    "DATA_PRICE_MISMATCH",
    "DATA_VOLUME_MISMATCH",
    "DATA_INVALID",
}

PRICE_DATA_STATES = {
    "PRICE_DATA_OK",
    "PRICE_DATA_STALE_SECONDARY",
    "PRICE_DATA_MISMATCH",
    "PRICE_DATA_INVALID",
}

VOLUME_DATA_STATES = {
    "VOLUME_DATA_OK",
    "VOLUME_DATA_STALE_SECONDARY",
    "VOLUME_DATA_MISMATCH",
    "VOLUME_DATA_INVALID",
}

SUPPLY_STATES = {
    "SUPPLY_OK",
    "SUPPLY_PARTIAL",
    "SUPPLY_MISSING",
}

CROSS_VALIDATION_STATES = {
    "CROSS_VALIDATION_OK",
    "CROSS_VALIDATION_PARTIAL",
    "CROSS_VALIDATION_STALE_SECONDARY",
    "CROSS_VALIDATION_MISMATCH",
    "CROSS_VALIDATION_INVALID",
}

FINAL_ACTION_STATES = {
    "NO_BUY_DATA_INVALID",
    "NO_BUY_BELOW_RECOVERY",
    "NO_BUY_BAD_RR",
    "NO_BUY_STRATEGY_INVALID",
    "NO_BUY_OVERHEATED_BAD_RR",
    "NO_BUY_HIGH_VOLUME_BEARISH",
    "WAIT_PULLBACK_SUPPORT",
    "WAIT_RECOVERY_CLOSE",
    "WATCH_INTRADAY_BREAKOUT",
    "HOLD_AND_TRAIL",
    "PARTIAL_PROFIT_NEAR_TARGET",
    "DEFENSE_REQUIRED",
}


def sanitize_filename(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", text).strip()


def ymd(d: date | datetime | pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%Y%m%d")


def iso(d: date | datetime | pd.Timestamp) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def today_kst() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Seoul"))
    except Exception:
        return datetime.now()


def latest_completed_candidate(now: datetime) -> date:
    current = now.date()
    if now.time() < time(15, 45):
        return current - timedelta(days=1)
    return current


def krx_tick(price: float) -> int:
    if price < 2000:
        return 1
    if price < 5000:
        return 5
    if price < 20000:
        return 10
    if price < 50000:
        return 50
    if price < 200000:
        return 100
    if price < 500000:
        return 500
    return 1000


def get_tick_unit(price: float) -> int:
    return krx_tick(float(price))


def round_to_tick(price: float, direction: str = "nearest") -> int:
    if not np.isfinite(price) or price <= 0:
        return 0
    tick = krx_tick(price)
    if direction == "up":
        return int(math.ceil(price / tick) * tick)
    if direction == "down":
        return int(math.floor(price / tick) * tick)
    return int(round(price / tick) * tick)


def separate_buy_high_from_breakout(buy_high: float, breakout_line: float, min_ticks: int = 3) -> int:
    """Keep support/re-entry ranges clearly below the daily breakout line."""
    if not np.isfinite(buy_high) or not np.isfinite(breakout_line) or breakout_line <= 0:
        return round_to_tick(buy_high, "nearest")
    tick = get_tick_unit(breakout_line)
    if abs(buy_high - breakout_line) <= tick or buy_high >= breakout_line:
        return round_to_tick(breakout_line - tick * min_ticks, "down")
    return round_to_tick(buy_high, "nearest")


def one_line(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def tick_aligned(price: float | int | None) -> bool:
    value = _finite_or_none(price)
    if value is None or value <= 0:
        return True
    return int(round(value)) == round_to_tick(value)


def classify_price_context(
    current_price: float,
    buy_low: float,
    buy_high: float,
    rebreak_line: float,
    breakout_line: float,
    target1: float,
    defense_line: float,
) -> dict[str, Any]:
    current = _finite_or_none(current_price)
    low = _finite_or_none(buy_low)
    high = _finite_or_none(buy_high)
    rebreak = _finite_or_none(rebreak_line)
    breakout = _finite_or_none(breakout_line)
    target = _finite_or_none(target1)
    defense = _finite_or_none(defense_line)
    tick = get_tick_unit(current or breakout or rebreak or high or low or 1)
    result: dict[str, Any] = {
        "tick": tick,
        "is_in_buy_zone": False,
        "is_above_buy_zone": False,
        "is_below_buy_zone": False,
        "is_above_rebreak": False,
        "is_at_rebreak": False,
        "is_below_rebreak": False,
        "is_above_breakout": False,
        "is_at_breakout": False,
        "is_below_breakout": False,
        "is_near_target1": False,
        "is_near_defense": False,
        "buy_zone_overlaps_breakout": False,
    }
    if current is None:
        return result
    if low is not None and high is not None:
        result["is_in_buy_zone"] = low <= current <= high
        result["is_above_buy_zone"] = current > high
        result["is_below_buy_zone"] = current < low
    if rebreak is not None:
        result["is_at_rebreak"] = abs(current - rebreak) <= tick
        result["is_above_rebreak"] = current > rebreak and not result["is_at_rebreak"]
        result["is_below_rebreak"] = current < rebreak and not result["is_at_rebreak"]
    if breakout is not None:
        result["is_at_breakout"] = abs(current - breakout) <= tick
        result["is_above_breakout"] = current > breakout and not result["is_at_breakout"]
        result["is_below_breakout"] = current < breakout and not result["is_at_breakout"]
    if target is not None and target > 0:
        result["is_near_target1"] = abs(target - current) / current <= 0.03
    if defense is not None and defense > 0:
        result["is_near_defense"] = current > defense and abs(current - defense) / current <= 0.03
    if high is not None and breakout is not None:
        result["buy_zone_overlaps_breakout"] = abs(high - breakout) <= tick
    return result


def buy_zone_sentence(price_context: dict[str, Any]) -> str:
    if price_context.get("is_in_buy_zone"):
        return "현재가는 눌림목 지지가 구간 안에 있으므로 지지 확인 후 분할매수를 검토합니다."
    if price_context.get("is_above_buy_zone"):
        return "현재가는 눌림목 지지가보다 위에 있어 추격매수보다 눌림 재형성을 기다립니다."
    if price_context.get("is_below_buy_zone"):
        return "현재가는 회복 확인가 아래에 있어 반등 회복 확인 전 신규매수는 제한합니다."
    return "눌림목 지지가는 지지 확인 구간으로만 봅니다."


def rebreak_sentence(price_context: dict[str, Any]) -> str:
    if price_context.get("is_at_rebreak"):
        return "현재가 부근 위 유지 시 단기 탄력 유지, 이탈 시 장중 돌파 실패 가능성 확인이 필요합니다."
    if price_context.get("is_above_rebreak"):
        return "현재가는 단기 재돌파선 위에 있어 단기 탄력 유지 여부를 확인합니다."
    if price_context.get("is_below_rebreak"):
        return "현재가는 단기 재돌파선 아래에 있어 재돌파 확인 전 추격매수는 제한합니다."
    return "단기 재돌파선은 분봉상 탄력 확인 신호로만 봅니다."


def breakout_sentence(price_context: dict[str, Any]) -> str:
    if price_context.get("is_above_breakout") or price_context.get("is_at_breakout"):
        return "장중 현재가는 일봉 돌파 확인가 위에서 거래되고 있어 장중 돌파 시도는 긍정적이나, 오늘 종가 유지 확인이 필요합니다."
    if price_context.get("is_below_breakout"):
        return "현재가는 돌파 확인가 아래에 있어 돌파 매수 조건이 아직 충족되지 않았습니다."
    return "돌파 확인가는 종가와 거래량으로 확인합니다."


def target_sentence(price_context: dict[str, Any]) -> str:
    if price_context.get("is_near_target1"):
        return "현재가는 1차 목표에 가까워 신규매수보다 보유자 일부 익절 관찰이 우선입니다."
    return "목표가 여유와 별개로 진입은 지지 또는 돌파 조건 확인 뒤에만 검토합니다."


def buy_zone_action(price_context: dict[str, Any]) -> str:
    if price_context.get("is_in_buy_zone"):
        return "지지 확인 후 분할매수"
    if price_context.get("is_above_buy_zone"):
        return "추격보다 눌림 재형성 대기"
    if price_context.get("is_below_buy_zone"):
        return "반등 회복 확인 전 신규매수 제한"
    return "지지 확인 후 분할매수"


def recovery_confirmation_level(
    current_price: float | None,
    support_high: float | None,
    ma20: float | None,
    bb_mid: float | None,
    breakout_line: float | None,
    *,
    allow_inside_support: bool = False,
) -> tuple[str, float | None]:
    current = _finite_or_none(current_price)
    high = _finite_or_none(support_high)
    breakout = _finite_or_none(breakout_line)
    if current is None or breakout is None or current >= breakout:
        return "해당 없음", None
    if high is not None and current <= high and not allow_inside_support:
        return "해당 없음", None

    candidates: list[float] = []
    for value in [ma20, bb_mid, breakout]:
        v = _finite_or_none(value)
        if v is not None and v > current:
            candidates.append(float(round_to_tick(v, "nearest")))
    if not candidates:
        candidates.append(float(round_to_tick(breakout, "nearest")))

    candidates = sorted(set(candidates), key=lambda v: abs(v - current))
    primary = candidates[0]
    nearby = sorted(
        value
        for value in candidates
        if abs(value - primary) <= max(get_tick_unit(primary), primary * 0.003)
    )
    low = min(nearby)
    high = max(nearby)
    return format_price_range(low, high, "단일 회복선"), low


def display_rebreak_line(
    rebreak_line: float | None,
    target1: float | None,
    target2: float | None = None,
) -> tuple[str, str, bool]:
    rebreak = _finite_or_none(rebreak_line)
    target = _finite_or_none(target1)
    second_target = _finite_or_none(target2)
    if rebreak is None:
        return "해당 없음", "단기 재돌파선 데이터 부족", False
    if second_target is not None and rebreak > second_target:
        return "이전 고점/강한 저항으로 재분류", "단기 재돌파선에서 제외하고 이전 고점/강한 저항으로만 관리", True
    if target is not None and rebreak > target:
        if second_target is not None and rebreak <= second_target * 1.03:
            return f"{money(rebreak)} - 강한 저항/2차 목표 전 확인선", "신규매수 기준이 아니라 보유자 익절/비중관리 기준으로만 사용", True
        return f"{money(rebreak)} - 강한 저항/목표권 확인선", "신규매수 기준이 아니라 보유자 익절/비중관리 기준으로만 사용", True
    if target is not None and abs(round_to_tick(rebreak) - round_to_tick(target)) <= get_tick_unit(rebreak):
        return "1차 목표/강한 저항과 중복되어 별도 표시 생략", "1차 목표/강한 저항으로만 관리", True
    return money(rebreak), rebreak_action(classify_price_context(rebreak, rebreak, rebreak, rebreak, rebreak, target or rebreak, rebreak)), False


def rebreak_display_label(rebreak_line: float | None, target1: float | None, target2: float | None = None) -> str:
    rebreak = _finite_or_none(rebreak_line)
    target = _finite_or_none(target1)
    second_target = _finite_or_none(target2)
    if rebreak is None:
        return "단기 재돌파선"
    if target is not None and rebreak > target:
        if second_target is not None and rebreak <= second_target * 1.03:
            return "강한 저항/2차 목표 전 확인선"
        return "강한 저항/목표권 확인선"
    return "단기 재돌파선"


def rebreak_action(price_context: dict[str, Any]) -> str:
    if price_context.get("is_at_rebreak") or price_context.get("is_above_rebreak"):
        return "위 유지 시 단기 탄력 확인"
    return "재돌파 확인 전 추격매수 제한"


def breakout_action(price_context: dict[str, Any]) -> str:
    if price_context.get("is_at_breakout") or price_context.get("is_above_breakout"):
        return "종가 유지와 거래량 증가 확인"
    return "종가 안착과 거래량 증가 시 돌파매수 검토"


def strategy_labels_by_price(
    current_price: float,
    buy_low: float,
    buy_high: float,
    rebreak_line: float,
    breakout_line: float,
    intraday_defense_line: float,
    swing_defense_line: float,
    data_reliability: str = "중간",
) -> dict[str, str]:
    current = _finite_or_none(current_price)
    low = _finite_or_none(buy_low)
    high = _finite_or_none(buy_high)
    breakout = _finite_or_none(breakout_line)
    intraday_defense = _finite_or_none(intraday_defense_line)
    if current is None or low is None or high is None or breakout is None or intraday_defense is None:
        return {"primary_strategy": "데이터 확인 대기", "final": "데이터 부족으로 분석 제한"}
    if data_reliability == "낮음":
        return {"primary_strategy": "데이터 확인 대기", "final": "데이터 부족으로 분석 제한"}
    if current < intraday_defense:
        return {"primary_strategy": "신규매수 금지, 장중 방어 우선", "final": "방어 확인 전 신규매수 금지"}
    if intraday_defense <= current < low:
        return {"primary_strategy": "회복 확인 대기", "final": "회복 확인가 도달 전 신규매수 금지"}
    if low <= current <= high:
        return {"primary_strategy": "눌림목 지지 확인", "final": "지지 확인 시 분할매수 검토"}
    if high < current < breakout:
        return {"primary_strategy": "추격 금지, 눌림 재형성 대기", "final": "돌파 전 추격매수 제한"}
    return {"primary_strategy": "종가 유지와 거래량 확인", "final": "돌파 유지 확인"}


def assess_intraday_overheated_breakout(
    current_price: float | None,
    daily_breakout_line: float | None,
    weighted_volume_ratio: float | None,
    rsi: float | None,
    intraday_rr: float | None,
    entry_rr: float | None,
    swing_rr: float | None,
) -> dict[str, Any]:
    current = _finite_or_none(current_price)
    breakout = _finite_or_none(daily_breakout_line)
    volume_ratio = _finite_or_none(weighted_volume_ratio)
    rsi_value = _finite_or_none(rsi)
    intraday_value = _finite_or_none(intraday_rr)
    entry_value = _finite_or_none(entry_rr)
    swing_value = _finite_or_none(swing_rr)
    above_breakout = current is not None and breakout is not None and current > breakout
    volume_ok = volume_ratio is not None and volume_ratio >= 1.2
    overheated = rsi_value is not None and rsi_value >= 70
    poor_intraday = intraday_value is not None and intraday_value < 1.2
    poor_entry = entry_value is not None and entry_value < 1.0
    poor_swing = swing_value is not None and swing_value < 1.0
    applies = above_breakout and volume_ok and overheated and (poor_entry or poor_swing)
    warnings: list[str] = []
    if poor_intraday:
        warnings.append("장중 신규 진입 매력 낮음")
    if poor_entry:
        warnings.append("돌파 추격매수 부적합")
    if poor_swing:
        warnings.append("스윙 신규매수 부적합")
    if overheated and any(v for v in [poor_intraday, poor_entry, poor_swing]):
        warnings.append("과열 추격 금지")
    final = "과열 추격 금지·손익비 부족으로 신규 추격매수 부적합" if applies else ""
    primary = "과열·손익비 부족으로 신규 추격매수 부적합" if applies else ""
    template = (
        "장중 돌파와 거래량은 긍정적이나 RSI 과열과 손익비 부족으로 신규 추격매수는 부적합합니다. "
        "보유자는 종가 유지 여부를 확인하고, 신규자는 눌림 또는 종가 확정 후 재판단합니다."
        if applies
        else ""
    )
    return {
        "applies": applies,
        "above_breakout": above_breakout,
        "volume_ok": volume_ok,
        "overheated": overheated,
        "poor_intraday_rr": poor_intraday,
        "poor_entry_rr": poor_entry,
        "poor_swing_rr": poor_swing,
        "warnings": list(dict.fromkeys(warnings)),
        "primary_strategy": primary,
        "final": final,
        "template": template,
        "no_buy_reason": "RSI 과열과 손익비 부족으로 신규 추격매수 부적합" if applies else "",
    }


def macd_grade_from_values(
    macd: float | None,
    signal: float | None,
    hist: float | None,
    current_price: float | None,
    breakout_line: float | None,
) -> str:
    macd_value = _finite_or_none(macd)
    signal_value = _finite_or_none(signal)
    hist_value = _finite_or_none(hist)
    current = _finite_or_none(current_price)
    breakout = _finite_or_none(breakout_line)
    if macd_value is None or signal_value is None or hist_value is None:
        return "데이터 부족"
    above_breakout = current is not None and breakout is not None and current >= breakout
    if macd_value > signal_value and hist_value > 0 and above_breakout:
        return "좋음"
    if macd_value > signal_value and hist_value > 0:
        return "혼조/개선 중"
    if macd_value > signal_value and hist_value <= 0:
        return "중립"
    if macd_value <= signal_value and hist_value > 0:
        return "중립/개선 시도"
    return "나쁨"


def rsi_grade_from_value(rsi: float | None) -> str:
    value = _finite_or_none(rsi)
    if value is None:
        return "데이터 부족"
    if value >= 70:
        return "과열"
    if value >= 60:
        return "강함"
    if value >= 55:
        return "양호"
    if value >= 50:
        return "중립 회복"
    if value >= 40:
        return "약한 반등"
    return "약세"


def reliability_breakdown(
    price_label: str,
    volume_label: str,
    supply_status: str,
    intraday_reliability: str,
    interpretation_complete: bool = True,
    validation_note: str = "",
) -> dict[str, str]:
    def map_label(label: str) -> str:
        text = str(label or "")
        if any(word in text for word in ["실패", "낮음"]):
            return "낮음"
        if any(word in text for word in ["경고", "중간", "부분"]):
            return "중간"
        if any(word in text for word in ["통과", "높음"]):
            return "높음"
        return "중간"

    supply_text = str(supply_status or "")
    supply_limited = any(word in supply_text for word in ["부족", "실패", "보류"])
    supply_reliability = "낮음" if supply_limited else "중간"
    interpretation = "중간" if supply_limited or not interpretation_complete else "높음"
    completeness = "중간" if any(word in str(validation_note) for word in ["지연", "stale", "제외", "불일치"]) else "높음"
    return {
        "가격 신뢰도": map_label(price_label),
        "거래량 신뢰도": map_label(volume_label),
        "지표 신뢰도": "높음" if interpretation_complete else "중간",
        "교차검증 완전성": completeness,
        "수급 신뢰도": supply_reliability,
        "장중 가격 신뢰도": map_label(intraday_reliability),
        "해석 완전성": interpretation,
    }


def monthly_chart_comment(monthly_cloud: str) -> str:
    if "데이터 부족" in str(monthly_cloud):
        return "월봉 일목균형표는 계산 표본 부족으로 이번 판단에서 제외합니다. 장기 판단은 주봉과 120/240일선 중심으로만 보조 참고합니다."
    return f"{monthly_cloud} 기준으로 장기 방향을 봅니다."


def pct(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float("nan")
    return (a - b) / b * 100


def downside_risk_pct(basis: float, defense: float) -> float:
    if not np.isfinite(basis) or not np.isfinite(defense) or basis == 0:
        return float("nan")
    return (basis - defense) / basis * 100


def money(v: float | int | None) -> str:
    if v is None or not np.isfinite(float(v)):
        return "데이터 부족"
    return f"{int(round(float(v))):,}원"


def format_price_range(
    low: float | int | None,
    high: float | int | None,
    equal_label: str = "단일 가격",
) -> str:
    if low is None or high is None or not np.isfinite(float(low)) or not np.isfinite(float(high)):
        return "데이터 부족"
    low_i = int(round(float(low)))
    high_i = int(round(float(high)))
    if low_i == high_i:
        return f"{money(low_i)} {equal_label}"
    return f"{money(low_i)}~{money(high_i)}"


def shares(v: float | int | None) -> str:
    if v is None or not np.isfinite(float(v)):
        return "데이터 부족"
    return f"{int(round(float(v))):,}주"


def fpct(v: float | None) -> str:
    if v is None or not np.isfinite(float(v)):
        return "데이터 부족"
    return f"{float(v):.2f}%"


def fratio(v: float | None) -> str:
    if v is None or not np.isfinite(float(v)):
        return "데이터 부족"
    return f"{float(v):.2f}배"


SECTOR_KEYWORDS = {
    "033100": "전력기기/변압기",
    "009150": "전자부품/MLCC",
    "267260": "전력기기",
    "010120": "전력기기/자동화",
    "298040": "전력기기/중공업",
    "403870": "반도체 장비/공정 장비",
    "005930": "반도체/IT 대형주",
    "000660": "반도체 메모리",
    "010140": "조선/중공업",
    "034020": "원전/에너지",
    "196170": "바이오",
    "247540": "2차전지 소재",
}


def infer_sector_label(code: str, stock_name: str) -> str:
    normalized_code = str(code).zfill(6)
    if normalized_code in SECTOR_KEYWORDS:
        return SECTOR_KEYWORDS[normalized_code]
    upper_name = str(stock_name).upper()
    if "HPSP" in upper_name:
        return "반도체 장비/공정 장비"
    if "전기" in str(stock_name) or "일렉트릭" in str(stock_name):
        return "전력기기"
    if "중공업" in str(stock_name) or "조선" in str(stock_name):
        return "조선/중공업"
    return "동종 업종"


def unique_price_levels(levels: list[float | int | None]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for raw in levels:
        value = _finite_or_none(raw)
        if value is None:
            continue
        rounded = int(round(value))
        if rounded in seen:
            continue
        seen.add(rounded)
        result.append(rounded)
    return result


def format_price_level_list(levels: list[float | int | None], fallback: str = "데이터 부족") -> str:
    unique = unique_price_levels(levels)
    if not unique:
        return fallback
    return ", ".join(money(level) for level in unique)


def rsi_comment(rsi: float | None) -> str:
    value = _finite_or_none(rsi)
    if value is None:
        return "RSI 데이터 부족으로 모멘텀 판단을 제한합니다."
    if value >= 70:
        return f"RSI {value:.2f}로 과열권에 진입해 추격매수보다 분할익절 또는 눌림 확인이 우선입니다."
    if value >= 60:
        return f"RSI {value:.2f}로 강한 모멘텀 구간이나 과열 접근 여부를 확인해야 합니다."
    if value >= 50:
        return f"RSI {value:.2f}로 50선을 회복한 상태이나, 강한 과열권은 아니므로 추가 탄력 확인이 필요합니다."
    if value >= 40:
        return f"RSI {value:.2f}로 약한 반등 시도 구간이며 50선 회복 확인이 필요합니다."
    return f"RSI {value:.2f}로 약세 구간이라 반등 확인 전 신규매수는 제한합니다."


def macd_comment(macd: float | None, signal: float | None, hist: float | None, current_price: float | None, rebreak_line: float | None, breakout_line: float | None) -> str:
    macd_value = _finite_or_none(macd)
    signal_value = _finite_or_none(signal)
    hist_value = _finite_or_none(hist)
    current = _finite_or_none(current_price)
    rebreak = _finite_or_none(rebreak_line)
    breakout = _finite_or_none(breakout_line)
    if macd_value is None or signal_value is None or hist_value is None:
        return "MACD 데이터 부족으로 모멘텀 해석을 제한합니다."
    below_lines = current is not None and (
        (rebreak is not None and current < rebreak) or (breakout is not None and current < breakout)
    )
    if macd_value > signal_value and hist_value > 0:
        if macd_value < 0:
            base = "MACD는 음수권에서 신호선 위로 회복 시도 중이고 히스토그램도 양수입니다."
        elif macd_value > 0:
            base = "MACD는 양수권에서 모멘텀 유지 중이고 신호선 위에 있으며 히스토그램도 양수입니다."
        else:
            base = "MACD는 0선 부근에서 신호선 위로 회복 시도 중이고 히스토그램도 양수입니다."
        if below_lines:
            return f"{base} 다만 현재가가 단기 재돌파선과 일봉 돌파 확인선 아래라 추격매수는 제한합니다."
        return base
    if macd_value > signal_value and hist_value <= 0:
        zone = "음수권" if macd_value < 0 else "양수권" if macd_value > 0 else "0선 부근"
        return f"MACD는 {zone}에서 신호선 위이나 히스토그램 탄력은 둔화 중입니다."
    if macd_value <= signal_value and hist_value > 0:
        zone = "음수권" if macd_value < 0 else "양수권" if macd_value > 0 else "0선 부근"
        return f"MACD는 {zone}에 있고 히스토그램은 개선 중이나 아직 신호선을 확실히 넘지 못했습니다."
    zone = "음수권" if macd_value < 0 else "양수권" if macd_value > 0 else "0선 부근"
    return f"MACD가 {zone}에서 신호선 아래이고 히스토그램도 약해 추세 확인이 필요합니다."


def assess_volume_candle(open_price: float | None, high_price: float | None, low_price: float | None, close_price: float | None, volume_ratio: float | None) -> dict[str, Any]:
    open_v = _finite_or_none(open_price)
    high_v = _finite_or_none(high_price)
    low_v = _finite_or_none(low_price)
    close_v = _finite_or_none(close_price)
    vol = _finite_or_none(volume_ratio)
    result = {
        "status": "보통",
        "comment": "거래량은 가격 방향과 함께 확인합니다.",
        "bearish_high_volume": False,
        "strong_distribution": False,
        "close_drawdown_from_high": None,
    }
    if vol is None:
        result["status"] = "데이터 부족"
        result["comment"] = "거래량 데이터 부족으로 거래량 판단을 제한합니다."
        return result
    if open_v is None or close_v is None:
        result["status"] = "좋음" if vol >= 1.2 else "보통"
        result["comment"] = f"20일 평균 대비 {fratio(vol)}이며 캔들 방향 확인이 필요합니다."
        return result

    drawdown = None
    if high_v is not None and low_v is not None and high_v > low_v and close_v <= high_v:
        drawdown = (high_v - close_v) / (high_v - low_v)
        result["close_drawdown_from_high"] = drawdown
    is_bearish = close_v < open_v
    if vol >= 1.5 and not is_bearish:
        result["status"] = "거래량 동반 상승"
        result["comment"] = f"20일 평균 대비 {fratio(vol)}이며 양봉으로 마감해 거래량 동반 상승으로 봅니다."
    elif vol >= 1.5 and is_bearish:
        result["status"] = "고거래량 음봉/매물 출회 경고"
        result["bearish_high_volume"] = True
        if vol >= 2.0 and drawdown is not None and drawdown >= 0.5:
            result["strong_distribution"] = True
            result["comment"] = (
                f"20일 평균 대비 {fratio(vol)}이지만 음봉이고 고가 대비 {drawdown * 100:.1f}% 밀려 "
                "강한 매물 출회 경고입니다. 거래량은 돌파 매수 근거가 아니라 매물 소화 확인 대상으로 봅니다."
            )
        else:
            result["comment"] = (
                f"20일 평균 대비 {fratio(vol)}이지만 음봉으로 마감해 고거래량 음봉/매물 출회 경고입니다. "
                "거래량은 돌파 매수 근거가 아니라 매물 소화 확인 대상으로 봅니다."
            )
    else:
        result["status"] = "좋음" if vol >= 1.2 else "보통"
        result["comment"] = f"20일 평균 대비 {fratio(vol)}이며 돌파 매수는 양봉 또는 고가권 마감 여부를 함께 확인해야 합니다."
    return result


def breakout_volume_condition_comment(volume_context: dict[str, Any]) -> str:
    if volume_context.get("bearish_high_volume"):
        return "거래량은 증가했지만 장대 음봉이라 돌파 매수 근거가 아니라 매물 소화 확인이 필요합니다."
    return "돌파 매수는 종가 안착, 거래량 1.2배 이상, 양봉 또는 고가권 마감이 함께 필요합니다."


def compressed_low_rr_warning(swing_rr: float | None, entry_rr: float | None) -> str:
    swing = _finite_or_none(swing_rr)
    entry = _finite_or_none(entry_rr)
    if swing is None and entry is None:
        return ""
    if (swing is not None and swing < 0.8) or (entry is not None and entry < 0.8):
        invalid_parts: list[str] = []
        if swing is not None and swing < 1.0:
            invalid_parts.append("스윙 신규매수 부적합")
        if entry is not None and entry < 1.0:
            invalid_parts.append("돌파 추격매수 부적합")
        invalid_text = f" {'; '.join(invalid_parts)}." if invalid_parts else ""
        if entry is not None and entry < 0.5:
            return f"스윙 손익비 {fratio(swing)}, 회복/돌파 진입 손익비 {fratio(entry)}로 손익비 부족이며 스윙/돌파 신규매수 매력 낮음, 신규매수와 돌파 추격매수 모두 부적합하고 돌파 매수 전략 성립 불가입니다.{invalid_text}"
        return f"스윙 손익비 {fratio(swing)}, 회복/돌파 진입 손익비 {fratio(entry)}로 손익비 부족이며 스윙/돌파 신규매수 매력 낮음, 신규매수 금지에 가까운 구간입니다.{invalid_text}"
    if (swing is not None and swing < 1.2) or (entry is not None and entry < 1.2):
        invalid_parts: list[str] = []
        if swing is not None and swing < 1.0:
            invalid_parts.append("스윙 신규매수 부적합")
        if entry is not None and entry < 1.0:
            invalid_parts.append("돌파 추격매수 부적합")
        invalid_text = f" {'; '.join(invalid_parts)}." if invalid_parts else ""
        return f"스윙 손익비 {fratio(swing)}, 회복/돌파 진입 손익비 {fratio(entry)}로 손익비 부족이며 스윙/돌파 신규매수 매력 낮음입니다.{invalid_text}"
    return ""


def classify_price_position_state(
    current_price: float | None,
    pullback_low: float | None,
    pullback_high: float | None,
    recovery_line: float | None,
    breakout_line: float | None,
    target1: float | None,
    defense_line: float | None,
    *,
    intraday_mode: bool = False,
    close_confirmed: bool = True,
    completed_daily: bool = True,
) -> str:
    current = _finite_or_none(current_price)
    low = _finite_or_none(pullback_low)
    high = _finite_or_none(pullback_high)
    recovery = _finite_or_none(recovery_line)
    breakout = _finite_or_none(breakout_line)
    target = _finite_or_none(target1)
    defense = _finite_or_none(defense_line)
    if current is None:
        return "BELOW_PULLBACK"
    tick = get_tick_unit(current)
    if defense is not None and current <= defense:
        return "BELOW_DEFENSE"
    if target is not None and current >= target * 0.97:
        return "NEAR_TARGET"
    if breakout is not None and current >= breakout:
        if intraday_mode and not close_confirmed:
            return "BREAKOUT_INTRADAY_UNCONFIRMED"
        if completed_daily or close_confirmed:
            return "BREAKOUT_CONFIRMED"
    if recovery is not None and abs(current - recovery) <= tick:
        return "RECOVERY_TEST"
    if recovery is not None and high is not None and high < current < recovery:
        return "ABOVE_PULLBACK_BELOW_RECOVERY"
    if recovery is not None and breakout is not None and recovery < current < breakout:
        return "ABOVE_RECOVERY_BELOW_BREAKOUT"
    if low is not None and current < low:
        return "BELOW_PULLBACK"
    if low is not None and high is not None and low <= current <= high:
        return "IN_PULLBACK"
    if high is not None and current > high:
        return "ABOVE_PULLBACK_BELOW_RECOVERY"
    return "BELOW_PULLBACK"


def classify_volume_state(
    open_price: float | None,
    high_price: float | None,
    low_price: float | None,
    close_price: float | None,
    volume_ratio20: float | None,
) -> str:
    open_v = _finite_or_none(open_price)
    high_v = _finite_or_none(high_price)
    low_v = _finite_or_none(low_price)
    close_v = _finite_or_none(close_price)
    vol = _finite_or_none(volume_ratio20)
    if vol is None:
        return "NORMAL_VOLUME"
    if vol < 0.8:
        return "LOW_VOLUME"
    if open_v is None or close_v is None:
        return "NORMAL_VOLUME" if vol < 1.5 else "VOLUME_SPIKE_WARNING"
    bearish = close_v < open_v
    bullish = close_v > open_v
    if vol >= 2.0 and bearish and high_v is not None and low_v is not None and high_v > low_v:
        lower_close = close_v <= low_v + (high_v - low_v) * 0.35
        high_to_close_drop_pct = (high_v - close_v) / high_v * 100 if high_v > 0 else 0
        if lower_close:
            return "HIGH_VOLUME_BEARISH_REVERSAL"
        if high_to_close_drop_pct >= 5:
            return "VOLUME_SPIKE_WARNING"
    if vol >= 1.5 and bullish:
        return "STRONG_VOLUME_UP"
    if vol >= 1.5 and bearish:
        return "STRONG_VOLUME_DOWN"
    return "NORMAL_VOLUME"


def classify_candle_state(open_price: float | None, high_price: float | None, low_price: float | None, close_price: float | None) -> str:
    open_v = _finite_or_none(open_price)
    high_v = _finite_or_none(high_price)
    low_v = _finite_or_none(low_price)
    close_v = _finite_or_none(close_price)
    if open_v is None or close_v is None:
        return "CANDLE_INDETERMINATE"
    if close_v > open_v:
        return "BULLISH_CANDLE"
    if close_v < open_v:
        if high_v is not None and low_v is not None and high_v > low_v and close_v <= low_v + (high_v - low_v) * 0.35:
            return "BEARISH_LONG_CANDLE"
        return "BEARISH_CANDLE"
    return "DOJI_CANDLE"


def classify_macd_state(macd: float | None, signal: float | None, hist: float | None) -> str:
    macd_v = _finite_or_none(macd)
    signal_v = _finite_or_none(signal)
    hist_v = _finite_or_none(hist)
    if macd_v is None or signal_v is None or hist_v is None:
        return "MACD_INDETERMINATE"
    if macd_v > 0 and macd_v > signal_v and hist_v > 0:
        return "MACD_POSITIVE_MOMENTUM"
    if macd_v < 0 and macd_v > signal_v and hist_v > 0:
        return "MACD_NEGATIVE_RECOVERY"
    if macd_v > signal_v and hist_v <= 0:
        return "MACD_WEAKENING"
    if macd_v <= signal_v and hist_v <= 0:
        return "MACD_BEARISH"
    return "MACD_INDETERMINATE"


def classify_rsi_state(rsi: float | None) -> str:
    value = _finite_or_none(rsi)
    if value is None:
        return "RSI_WEAK_REBOUND"
    if value >= 70:
        return "RSI_OVERHEATED"
    if value >= 60:
        return "RSI_STRONG"
    if value >= 55:
        return "RSI_POSITIVE"
    if value >= 50:
        return "RSI_NEUTRAL_RECOVERY"
    if value >= 40:
        return "RSI_WEAK_REBOUND"
    return "RSI_BEARISH"


def classify_risk_reward_state(entry_rr: float | None, swing_rr: float | None, intraday_rr: float | None = None) -> str:
    entry = _finite_or_none(entry_rr)
    swing = _finite_or_none(swing_rr)
    intraday = _finite_or_none(intraday_rr)
    if (entry is not None and entry < 0.5) or (swing is not None and swing < 0.8):
        return "RR_STRATEGY_INVALID"
    if (entry is not None and entry < 1.0) or (swing is not None and swing < 1.0):
        return "RR_BAD"
    if intraday is not None and intraday >= 2.0 and swing is not None and swing < 1.2:
        return "RR_ACCEPTABLE_INTRADAY_ONLY"
    if (entry is not None and entry < 1.2) or (swing is not None and swing < 1.2):
        return "RR_WEAK"
    if (entry is not None and entry >= 1.2) or (swing is not None and swing >= 1.2):
        return "RR_GOOD"
    if entry is not None and swing is not None and entry >= 1.8 and swing >= 1.5:
        return "RR_GOOD"
    return "RR_WEAK"


def classify_data_state(
    price_label: str = "",
    volume_label: str = "",
    validation_note: str = "",
    reliability: str = "",
    supply_status: str = "",
    stop_precision: bool = False,
) -> str:
    note = str(validation_note or "")
    price = str(price_label or "")
    volume = str(volume_label or "")
    supply = str(supply_status or "")
    if stop_precision or reliability == "낮음" or "대표 가격 산정 불가" in note:
        return "DATA_INVALID"
    if "가격" in note and "불일치" in note or price == "실패":
        return "DATA_PRICE_MISMATCH"
    if "거래량" in note and "불일치" in note or volume == "실패":
        return "DATA_VOLUME_MISMATCH"
    if "보조 소스" in note and ("지연" in note or "stale" in note):
        return "DATA_STALE_SECONDARY_SOURCE"
    if any(word in supply for word in ["부족", "실패", "보류"]):
        return "DATA_PARTIAL_SUPPLY_MISSING"
    return "DATA_OK"


def classify_price_data_state(
    price_label: str = "",
    validation_note: str = "",
    reliability: str = "",
    stop_precision: bool = False,
) -> str:
    note = str(validation_note or "")
    price = str(price_label or "")
    if stop_precision or reliability == "낮음" or "대표 가격 산정 불가" in note:
        return "PRICE_DATA_INVALID"
    if ("가격" in note and "불일치" in note) or price == "실패":
        return "PRICE_DATA_MISMATCH"
    if "보조 소스" in note and ("지연" in note or "stale" in note):
        return "PRICE_DATA_STALE_SECONDARY"
    return "PRICE_DATA_OK"


def classify_volume_data_state(
    volume_label: str = "",
    validation_note: str = "",
    reliability: str = "",
    stop_precision: bool = False,
) -> str:
    note = str(validation_note or "")
    volume = str(volume_label or "")
    if stop_precision or reliability == "낮음" or "대표 가격 산정 불가" in note:
        return "VOLUME_DATA_INVALID"
    if ("거래량" in note and "불일치" in note) or volume == "실패":
        return "VOLUME_DATA_MISMATCH"
    if "보조 소스" in note and ("지연" in note or "stale" in note):
        return "VOLUME_DATA_STALE_SECONDARY"
    return "VOLUME_DATA_OK"


def classify_supply_state(supply_status: str = "") -> str:
    supply = str(supply_status or "")
    if any(word in supply for word in ["부족", "실패", "보류"]):
        return "SUPPLY_MISSING"
    if any(word in supply for word in ["부분", "제한", "참고"]):
        return "SUPPLY_PARTIAL"
    return "SUPPLY_OK"


def classify_cross_validation_state(
    price_data_state: str,
    volume_data_state: str,
    supply_state: str,
    validation_note: str = "",
) -> str:
    note = str(validation_note or "")
    if price_data_state == "PRICE_DATA_INVALID" or volume_data_state == "VOLUME_DATA_INVALID":
        return "CROSS_VALIDATION_INVALID"
    if price_data_state == "PRICE_DATA_MISMATCH" or volume_data_state == "VOLUME_DATA_MISMATCH":
        return "CROSS_VALIDATION_MISMATCH"
    if price_data_state == "PRICE_DATA_STALE_SECONDARY" or volume_data_state == "VOLUME_DATA_STALE_SECONDARY":
        return "CROSS_VALIDATION_STALE_SECONDARY"
    if supply_state != "SUPPLY_OK" or "부분" in note or "제외" in note:
        return "CROSS_VALIDATION_PARTIAL"
    return "CROSS_VALIDATION_OK"


def classify_momentum_state(macd_state: str, rsi_state: str) -> str:
    if macd_state in {"MACD_POSITIVE_MOMENTUM", "MACD_NEGATIVE_RECOVERY"} and rsi_state in {"RSI_STRONG", "RSI_POSITIVE", "RSI_NEUTRAL_RECOVERY"}:
        return "MOMENTUM_IMPROVING"
    if rsi_state == "RSI_OVERHEATED":
        return "MOMENTUM_OVERHEATED"
    if macd_state == "MACD_BEARISH" or rsi_state == "RSI_BEARISH":
        return "MOMENTUM_BEARISH"
    return "MOMENTUM_MIXED"


def determine_final_action_state(
    price_position_state: str,
    volume_state: str,
    rsi_state: str,
    risk_reward_state: str,
    data_state: str,
) -> str:
    if data_state == "DATA_INVALID":
        return "NO_BUY_DATA_INVALID"
    if price_position_state == "BELOW_DEFENSE":
        return "DEFENSE_REQUIRED"
    if risk_reward_state == "RR_STRATEGY_INVALID":
        return "NO_BUY_STRATEGY_INVALID"
    if volume_state == "HIGH_VOLUME_BEARISH_REVERSAL":
        return "NO_BUY_HIGH_VOLUME_BEARISH"
    if rsi_state == "RSI_OVERHEATED" and risk_reward_state in {"RR_BAD", "RR_STRATEGY_INVALID", "RR_WEAK"}:
        return "NO_BUY_OVERHEATED_BAD_RR"
    if data_state == "DATA_PRICE_MISMATCH":
        return "NO_BUY_DATA_INVALID"
    if risk_reward_state == "RR_BAD":
        return "NO_BUY_BAD_RR"
    if price_position_state == "BELOW_PULLBACK":
        return "NO_BUY_BELOW_RECOVERY"
    if risk_reward_state == "RR_WEAK":
        return "NO_BUY_BAD_RR"
    if price_position_state in {"BELOW_PULLBACK", "ABOVE_PULLBACK_BELOW_RECOVERY", "RECOVERY_TEST", "ABOVE_RECOVERY_BELOW_BREAKOUT"}:
        return "WAIT_RECOVERY_CLOSE" if price_position_state != "BELOW_PULLBACK" else "NO_BUY_BELOW_RECOVERY"
    if price_position_state == "IN_PULLBACK":
        return "WAIT_PULLBACK_SUPPORT"
    if price_position_state == "BREAKOUT_INTRADAY_UNCONFIRMED":
        return "WATCH_INTRADAY_BREAKOUT"
    if price_position_state == "NEAR_TARGET":
        return "PARTIAL_PROFIT_NEAR_TARGET"
    if price_position_state == "BREAKOUT_CONFIRMED":
        return "HOLD_AND_TRAIL"
    return "WAIT_RECOVERY_CLOSE"


FINAL_ACTION_TEMPLATES = {
    "NO_BUY_DATA_INVALID": {
        "final": "데이터 불일치로 정밀 판단 중단",
        "primary": "데이터 확인 대기",
        "reason": "대표 가격 또는 핵심 데이터 검증이 완료되지 않아 정상 매매 타점 확정이 불가합니다.",
        "new": "신규매수자는 데이터 검증 통과 전까지 진입하지 않습니다.",
    },
    "NO_BUY_BELOW_RECOVERY": {
        "final": "회복 확인가 회복 전 신규매수 금지",
        "primary": "회복 확인 대기",
        "reason": "현재가가 회복 확인가 아래에 있어 추격매수보다 회복 후 지지 확인이 우선입니다.",
        "new": "신규매수자는 회복 확인가 종가 안착 또는 눌림목 재지지 확인 전까지 대기합니다.",
    },
    "NO_BUY_BAD_RR": {
        "final": "신규매수 금지, 돌파 추격 전략 성립 불가",
        "primary": "손익비 부족으로 신규매수 부적합",
        "reason": "가격 조건 일부가 좋아도 손익비가 맞지 않아 신규매수는 부적합합니다.",
        "new": "신규매수자는 눌림 재형성 전까지 대기합니다.",
    },
    "NO_BUY_STRATEGY_INVALID": {
        "final": "신규매수 금지, 돌파 추격 전략 성립 불가",
        "primary": "손익비 전략 성립 불가로 신규매수 금지",
        "reason": "손익비가 전략 성립 기준을 밑돌아 신규 진입 전략으로 사용할 수 없습니다.",
        "new": "신규매수자는 회복/돌파 추격을 하지 않고 손익비가 재형성될 때까지 대기합니다.",
    },
    "NO_BUY_OVERHEATED_BAD_RR": {
        "final": "과열 추격 금지·손익비 부족으로 신규 추격매수 부적합",
        "primary": "과열·손익비 부족으로 신규 추격매수 부적합",
        "reason": "RSI 과열과 낮은 손익비가 동시에 나타나 추격매수 조건이 아닙니다.",
        "new": "신규매수자는 눌림 또는 종가 확정 후 다음 거래일 재판단합니다.",
    },
    "NO_BUY_HIGH_VOLUME_BEARISH": {
        "final": "신규매수 금지, 고거래량 음봉 매물 소화 대기",
        "primary": "고거래량 음봉 경고, 회복 전 신규매수 금지",
        "reason": "거래량은 급증했지만 장대 음봉으로 마감되어 매물 출회 경고가 우선입니다.",
        "new": "신규매수는 금지에 가깝고, 회복 확인가 종가 회복 전까지 관망합니다.",
    },
    "WAIT_PULLBACK_SUPPORT": {
        "final": "지지와 손익비 확인 시 분할매수 검토",
        "primary": "눌림목 지지 확인",
        "reason": "현재가가 눌림목 지지가 안에 있어 종가 지지와 거래량 둔화를 확인해야 합니다.",
        "new": "신규매수자는 눌림목 지지가 확인될 때만 분할 접근합니다.",
    },
    "WAIT_RECOVERY_CLOSE": {
        "final": "회복 확인 전 추격매수 제한·손익비 확인 필요",
        "primary": "회복 확인 대기",
        "reason": "현재가는 눌림목 지지가 위에 있으나 회복 확인가 아래입니다.",
        "new": "회복 확인가 종가 안착 전 추격매수는 제한합니다.",
    },
    "WATCH_INTRADAY_BREAKOUT": {
        "final": "종가 유지 전 신규매수 보류·손익비 확인 필요",
        "primary": "장중 돌파 시도, 종가 유지 확인",
        "reason": "장중 돌파 시도는 확인되지만 종가 확정 전입니다.",
        "new": "신규자는 종가 유지 또는 다음 거래일 눌림 확인 전까지 대기합니다.",
    },
    "HOLD_AND_TRAIL": {
        "final": "돌파 유지 확인",
        "primary": "종가 유지와 거래량 확인",
        "reason": "돌파 확인가 위에서 마감했으나 신규매수는 손익비와 눌림 여부를 함께 봅니다.",
        "new": "신규매수자는 추격보다 다음 눌림 확인 후 검토합니다.",
    },
    "PARTIAL_PROFIT_NEAR_TARGET": {
        "final": "보유하되 일부 익절 우선",
        "primary": "목표 접근, 보유자 익절 우선",
        "reason": "현재가가 1차 목표에 가까워 신규매수보다 보유자 익절 판단이 우선입니다.",
        "new": "신규매수자는 목표 부근 추격을 피하고 눌림을 기다립니다.",
    },
    "DEFENSE_REQUIRED": {
        "final": "방어 확인 전 신규매수 금지",
        "primary": "신규매수 금지, 방어 우선",
        "reason": "현재가가 방어선 이하라 신규매수보다 손절/비중 축소 판단이 우선입니다.",
        "new": "신규매수자는 방어선 회복 전까지 진입하지 않습니다.",
    },
}


def render_trade_state_actions(final_action_state: str, price_texts: dict[str, str] | None = None) -> dict[str, str]:
    texts = price_texts or {}
    template = FINAL_ACTION_TEMPLATES.get(final_action_state, FINAL_ACTION_TEMPLATES["WAIT_RECOVERY_CLOSE"])
    recovery = texts.get("recovery", "회복 확인가")
    pullback = texts.get("pullback", "눌림목 지지가")
    target1 = texts.get("target1", "1차 목표")
    defense = texts.get("defense", "방어선")
    return {
        "final_judgment": template["final"],
        "primary_strategy": template["primary"],
        "now_buy": "가능" if final_action_state == "WAIT_PULLBACK_SUPPORT" and texts.get("allow_now_buy") == "true" else "불가",
        "no_buy_reason": template["reason"],
        "new_buyer_action": template["new"],
        "holder_action": f"보유자는 {recovery} 회복 실패 시 추가매수 보류, {pullback} 재이탈 시 단기 비중 축소 검토, {target1} 접근 시 일부 익절, {defense} 이탈 시 방어/손절합니다.",
        "add_buyer_action": f"추가매수자는 {recovery} 종가 안착과 거래량 확인 전까지 보류합니다.",
        "stop_loss_action": f"{defense} 이탈 시 방어/손절을 우선하고, {pullback} 재이탈 시 단기 비중 축소를 검토합니다.",
    }


def trade_state_invariants(
    *,
    current_price: float | None = None,
    recovery_line: float | None = None,
    breakout_line: float | None = None,
    short_rebreak_line: float | None = None,
    target1: float | None = None,
    target2: float | None = None,
    entry_rr: float | None = None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings_list: list[str] = []
    current = _finite_or_none(current_price)
    recovery = _finite_or_none(recovery_line)
    breakout = _finite_or_none(breakout_line)
    rebreak = _finite_or_none(short_rebreak_line)
    t1 = _finite_or_none(target1)
    t2 = _finite_or_none(target2)
    rr = _finite_or_none(entry_rr)
    if rebreak is not None and t2 is not None and rebreak > t2:
        errors.append("단기 재돌파선이 2차 목표보다 높아 매매 타점으로 사용할 수 없습니다.")
    if t1 is not None and t2 is not None and t1 >= t2:
        errors.append("1차 목표가가 2차 목표가보다 높거나 같습니다.")
    if breakout is not None and t1 is not None and rr is not None and breakout > t1 and rr < 1.0:
        errors.append("돌파 확인가가 1차 목표보다 높은데 회복/돌파 진입 손익비가 1.0 미만입니다.")
    if recovery is not None and breakout is not None and same_price_level(recovery, breakout):
        warnings_list.append("회복 확인가와 일봉 돌파 확인가가 같아 회복/돌파 공통 확인가로 통합해야 합니다.")
    if current is not None and rebreak is not None and abs(current - rebreak) <= get_tick_unit(current):
        warnings_list.append("단기 재돌파선이 현재가와 같아 장중 현재가 유지 기준으로 표시해야 합니다.")
    return errors, warnings_list


def build_trade_state(
    *,
    current_price: float | None,
    pullback_low: float | None,
    pullback_high: float | None,
    recovery_line: float | None,
    breakout_line: float | None,
    target1: float | None,
    target2: float | None,
    defense_line: float | None,
    short_rebreak_line: float | None = None,
    open_price: float | None = None,
    high_price: float | None = None,
    low_price: float | None = None,
    close_price: float | None = None,
    volume_ratio20: float | None = None,
    macd: float | None = None,
    macd_signal: float | None = None,
    macd_hist: float | None = None,
    rsi: float | None = None,
    entry_rr: float | None = None,
    swing_rr: float | None = None,
    intraday_rr: float | None = None,
    price_label: str = "",
    volume_label: str = "",
    validation_note: str = "",
    reliability: str = "",
    supply_status: str = "",
    stop_precision: bool = False,
    intraday_mode: bool = False,
    close_confirmed: bool = True,
    completed_daily: bool = True,
    trend_state: str = "TREND_MIXED",
) -> TradeState:
    price_position_state = classify_price_position_state(
        current_price,
        pullback_low,
        pullback_high,
        recovery_line,
        breakout_line,
        target1,
        defense_line,
        intraday_mode=intraday_mode,
        close_confirmed=close_confirmed,
        completed_daily=completed_daily,
    )
    volume_state = classify_volume_state(open_price, high_price, low_price, close_price, volume_ratio20)
    candle_state = classify_candle_state(open_price, high_price, low_price, close_price)
    macd_state = classify_macd_state(macd, macd_signal, macd_hist)
    rsi_state = classify_rsi_state(rsi)
    momentum_state = classify_momentum_state(macd_state, rsi_state)
    risk_reward_state = classify_risk_reward_state(entry_rr, swing_rr, intraday_rr)
    data_state = classify_data_state(price_label, volume_label, validation_note, reliability, supply_status, stop_precision)
    price_data_state = classify_price_data_state(price_label, validation_note, reliability, stop_precision)
    volume_data_state = classify_volume_data_state(volume_label, validation_note, reliability, stop_precision)
    supply_state = classify_supply_state(supply_status)
    cross_validation_state = classify_cross_validation_state(price_data_state, volume_data_state, supply_state, validation_note)
    final_action_state = determine_final_action_state(price_position_state, volume_state, rsi_state, risk_reward_state, data_state)
    blocking, warnings_list = trade_state_invariants(
        current_price=current_price,
        recovery_line=recovery_line,
        breakout_line=breakout_line,
        short_rebreak_line=short_rebreak_line,
        target1=target1,
        target2=target2,
        entry_rr=entry_rr,
    )
    if data_state == "DATA_INVALID" or cross_validation_state == "CROSS_VALIDATION_INVALID":
        blocking.append("데이터 검증 무효 상태에서는 정상 보고서를 저장할 수 없습니다.")
    actions = render_trade_state_actions(
        final_action_state,
        {
            "recovery": money(recovery_line) if _finite_or_none(recovery_line) is not None else "회복 확인가",
            "pullback": format_price_range(pullback_low, pullback_high, "단일 지지선"),
            "target1": money(target1),
            "defense": money(defense_line),
        },
    )
    return TradeState(
        price_position_state=price_position_state,
        trend_state=trend_state,
        volume_state=volume_state,
        candle_state=candle_state,
        momentum_state=momentum_state,
        macd_state=macd_state,
        rsi_state=rsi_state,
        risk_reward_state=risk_reward_state,
        price_data_state=price_data_state,
        volume_data_state=volume_data_state,
        supply_state=supply_state,
        cross_validation_state=cross_validation_state,
        final_action_state=final_action_state,
        new_buyer_action=actions["new_buyer_action"],
        holder_action=actions["holder_action"],
        add_buyer_action=actions["add_buyer_action"],
        stop_loss_action=actions["stop_loss_action"],
        blocking_errors=blocking,
        warnings=warnings_list,
    )


def trade_state_to_dict(state: TradeState) -> dict[str, Any]:
    return {
        "price_position_state": state.price_position_state,
        "trend_state": state.trend_state,
        "volume_state": state.volume_state,
        "candle_state": state.candle_state,
        "momentum_state": state.momentum_state,
        "macd_state": state.macd_state,
        "rsi_state": state.rsi_state,
        "risk_reward_state": state.risk_reward_state,
        "price_data_state": state.price_data_state,
        "volume_data_state": state.volume_data_state,
        "supply_state": state.supply_state,
        "cross_validation_state": state.cross_validation_state,
        "final_action_state": state.final_action_state,
        "new_buyer_action": state.new_buyer_action,
        "holder_action": state.holder_action,
        "add_buyer_action": state.add_buyer_action,
        "stop_loss_action": state.stop_loss_action,
        "blocking_errors": list(state.blocking_errors),
        "warnings": list(state.warnings),
        "qa_blocking_errors": list(state.blocking_errors),
        "qa_warnings": list(state.warnings),
    }


def state_code_report_rows(state: TradeState) -> str:
    rows = [
        ("가격 위치", state.price_position_state),
        ("추세", state.trend_state),
        ("거래량", state.volume_state),
        ("캔들", state.candle_state),
        ("모멘텀", state.momentum_state),
        ("MACD", state.macd_state),
        ("RSI", state.rsi_state),
        ("손익비", state.risk_reward_state),
        ("가격 데이터", state.price_data_state),
        ("거래량 데이터", state.volume_data_state),
        ("수급", state.supply_state),
        ("교차검증", state.cross_validation_state),
        ("최종 행동", state.final_action_state),
    ]
    return "\n".join(["| 상태 항목 | 상태코드 |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows])


def daily_trend_state_from_values(current_price: float | None, row: Any, macd: float | None, signal: float | None, rsi: float | None, fallback_state: str = "중립") -> str:
    current = _finite_or_none(current_price)
    ma20 = last_valid(row, "MA20") if row is not None else float("nan")
    ma60 = last_valid(row, "MA60") if row is not None else float("nan")
    macd_value = _finite_or_none(macd)
    signal_value = _finite_or_none(signal)
    rsi_value = _finite_or_none(rsi)
    below_ma20 = current is not None and np.isfinite(ma20) and current < ma20
    above_ma60 = current is not None and np.isfinite(ma60) and current > ma60
    weak_momentum = (
        (macd_value is not None and signal_value is not None and macd_value < signal_value)
        and (rsi_value is not None and rsi_value < 50)
    )
    if below_ma20 and above_ma60:
        return "중기 상승 속 단기 조정"
    if below_ma20 and weak_momentum:
        return "단기 조정"
    return fallback_state


def same_price_level(a: float | int | None, b: float | int | None) -> bool:
    first = _finite_or_none(a)
    second = _finite_or_none(b)
    if first is None or second is None:
        return False
    tick = get_tick_unit(max(first, second, 1))
    return abs(round_to_tick(first) - round_to_tick(second)) <= tick


def assess_volume_momentum_conflict(
    current_price: float | None,
    pullback_high: float | None,
    recovery_line: float | None,
    weighted_volume_ratio: float | None,
    rsi: float | None,
    macd: float | None,
    signal: float | None,
) -> dict[str, Any]:
    current = _finite_or_none(current_price)
    pull_high = _finite_or_none(pullback_high)
    recovery = _finite_or_none(recovery_line)
    vol = _finite_or_none(weighted_volume_ratio)
    rsi_value = _finite_or_none(rsi)
    macd_value = _finite_or_none(macd)
    signal_value = _finite_or_none(signal)
    applies = (
        current is not None
        and pull_high is not None
        and recovery is not None
        and vol is not None
        and rsi_value is not None
        and macd_value is not None
        and signal_value is not None
        and current > pull_high
        and current < recovery
        and vol >= 1.5
        and rsi_value < 50
        and macd_value < signal_value
    )
    return {
        "applies": applies,
        "state": "거래량 동반 반등 시도이나 모멘텀 미회복" if applies else "",
        "primary_strategy": "거래량 강하지만 모멘텀 미회복, 회복 확인 전 추격 금지" if applies else "",
        "final": "모멘텀 확인 전 추격 금지" if applies else "",
        "template": (
            "거래량은 강하지만 가격은 회복 확인가 아래이고 RSI/MACD가 아직 약합니다. "
            "신규매수자는 회복 확인가 종가 안착 또는 눌림목 재지지 전까지 추격하지 않습니다."
        )
        if applies
        else "",
    }


def bollinger_comment(current_price: float | None, bb_mid: float | None, bb_upper: float | None, bb_lower: float | None, rebreak_line: float | None = None, breakout_line: float | None = None) -> str:
    current = _finite_or_none(current_price)
    mid = _finite_or_none(bb_mid)
    upper = _finite_or_none(bb_upper)
    lower = _finite_or_none(bb_lower)
    if current is None or mid is None:
        return "볼린저밴드 데이터 부족으로 위치 해석을 제한합니다."
    resistance_text = ""
    rebreak = _finite_or_none(rebreak_line)
    breakout = _finite_or_none(breakout_line)
    if rebreak is not None and breakout is not None:
        resistance_text = f" {money(rebreak)}/{money(breakout)} 저항 돌파가 먼저 필요합니다."
    if upper is not None and current > upper:
        return "현재가는 볼린저밴드 상단 위의 과열 또는 상단 돌파 구간이라 윗꼬리와 거래량 확인이 필요합니다."
    if upper is not None and mid < current <= upper:
        return f"현재가는 볼린저밴드 중심선 위에 있어 중기 반등 흐름은 유지 중이나, 상단 접근 전 주요 저항 돌파 확인이 필요합니다.{resistance_text}"
    if abs(current - mid) / mid <= 0.02:
        return "현재가는 볼린저밴드 중심선 근처 공방 구간입니다."
    if lower is not None and lower <= current < mid:
        return "현재가는 볼린저밴드 중심선 아래에 있어 반등 확인이 필요합니다."
    if lower is not None and current < lower:
        return "현재가는 볼린저밴드 하단 이탈 또는 과매도 구간이라 반등 확인 전 신규매수는 제한합니다."
    return "현재가와 볼린저밴드 위치를 추가 확인해야 합니다."


def trading_score_label(total: float | int | None) -> str:
    if total is None or not np.isfinite(float(total)):
        return "데이터 부족"
    score = float(total)
    if score >= 85:
        return "매우 우수. 단, 조건 충족 시에만 진입"
    if score >= 70:
        return "양호. 분할 진입 가능"
    if score >= 55:
        return "관망 우위. 조건 확인 필요"
    if score >= 40:
        return "매수 금지. 반등 확인 후 대기"
    return "위험. 방어 또는 관망"


def calculate_trading_scores(
    decision: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    metrics = metrics or {}
    context = context or {}
    final = str(decision.get("최종판단") or decision.get("최종 판단") or "")
    now_buy = str(decision.get("지금바로매수") or decision.get("지금 바로 매수") or "")
    reliability = str(decision.get("데이터신뢰도") or context.get("data_reliability") or "")
    rr1 = _finite_or_none(metrics.get("rr1")) or _finite_or_none(decision.get("손익비1")) or 0.0
    rr2 = _finite_or_none(metrics.get("rr2")) or _finite_or_none(decision.get("손익비2")) or 0.0
    reward1 = (
        _finite_or_none(metrics.get("reward1"))
        or _finite_or_none(decision.get("신규예상수익률1"))
        or _finite_or_none(decision.get("예상수익률1"))
        or 0.0
    )
    vol_ratio = _finite_or_none(decision.get("거래량비율")) or _finite_or_none(decision.get("장중거래량비율")) or 0.0
    rsi = _finite_or_none(decision.get("RSI"))
    macd_improving = bool(decision.get("MACD개선"))
    ma_text = str(decision.get("MA배열") or "")
    cloud_text = " ".join(str(decision.get(k) or "") for k in ["일봉구름", "주봉구름", "월봉구름"])

    trend = 8.0
    if any(word in ma_text for word in ["정배열", "상승", "강세"]):
        trend += 6
    if "구름 위" in cloud_text:
        trend += 4
    if any(word in final for word in ["방어", "데이터 불일치", "데이터 부족"]):
        trend -= 4
    trend = min(20.0, max(0.0, trend))

    momentum = 5.0
    if rsi is not None:
        if rsi >= 60:
            momentum += 6
        elif rsi >= 50:
            momentum += 4
        elif rsi >= 40:
            momentum += 2
    if macd_improving:
        momentum += 4
    momentum = min(15.0, max(0.0, momentum))

    if vol_ratio >= 1.5:
        volume = 14.0
    elif vol_ratio >= 1.2:
        volume = 12.0
    elif vol_ratio >= 1.0:
        volume = 10.0
    elif vol_ratio >= 0.5:
        volume = 7.0
    else:
        volume = 5.0
    current_price = _finite_or_none(metrics.get("current_price")) or _finite_or_none(decision.get("현재가")) or _finite_or_none(decision.get("기준가"))
    intraday_high = _finite_or_none(metrics.get("intraday_high")) or _finite_or_none(decision.get("고가"))
    breakout_line = (
        _finite_or_none(metrics.get("breakout_line"))
        or _finite_or_none(metrics.get("close_confirm_line"))
        or _finite_or_none(decision.get("일봉돌파확인선"))
        or _finite_or_none(decision.get("종가유지확인선"))
        or _finite_or_none(decision.get("돌파가격"))
    )
    rebreak_line = (
        _finite_or_none(metrics.get("rebreak_line"))
        or _finite_or_none(decision.get("단기재돌파확인선"))
        or _finite_or_none(decision.get("재돌파가격"))
    )
    close_confirmed_breakout = bool(decision.get("종가돌파확인")) or (
        now_buy == "가능" and current_price is not None and breakout_line is not None and current_price >= breakout_line
    )
    if vol_ratio >= 1.2 and not close_confirmed_breakout:
        volume = min(volume, 12.0)
    if intraday_high is not None and breakout_line is not None and current_price is not None:
        if intraday_high > breakout_line and current_price < breakout_line:
            volume = min(volume, 11.0)
    if intraday_high is not None and rebreak_line is not None and current_price is not None:
        if intraday_high > rebreak_line and current_price < rebreak_line:
            volume = min(volume, 12.0)

    def rr_to_score(rr_value: float, fallback_reward: float = 0.0) -> float:
        if rr_value >= 2.5:
            return 20.0
        if rr_value >= 2.0:
            return 18.0
        if rr_value >= 1.5:
            return 15.0
        if rr_value >= 1.0:
            return 9.0
        if fallback_reward > 0:
            return 6.0
        return 4.0

    def price_rr_score(target: float | None, entry: float | None, defense: float | None) -> float | None:
        if target is None or entry is None or defense is None or entry <= 0 or defense >= entry:
            return None
        reward = (target - entry) / entry * 100
        risk = (entry - defense) / entry * 100
        if risk <= 0 or not np.isfinite(reward) or not np.isfinite(risk):
            return None
        return rr_to_score(reward / risk, reward)

    best_rr = max(rr1, rr2)
    current_rr_score = rr_to_score(best_rr, reward1)
    if now_buy == "불가":
        current_rr_score = min(current_rr_score, 14.0)

    target_for_score = _finite_or_none(metrics.get("target1")) or _finite_or_none(decision.get("신규1차목표")) or _finite_or_none(decision.get("1차목표"))
    pullback_entry = (
        _finite_or_none(metrics.get("pullback_entry"))
        or _finite_or_none(metrics.get("pullback_low"))
        or _finite_or_none(decision.get("얕은눌림하단"))
        or _finite_or_none(decision.get("눌림하단"))
    )
    pullback_defense = (
        _finite_or_none(metrics.get("pullback_defense"))
        or _finite_or_none(metrics.get("intraday_defense_line"))
        or _finite_or_none(decision.get("장중방어선"))
        or _finite_or_none(decision.get("방어선"))
    )
    breakout_entry = (
        _finite_or_none(metrics.get("breakout_entry"))
        or _finite_or_none(metrics.get("breakout_line"))
        or _finite_or_none(decision.get("일봉돌파확인선"))
        or _finite_or_none(decision.get("돌파가격"))
    )
    breakout_defense = (
        _finite_or_none(metrics.get("breakout_defense"))
        or _finite_or_none(metrics.get("breakout_line"))
        or _finite_or_none(decision.get("일봉돌파확인선"))
        or _finite_or_none(decision.get("돌파가격"))
    )
    pullback_rr_score = price_rr_score(target_for_score, pullback_entry, pullback_defense)
    breakout_rr_score = price_rr_score(
        _finite_or_none(metrics.get("target2")) or _finite_or_none(decision.get("신규2차목표")) or target_for_score,
        breakout_entry,
        breakout_defense,
    )
    if pullback_rr_score is None:
        pullback_rr_score = current_rr_score
    if breakout_rr_score is None:
        breakout_rr_score = current_rr_score
    risk_reward = round(current_rr_score * 0.3 + pullback_rr_score * 0.4 + breakout_rr_score * 0.3, 1)

    market_score = 3.0 if context.get("market_index_invalid") else (8.0 if context.get("market_rel_strong") else 6.0)
    supply_score = 4.0 if context.get("supply_failed") else 6.0

    if now_buy == "가능":
        position = 7.0
    elif any(word in final for word in ["눌림목 우선", "대기", "재확인"]):
        position = 5.0
    else:
        position = 4.0
    if any(word in final for word in ["방어", "매수 금지", "데이터 불일치"]):
        position = 2.0
    if reliability == "낮음":
        position = min(position, 2.0)

    total = round(trend + momentum + volume + risk_reward + market_score + supply_score + position)
    return {
        "추세 점수": round(trend, 1),
        "모멘텀 점수": round(momentum, 1),
        "거래량 점수": round(volume, 1),
        "현재가 기준 손익비 점수": round(current_rr_score, 1),
        "눌림목 진입 기준 손익비 점수": round(pullback_rr_score, 1),
        "돌파 진입 기준 손익비 점수": round(breakout_rr_score, 1),
        "손익비 점수": round(risk_reward, 1),
        "시장/섹터 점수": round(market_score, 1),
        "수급 점수": round(supply_score, 1),
        "위치 점수": round(position, 1),
        "총점": float(total),
        "판정": trading_score_label(total),
    }


class ReportValidationError(Exception):
    pass


def extract_price_values(text: str) -> list[int]:
    number = r"\d{1,3}(?:,\d{3})+|\d{4,7}"
    found: list[tuple[int, int]] = []
    range_pattern = re.compile(rf"(?<![\d.])({number})\s*(?:원\s*)?[~～-]\s*({number})\s*원")
    single_pattern = re.compile(rf"(?<![\d.])({number})\s*원")

    for match in range_pattern.finditer(text):
        for group_no in (1, 2):
            raw = match.group(group_no).replace(",", "")
            try:
                value = int(raw)
            except ValueError:
                continue
            if value > 0:
                found.append((match.start(group_no), value))

    for match in single_pattern.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            found.append((match.start(1), value))

    prices: list[int] = []
    seen: set[tuple[int, int]] = set()
    for item in sorted(found, key=lambda x: x[0]):
        if item in seen:
            continue
        seen.add(item)
        prices.append(item[1])
    return prices


def _finite_or_none(value: Any) -> float | None:
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _round_price(value: Any) -> int | None:
    v = _finite_or_none(value)
    if v is None or v <= 0:
        return None
    return int(round(v))


def _add_price(prices: set[int], value: Any) -> None:
    price = _round_price(value)
    if price is not None:
        prices.add(price)


def _add_price_range(ranges: set[tuple[int, int]], low: Any, high: Any) -> None:
    lo = _round_price(low)
    hi = _round_price(high)
    if lo is None or hi is None:
        return
    a, b = sorted((lo, hi))
    ranges.add((a, b))


PRICE_KEY_HINTS = (
    "가",
    "가격",
    "목표",
    "선",
    "지지",
    "저항",
    "방어",
    "손절",
    "이탈",
    "돌파",
    "확인",
    "눌림",
    "고가",
    "저가",
    "시가",
    "종가",
    "현재",
    "기준",
    "MA",
    "BB",
    "ATR",
    "profile",
    "level",
    "support",
    "resistance",
    "high",
    "low",
    "open",
    "close",
    "basis",
    "target",
    "warning",
    "defense",
    "stop",
    "breakout",
    "pull",
    "price",
    "line",
)
NON_PRICE_KEY_HINTS = (
    "거래량",
    "비율",
    "수익률",
    "손익비",
    "위험률",
    "거래대금",
    "volume",
    "ratio",
    "reward",
    "risk",
    "rr",
)


def _is_price_key(key: Any) -> bool:
    text = str(key)
    if text.startswith("approved"):
        return True
    lowered = text.lower()
    if any(skip.lower() in lowered for skip in NON_PRICE_KEY_HINTS):
        return False
    return any(hint.lower() in lowered for hint in PRICE_KEY_HINTS)


def _collect_price_like(value: Any, prices: set[int], ranges: set[tuple[int, int]]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        for price in extract_price_values(value):
            prices.add(price)
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        _add_price(prices, value)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if _is_price_key(k):
                _collect_price_like(v, prices, ranges)
        return
    if isinstance(value, (list, tuple, set)):
        if len(value) == 2:
            first, second = list(value)
            if _finite_or_none(first) is not None and _finite_or_none(second) is not None:
                _add_price_range(ranges, first, second)
        for item in value:
            _collect_price_like(item, prices, ranges)


def _collect_approved_from_mapping(source: dict[str, Any] | None, prices: set[int], ranges: set[tuple[int, int]]) -> None:
    if not source:
        return
    for key, value in source.items():
        if key in {"approved_price_set", "approved_long_term_levels", "approved_profile_levels"}:
            _collect_price_like(value, prices, ranges)
        elif key == "approved_price_range_set":
            for item in value or []:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    _add_price_range(ranges, item[0], item[1])
        elif _is_price_key(key):
            _collect_price_like(value, prices, ranges)


def build_approved_price_sets(
    context: dict[str, Any],
    indicators: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> tuple[set[int], set[tuple[int, int]]]:
    prices: set[int] = set()
    ranges: set[tuple[int, int]] = set()
    for source in (context, indicators or {}, decision or {}, metrics or {}):
        _collect_approved_from_mapping(source, prices, ranges)

    range_pairs = [
        ("current_price_min", "current_price_max"),
        ("shallow_pull_low", "shallow_pull_high"),
        ("deep_pull_low", "deep_pull_high"),
        ("pull_low", "pull_high"),
        ("price_min", "price_max"),
    ]
    for source in (context, indicators or {}, decision or {}, metrics or {}):
        for low_key, high_key in range_pairs:
            if low_key in source and high_key in source:
                _add_price_range(ranges, source.get(low_key), source.get(high_key))

    for lo, hi in ranges:
        prices.add(lo)
        prices.add(hi)
    return prices, ranges


def _price_is_approved(price: int, prices: set[int], ranges: set[tuple[int, int]]) -> bool:
    if price in prices:
        return True
    return any(lo <= price <= hi for lo, hi in ranges)


def build_indicator_snapshot(
    daily: pd.DataFrame,
    levels: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    if daily is not None and not daily.empty:
        row = daily.iloc[-1]
        for col in [
            "Open",
            "High",
            "Low",
            "Close",
            "MA5",
            "MA10",
            "MA20",
            "MA60",
            "MA120",
            "MA240",
            "전환선",
            "기준선",
            "선행스팬1",
            "선행스팬2",
            "ATR14",
            "BB상단",
            "BB중심",
            "BB하단",
            "RSI14",
            "MACD",
            "MACD신호",
            "MACD히스토그램",
        ]:
            if col in row and _finite_or_none(row[col]) is not None:
                snapshot[col] = float(row[col])

    if levels:
        for key in [
            "basis",
            "pull_low",
            "pull_high",
            "nearest_support",
            "nearest_resistance",
            "breakout",
            "target1",
            "target2",
            "warning",
            "defense",
            "recent5_high",
            "recent5_low",
            "recent10_high",
            "recent10_low",
            "recent20_high",
            "recent20_low",
            "recent60_high",
            "recent60_low",
            "high52",
            "low52",
        ]:
            if key in levels:
                snapshot[key] = levels[key]
        snapshot["supports"] = levels.get("supports", [])
        snapshot["resistances"] = levels.get("resistances", [])
        profile = levels.get("profile")
        if isinstance(profile, pd.DataFrame) and not profile.empty and "중심" in profile.columns:
            profile_levels: list[int] = []
            for center in profile["중심"].head(24):
                if _finite_or_none(center) is not None:
                    profile_levels.append(int(round(center)))
                    profile_levels.append(round_to_tick(float(center)))
            snapshot["approved_profile_levels"] = profile_levels

    if extra:
        snapshot.update(extra)
    return snapshot


def build_report_qa_section(
    data_reliability: str = "확인 필요",
    validation_error_count: int | None = 0,
    reliability_details: dict[str, str] | None = None,
) -> str:
    if validation_error_count is None:
        qa_status = "검증 중"
    elif validation_error_count == 0:
        qa_status = "통과"
    else:
        qa_status = "실패"
    details = reliability_details or {}
    price_rel = details.get("가격 신뢰도", data_reliability)
    volume_rel = details.get("거래량 신뢰도", data_reliability)
    indicator_rel = details.get("지표 신뢰도", "확인 필요")
    cross_rel = details.get("교차검증 완전성", "확인 필요")
    supply_rel = details.get("수급 신뢰도", "확인 필요")
    interpretation_rel = details.get("해석 완전성", "확인 필요")
    return f"""## 내부 검증

내부 검증: {qa_status}
가격 신뢰도: {price_rel}
거래량 신뢰도: {volume_rel}
지표 신뢰도: {indicator_rel}
교차검증 완전성: {cross_rel}
수급 신뢰도: {supply_rel}
해석 완전성: {interpretation_rel}"""


def practical_state_from_text(*texts: Any) -> str:
    combined = " ".join(str(text) for text in texts if text is not None)
    if not combined or "데이터 부족" in combined or "사용 불가" in combined:
        return "데이터 부족"
    if any(token in combined for token in ["구름 아래", "역배열", "하락", "이탈", "약세"]):
        return "하락"
    if any(token in combined for token in ["구름 위", "정배열", "상승", "회복", "개선"]):
        return "상승"
    return "중립"


def practical_grade_from_text(*texts: Any) -> str:
    combined = " ".join(str(text) for text in texts if text is not None)
    if not combined or "데이터 부족" in combined or "사용 불가" in combined:
        return "데이터 부족"
    bad_tokens = ["부족", "실패", "이탈", "아래", "둔화", "약세", "하락", "부정"]
    good_tokens = ["양호", "개선", "회복", "상승", "위", "강함", "돌파"]
    if any(token in combined for token in bad_tokens):
        return "나쁨"
    if any(token in combined for token in good_tokens):
        return "좋음"
    return "보통"


def validate_decision_consistency(decision: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    now_buy = decision.get("지금바로매수") or decision.get("지금 바로 매수")
    final_rating = decision.get("최종판단") or decision.get("최종 판단")
    rr1 = _finite_or_none(metrics.get("rr1"))
    rr2 = _finite_or_none(metrics.get("rr2"))
    reward1 = _finite_or_none(metrics.get("reward1"))
    current_price = _finite_or_none(metrics.get("current_price"))
    target1 = _finite_or_none(metrics.get("target1"))
    breakout_line = _finite_or_none(metrics.get("breakout_line"))

    if now_buy == "가능" and (rr1 is None or rr1 < 1.5):
        errors.append("지금 바로 매수 가능 판단인데 1차 손익비가 1.5 미만입니다.")

    buyable_ratings = {"공격 매수 가능", "돌파 매수 가능", "눌림목 매수 가능", "눌림목 매수만 가능", "돌파 매수만 가능"}
    if now_buy == "불가" and final_rating in buyable_ratings:
        errors.append("지금 바로 매수 불가 판단인데 최종 판단은 매수 가능 등급입니다.")

    if final_rating in buyable_ratings and max(rr1 or 0, rr2 or 0) < 1.5:
        errors.append("매수 가능 최종 판단인데 유효 손익비가 1.5 미만입니다.")

    if now_buy == "불가" and final_rating == "눌림목 매수만 가능" and breakout_line is not None:
        errors.append("돌파 조건을 함께 제시하면서 최종 판단을 눌림목 매수만 가능으로 표시했습니다.")

    if final_rating == "돌파 매수만 가능" and now_buy == "불가" and reward1 is not None and reward1 < 3:
        errors.append("1차 목표가 너무 가까워 지금은 대기 또는 돌파 재확인 대기가 맞습니다.")

    if final_rating == "돌파 매수만 가능" and current_price is not None and breakout_line is not None and current_price >= breakout_line and reward1 is not None and reward1 < 3:
        errors.append("현재가가 돌파선 위이고 1차 목표가 가까운데 돌파 매수만 가능으로 판단했습니다.")

    if target1 is not None and current_price is not None and target1 <= current_price:
        errors.append("1차 목표가가 현재가 이하입니다.")
    return errors


def validate_intraday_lines(metrics: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    intraday_warning = _finite_or_none(metrics.get("intraday_warning_line") or decision.get("장중주의선"))
    intraday_defense = _finite_or_none(metrics.get("intraday_defense_line") or decision.get("장중방어선"))
    if intraday_warning is not None and intraday_defense is not None and intraday_warning <= intraday_defense:
        errors.append("장중 주의선은 장중 방어선보다 높아야 합니다.")
    return errors


def _extract_percent_values(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*%", text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def _extract_ratio_values(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"([-+]?\d+(?:\.\d+)?)\s*배", text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def _extract_prices_as_float(text: str) -> list[float]:
    values: list[float] = []
    for price in extract_price_values(text):
        values.append(float(price))
    return values


def _iter_markdown_rows(report_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


def _find_table_price(report_text: str, labels: tuple[str, ...]) -> float | None:
    for cells in _iter_markdown_rows(report_text):
        if len(cells) < 2:
            continue
        label = cells[0].strip()
        if label in labels:
            prices = _extract_prices_as_float(" | ".join(cells[1:]))
            if prices:
                return prices[-1] if "방어" in label or "손절" in label else prices[0]
    return None


def _report_section(report_text: str, heading: str) -> str:
    escaped = re.escape(heading)
    pattern = re.compile(rf"^##\s+(?:\d+\.\s+)?{escaped}\s*$", re.MULTILINE)
    match = pattern.search(report_text)
    if not match and ". " in heading:
        title_without_number = re.escape(heading.split(". ", 1)[1])
        pattern = re.compile(rf"^##\s+\d+\.\s+{title_without_number}\s*$", re.MULTILINE)
        match = pattern.search(report_text)
    if not match:
        return ""
    rest = report_text[match.end() :]
    next_heading = re.search(r"\n##\s+", rest)
    if next_heading:
        return rest[: next_heading.start()]
    return rest


def _close_enough(displayed: float, expected: float, tolerance: float = 0.1) -> bool:
    return np.isfinite(displayed) and np.isfinite(expected) and abs(displayed - expected) <= tolerance + 1e-9


def validate_recalculated_report_metrics(
    report_text: str,
    decision: dict[str, Any],
    metrics: dict[str, Any],
    context: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    basis = (
        _find_table_price(report_text, ("대표 기준가", "기준가"))
        or _find_table_price(report_text, ("현재가",))
        or _finite_or_none(metrics.get("current_price"))
        or _finite_or_none(decision.get("기준가"))
        or _finite_or_none(decision.get("현재가"))
        or _finite_or_none(context.get("current_price"))
    )
    if basis is None or basis <= 0:
        return ["수익률/위험률 재계산 QA 실패: 보고서 기준가를 확인할 수 없습니다."]

    returns_section = _report_section(report_text, "7. 예상 수익률과 하락 위험")
    if not returns_section:
        return errors

    risk_abs: float | None = None
    target_ratios: dict[str, float] = {}
    rows = _iter_markdown_rows(returns_section)
    for cells in rows:
        if len(cells) < 3 or cells[0] == "시나리오":
            continue
        scenario = cells[0]
        price_values = _extract_prices_as_float(cells[1])
        displayed_pcts = _extract_percent_values(cells[2])
        if not price_values or not displayed_pcts:
            continue
        price = price_values[0]
        displayed_pct = displayed_pcts[0]
        if any(token in scenario for token in ["손절", "방어"]):
            expected_signed = (price - basis) / basis * 100
            expected_risk = (basis - price) / basis * 100
            risk_abs = expected_risk
            if not _close_enough(displayed_pct, expected_signed):
                errors.append(
                    f"손절/방어 수익률 불일치: 표시 {displayed_pct:.2f}%, 재계산 {expected_signed:.2f}% "
                    f"(기준가 {basis:,.0f}원, 방어선 {price:,.0f}원)"
                )
            continue

        if any(token in scenario for token in ["목표", "익절", "저항"]):
            expected_reward = (price - basis) / basis * 100
            if not _close_enough(displayed_pct, expected_reward):
                errors.append(
                    f"{scenario} 수익률 불일치: 표시 {displayed_pct:.2f}%, 재계산 {expected_reward:.2f}% "
                    f"(기준가 {basis:,.0f}원, 목표가 {price:,.0f}원)"
                )
            key = "2차" if "2차" in scenario else "신규1차" if "신규" in scenario and "1차" in scenario else "1차"
            target_ratios[key] = expected_reward

    if risk_abs is None:
        defense = _find_table_price(report_text, ("스윙 최종 방어선", "방어선")) or _finite_or_none(decision.get("방어선"))
        if defense is not None:
            risk_abs = (basis - defense) / basis * 100
    if risk_abs is None or risk_abs <= 0:
        errors.append("하락 위험률 재계산 QA 실패: 방어선을 확인할 수 없거나 기준가보다 낮지 않습니다.")
        return errors

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if "하락 위험률" in line:
            for displayed in _extract_percent_values(line):
                if not _close_enough(displayed, risk_abs):
                    errors.append(f"하락 위험률 불일치: 표시 {displayed:.2f}%, 재계산 {risk_abs:.2f}%")
        if "기준가 대비" in line and "목표" in line:
            prices = _extract_prices_as_float(line)
            displayed_pcts = _extract_percent_values(line)
            for price, displayed in zip(prices, displayed_pcts):
                expected = (price - basis) / basis * 100
                if not _close_enough(displayed, expected):
                    errors.append(
                        f"본문 목표 수익률 불일치: 표시 {displayed:.2f}%, 재계산 {expected:.2f}% "
                        f"(기준가 {basis:,.0f}원, 목표가 {price:,.0f}원)"
                    )
        if "손익비" in line:
            ratios = _extract_ratio_values(line)
            if not ratios:
                continue
            if "2차" in line:
                reward = target_ratios.get("2차")
                label = "2차 목표 손익비"
            elif "신규" in line and "1차" in line:
                reward = target_ratios.get("신규1차") or target_ratios.get("1차")
                label = "신규매수 기준 1차 목표 손익비"
            elif "1차" in line:
                reward = target_ratios.get("1차") or target_ratios.get("신규1차")
                label = "1차 목표 손익비"
            else:
                reward = None
                label = "손익비"
            if reward is None:
                continue
            expected_ratio = reward / risk_abs if risk_abs > 0 else np.nan
            for displayed in ratios:
                if not _close_enough(displayed, expected_ratio):
                    errors.append(f"{label} 불일치: 표시 {displayed:.2f}배, 재계산 {expected_ratio:.2f}배")

    return errors


def validate_market_index_qa(report_text: str, context: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    market = context.get("market")
    if market not in {"KOSPI", "KOSDAQ"}:
        return errors
    expected_name = market
    opposite_name = "KOSDAQ" if market == "KOSPI" else "KOSPI"
    source_name = str(context.get("market_index_source", ""))
    source_symbol = str(context.get("market_index_symbol", ""))
    index_value = _finite_or_none(context.get("market_index_value"))
    invalid_source = bool(context.get("market_index_invalid"))

    if source_symbol and source_symbol not in {"KS11", "KQ11", "1001", "2001"}:
        errors.append(f"국내 시장 지수 심볼이 허용 목록이 아닙니다: {source_symbol}")
    if market == "KOSPI" and source_symbol in {"KQ11", "2001"}:
        errors.append("KOSPI 종목인데 KOSDAQ 지수를 주 비교 대상으로 사용했습니다.")
    if market == "KOSDAQ" and source_symbol in {"KS11", "1001"}:
        errors.append("KOSDAQ 종목인데 KOSPI 지수를 주 비교 대상으로 사용했습니다.")
    if invalid_source:
        errors.append(f"{expected_name} 시장 지수 데이터가 비정상 범위이거나 잘못된 심볼입니다.")
    if index_value is not None:
        lo, hi = (1000, 6000) if market == "KOSPI" else (300, 2000)
        if not (lo <= index_value <= hi):
            errors.append(f"{expected_name} 지수 값이 비정상 범위입니다: {index_value:,.2f}")
    if f"{opposite_name} 대비" in report_text and f"{expected_name} 대비" not in report_text:
        errors.append(f"{market} 종목의 주 비교 지수가 {opposite_name}로 표시되었습니다.")
    return errors


def validate_qa_section(report_text: str) -> list[str]:
    errors: list[str] = []
    if "## 보고서 QA 점검 결과" in report_text:
        errors.append("장문 QA 표가 보고서에 출력되었습니다. 내부 검증 요약만 표시해야 합니다.")
    if "## 내부 검증" not in report_text:
        errors.append("내부 검증 섹션이 누락되었습니다.")
    if "내부 검증: 통과" not in report_text:
        errors.append("내부 검증 통과 문구가 누락되었습니다.")
    for label in ["가격 신뢰도:", "거래량 신뢰도:", "지표 신뢰도:", "교차검증 완전성:", "수급 신뢰도:", "해석 완전성:"]:
        if label not in report_text:
            errors.append(f"내부 검증 신뢰도 분리 항목이 누락되었습니다: {label}")
    if re.search(r"수급 신뢰도:\s*(낮음|데이터 부족)", report_text) and re.search(r"데이터 신뢰도(?:는|:)\s*높음", report_text):
        errors.append("수급 데이터 부족 상태인데 전체 데이터 신뢰도만 높음으로 표시했습니다.")
    return errors


def validate_intraday_breakout_wording(report_text: str, decision: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    intraday_high = (
        _finite_or_none(metrics.get("intraday_high"))
        or _finite_or_none(decision.get("고가"))
        or _finite_or_none(context.get("high_price"))
    )
    close_confirm_line = (
        _finite_or_none(metrics.get("close_confirm_line"))
        or _finite_or_none(metrics.get("breakout_line"))
        or _finite_or_none(decision.get("종가유지확인선"))
    )
    if intraday_high is not None and close_confirm_line is not None and intraday_high < close_confirm_line:
        if "돌파 후 유지 실패" in report_text:
            errors.append("종가 유지 확인선을 실제로 돌파하지 않았는데 돌파 실패 문구를 사용했습니다.")
    if re.search(r"당일 고가\s+\d[\d,]*원\s+재돌파\s+또는\s+\d[\d,]*원\s+안착", report_text):
        errors.append("단기 재돌파선과 일봉 돌파 확인선을 같은 레벨로 묶어 표시했습니다.")
    return errors


def validate_single_price_ranges(report_text: str) -> list[str]:
    if re.search(r"(\d[\d,]*)\s*원?\s*~\s*\1\s*원", report_text):
        return ["같은 가격을 범위로 표시했습니다. 단일 지지선으로 표시해야 합니다."]
    return []


def validate_moving_average_wording(report_text: str, metrics: dict[str, Any], context: dict[str, Any], indicators: dict[str, Any] | None) -> list[str]:
    current_price = (
        _finite_or_none(metrics.get("current_price"))
        or _finite_or_none(context.get("current_price"))
        or _find_table_price(report_text, ("대표 기준가", "기준가", "현재가"))
    )
    indicators = indicators or {}
    ma20 = _finite_or_none(indicators.get("MA20"))
    ma60 = _finite_or_none(indicators.get("MA60"))
    if current_price is not None and ma20 is not None and ma60 is not None:
        if current_price > ma20 and current_price > ma60 and "20일/60일선 회복 확인" in report_text:
            return ["현재가가 20일·60일선 위인데 회복 확인 필요 문구가 사용되었습니다."]
    return []


def validate_nearby_profile_wording(report_text: str, metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    current_price = (
        _finite_or_none(metrics.get("current_price"))
        or _finite_or_none(context.get("current_price"))
        or _find_table_price(report_text, ("대표 기준가", "기준가", "현재가"))
    )
    if current_price is None or current_price <= 0:
        return errors
    for line in report_text.splitlines():
        if "현재가 주변" not in line or "매물대" not in line:
            continue
        for level in _extract_prices_as_float(line):
            if abs(level - current_price) / current_price > 0.25:
                errors.append("현재가 주변 매물대 표시에 현재가와 25% 이상 떨어진 가격이 포함되었습니다.")
                return errors
    return errors


def validate_yfinance_minute_status(report_text: str) -> list[str]:
    errors: list[str] = []
    yfinance_failed = any(
        "yfinance 1분봉" in line and any(phrase in line for phrase in ["수집 실패", "데이터 없음"])
        for line in report_text.splitlines()
    )
    if yfinance_failed and "yfinance 분봉: 정상" in report_text:
        errors.append("yfinance 1분봉 수집 실패 상태에서 yfinance 분봉 정상 문구가 사용되었습니다.")
    if yfinance_failed and "| 분봉 데이터 신뢰도 | 통과 |" in report_text:
        errors.append("yfinance 1분봉 수집 실패 상태에서 분봉 데이터 신뢰도를 통과로 표시했습니다.")
    return errors


def validate_target_profit_labels(report_text: str) -> list[str]:
    errors: list[str] = []
    forbidden_labels = ["1차 익절 가격", "2차 익절 가격", "1차 목표가", "2차 목표가"]
    for label in forbidden_labels:
        if label in report_text:
            errors.append(f"목표가/익절 라벨이 표준 표현과 어긋났습니다: {label}")
    return errors


def validate_intraday_defense_labels(report_text: str) -> list[str]:
    errors: list[str] = []
    if "## 1. 장중 매매 판단" not in report_text:
        return errors
    required_labels = ["장중 주의선", "장중 방어선", "스윙 최종 방어선", "전량 이탈 조건"]
    for label in required_labels:
        if label not in report_text:
            errors.append(f"장중 보고서에 필수 방어 라벨이 없습니다: {label}")
    for label in ["| 주의선 |", "| 방어선 |"]:
        if label in report_text:
            errors.append(f"장중 보고서에서 모호한 방어 라벨을 사용했습니다: {label}")
    return errors


def _extract_analysis_time(context: dict[str, Any], metrics: dict[str, Any], report_text: str) -> datetime | None:
    raw = context.get("analysis_time") or metrics.get("analysis_time")
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        for fmt in ("%Y-%m-%d %H:%M KST", "%Y-%m-%d %H:%M:%S KST", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                pass
    match = re.search(r"\|\s*(?:분석 실행 시각|현재 시각)\s*\|\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})(?::\d{2})?\s*KST\s*\|", report_text)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M")
        except ValueError:
            return None
    return None


def validate_early_volume_wording(report_text: str, metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    analysis_dt = _extract_analysis_time(context, metrics, report_text)
    if analysis_dt is None or analysis_dt.time() >= time(9, 30):
        return []
    errors: list[str] = []
    forbidden_volume_phrases = ["거래량 급증 확정", "강한 거래량 확인", "거래량 동반 확정"]
    for phrase in forbidden_volume_phrases:
        if phrase in report_text:
            errors.append(f"장초반 시간가중 거래량을 확정 신호로 표현했습니다: {phrase}")
    if "시간가중 환산 거래량" in report_text and "장초반 30분 이내 시간가중 환산 거래량은 과장될 수 있으므로 참고값으로만 봅니다." not in report_text:
        errors.append("장초반 시간가중 환산 거래량 참고값 안내 문구가 누락되었습니다.")
    return errors


def _extract_trading_score(report_text: str) -> float | None:
    section = _report_section(report_text, "트레이딩 점수")
    if not section:
        return None
    match = re.search(r"\|\s*총점\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*점\s*\|", section)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _extract_trading_score_item(report_text: str, label: str) -> float | None:
    section = _report_section(report_text, "트레이딩 점수")
    if not section:
        return None
    for cells in _iter_markdown_rows(section):
        if len(cells) < 2 or cells[0].strip() != label:
            continue
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)", cells[1])
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _extract_scenario_row(report_text: str, scenario_name: str) -> list[str] | None:
    section = _report_section(report_text, "매매 시나리오")
    for cells in _iter_markdown_rows(section):
        if cells and cells[0].strip() == scenario_name:
            return cells
    return None


def validate_pro_trader_layer(report_text: str, decision: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_sections = [
        "## 1. 최종 결론",
        "## 2. 프로 트레이더 관점",
        "## 3. 차트 분석",
        "## 4. 보조지표 판단",
        "## 5. 매매 타점",
        "## 6. 최종 한 문단",
    ]
    for section in required_sections:
        if section not in report_text:
            errors.append(f"실전 매매 판단 리포트 필수 섹션이 없습니다: {section}")

    now_buy = decision.get("지금바로매수") or decision.get("지금 바로 매수")
    if now_buy == "불가":
        if re.search(r"\|\s*지금\s*매수\s*\|\s*가능\s*\|", report_text):
            errors.append("지금 매수 불가 상태인데 최종 결론 표에 가능으로 표시되었습니다.")

    rr1 = _finite_or_none(metrics.get("rr1")) or _finite_or_none(decision.get("손익비1"))
    if rr1 is not None and rr1 < 1.5:
        if re.search(r"\|\s*지금\s*매수\s*\|\s*가능\s*\|", report_text) or "공격 매수 가능" in report_text:
            errors.append("손익비 1.5 미만인데 매수 가능으로 표현했습니다.")
    return errors


def validate_report_text(
    report_text: str,
    context: dict[str, Any],
    approved_price_set: set[int],
    approved_price_range_set: set[tuple[int, int]],
) -> list[str]:
    errors: list[str] = []
    market = context["market"]
    code = context["code"]
    suffix = context["suffix"]

    unapproved = sorted(
        {
            price
            for price in extract_price_values(report_text)
            if not _price_is_approved(price, approved_price_set, approved_price_range_set)
        }
    )
    if unapproved:
        sample = ", ".join(f"{price:,}원" for price in unapproved[:20])
        errors.append(f"가격 whitelist에 없는 원화 가격이 보고서에 포함되었습니다: {sample}")

    if market == "KOSPI" and f"{code}.KQ" in report_text:
        errors.append("KOSPI 종목인데 Yahoo suffix가 .KQ로 들어갔습니다.")
    if market == "KOSDAQ" and f"{code}.KS" in report_text:
        errors.append("KOSDAQ 종목인데 Yahoo suffix가 .KS로 들어갔습니다.")
    expected_symbol = f"{code}{suffix}"
    if f"https://finance.yahoo.com/quote/{code}." in report_text and expected_symbol not in report_text:
        errors.append("Yahoo Finance URL의 시장 suffix가 현재 종목 시장과 일치하지 않습니다.")

    supply_failed = context.get("supply_failed")
    if supply_failed is None:
        supply_failed = any(phrase in report_text for phrase in ["수급 데이터 수집 실패", "수급 데이터 부족", "수급 판단 보류"])
    if supply_failed:
        forbidden_phrases = ["매도 흐름", "동반 순매수", "동반 순매도", "수급 우호", "수급 악화"]
        for phrase in forbidden_phrases:
            if phrase in report_text:
                errors.append(f"수급 수집 실패 상태에서 수급 방향을 단정했습니다: {phrase}")

    if market in ["KOSPI", "KOSDAQ"]:
        for phrase in ["미국 주식:", "S&P 500:", "Nasdaq:", "섹터 ETF:"]:
            if phrase in report_text:
                errors.append(f"국내 주식 보고서에 미국 주식 섹션이 포함되었습니다: {phrase}")

    return errors


def validate_alert_price_section(report_text: str) -> list[str]:
    errors: list[str] = []
    if "## 눌림 설정 가격" in report_text:
        errors.append("섹션명 '눌림 설정 가격'은 사용하지 않습니다. '알림 설정 가격'으로 변경해야 합니다.")
    if "| 눌림 가격 |" in report_text:
        errors.append("표 헤더 '눌림 가격'은 사용하지 않습니다. '알림 가격'으로 변경해야 합니다.")
    return errors


def validate_pullback_below_wording(report_text: str, decision: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    current_price = (
        _finite_or_none(metrics.get("current_price"))
        or _finite_or_none(decision.get("현재가"))
        or _finite_or_none(decision.get("기준가"))
        or _finite_or_none(context.get("current_price"))
    )
    pullback_low = (
        _finite_or_none(metrics.get("pullback_low"))
        or _finite_or_none(decision.get("얕은눌림하단"))
        or _finite_or_none(decision.get("눌림하단"))
        or _finite_or_none(context.get("pullback_low"))
    )
    if current_price is None or pullback_low is None or current_price >= pullback_low:
        return []
    errors: list[str] = []
    has_recovery_wording = "회복 후 지지 확인" in report_text or "재진입" in report_text
    for phrase in ["눌림목 지지 중", "지지 확인 전까지 대기"]:
        if phrase in report_text and not has_recovery_wording:
            errors.append("현재가가 눌림목 아래인데 지지 확인 문구가 부정확합니다.")
            break
    return errors


def validate_breakout_failed_volume_score(report_text: str, decision: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    intraday_high = (
        _finite_or_none(metrics.get("intraday_high"))
        or _finite_or_none(decision.get("고가"))
        or _finite_or_none(context.get("high_price"))
    )
    current_price = (
        _finite_or_none(metrics.get("current_price"))
        or _finite_or_none(decision.get("현재가"))
        or _finite_or_none(decision.get("기준가"))
        or _finite_or_none(context.get("current_price"))
    )
    breakout_line = (
        _finite_or_none(metrics.get("breakout_line"))
        or _finite_or_none(metrics.get("close_confirm_line"))
        or _finite_or_none(decision.get("일봉돌파확인선"))
        or _finite_or_none(decision.get("종가유지확인선"))
        or _finite_or_none(decision.get("돌파가격"))
    )
    if intraday_high is None or current_price is None or breakout_line is None:
        return []
    if intraday_high > breakout_line and current_price < breakout_line:
        volume_score = _extract_trading_score_item(report_text, "거래량 점수")
        if volume_score is not None and volume_score > 12:
            return ["돌파 유지 실패 상태에서 거래량 점수가 과도하게 높습니다."]
    return []


def validate_deep_pullback_status_wording(report_text: str, decision: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    current_price = (
        _finite_or_none(metrics.get("current_price"))
        or _finite_or_none(decision.get("현재가"))
        or _finite_or_none(decision.get("기준가"))
        or _finite_or_none(context.get("current_price"))
    )
    deep_low = (
        _finite_or_none(metrics.get("deep_pull_low"))
        or _finite_or_none(decision.get("깊은눌림하단"))
        or _finite_or_none(context.get("deep_pull_low"))
    )
    deep_high = (
        _finite_or_none(metrics.get("deep_pull_high"))
        or _finite_or_none(decision.get("깊은눌림상단"))
        or _finite_or_none(context.get("deep_pull_high"))
    )
    shallow_low = (
        _finite_or_none(metrics.get("shallow_pull_low"))
        or _finite_or_none(decision.get("얕은눌림하단"))
        or _finite_or_none(decision.get("눌림하단"))
        or _finite_or_none(context.get("shallow_pull_low"))
    )
    shallow_high = (
        _finite_or_none(metrics.get("shallow_pull_high"))
        or _finite_or_none(decision.get("얕은눌림상단"))
        or _finite_or_none(decision.get("눌림상단"))
        or _finite_or_none(context.get("shallow_pull_high"))
    )
    if current_price is None or deep_low is None or deep_high is None:
        return []
    in_deep = deep_low <= current_price <= deep_high
    in_shallow = shallow_low is not None and shallow_high is not None and shallow_low <= current_price <= shallow_high
    if in_shallow and in_deep:
        if "겹치는 구간" not in report_text or "반등 확인 후" not in report_text:
            return ["현재가가 얕은 눌림목과 깊은 눌림목에 동시에 포함되는데 겹침 구간 설명이 없습니다."]
    elif in_deep:
        required_phrases = ["깊은 눌림목 구간 안", "반등 확인 후"]
        if not all(phrase in report_text for phrase in required_phrases):
            return ["현재가가 깊은 눌림목 구간 안인데 해당 상태 설명이 부족합니다."]
    return []


def validate_rr_buy_unavailable_wording(report_text: str, decision: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    now_buy = decision.get("지금바로매수") or decision.get("지금 바로 매수")
    current_rr = (
        _finite_or_none(metrics.get("current_rr"))
        or _finite_or_none(metrics.get("rr1"))
        or _finite_or_none(decision.get("손익비1"))
    )
    if now_buy == "불가" and current_rr is not None and current_rr >= 1.5:
        required = "손익비는" in report_text and "지금 바로 신규매수 조건은 충족하지 못했습니다" in report_text
        if not required:
            return ["매수 불가 상태에서 양호한 손익비에 대한 제한 설명이 없습니다."]
    return []


def validate_duplicate_status_wording(report_text: str) -> list[str]:
    errors: list[str] = []
    for phrase in ["장중 장중", "완료 일봉 완료 일봉", "분봉 분봉"]:
        if phrase in report_text:
            errors.append(f"중복 상태 문구가 발견되었습니다: {phrase}")
    return errors


def validate_holder_defense_wording(report_text: str, decision: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    current_price = (
        _finite_or_none(metrics.get("current_price"))
        or _finite_or_none(decision.get("현재가"))
        or _finite_or_none(decision.get("기준가"))
        or _finite_or_none(context.get("current_price"))
    )
    intraday_warning = (
        _finite_or_none(metrics.get("intraday_warning_line"))
        or _finite_or_none(decision.get("장중주의선"))
        or _finite_or_none(context.get("intraday_warning_line"))
    )
    if current_price is not None and intraday_warning is not None and current_price < intraday_warning:
        if "방어 관찰이 우선" not in report_text:
            return ["현재가가 장중 주의선 아래인데 보유자 방어 관찰 우선 문구가 없습니다."]
    return []


def validate_sector_wording(report_text: str, context: dict[str, Any]) -> list[str]:
    code = str(context.get("code") or "").zfill(6)
    errors: list[str] = []
    if code != "033100" and "전력기기/변압기" in report_text:
        errors.append("전력기기/변압기 문구가 비해당 종목 보고서에 출력되었습니다.")
    if code == "403870" and "전력기기/변압기" in report_text:
        errors.append("HPSP 보고서에 제룡전기 섹터 문구가 남아 있습니다.")
    return errors


def validate_indicator_wording(report_text: str, metrics: dict[str, Any], context: dict[str, Any], indicators: dict[str, Any] | None) -> list[str]:
    indicators = indicators or {}
    errors: list[str] = []
    rsi = _finite_or_none(metrics.get("rsi")) or _finite_or_none(indicators.get("RSI14"))
    if rsi is not None:
        if rsi >= 50 and "50 아래" in report_text:
            errors.append("RSI가 50 이상인데 50 아래 해석 문구가 사용되었습니다.")
        if rsi < 50 and "50선을 회복한 상태" in report_text:
            errors.append("RSI가 50 미만인데 50선 회복 문구가 사용되었습니다.")

    macd = _finite_or_none(metrics.get("macd")) or _finite_or_none(indicators.get("MACD"))
    signal = _finite_or_none(metrics.get("macd_signal")) or _finite_or_none(indicators.get("MACD신호"))
    hist = _finite_or_none(metrics.get("macd_hist")) or _finite_or_none(indicators.get("MACD히스토그램"))
    if macd is not None and signal is not None and hist is not None and macd > signal and hist > 0:
        for phrase in ["히스토그램 개선 확인 전", "MACD 개선 확인 전"]:
            if phrase in report_text:
                errors.append("MACD가 신호선 위이고 히스토그램 양수인데 부정확한 개선 대기 문구가 사용되었습니다.")
                break

    current_price = (
        _finite_or_none(metrics.get("current_price"))
        or _finite_or_none(context.get("current_price"))
        or _finite_or_none(indicators.get("current_price"))
        or _finite_or_none(indicators.get("Close"))
    )
    bb_mid = _finite_or_none(metrics.get("bb_mid")) or _finite_or_none(indicators.get("BB중심"))
    if current_price is not None and bb_mid is not None and bb_mid > 0:
        if current_price > bb_mid * 1.02 and "중심선 근처" in report_text:
            errors.append("현재가가 볼린저밴드 중심선보다 충분히 위인데 중심선 근처 문구가 사용되었습니다.")
        if current_price < bb_mid * 0.98 and "중심선 위" in report_text:
            errors.append("현재가가 볼린저밴드 중심선 아래인데 중심선 위 문구가 사용되었습니다.")
    return errors


def validate_support_resistance_duplicates(report_text: str) -> list[str]:
    errors: list[str] = []
    duplicate_resistance_pattern = re.compile(r"주요 저항:\s*([0-9,]+원),\s*\1")
    duplicate_support_pattern = re.compile(r"주요 지지:\s*([0-9,]+원),\s*\1")
    if duplicate_resistance_pattern.search(report_text):
        errors.append("주요 저항 가격이 중복 출력되었습니다.")
    if duplicate_support_pattern.search(report_text):
        errors.append("주요 지지 가격이 중복 출력되었습니다.")
    return errors


def validate_tick_unit_prices(report_text: str, context: dict[str, Any]) -> list[str]:
    if context.get("market") not in {"KOSPI", "KOSDAQ"}:
        return []
    errors: list[str] = []
    for price in sorted(set(extract_price_values(report_text))):
        if price != round_to_tick(price):
            errors.append(f"호가단위에 맞지 않는 가격이 출력되었습니다: {money(price)}")
    return errors


def validate_price_context_qa(report_text: str, decision: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    current = (
        _finite_or_none(metrics.get("current_price"))
        or _finite_or_none(decision.get("현재가"))
        or _finite_or_none(decision.get("기준가"))
        or _finite_or_none(context.get("current_price"))
    )
    buy_low = (
        _finite_or_none(metrics.get("shallow_pull_low"))
        or _finite_or_none(metrics.get("pullback_low"))
        or _finite_or_none(decision.get("얕은눌림하단"))
        or _finite_or_none(decision.get("눌림하단"))
    )
    buy_high = (
        _finite_or_none(metrics.get("shallow_pull_high"))
        or _finite_or_none(metrics.get("pullback_high"))
        or _finite_or_none(decision.get("얕은눌림상단"))
        or _finite_or_none(decision.get("눌림상단"))
    )
    rebreak = (
        _finite_or_none(metrics.get("rebreak_line"))
        or _finite_or_none(metrics.get("near_high_rebreak"))
        or _finite_or_none(decision.get("단기재돌파확인선"))
        or _finite_or_none(decision.get("당일고가재돌파"))
    )
    breakout = (
        _finite_or_none(metrics.get("breakout_line"))
        or _finite_or_none(metrics.get("close_confirm_line"))
        or _finite_or_none(decision.get("일봉돌파확인선"))
        or _finite_or_none(decision.get("종가유지확인선"))
        or _finite_or_none(decision.get("돌파가격"))
    )
    target1 = _finite_or_none(metrics.get("target1")) or _finite_or_none(decision.get("신규1차목표")) or _finite_or_none(decision.get("1차목표"))
    defense = _finite_or_none(metrics.get("defense_line")) or _finite_or_none(decision.get("방어선"))
    if current is None:
        return errors
    if buy_high is not None and breakout is not None and abs(buy_high - breakout) <= get_tick_unit(current):
        errors.append("매수 관심가 상단과 돌파 확인가가 사실상 같은 구간입니다.")
    if current is not None and breakout is not None and current > breakout:
        forbidden = ["돌파 확인가를 기다", "돌파 확인이 먼저", "종가 안착 확인이 먼저"]
        for phrase in forbidden:
            if phrase in report_text:
                errors.append("현재가가 돌파 확인가 위인데 돌파 확인가 대기 문구가 사용되었습니다.")
                break
    if current is not None and rebreak is not None and abs(current - rebreak) <= get_tick_unit(current):
        if re.search(r"단기\s*재돌파(?:\s*확인)?선[^.\n|]*회복|회복[^.\n|]*단기\s*재돌파(?:\s*확인)?선", report_text):
            errors.append("현재가와 단기 재돌파선이 같은데 회복 문구가 사용되었습니다.")
    if None not in (current, buy_low, buy_high, rebreak, breakout, target1, defense):
        ctx = classify_price_context(current, buy_low, buy_high, rebreak, breakout, target1, defense)
        if ctx.get("is_near_target1") and "신규매수보다 보유자 일부 익절 관찰이 우선" not in report_text:
            errors.append("현재가가 1차 목표에 가까운데 익절 관찰 우선 문구가 없습니다.")
    return errors


def validate_markdown_table_integrity(report_text: str) -> list[str]:
    errors: list[str] = []
    lines = report_text.splitlines()
    in_table = False
    expected_pipes = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            pipe_count = stripped.count("|")
            if not in_table:
                in_table = True
                expected_pipes = pipe_count
            elif pipe_count != expected_pipes:
                errors.append("Markdown 표의 열 개수가 일치하지 않습니다.")
                break
            continue
        if in_table and stripped:
            errors.append("Markdown 표 안에 줄바꿈 문장이 포함되어 HTML 표가 깨질 수 있습니다.")
            break
        if not stripped:
            in_table = False
            expected_pipes = 0
    return errors


def _is_markdown_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _markdown_table_blocks(section_text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in section_text.splitlines():
        if _is_markdown_table_line(line):
            current.append(line.strip())
            continue
        if current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def validate_markdown_html_table_rendering_qa(report_text: str) -> list[str]:
    """Validate that the trading-point table stays as one Markdown table.

    Markdown-level integrity is mandatory. HTML table checks are applied only
    when the local Markdown renderer actually emits <table> tags; otherwise a
    valid Markdown table must not be falsely failed by renderer differences.
    """

    errors: list[str] = []
    section = _report_section(report_text, "매매 타점")
    if not section:
        return ["매매 타점 섹션이 없습니다."]

    blocks = _markdown_table_blocks(section)
    trading_block: list[str] = []
    for block in blocks:
        joined = "\n".join(block)
        if "구분" in joined and "가격" in joined and "행동" in joined:
            trading_block = block
            break
    if not trading_block:
        return ["매매 타점 표가 Markdown 표로 렌더링되지 않았습니다."]

    required_labels = ["눌림목 지지가", "1차 목표", "2차 목표", "장중 방어선", "스윙 손절선"]
    required_label_groups = {
        "회복/돌파 확인가": ["회복/돌파 공통 확인가", "회복 확인가", "일봉 돌파 확인가"],
        "재돌파선 또는 강한 저항": ["단기 재돌파선", "장중 현재가 유지 기준", "강한 저항/목표권 확인선", "강한 저항/2차 목표 전 확인선"],
    }
    joined_block = "\n".join(trading_block)

    for label in required_labels:
        if not re.search(rf"^\|\s*{re.escape(label)}\s*\|", joined_block, re.MULTILINE):
            errors.append(f"매매 타점 표 안에 필수 행이 없습니다: {label}")
    for group_name, labels in required_label_groups.items():
        if not any(re.search(rf"^\|\s*{re.escape(label)}\s*\|", joined_block, re.MULTILINE) for label in labels):
            errors.append(f"매매 타점 표 안에 필수 행 그룹이 없습니다: {group_name}")

    for block in blocks:
        if block is trading_block:
            continue
        other_joined = "\n".join(block)
        for label in required_labels:
            if re.search(rf"^\|\s*{re.escape(label)}\s*\|", other_joined, re.MULTILINE):
                errors.append(f"매매 타점 필수 행이 주 표 밖 별도 표로 분리되었습니다: {label}")
        for group_name, labels in required_label_groups.items():
            if any(re.search(rf"^\|\s*{re.escape(label)}\s*\|", other_joined, re.MULTILINE) for label in labels):
                errors.append(f"매매 타점 필수 행 그룹이 주 표 밖 별도 표로 분리되었습니다: {group_name}")

    in_code = False
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped or _is_markdown_table_line(line):
            continue
        if "|" in stripped:
            errors.append("Markdown 표 밖 문단에 파이프(|) 문자가 남아 HTML 문단으로 렌더링될 수 있습니다.")
            break

    try:
        html = html_from_markdown(report_text, "QA")
    except Exception:
        html = ""

    if html and "<table" in html.lower():
        if re.search(r"<p\b[^>]*>[^<]*\|[^<]*</p>", html):
            errors.append("HTML 문단에 파이프(|) 문자가 그대로 출력되었습니다.")
        for label in required_labels:
            if not re.search(rf"<tr[^>]*>.*?<td[^>]*>\s*{re.escape(label)}\s*</td>.*?</tr>", html, re.DOTALL):
                errors.append(f"HTML table 안에 필수 행이 없습니다: {label}")
        for group_name, labels in required_label_groups.items():
            if not any(re.search(rf"<tr[^>]*>.*?<td[^>]*>\s*{re.escape(label)}\s*</td>.*?</tr>", html, re.DOTALL) for label in labels):
                errors.append(f"HTML table 안에 필수 행 그룹이 없습니다: {group_name}")

    return errors


def _report_table_row(report_text: str, label: str) -> list[str]:
    for line in report_text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and cells[0] == label:
            return cells
    return []


def validate_strategy_labeling_qa(report_text: str, decision: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    current = (
        _finite_or_none(metrics.get("current_price"))
        or _finite_or_none(decision.get("현재가"))
        or _finite_or_none(decision.get("기준가"))
        or _finite_or_none(context.get("current_price"))
    )
    buy_low = (
        _finite_or_none(metrics.get("buy_low"))
        or _finite_or_none(metrics.get("shallow_pull_low"))
        or _finite_or_none(metrics.get("pullback_low"))
        or _finite_or_none(decision.get("얕은눌림하단"))
        or _finite_or_none(decision.get("눌림하단"))
    )
    primary_strategy = str(decision.get("주전략") or decision.get("주 전략") or "")
    if current is None or buy_low is None or current >= buy_low:
        return errors
    if primary_strategy == "눌림목 대기":
        errors.append("현재가가 매수 관심가 하단보다 낮은데 주 전략이 눌림목 대기로 표시되었습니다.")
    if re.search(r"\|\s*매수 관심가\s*\|", report_text):
        errors.append("현재가가 매수 관심가보다 낮은데 매수 관심가를 단일 눌림목 명칭처럼 표시했습니다.")
    return errors


def validate_forbidden_interpretation_wording(report_text: str) -> list[str]:
    if "데이터 부족 기준으로" in report_text:
        return ["금지 문구가 사용되었습니다: 데이터 부족 기준으로"]
    return []


def validate_numeric_indicator_grades(report_text: str, metrics: dict[str, Any], indicators: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    indicators = indicators or {}
    rsi = _finite_or_none(metrics.get("rsi")) or _finite_or_none(indicators.get("RSI14"))
    rsi_row = _report_table_row(report_text, "RSI")
    if rsi is not None and len(rsi_row) >= 2:
        rsi_state = rsi_row[1]
        expected_rsi = rsi_grade_from_value(rsi)
        if 50 <= rsi < 55 and rsi_state == "좋음":
            errors.append("RSI 50~55 구간인데 상태가 좋음으로 표시되었습니다.")
        if rsi_state in {"좋음", "강함", "양호", "중립 회복", "약한 반등", "약세", "과열"} and rsi_state != expected_rsi:
            errors.append(f"RSI 상태 라벨이 수치와 다릅니다. 기대값 {expected_rsi}, 실제값 {rsi_state}.")

    macd = _finite_or_none(metrics.get("macd")) or _finite_or_none(indicators.get("MACD"))
    signal = _finite_or_none(metrics.get("macd_signal")) or _finite_or_none(indicators.get("MACD신호"))
    hist = _finite_or_none(metrics.get("macd_hist")) or _finite_or_none(indicators.get("MACD히스토그램"))
    current = _finite_or_none(metrics.get("current_price"))
    breakout = _finite_or_none(metrics.get("breakout_line")) or _finite_or_none(metrics.get("close_confirm_line"))
    macd_row = _report_table_row(report_text, "MACD")
    if len(macd_row) >= 3:
        macd_state = macd_row[1]
        macd_comment_text = " ".join(macd_row[2:])
        if macd_state == "나쁨" and "모멘텀은 살아" in macd_comment_text:
            errors.append("MACD 설명은 모멘텀이 살아 있다고 하면서 상태는 나쁨으로 표시되었습니다.")
        if macd is not None and signal is not None and hist is not None:
            expected_macd = macd_grade_from_values(macd, signal, hist, current, breakout)
            if macd_state in {"좋음", "혼조/개선 중", "중립", "중립/개선 시도", "나쁨"} and macd_state != expected_macd:
                errors.append(f"MACD 상태 라벨이 수치와 다릅니다. 기대값 {expected_macd}, 실제값 {macd_state}.")
    return errors


def validate_near_intraday_defense_warning(report_text: str, decision: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    if "장중방어선" not in decision:
        return []
    current = _finite_or_none(metrics.get("current_price")) or _finite_or_none(decision.get("현재가"))
    intraday_defense = _finite_or_none(decision.get("장중방어선")) or _finite_or_none(metrics.get("intraday_defense_line"))
    if current is None or intraday_defense is None or current <= 0:
        return []
    if abs(current - intraday_defense) / current <= 0.005:
        required = "장중 방어선이 가까워 신규 진입은 손익비보다 실패 확인 리스크가 더 큼"
        if required not in report_text:
            return ["장중 방어선이 현재가 0.5% 이내인데 실패 확인 리스크 경고가 없습니다."]
    return []


def validate_recovery_confirmation_qa(report_text: str, decision: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    current = _finite_or_none(metrics.get("current_price")) or _finite_or_none(decision.get("현재가")) or _finite_or_none(decision.get("기준가"))
    buy_high = (
        _finite_or_none(metrics.get("buy_high"))
        or _finite_or_none(metrics.get("shallow_pull_high"))
        or _finite_or_none(decision.get("얕은눌림상단"))
        or _finite_or_none(decision.get("눌림상단"))
    )
    breakout = (
        _finite_or_none(metrics.get("breakout_line"))
        or _finite_or_none(metrics.get("close_confirm_line"))
        or _finite_or_none(decision.get("일봉돌파확인선"))
        or _finite_or_none(decision.get("돌파가격"))
    )
    recovery_row = _report_table_row(report_text, "회복 확인가")
    if not recovery_row:
        recovery_row = _report_table_row(report_text, "회복/돌파 공통 확인가")
    recovery_text = " | ".join(recovery_row[1:]) if len(recovery_row) >= 2 else ""
    if current is not None and buy_high is not None and breakout is not None:
        if buy_high < current < breakout and ("해당 없음" in recovery_text or not recovery_text):
            errors.append("회복 확인가가 필요한 가격 구조인데 회복 확인가가 해당 없음으로 표시되었습니다.")
    return errors


def validate_rebreak_target_duplicate_qa(report_text: str, decision: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rebreak_row = _report_table_row(report_text, "단기 재돌파선")
    target_row = _report_table_row(report_text, "1차 목표")
    if not target_row:
        target_row = _report_table_row(report_text, "신규매수 기준 1차 목표")
    rebreak_prices = extract_price_values(" | ".join(rebreak_row[1:])) if len(rebreak_row) >= 2 else []
    target_prices = extract_price_values(" | ".join(target_row[1:])) if len(target_row) >= 2 else []
    if rebreak_prices and target_prices:
        rebreak = rebreak_prices[0]
        target = target_prices[0]
        if abs(round_to_tick(rebreak) - round_to_tick(target)) <= get_tick_unit(rebreak):
            errors.append("단기 재돌파선과 1차 목표가 같은 가격으로 중복 표시되었습니다.")
    return errors


def validate_rebreak_target_order_qa(report_text: str, decision: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rebreak = (
        _finite_or_none(metrics.get("rebreak_line"))
        or _finite_or_none(decision.get("단기재돌파확인선"))
        or _finite_or_none(decision.get("재돌파가격"))
    )
    target1 = _finite_or_none(metrics.get("target1")) or _finite_or_none(decision.get("신규1차목표")) or _finite_or_none(decision.get("1차목표"))
    target2 = _finite_or_none(metrics.get("target2")) or _finite_or_none(decision.get("신규2차목표")) or _finite_or_none(decision.get("2차목표"))
    row = _report_table_row(report_text, "단기 재돌파선")
    row_text = " | ".join(row[1:]) if len(row) >= 2 else ""
    row_prices = extract_price_values(row_text)
    strong_row = _report_table_row(report_text, "강한 저항/목표권 확인선")
    if not strong_row:
        strong_row = _report_table_row(report_text, "강한 저항/2차 목표 전 확인선")
    strong_text = " | ".join(strong_row[1:]) if len(strong_row) >= 2 else ""
    if rebreak is not None and target2 is not None and rebreak > target2:
        if row_prices and any(same_price_level(price, rebreak) for price in row_prices):
            errors.append("단기 재돌파선이 2차 목표보다 높은데 단기 재돌파선으로 출력되었습니다.")
        if row and "이전 고점/강한 저항" not in row_text and "강한 저항" not in row_text:
            errors.append("2차 목표보다 높은 단기 재돌파선은 이전 고점/강한 저항으로 재분류해야 합니다.")
    if rebreak is not None and target1 is not None and rebreak > target1 and row_prices:
        errors.append("단기 재돌파선이 1차 목표보다 높은데 단기 재돌파선 행으로 출력되었습니다.")
    if rebreak is not None and target1 is not None and rebreak > target1:
        if not strong_text or not any(word in strong_text for word in ["보유자", "비중관리", "익절", "강한 저항"]):
            errors.append("1차 목표보다 높은 단기 재돌파선은 강한 저항/목표권 확인선으로 재분류해야 합니다.")
    return errors


def validate_holder_recovery_profit_confusion(report_text: str, decision: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    recovery_row = _report_table_row(report_text, "회복 확인가")
    if not recovery_row:
        recovery_row = _report_table_row(report_text, "회복/돌파 공통 확인가")
    recovery_prices = extract_price_values(" | ".join(recovery_row[1:])) if len(recovery_row) >= 2 else []
    if not recovery_prices:
        return errors
    holder_parts = []
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- 보유자:") or stripped.startswith("보유자:"):
            holder_parts.append(stripped)
    for label in ["기존 보유자", "보유자 기준 대응"]:
        row = _report_table_row(report_text, label)
        if len(row) >= 2:
            holder_parts.append(" | ".join(row[1:]))
    holder_text = " ".join(holder_parts)
    if not holder_text:
        return errors
    for price in recovery_prices:
        token = money(price)
        if token not in holder_text:
            continue
        for sentence in re.split(r"[.。]\s*", holder_text):
            if token in sentence and "익절" in sentence and not any(word in sentence for word in ["회복 실패", "저항 확인", "회복 확인"]):
                errors.append("보유자 문장에서 회복 확인가를 일부 익절가처럼 표현했습니다.")
                return errors
    return errors


def validate_terminology_consistency(report_text: str) -> list[str]:
    if "눌림목 지지가" in report_text and "매수 관심가" in report_text:
        return ["보고서에서 눌림목 지지가와 매수 관심가 표현이 혼용되었습니다."]
    return []


def validate_today_action_has_price(report_text: str) -> list[str]:
    action_texts: list[str] = []
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- 지금 할 행동:"):
            action_texts.append(stripped)
    row = _report_table_row(report_text, "오늘 할 행동")
    if len(row) >= 2:
        action_texts.append(" | ".join(row[1:]))
    if not action_texts:
        return []
    if not extract_price_values(" ".join(action_texts)):
        return ["지금 할 행동에 구체 가격이 포함되지 않았습니다."]
    return []


def validate_precision_limited_points(report_text: str, decision: dict[str, Any]) -> list[str]:
    final = str(decision.get("최종판단") or "")
    if "정밀 판단 중단" not in final:
        return []
    errors: list[str] = []
    for label in ["회복 확인가", "눌림목 지지가", "일봉 돌파 확인가", "1차 목표", "2차 목표", "스윙 손절선"]:
        row = _report_table_row(report_text, label)
        if len(row) >= 2:
            value = row[1]
            if extract_price_values(value) and not value.startswith("참고 "):
                errors.append(f"정밀 판단 중단 상태인데 {label}이 참고 라벨 없이 확정값처럼 출력되었습니다.")
    return errors


def validate_stale_yfinance_explanation(report_text: str, context: dict[str, Any]) -> list[str]:
    validation_note = str(context.get("validation_note") or "")
    if "yfinance" in validation_note and "지연" in validation_note:
        if "yfinance" not in report_text or "보조 소스" not in report_text or "지연" not in report_text:
            return ["yfinance 최신거래일 지연이 있는데 원인 설명이 보고서에 없습니다."]
        if re.search(r"가격 신뢰도:\s*낮음", report_text) or re.search(r"\|\s*가격 신뢰도\s*\|\s*낮음\s*\|", report_text):
            return ["pykrx/FDR 일치 + yfinance stale 상황에서 가격 신뢰도를 낮음으로 표시했습니다."]
    return []


def validate_low_rr_wording(report_text: str, metrics: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rr1 = _finite_or_none(metrics.get("rr1")) or _finite_or_none(decision.get("손익비1"))
    intraday_rr = _finite_or_none(metrics.get("intraday_rr")) or _finite_or_none(decision.get("장중방어손익비"))
    confirm_rr = _finite_or_none(metrics.get("confirm_rr")) or _finite_or_none(decision.get("확인진입손익비"))
    rsi = _finite_or_none(metrics.get("rsi")) or _finite_or_none(decision.get("RSI"))
    final = str(decision.get("최종판단") or "")
    if intraday_rr is not None and intraday_rr < 1.2 and "장중 신규 진입 매력 낮음" not in report_text:
        errors.append("장중 방어선 기준 손익비가 1.2 미만인데 장중 신규 진입 매력 낮음 문구가 없습니다.")
    if rr1 is not None and rr1 < 1.2 and "손익비 부족" not in report_text:
        errors.append("스윙 손절선 기준 손익비가 1.2 미만인데 손익비 부족 문구가 없습니다.")
    if rr1 is not None and rr1 < 1.0 and not any(
        phrase in report_text
        for phrase in ["스윙 신규매수 부적합", "신규매수와 돌파 추격매수 모두 부적합", "신규매수와 돌파 추격 모두 부적합"]
    ):
        errors.append("스윙 손절선 기준 손익비가 1.0 미만인데 스윙 신규매수 부적합 문구가 없습니다.")
    if confirm_rr is not None and confirm_rr < 1.2 and not any(
        phrase in report_text
        for phrase in [
            "스윙/돌파 신규매수 매력 낮음",
            "스윙/돌파 추격 손익비는 부족",
            "돌파 추격매수 부적합",
            "신규매수와 돌파 추격매수 모두 부적합",
            "신규매수와 돌파 추격 모두 부적합",
            "돌파 매수 전략 성립 불가",
            "돌파 추격 전략 성립 불가",
        ]
    ):
        errors.append("회복/돌파 진입 기준 손익비가 1.2 미만인데 돌파/추격 신규매수 매력 낮음 문구가 없습니다.")
    if confirm_rr is not None and confirm_rr < 1.0 and not any(
        phrase in report_text
        for phrase in [
            "돌파 추격매수 부적합",
            "신규매수와 돌파 추격매수 모두 부적합",
            "신규매수와 돌파 추격 모두 부적합",
            "돌파 매수 전략 성립 불가",
            "돌파 추격 전략 성립 불가",
        ]
    ):
        errors.append("회복/돌파 진입 기준 손익비가 1.0 미만인데 돌파 추격매수 부적합 문구가 없습니다.")
    if rsi is not None and rsi >= 70 and any(v is not None and v < 1.2 for v in [intraday_rr, confirm_rr, rr1]):
        if "과열 추격 금지" not in report_text and "과열권에서 장중 급등 가격을 추격매수하지 않습니다" not in report_text:
            errors.append("RSI 과열과 낮은 손익비가 겹쳤는데 과열 추격 금지 문구가 없습니다.")
    if rr1 is not None and rr1 < 1.2 and any(word in final for word in ["매수 가능", "조건 충족"]):
        errors.append("손익비가 낮은데 최종 판단이 신규매수 가능처럼 긍정적으로 표시되었습니다.")
    return errors


def validate_daily_trend_label_qa(report_text: str, metrics: dict[str, Any], decision: dict[str, Any], indicators: dict[str, Any] | None) -> list[str]:
    indicators = indicators or {}
    row = _report_table_row(report_text, "일봉")
    if len(row) < 3:
        return []
    state = row[1]
    comment = " ".join(row[2:])
    errors: list[str] = []
    current = _finite_or_none(metrics.get("current_price")) or _finite_or_none(decision.get("현재가")) or _finite_or_none(indicators.get("current_price"))
    ma20 = _finite_or_none(metrics.get("ma20")) or _finite_or_none(indicators.get("MA20"))
    ma60 = _finite_or_none(metrics.get("ma60")) or _finite_or_none(indicators.get("MA60"))
    macd = _finite_or_none(metrics.get("macd")) or _finite_or_none(indicators.get("MACD"))
    signal = _finite_or_none(metrics.get("macd_signal")) or _finite_or_none(indicators.get("MACD신호"))
    rsi = _finite_or_none(metrics.get("rsi")) or _finite_or_none(indicators.get("RSI14"))
    if state == "상승" and any(phrase in comment for phrase in ["20일선 아래", "단기 추세 회복 확인 필요"]):
        errors.append("일봉 상태가 상승인데 20일선 아래/단기 추세 회복 확인 필요 문장이 함께 출력되었습니다.")
    if current is not None and ma20 is not None and ma60 is not None:
        if current < ma20 and current > ma60 and state == "상승":
            errors.append("현재가가 20일선 아래·60일선 위인데 일봉 상승 단독 라벨이 출력되었습니다.")
        weak_momentum = macd is not None and signal is not None and rsi is not None and macd < signal and rsi < 50
        if current < ma20 and weak_momentum and state == "상승":
            errors.append("현재가가 20일선 아래이고 MACD/RSI가 약한데 일봉 상승 단독 라벨이 출력되었습니다.")
    return errors


def validate_common_confirmation_price_qa(report_text: str, decision: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    recovery = (
        _finite_or_none(metrics.get("recovery_line"))
        or _finite_or_none(decision.get("회복확인선"))
        or _finite_or_none(decision.get("회복/돌파공통확인선"))
    )
    breakout = (
        _finite_or_none(metrics.get("breakout_line"))
        or _finite_or_none(metrics.get("close_confirm_line"))
        or _finite_or_none(decision.get("일봉돌파확인선"))
        or _finite_or_none(decision.get("종가유지확인선"))
    )
    if recovery is None or breakout is None or not same_price_level(recovery, breakout):
        return []
    errors: list[str] = []
    common_row = _report_table_row(report_text, "회복/돌파 공통 확인가")
    recovery_row = _report_table_row(report_text, "회복 확인가")
    breakout_row = _report_table_row(report_text, "일봉 돌파 확인가")
    if not common_row:
        errors.append("회복 확인가와 일봉 돌파 확인가가 같은데 회복/돌파 공통 확인가로 통합되지 않았습니다.")
    elif "종가 안착 + 거래량 유지 확인" not in " ".join(common_row[1:]):
        errors.append("회복/돌파 공통 확인가의 행동 문구가 종가 안착 + 거래량 유지 확인으로 통합되지 않았습니다.")
    recovery_prices = extract_price_values(" ".join(recovery_row[1:])) if len(recovery_row) >= 2 else []
    breakout_prices = extract_price_values(" ".join(breakout_row[1:])) if len(breakout_row) >= 2 else []
    if recovery_prices and breakout_prices and same_price_level(recovery_prices[0], breakout_prices[0]):
        errors.append("같은 가격의 회복 확인가와 일봉 돌파 확인가가 별도 항목으로 반복 출력되었습니다.")
    return errors


def validate_volume_momentum_conflict_qa(report_text: str, decision: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    current = _finite_or_none(metrics.get("current_price")) or _finite_or_none(decision.get("현재가"))
    pull_high = (
        _finite_or_none(metrics.get("pullback_high"))
        or _finite_or_none(metrics.get("shallow_pull_high"))
        or _finite_or_none(decision.get("얕은눌림상단"))
    )
    recovery = (
        _finite_or_none(metrics.get("recovery_line"))
        or _finite_or_none(decision.get("회복확인선"))
        or _finite_or_none(decision.get("회복/돌파공통확인선"))
        or _finite_or_none(metrics.get("breakout_line"))
    )
    vol = _finite_or_none(metrics.get("weighted_volume_ratio")) or _finite_or_none(decision.get("장중거래량비율"))
    rsi = _finite_or_none(metrics.get("rsi")) or _finite_or_none(decision.get("RSI"))
    macd = _finite_or_none(metrics.get("macd"))
    signal = _finite_or_none(metrics.get("macd_signal"))
    conflict = assess_volume_momentum_conflict(current, pull_high, recovery, vol, rsi, macd, signal)
    if not conflict["applies"]:
        return []
    errors: list[str] = []
    if "거래량 동반 반등 시도이나 모멘텀 미회복" not in report_text:
        errors.append("거래량 강세와 모멘텀 약세가 동시에 발생했는데 모멘텀 미회복 문구가 없습니다.")
    if "모멘텀 확인 전 추격 금지" not in report_text:
        errors.append("거래량 강세·모멘텀 약세 구간인데 최종 판단에 모멘텀 확인 전 추격 금지가 없습니다.")
    return errors


def validate_macd_zone_wording_qa(report_text: str, metrics: dict[str, Any], indicators: dict[str, Any] | None = None) -> list[str]:
    indicators = indicators or {}
    macd = _finite_or_none(metrics.get("macd")) or _finite_or_none(indicators.get("MACD"))
    signal = _finite_or_none(metrics.get("macd_signal")) or _finite_or_none(indicators.get("MACD신호"))
    hist = _finite_or_none(metrics.get("macd_hist")) or _finite_or_none(indicators.get("MACD히스토그램"))
    if macd is None:
        return []
    errors: list[str] = []
    if macd < 0 and "양수권" in report_text:
        errors.append("MACD가 음수인데 양수권 문구가 사용되었습니다.")
    if macd > 0 and "MACD는 음수권" in report_text:
        errors.append("MACD가 양수인데 음수권 문구가 사용되었습니다.")
    if macd < 0 and signal is not None and hist is not None and macd > signal and hist > 0:
        if "음수권에서 신호선 위로 회복 시도" not in report_text:
            errors.append("MACD 음수권 개선 조건인데 음수권에서 신호선 위로 회복 시도 문구가 없습니다.")
    return errors


def validate_volume_candle_qa(report_text: str, metrics: dict[str, Any], indicators: dict[str, Any] | None = None) -> list[str]:
    indicators = indicators or {}
    open_price = _finite_or_none(metrics.get("open_price")) or _finite_or_none(indicators.get("Open"))
    high_price = _finite_or_none(metrics.get("high_price")) or _finite_or_none(indicators.get("High"))
    low_price = _finite_or_none(metrics.get("low_price")) or _finite_or_none(indicators.get("Low"))
    close_price = _finite_or_none(metrics.get("close_price")) or _finite_or_none(metrics.get("current_price")) or _finite_or_none(indicators.get("Close"))
    volume_ratio = _finite_or_none(metrics.get("volume_ratio20")) or _finite_or_none(metrics.get("weighted_volume_ratio")) or _finite_or_none(indicators.get("거래량비율20"))
    context = assess_volume_candle(open_price, high_price, low_price, close_price, volume_ratio)
    if not context.get("bearish_high_volume"):
        return []
    errors: list[str] = []
    if "고거래량 음봉/매물 출회 경고" not in report_text:
        errors.append("고거래량 음봉인데 거래량 상태가 고거래량 음봉/매물 출회 경고로 표시되지 않았습니다.")
    if context.get("strong_distribution") and "강한 매물 출회 경고" not in report_text:
        errors.append("고거래량 장대음봉인데 강한 매물 출회 경고 문구가 없습니다.")
    if "돌파 매수는 1.2배 이상이 필요합니다" in report_text and "매물 소화 확인" not in report_text:
        errors.append("장대 음봉 거래량을 돌파 매수 근거처럼 해석했습니다.")
    return errors


def validate_clean_data_action_wording(report_text: str) -> list[str]:
    high_patterns = [
        r"가격 신뢰도\s*[:|]\s*높음",
        r"거래량 신뢰도\s*[:|]\s*높음",
        r"교차검증 완전성\s*[:|]\s*높음",
    ]
    all_high = all(re.search(pattern, report_text) for pattern in high_patterns)
    if all_high and "소스 지연 여부 확인" in report_text:
        return ["가격/거래량/교차검증 신뢰도가 모두 높음인데 소스 지연 여부 확인 문구가 출력되었습니다."]
    return []


def validate_low_rr_priority_qa(report_text: str, metrics: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    swing_rr = _finite_or_none(metrics.get("rr1")) or _finite_or_none(decision.get("스윙손절손익비")) or _finite_or_none(decision.get("손익비1"))
    entry_rr = _finite_or_none(metrics.get("confirm_rr")) or _finite_or_none(decision.get("확인진입손익비"))
    final = str(decision.get("최종판단") or "")
    errors: list[str] = []
    if any(v is not None and v < 1.2 for v in [swing_rr, entry_rr]):
        if not any(
            phrase in report_text
            for phrase in [
                "스윙/돌파 신규매수 매력 낮음",
                "스윙/돌파 추격 손익비는 부족",
                "단기 트레이딩 손익비는 가능하나",
                "신규매수와 돌파 추격매수 모두 부적합",
                "신규매수와 돌파 추격 모두 부적합",
                "돌파 매수 전략 성립 불가",
                "돌파 추격 전략 성립 불가",
            ]
        ):
            errors.append("스윙 또는 회복/돌파 진입 손익비가 1.2 미만인데 스윙/돌파 신규매수 매력 낮음 문구가 없습니다.")
        if "손익비" not in final and not any(phrase in final for phrase in ["신규매수 금지", "전략 성립 불가", "부적합"]):
            errors.append("스윙 또는 회복/돌파 진입 손익비가 1.2 미만인데 최종 판단에 손익비 부족이 반영되지 않았습니다.")
    target_phrase_idx = report_text.find("1차 목표까지 여유")
    warning_candidates = [
        idx for idx in [report_text.find("손익비 부족"), report_text.find("스윙/돌파"), report_text.find("추격 손익비는 부족")] if idx >= 0
    ]
    if target_phrase_idx >= 0 and (not warning_candidates or target_phrase_idx < min(warning_candidates)):
        errors.append("1차 목표까지 여유 문구가 손익비 부족 경고보다 앞서 과도하게 강조되었습니다.")
    if any(v is not None and v < 0.8 for v in [swing_rr, entry_rr]):
        if not any(phrase in report_text for phrase in ["신규매수 금지에 가까움", "신규매수 금지", "신규매수와 돌파 추격매수 모두 부적합", "신규매수와 돌파 추격 모두 부적합"]):
            errors.append("스윙 또는 회복/돌파 진입 손익비가 0.8 미만인데 신규매수 금지에 가까움 문구가 없습니다.")
    if entry_rr is not None and entry_rr < 0.5:
        if not any(phrase in report_text for phrase in ["돌파 매수 전략 성립 불가", "돌파 추격 전략 성립 불가"]):
            errors.append("회복/돌파 진입 손익비가 0.5 미만인데 돌파 매수 전략 성립 불가 문구가 없습니다.")
    return errors


def validate_rr_warning_dedup_qa(report_text: str) -> list[str]:
    final_section = _report_section(report_text, "최종 한 문단") or report_text
    duplicate_markers = [
        "스윙 손절선 기준 손익비 부족",
        "스윙 손절선 기준 손익비 1.0 미만",
        "스윙/돌파 신규매수 매력 낮음",
        "회복/돌파 진입 기준 손익비 1.0 미만",
        "돌파 추격매수 부적합",
    ]
    count = sum(1 for marker in duplicate_markers if marker in final_section)
    if count >= 3:
        return ["최종 문단에서 같은 의미의 손익비 경고가 2회 이상 반복되었습니다."]
    return []


def validate_intraday_overheated_breakout_qa(report_text: str, metrics: dict[str, Any], decision: dict[str, Any], context: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    current = _finite_or_none(metrics.get("current_price")) or _finite_or_none(decision.get("현재가")) or _finite_or_none(context.get("current_price"))
    breakout = (
        _finite_or_none(metrics.get("breakout_line"))
        or _finite_or_none(metrics.get("close_confirm_line"))
        or _finite_or_none(decision.get("일봉돌파확인선"))
        or _finite_or_none(decision.get("종가유지확인선"))
    )
    rebreak = _finite_or_none(metrics.get("rebreak_line")) or _finite_or_none(decision.get("단기재돌파확인선"))
    target1 = _finite_or_none(metrics.get("target1")) or _finite_or_none(decision.get("신규1차목표")) or _finite_or_none(decision.get("1차목표"))
    rsi = _finite_or_none(metrics.get("rsi")) or _finite_or_none(decision.get("RSI"))
    entry_rr = _finite_or_none(metrics.get("confirm_rr")) or _finite_or_none(decision.get("확인진입손익비"))
    swing_rr = _finite_or_none(metrics.get("rr1")) or _finite_or_none(decision.get("스윙손절손익비")) or _finite_or_none(decision.get("손익비1"))
    intraday_rr = _finite_or_none(metrics.get("intraday_rr")) or _finite_or_none(decision.get("장중방어손익비"))
    final = str(decision.get("최종판단") or "")
    now_buy = str(decision.get("지금바로매수") or "")

    if now_buy == "불가" and final in {"돌파 유지 확인", "돌파 매수 가능", "공격 매수 가능", "조건 충족 시 분할 매수"}:
        errors.append("지금 매수 불가인데 최종 판단이 긍정 문장으로만 끝났습니다.")
    if rsi is not None and rsi >= 70 and entry_rr is not None and entry_rr < 1.0 and "신규 추격매수 부적합" not in report_text:
        errors.append("RSI 과열과 회복/돌파 진입 손익비 1.0 미만인데 신규 추격매수 부적합 문구가 없습니다.")
    if rsi is not None and rsi >= 70 and any(v is not None and v < 1.2 for v in [intraday_rr, entry_rr, swing_rr]):
        if "과열 추격 금지" not in report_text and "과열권에서 장중 급등 가격을 추격매수하지 않습니다" not in report_text:
            errors.append("RSI 과열과 낮은 손익비가 겹쳤는데 과열 추격 금지 문구가 없습니다.")
    if current is not None and rebreak is not None and abs(current - rebreak) <= get_tick_unit(current):
        if "현재가는 단기 재돌파선에 걸쳐" in report_text:
            errors.append("단기 재돌파선이 현재가와 같은데 기계적 재돌파선 문구가 사용되었습니다.")
        rebreak_row = _report_table_row(report_text, "단기 재돌파선")
        rebreak_text = " | ".join(rebreak_row[1:]) if len(rebreak_row) >= 2 else ""
        if rebreak_row and "장중 현재가 유지 기준" not in rebreak_text:
            errors.append("단기 재돌파선이 현재가와 같은데 장중 현재가 유지 기준으로 표시하지 않았습니다.")
    recovery_row = _report_table_row(report_text, "회복 확인가")
    if len(recovery_row) >= 3 and "해당 없음" in recovery_row[1] and "현재가보다 위에 있는 재진입 확인 가격" in " ".join(recovery_row[2:]):
        errors.append("회복 확인가가 해당 없음인데 고정 재진입 설명이 출력되었습니다.")
    if current is not None and breakout is not None and current > breakout:
        for phrase in ["돌파 확정", "조건 충족 매수 가능", "조건 충족 시 분할 매수"]:
            if phrase in report_text:
                errors.append("장중 현재가가 돌파 확인가 위라는 이유만으로 확정/매수 가능 문구가 출력되었습니다.")
                break
    if current is not None and target1 is not None and current > 0 and (target1 - current) / current >= 0.05:
        if "근접 저항 바로 아래" in report_text:
            errors.append("현재가와 1차 목표 차이가 5% 이상인데 근접 저항 바로 아래라고 표현했습니다.")
    if any(v is not None and v < 1.0 for v in [entry_rr, swing_rr]):
        if not any(token in final for token in ["손익비", "부적합", "금지", "보류", "대기", "제한"]):
            errors.append("손익비 1.0 미만인데 최종 판단에 손익비 부족이 우선 반영되지 않았습니다.")
    return errors


def validate_holder_action_split(report_text: str) -> list[str]:
    holder_parts: list[str] = []
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- 보유자:") or stripped.startswith("보유자:"):
            holder_parts.append(stripped)
    for label in ["기존 보유자", "보유자 기준 대응"]:
        row = _report_table_row(report_text, label)
        if len(row) >= 2:
            holder_parts.append(" | ".join(row[1:]))
    holder_text = " ".join(holder_parts)
    if not holder_text:
        return []
    required = ["추가매수 보류", "단기 비중 축소", "방어/손절"]
    missing = [word for word in required if word not in holder_text]
    if missing:
        return [f"보유자 대응이 회복 확인가/눌림목 지지가/스윙 손절선 대응을 분리하지 않았습니다: {', '.join(missing)}"]
    return []


def validate_trade_state_qa(report_text: str, decision: dict[str, Any], metrics: dict[str, Any], context: dict[str, Any]) -> list[str]:
    state = decision.get("상태코드") or context.get("trade_state") or {}
    if not isinstance(state, dict) or not state:
        return ["TradeState 상태코드가 생성되지 않았습니다."]
    errors: list[str] = []
    for field, allowed in [
        ("price_position_state", PRICE_POSITION_STATES),
        ("volume_state", VOLUME_STATES),
        ("macd_state", MACD_STATES),
        ("rsi_state", RSI_STATES),
        ("risk_reward_state", RISK_REWARD_STATES),
        ("price_data_state", PRICE_DATA_STATES),
        ("volume_data_state", VOLUME_DATA_STATES),
        ("supply_state", SUPPLY_STATES),
        ("cross_validation_state", CROSS_VALIDATION_STATES),
        ("final_action_state", FINAL_ACTION_STATES),
    ]:
        value = str(state.get(field) or "")
        if value not in allowed:
            errors.append(f"TradeState {field} 값이 허용 상태코드가 아닙니다: {value}")
    blocking_errors = state.get("blocking_errors") or state.get("qa_blocking_errors") or []
    if blocking_errors:
        errors.extend([f"TradeState blocking error: {err}" for err in blocking_errors])
    final_action = str(state.get("final_action_state") or "")
    expected_final = FINAL_ACTION_TEMPLATES.get(final_action, {}).get("final")
    actual_final = str(decision.get("최종판단") or "")
    if expected_final and actual_final != expected_final:
        errors.append("최종 판단이 final_action_state 템플릿과 일치하지 않습니다.")
    if state.get("risk_reward_state") == "RR_STRATEGY_INVALID" and final_action != "NO_BUY_STRATEGY_INVALID":
        errors.append("RR_STRATEGY_INVALID인데 final_action_state가 전략 무효 전용 코드가 아닙니다.")
    price_data_state = str(state.get("price_data_state") or "")
    volume_data_state = str(state.get("volume_data_state") or "")
    supply_state = str(state.get("supply_state") or "")
    cross_validation_state = str(state.get("cross_validation_state") or "")
    if price_data_state == "PRICE_DATA_OK" and volume_data_state == "VOLUME_DATA_OK" and cross_validation_state == "CROSS_VALIDATION_OK":
        if any(phrase in report_text for phrase in ["소스 지연 여부 확인", "대표 가격 소스 일치 여부 확인"]):
            errors.append("가격/거래량/교차검증 정상 상태인데 데이터 소스 지연/일치 확인 문구가 출력되었습니다.")
    if price_data_state == "PRICE_DATA_OK" and volume_data_state == "VOLUME_DATA_OK" and supply_state == "SUPPLY_MISSING":
        if "| 수급 | SUPPLY_MISSING |" not in report_text:
            errors.append("가격/거래량 데이터는 정상이고 수급만 부족한데 수급 상태코드가 별도 출력되지 않았습니다.")
        if re.search(r"\|\s*데이터\s*\|\s*DATA_OK\s*\|", report_text):
            errors.append("데이터: DATA_OK 단일 상태처럼 보이면서 수급 신뢰도 낮음이 함께 출력되었습니다.")
    if re.search(r"\|\s*데이터\s*\|\s*DATA_OK\s*\|", report_text):
        errors.append("상태코드 표에 단일 DATA_OK 데이터 행이 출력되었습니다.")
    if price_data_state == "PRICE_DATA_INVALID" or volume_data_state == "VOLUME_DATA_INVALID" or cross_validation_state == "CROSS_VALIDATION_INVALID":
        errors.append("DATA_INVALID 상태에서는 정상 보고서를 저장할 수 없습니다.")
    if (price_data_state == "PRICE_DATA_MISMATCH" or volume_data_state == "VOLUME_DATA_MISMATCH") and "참고 " not in report_text and "정밀 판단 중단" not in actual_final:
        errors.append("DATA_PRICE_MISMATCH 상태인데 정밀 타점을 확정값처럼 출력했습니다.")
    if state.get("macd_state") == "MACD_NEGATIVE_RECOVERY" and "음수권에서 신호선 위로 회복 시도" not in report_text:
        errors.append("MACD_NEGATIVE_RECOVERY 상태인데 전용 문구가 출력되지 않았습니다.")
    if state.get("rsi_state") == "RSI_OVERHEATED" and state.get("risk_reward_state") in {"RR_BAD", "RR_STRATEGY_INVALID", "RR_WEAK"}:
        if "과열 추격 금지" not in report_text and final_action != "NO_BUY_OVERHEATED_BAD_RR":
            errors.append("RSI_OVERHEATED와 낮은 손익비가 겹쳤는데 과열 추격 금지 판단이 없습니다.")
    if state.get("risk_reward_state") in {"RR_BAD", "RR_STRATEGY_INVALID"}:
        if any(phrase in report_text for phrase in ["조건 충족 시 매수 가능", "공격 매수 가능", "지금 매수 | 가능"]):
            errors.append("RR_BAD/RR_STRATEGY_INVALID 상태인데 신규매수 긍정 문구가 출력되었습니다.")
    if state.get("volume_state") == "HIGH_VOLUME_BEARISH_REVERSAL":
        if "고거래량 음봉" not in report_text or "매물" not in report_text:
            errors.append("HIGH_VOLUME_BEARISH_REVERSAL 상태인데 고거래량 음봉/매물 출회 문구가 없습니다.")
    return errors


def run_report_qa(
    report_text: str,
    decision: dict[str, Any],
    metrics: dict[str, Any],
    context: dict[str, Any],
    indicators: dict[str, Any] | None = None,
) -> None:
    errors: list[str] = []
    approved_prices, approved_ranges = build_approved_price_sets(context, indicators, decision, metrics)
    errors.extend(validate_decision_consistency(decision, metrics))
    errors.extend(validate_intraday_lines(metrics, decision))
    errors.extend(validate_recalculated_report_metrics(report_text, decision, metrics, context))
    errors.extend(validate_market_index_qa(report_text, context))
    errors.extend(validate_qa_section(report_text))
    errors.extend(validate_intraday_breakout_wording(report_text, decision, metrics, context))
    errors.extend(validate_single_price_ranges(report_text))
    errors.extend(validate_moving_average_wording(report_text, metrics, context, indicators))
    errors.extend(validate_nearby_profile_wording(report_text, metrics, context))
    errors.extend(validate_yfinance_minute_status(report_text))
    errors.extend(validate_target_profit_labels(report_text))
    errors.extend(validate_intraday_defense_labels(report_text))
    errors.extend(validate_early_volume_wording(report_text, metrics, context))
    errors.extend(validate_pro_trader_layer(report_text, decision, metrics))
    errors.extend(validate_pullback_below_wording(report_text, decision, metrics, context))
    errors.extend(validate_breakout_failed_volume_score(report_text, decision, metrics, context))
    errors.extend(validate_alert_price_section(report_text))
    errors.extend(validate_deep_pullback_status_wording(report_text, decision, metrics, context))
    errors.extend(validate_rr_buy_unavailable_wording(report_text, decision, metrics, context))
    errors.extend(validate_duplicate_status_wording(report_text))
    errors.extend(validate_holder_defense_wording(report_text, decision, metrics, context))
    errors.extend(validate_sector_wording(report_text, context))
    errors.extend(validate_indicator_wording(report_text, metrics, context, indicators))
    errors.extend(validate_support_resistance_duplicates(report_text))
    errors.extend(validate_tick_unit_prices(report_text, context))
    errors.extend(validate_price_context_qa(report_text, decision, metrics, context))
    errors.extend(validate_markdown_table_integrity(report_text))
    errors.extend(validate_markdown_html_table_rendering_qa(report_text))
    errors.extend(validate_strategy_labeling_qa(report_text, decision, metrics, context))
    errors.extend(validate_forbidden_interpretation_wording(report_text))
    errors.extend(validate_numeric_indicator_grades(report_text, metrics, indicators))
    errors.extend(validate_daily_trend_label_qa(report_text, metrics, decision, indicators))
    errors.extend(validate_near_intraday_defense_warning(report_text, decision, metrics))
    errors.extend(validate_recovery_confirmation_qa(report_text, decision, metrics))
    errors.extend(validate_common_confirmation_price_qa(report_text, decision, metrics))
    errors.extend(validate_rebreak_target_duplicate_qa(report_text, decision, metrics))
    errors.extend(validate_rebreak_target_order_qa(report_text, decision, metrics))
    errors.extend(validate_holder_recovery_profit_confusion(report_text, decision, metrics))
    errors.extend(validate_terminology_consistency(report_text))
    errors.extend(validate_today_action_has_price(report_text))
    errors.extend(validate_precision_limited_points(report_text, decision))
    errors.extend(validate_stale_yfinance_explanation(report_text, context))
    errors.extend(validate_low_rr_wording(report_text, metrics, decision))
    errors.extend(validate_low_rr_priority_qa(report_text, metrics, decision))
    errors.extend(validate_rr_warning_dedup_qa(report_text))
    errors.extend(validate_intraday_overheated_breakout_qa(report_text, metrics, decision, context))
    errors.extend(validate_volume_momentum_conflict_qa(report_text, decision, metrics))
    errors.extend(validate_macd_zone_wording_qa(report_text, metrics, indicators))
    errors.extend(validate_volume_candle_qa(report_text, metrics, indicators))
    errors.extend(validate_clean_data_action_wording(report_text))
    errors.extend(validate_holder_action_split(report_text))
    errors.extend(validate_trade_state_qa(report_text, decision, metrics, context))
    errors.extend(validate_report_text(report_text, context, approved_prices, approved_ranges))
    if errors:
        raise ReportValidationError("\n".join(errors))


def save_qa_failure(out_dir: Path, safe_name: str, code: str, errors: str, draft: str = "") -> Path:
    qa_fail_path = out_dir / f"{safe_name}_{code}_보고서_QA실패.md"
    body = f"# 보고서 QA 실패\n\n## 실패 사유\n\n{errors}\n"
    if draft:
        body += "\n## 생성 중단된 초안\n\n" + draft
    qa_fail_path.write_text(body, encoding="utf-8-sig")
    return qa_fail_path


def normalize_ohlcv(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]
    rename = {
        "시가": "Open",
        "고가": "High",
        "저가": "Low",
        "종가": "Close",
        "거래량": "Volume",
        "등락률": "Change",
        "Adj Close": "Adj Close",
    }
    out = out.rename(columns=rename)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "Change", "Adj Close"] if c in out.columns]
    out = out[keep].copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out = out.sort_index()
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" not in out.columns:
        out["Volume"] = np.nan
    out = out[out["Close"] > 0]
    out.attrs["source"] = source
    return out


def load_pykrx(code: str, start: date, end: date) -> SourceFrame:
    try:
        with suppress_external_output():
            from pykrx import stock
            df = stock.get_market_ohlcv_by_date(ymd(start), ymd(end), code)
        return SourceFrame("pykrx", normalize_ohlcv(df, "pykrx"))
    except Exception as e:
        return SourceFrame("pykrx", pd.DataFrame(), f"{type(e).__name__}: {e}")


def load_fdr(code: str, start: date, end: date) -> SourceFrame:
    try:
        import FinanceDataReader as fdr

        # FinanceDataReader end is inclusive for Korean equities in current versions.
        df = fdr.DataReader(code, iso(start), iso(end))
        return SourceFrame("FinanceDataReader", normalize_ohlcv(df, "FinanceDataReader"))
    except Exception as e:
        return SourceFrame("FinanceDataReader", pd.DataFrame(), f"{type(e).__name__}: {e}")


def load_yfinance(ticker: str, start: date, end: date, name: str = "yfinance") -> SourceFrame:
    try:
        import yfinance as yf

        df = yf.download(
            ticker,
            start=iso(start),
            end=iso(end + timedelta(days=1)),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        return SourceFrame(name, normalize_ohlcv(df, name), ticker)
    except Exception as e:
        return SourceFrame(name, pd.DataFrame(), f"{type(e).__name__}: {e}")


def load_stooq(ticker: str, start: date, end: date) -> SourceFrame:
    symbol = ticker.lower()
    if not symbol.endswith(".us") and not symbol.startswith("^"):
        symbol = f"{symbol}.us"
    try:
        url = f"https://stooq.com/q/d/l/?s={symbol}&d1={ymd(start)}&d2={ymd(end)}&i=d"
        df = pd.read_csv(url)
        if df.empty or "Date" not in df.columns:
            return SourceFrame("Stooq", pd.DataFrame(), symbol)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        return SourceFrame("Stooq", normalize_ohlcv(df, "Stooq"), symbol)
    except Exception as e:
        return SourceFrame("Stooq", pd.DataFrame(), f"{type(e).__name__}: {e}")


def load_yfinance_intraday(ticker: str, interval: str, period: str) -> SourceFrame:
    try:
        import yfinance as yf

        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty:
            return SourceFrame(f"yfinance {interval}", pd.DataFrame(), ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns={"Adj Close": "Adj Close"})
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "Adj Close"] if c in df.columns]
        df = df[keep].copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        try:
            df.index = pd.to_datetime(df.index).tz_convert("Asia/Seoul").tz_localize(None)
        except Exception:
            df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        return SourceFrame(f"yfinance {interval}", df, ticker)
    except Exception as e:
        return SourceFrame(f"yfinance {interval}", pd.DataFrame(), f"{type(e).__name__}: {e}")


def detect_name_market(code: str, fallback_name: str | None, end: date) -> tuple[str, str, str]:
    code = code.strip()
    if not (code.isdigit() and len(code) == 6):
        ticker = code.upper()
        return fallback_name or ticker, "US", ""

    name = fallback_name or code
    market = "KOSDAQ"
    yf_suffix = ".KQ"
    try:
        import requests

        url = f"https://m.stock.naver.com/api/stock/{code}/basic"
        js = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).json()
        if js.get("stockName"):
            name = js["stockName"]
        exchange_code = (js.get("stockExchangeType") or {}).get("code")
        exchange_name = js.get("stockExchangeName") or (js.get("stockExchangeType") or {}).get("name")
        if exchange_code == "KS" or exchange_name == "KOSPI":
            return name, "KOSPI", ".KS"
        if exchange_code == "KQ" or exchange_name == "KOSDAQ":
            return name, "KOSDAQ", ".KQ"
    except Exception:
        pass
    try:
        with suppress_external_output():
            from pykrx import stock

            krx_name = stock.get_market_ticker_name(code)
            if krx_name:
                name = krx_name
            for m, suffix in [("KOSDAQ", ".KQ"), ("KOSPI", ".KS")]:
                tickers = stock.get_market_ticker_list(ymd(end), market=m)
                if code in tickers:
                    return name, m, suffix
    except Exception:
        pass
    return name, market, yf_suffix


def add_indicators(df: pd.DataFrame, ma_periods: list[int]) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"].fillna(0)

    for p in ma_periods:
        out[f"MA{p}"] = close.rolling(p).mean()

    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    chikou = close.shift(-26)
    out["전환선"] = tenkan
    out["기준선"] = kijun
    out["선행스팬1"] = span_a
    out["선행스팬2"] = span_b
    out["후행스팬"] = chikou

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD신호"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD히스토그램"] = out["MACD"] - out["MACD신호"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    fast_k = (close - low14) / (high14 - low14).replace(0, np.nan) * 100
    out["StochK"] = fast_k.rolling(3).mean()
    out["StochD"] = out["StochK"].rolling(3).mean()
    out["WilliamsR"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)

    tp = (high + low + close) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True)
    out["CCI20"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    out["ROC12"] = close.pct_change(12) * 100

    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    out["BB중심"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["BB상단"] = out["BB중심"] + 2 * bb_std
    out["BB하단"] = out["BB중심"] - 2 * bb_std
    out["Donchian상단20"] = high.rolling(20).max()
    out["Donchian하단20"] = low.rolling(20).min()

    obv = [0.0]
    for i in range(1, len(out)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    out["OBV"] = obv

    raw_money_flow = tp * volume
    positive_flow = raw_money_flow.where(tp.diff() > 0, 0.0)
    negative_flow = raw_money_flow.where(tp.diff() < 0, 0.0)
    mfi_ratio = positive_flow.rolling(14).sum() / negative_flow.rolling(14).sum().replace(0, np.nan)
    out["MFI14"] = 100 - 100 / (1 + mfi_ratio)

    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    out["CMF20"] = (mf_multiplier * volume).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    out["거래량20평균"] = volume.rolling(20).mean()
    out["거래량비율20"] = volume / out["거래량20평균"].replace(0, np.nan)
    out["거래대금"] = close * volume

    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    return out


def ichimoku_position(row: pd.Series) -> str:
    c = row.get("Close", np.nan)
    a = row.get("선행스팬1", np.nan)
    b = row.get("선행스팬2", np.nan)
    if not np.isfinite(c) or not np.isfinite(a) or not np.isfinite(b):
        return "데이터 부족"
    upper, lower = max(a, b), min(a, b)
    if c > upper:
        return "구름 위"
    if c < lower:
        return "구름 아래"
    return "구름 안"


def ma_alignment(row: pd.Series, periods: list[int]) -> str:
    vals = [row.get(f"MA{p}", np.nan) for p in periods]
    vals = [float(v) for v in vals if np.isfinite(v)]
    if len(vals) < 3:
        return "데이터 부족"
    if all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
        return "정배열"
    if all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)):
        return "역배열"
    return "혼조"


def volume_profile(df: pd.DataFrame, bins: int = 24) -> pd.DataFrame:
    base = df.dropna(subset=["Close", "Volume"]).tail(120).copy()
    if base.empty:
        return pd.DataFrame(columns=["하단", "상단", "중심", "거래량"])
    lo, hi = base["Low"].min(), base["High"].max()
    if lo == hi:
        return pd.DataFrame(columns=["하단", "상단", "중심", "거래량"])
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    assigned = pd.cut(base["Close"], bins=edges, include_lowest=True, labels=False)
    rows = []
    for i, center in enumerate(centers):
        vol = float(base.loc[assigned == i, "Volume"].sum())
        rows.append({"하단": edges[i], "상단": edges[i + 1], "중심": center, "거래량": vol})
    return pd.DataFrame(rows).sort_values("거래량", ascending=False)


def nearby_profile_levels(profile: pd.DataFrame, current_price: float, limit: int = 3) -> dict[str, list[int]]:
    result = {"upper": [], "lower": []}
    if profile is None or profile.empty or not np.isfinite(current_price) or current_price <= 0:
        return result
    df = profile.copy()
    if "중심" not in df.columns:
        return result
    df["중심"] = pd.to_numeric(df["중심"], errors="coerce")
    df = df.dropna(subset=["중심"])
    df = df[(df["중심"] - current_price).abs() / current_price <= 0.25]
    if df.empty:
        return result
    upper = df[df["중심"] >= current_price].sort_values("중심").head(limit)
    lower = df[df["중심"] < current_price].sort_values("중심", ascending=False).head(limit)
    result["upper"] = [round_to_tick(float(v), "nearest") for v in upper["중심"]]
    result["lower"] = [round_to_tick(float(v), "nearest") for v in lower["중심"]]
    return result


def format_nearby_profile(profile: pd.DataFrame, current_price: float) -> str:
    levels = nearby_profile_levels(profile, current_price)
    upper = ", ".join(money(v) for v in levels["upper"]) or "데이터 부족"
    lower = ", ".join(money(v) for v in levels["lower"]) or "데이터 부족"
    return f"현재가 주변 상단 매물대: {upper}; 현재가 주변 하단 지지 매물대: {lower}"


def moving_average_comment(current_price: float, row: pd.Series) -> str:
    ma20 = last_valid(row, "MA20")
    ma60 = last_valid(row, "MA60")
    if not np.isfinite(current_price):
        return "현재가 데이터 부족으로 이동평균선 해석을 제한합니다."
    above_ma20 = np.isfinite(ma20) and current_price > ma20
    above_ma60 = np.isfinite(ma60) and current_price > ma60
    if above_ma20 and above_ma60:
        return "20일·60일선은 이미 회복했으며, 단기적으로 5일·10일선 지지 여부가 중요합니다."
    if above_ma20 and not above_ma60:
        return "20일선은 회복했지만 60일선 회복 확인이 필요합니다."
    return "20일선 아래에 있어 단기 추세 회복 확인이 필요합니다."


def last_valid(row: pd.Series, key: str) -> float:
    try:
        v = row[key]
        return float(v) if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def nearest_levels(df: pd.DataFrame, ind: pd.DataFrame) -> dict[str, Any]:
    row = ind.iloc[-1]
    basis = float(row["Close"])
    atr = last_valid(row, "ATR14")
    tick = krx_tick(basis)
    recent20_high = float(ind["High"].tail(20).max())
    recent20_low = float(ind["Low"].tail(20).min())
    recent10_high = float(ind["High"].tail(10).max())
    recent10_low = float(ind["Low"].tail(10).min())
    recent5_high = float(ind["High"].tail(5).max())
    recent5_low = float(ind["Low"].tail(5).min())
    recent60_high = float(ind["High"].tail(60).max())
    recent60_low = float(ind["Low"].tail(60).min())
    high52 = float(ind["High"].tail(252).max())
    low52 = float(ind["Low"].tail(252).min())

    profile = volume_profile(ind)
    profile_centers = profile["중심"].tolist() if not profile.empty else []
    supports = []
    resistances = []
    for key in ["MA5", "MA10", "MA20", "MA60", "MA120", "MA240", "기준선", "전환선", "BB하단", "Donchian하단20"]:
        v = last_valid(row, key)
        if np.isfinite(v):
            (supports if v < basis else resistances).append(v)
    for v in [recent5_low, recent10_low, recent20_low, recent60_low, low52] + profile_centers:
        if np.isfinite(v) and v < basis:
            supports.append(v)
    for v in [
        recent5_high,
        recent10_high,
        recent20_high,
        recent60_high,
        high52,
        last_valid(row, "BB상단"),
        last_valid(row, "Donchian상단20"),
    ] + profile_centers:
        if np.isfinite(v) and v > basis:
            resistances.append(v)

    supports = sorted({round_to_tick(x, "nearest") for x in supports if x > 0 and x < basis * 0.999}, reverse=True)
    resistances = sorted({round_to_tick(x, "nearest") for x in resistances if x > basis * 1.001})

    nearest_support = supports[0] if supports else round_to_tick(basis - atr, "down")
    deeper_support = supports[1] if len(supports) > 1 else round_to_tick(basis - 1.5 * atr, "down")
    immediate_resistance = resistances[0] if resistances else round_to_tick(basis + atr, "up")
    meaningful_resistances = [r for r in resistances if r >= basis * 1.03]
    nearest_resistance = meaningful_resistances[0] if meaningful_resistances else immediate_resistance
    higher_key_resistances = [r for r in resistances if r > nearest_resistance * 1.02]
    next_resistance = higher_key_resistances[0] if higher_key_resistances else round_to_tick(
        nearest_resistance + max(atr, basis * 0.05), "up"
    )

    breakout = round_to_tick(nearest_resistance + tick, "up")
    target1_candidates = [r for r in resistances if r >= breakout * 1.025]
    target1 = target1_candidates[0] if target1_candidates else round_to_tick(max(nearest_resistance, breakout + max(atr * 0.35, basis * 0.035)), "up")
    if target1 <= breakout:
        target1 = round_to_tick(breakout + max(atr * 0.35, basis * 0.035), "up")
    higher_resistances = [r for r in resistances if r > target1 * 1.08]
    if higher_resistances:
        target2 = higher_resistances[0]
    else:
        target2 = round_to_tick(target1 + max(atr, basis * 0.05), "up")
    warning = round_to_tick(max(nearest_support, min(row.get("MA10", nearest_support), basis)), "down")
    defense_candidates = [deeper_support]
    ma20 = last_valid(row, "MA20")
    if np.isfinite(ma20):
        defense_candidates.append(ma20)
    if np.isfinite(atr):
        defense_candidates.append(basis - 1.2 * atr)
    defense = round_to_tick(min([x for x in defense_candidates if np.isfinite(x) and x < basis]), "down")

    # 눌림목은 지지대 확인을 전제로 하므로 한 가격보다 범위가 더 실전적이다.
    if np.isfinite(atr):
        pull_low = round_to_tick(max(defense + tick, nearest_support - 0.35 * atr), "down")
        pull_high = round_to_tick(min(basis - tick, nearest_support + 0.15 * atr), "nearest")
    else:
        pull_low = round_to_tick(nearest_support, "down")
        pull_high = round_to_tick(nearest_support, "nearest")
    if pull_high >= basis:
        pull_high = round_to_tick(basis - tick, "down")
    if pull_low > pull_high:
        pull_low, pull_high = pull_high, pull_low

    return {
        "profile": profile,
        "supports": supports[:6],
        "resistances": resistances[:6],
        "nearest_support": nearest_support,
        "deeper_support": deeper_support,
        "immediate_resistance": immediate_resistance,
        "nearest_resistance": nearest_resistance,
        "next_resistance": next_resistance,
        "pull_low": pull_low,
        "pull_high": pull_high,
        "breakout": breakout,
        "target1": target1,
        "target2": target2,
        "warning": warning,
        "defense": defense,
        "recent20_high": round_to_tick(recent20_high),
        "recent20_low": round_to_tick(recent20_low),
        "recent10_high": round_to_tick(recent10_high),
        "recent10_low": round_to_tick(recent10_low),
        "recent5_high": round_to_tick(recent5_high),
        "recent5_low": round_to_tick(recent5_low),
        "recent60_high": round_to_tick(recent60_high),
        "recent60_low": round_to_tick(recent60_low),
        "high52": round_to_tick(high52),
        "low52": round_to_tick(low52),
    }


def source_validation(sources: list[SourceFrame], end_limit: date) -> tuple[pd.DataFrame, str, bool]:
    rows = []
    for src in sources:
        is_core = src.name in {"pykrx", "FinanceDataReader"}
        if src.data is None or src.data.empty:
            rows.append(
                {
                    "소스": src.name,
                    "최신거래일": "수집 실패",
                    "시가": "",
                    "고가": "",
                    "저가": "",
                    "종가": "",
                    "거래량": "",
                    "검증유형": "핵심 소스 수집 실패" if is_core else "보조 소스 수집 실패",
                    "대표가격사용": "아니오",
                    "비고": src.note,
                }
            )
            continue
        data = src.data[src.data.index.date <= end_limit]
        if data.empty:
            rows.append(
                {
                    "소스": src.name,
                    "최신거래일": "기준일 이전 데이터 없음",
                    "시가": "",
                    "고가": "",
                    "저가": "",
                    "종가": "",
                    "거래량": "",
                    "검증유형": "핵심 소스 수집 실패" if is_core else "보조 소스 수집 실패",
                    "대표가격사용": "아니오",
                    "비고": src.note,
                }
            )
            continue
        last = data.iloc[-1]
        rows.append(
            {
                "소스": src.name,
                "최신거래일": iso(data.index[-1]),
                "시가": int(last["Open"]),
                "고가": int(last["High"]),
                "저가": int(last["Low"]),
                "종가": int(last["Close"]),
                "거래량": int(last["Volume"]) if np.isfinite(last["Volume"]) else "",
                "검증유형": "검증 대기",
                "대표가격사용": "검토",
                "비고": src.note,
            }
        )

    table = pd.DataFrame(rows)
    usable = table[pd.to_numeric(table["종가"], errors="coerce").notna()].copy()
    reliability = "낮음"
    stop_precision = False
    if usable.empty:
        return table, reliability, True

    for idx in table.index:
        if table.at[idx, "검증유형"] == "검증 대기":
            table.at[idx, "검증유형"] = "보조 확인" if table.at[idx, "소스"] == "yfinance" else "핵심 확인"

    def row_date(row: pd.Series) -> date | None:
        try:
            return datetime.fromisoformat(str(row["최신거래일"])).date()
        except Exception:
            return None

    usable["_date"] = usable.apply(row_date, axis=1)
    latest_date = max([d for d in usable["_date"] if d is not None], default=None)
    latest_rows = usable[usable["_date"] == latest_date].copy() if latest_date is not None else usable.iloc[0:0].copy()
    core = usable[usable["소스"].isin(["pykrx", "FinanceDataReader"])].copy()
    core_latest = core[core["_date"] == latest_date].copy() if latest_date is not None else core.iloc[0:0].copy()

    def price_volume_diff(frame: pd.DataFrame) -> tuple[float, float]:
        if len(frame) < 2:
            return np.nan, np.nan
        price_diffs: list[float] = []
        for col in ["시가", "고가", "저가", "종가"]:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if len(values) >= 2 and values.mean():
                price_diffs.append((values.max() - values.min()) / values.mean() * 100)
        vols = pd.to_numeric(frame["거래량"], errors="coerce")
        price_diff = max(price_diffs) if price_diffs else np.nan
        vol_diff = (vols.max() - vols.min()) / vols.mean() * 100 if vols.notna().sum() >= 2 and vols.mean() else np.nan
        return price_diff, vol_diff

    core_close_diff, core_vol_diff = price_volume_diff(core_latest)
    latest_close_diff, latest_vol_diff = price_volume_diff(latest_rows)
    core_agree = (
        len(core_latest) >= 2
        and np.isfinite(core_close_diff)
        and core_close_diff <= 0.5
        and (not np.isfinite(core_vol_diff) or core_vol_diff <= 10)
    )
    latest_agree = (
        len(latest_rows) >= 2
        and np.isfinite(latest_close_diff)
        and latest_close_diff <= 0.5
        and (not np.isfinite(latest_vol_diff) or latest_vol_diff <= 10)
    )
    stale_mask = usable["_date"].notna() & (usable["_date"] < latest_date) if latest_date is not None else pd.Series(False, index=usable.index)
    stale_sources = set(usable.loc[stale_mask, "소스"].astype(str))
    only_yfinance_stale = bool(stale_sources) and stale_sources <= {"yfinance"}

    if core_agree:
        representative_sources = set(core_latest["소스"].astype(str))
        reliability = "중간" if only_yfinance_stale else "높음"
        stop_precision = False
    elif latest_agree:
        representative_sources = set(latest_rows["소스"].astype(str))
        reliability = "중간" if stale_sources else "높음"
        stop_precision = False
    else:
        representative_sources = set()
        reliability = "낮음"
        stop_precision = True

    unique_dates = {d for d in usable["_date"] if d is not None}
    latest_mismatch_type = "가격/최신거래일 불일치"
    if len(latest_rows) >= 2 and np.isfinite(latest_close_diff) and latest_close_diff > 0.5:
        latest_mismatch_type = "동일 거래일 가격 불일치"
    elif len(latest_rows) >= 2 and np.isfinite(latest_vol_diff) and latest_vol_diff > 10:
        latest_mismatch_type = "동일 거래일 거래량 불일치"
    elif len(unique_dates) > 1:
        latest_mismatch_type = "최신거래일 불일치"

    for idx, row in table.iterrows():
        source = str(row.get("소스", ""))
        if source in representative_sources:
            table.at[idx, "대표가격사용"] = "예"
        elif source == "yfinance" and source in stale_sources:
            table.at[idx, "검증유형"] = "보조 소스 최신거래일 지연"
            table.at[idx, "대표가격사용"] = "제외"
            table.at[idx, "비고"] = one_line(f"{row.get('비고', '')} 대표 가격 산정에서 제외")
        elif source in stale_sources:
            table.at[idx, "검증유형"] = "보조 소스 최신거래일 지연"
            table.at[idx, "대표가격사용"] = "제외" if not stop_precision else "아니오"
        elif str(row.get("최신거래일")) in {"수집 실패", "기준일 이전 데이터 없음"}:
            table.at[idx, "대표가격사용"] = "아니오"
        elif stop_precision:
            table.at[idx, "검증유형"] = latest_mismatch_type
            table.at[idx, "대표가격사용"] = "아니오"
        else:
            table.at[idx, "대표가격사용"] = "보조"

    if len(usable) >= 2:
        pass
    elif len(usable) == 1:
        reliability = "낮음"
        stop_precision = True
    return table.drop(columns=[c for c in ["_date"] if c in table.columns], errors="ignore"), reliability, stop_precision


def validation_labels(validation: pd.DataFrame) -> tuple[str, str, str]:
    usable = validation[pd.to_numeric(validation["종가"], errors="coerce").notna()].copy()
    if len(usable) < 2:
        return "실패", "실패", "수집 가능한 비교 소스가 2개 미만입니다."
    representative = usable[usable["대표가격사용"] == "예"].copy() if "대표가격사용" in usable.columns else usable
    compare = representative if len(representative) >= 2 else usable
    closes = pd.to_numeric(compare["종가"])
    vols = pd.to_numeric(compare["거래량"], errors="coerce")
    close_diff = (closes.max() - closes.min()) / closes.mean() * 100 if closes.mean() else np.nan
    vol_diff = (vols.max() - vols.min()) / vols.mean() * 100 if vols.notna().sum() >= 2 and vols.mean() else np.nan
    dates = compare["최신거래일"].unique()
    stale_aux = "검증유형" in usable.columns and usable["검증유형"].astype(str).str.contains("보조 소스 최신거래일 지연", na=False).any()
    price_label = "실패" if close_diff > 1.0 else ("경고" if close_diff > 0.5 or len(dates) > 1 else "통과")
    vol_label = "경고" if np.isfinite(vol_diff) and vol_diff > 10 else "통과"
    note = f"대표 가격 소스 가격 차이 {close_diff:.2f}%, 거래량 차이 {vol_diff:.2f}%"
    if stale_aux:
        note += ", yfinance 보조 소스 최신거래일 지연으로 대표 가격 산정에서 제외"
    elif len(usable["최신거래일"].unique()) > 1:
        note += ", 최신 거래일 불일치"
    return price_label, vol_label, note


def naver_investor_table(code: str) -> dict[str, Any]:
    url = f"https://finance.naver.com/item/frgn.naver?code={code}&page=1"
    result: dict[str, Any] = {"url": url, "status": "데이터 부족"}
    errors: list[str] = []
    try:
        dfs = pd.read_html(url, encoding="euc-kr")
        candidates = []
        for df in dfs:
            flat_cols = [
                " ".join(str(part) for part in col if str(part) != "nan")
                if isinstance(col, tuple)
                else str(col)
                for col in df.columns
            ]
            if any("외국인" in c for c in flat_cols) and len(df) > 0:
                candidates.append(df)
        if not candidates:
            raise ValueError("네이버 투자자별 매매동향 표 없음")
        df = candidates[0].dropna(how="all").copy()
        result["raw"] = df.head(10)
        result["status"] = "일별 외국인 보유/순매매 표 확인"
        return result
    except Exception as e:
        errors.append(f"네이버 투자자별 매매동향 실패: {type(e).__name__}")

    try:
        with suppress_external_output():
            from pykrx import stock

            end = today_kst().date()
            start = end - timedelta(days=21)
            df = stock.get_market_trading_volume_by_date(ymd(start), ymd(end), code)
        if df is None or df.empty:
            raise ValueError("pykrx 투자자별 순매수 데이터 없음")
        recent = df.tail(5).copy()
        result["raw"] = recent
        foreign_cols = [c for c in recent.columns if "외국인" in str(c)]
        inst_cols = [c for c in recent.columns if "기관" in str(c)]
        retail_cols = [c for c in recent.columns if "개인" in str(c)]
        parts = ["pykrx 투자자별 순매수 확인"]
        if foreign_cols:
            parts.append(f"최근 5거래일 외국인 {shares(pd.to_numeric(recent[foreign_cols[0]], errors='coerce').sum())}")
        if inst_cols:
            parts.append(f"기관 {shares(pd.to_numeric(recent[inst_cols[0]], errors='coerce').sum())}")
        if retail_cols:
            parts.append(f"개인 {shares(pd.to_numeric(recent[retail_cols[0]], errors='coerce').sum())}")
        result["status"] = ", ".join(parts)
        return result
    except Exception as e:
        errors.append(f"pykrx 투자자별 순매수 실패: {type(e).__name__}")

    result["status"] = "수급 데이터 부족으로 수급 판단 보류"
    result["errors"] = errors
    return result


def relative_returns(stock_df: pd.DataFrame, index_df: pd.DataFrame) -> dict[str, float]:
    out = {}
    if stock_df.empty or index_df.empty:
        return out
    s = stock_df["Close"].dropna()
    i = index_df["Close"].dropna()
    common = pd.concat([s, i], axis=1, join="inner").dropna()
    common.columns = ["stock", "index"]
    for p in [20, 60, 120]:
        if len(common) > p:
            out[f"{p}일 종목"] = pct(common["stock"].iloc[-1], common["stock"].iloc[-p - 1])
            out[f"{p}일 지수"] = pct(common["index"].iloc[-1], common["index"].iloc[-p - 1])
            out[f"{p}일 초과"] = out[f"{p}일 종목"] - out[f"{p}일 지수"]
    return out


def domestic_index_symbol(market: str) -> str:
    return "KS11" if market == "KOSPI" else "KQ11"


def domestic_index_range(market: str) -> tuple[float, float]:
    return (1000.0, 6000.0) if market == "KOSPI" else (300.0, 2000.0)


def market_index_value_is_valid(market: str, value: Any) -> bool:
    v = _finite_or_none(value)
    if v is None:
        return False
    lo, hi = domestic_index_range(market)
    return lo <= v <= hi


def market_index_frame_is_valid(market: str, df: pd.DataFrame) -> bool:
    if df is None or df.empty or "Close" not in df.columns:
        return False
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return False
    return market_index_value_is_valid(market, close.iloc[-1])


def load_kosdaq_index(start: date, end: date) -> SourceFrame:
    try:
        with suppress_external_output():
            from pykrx import stock
            df = stock.get_index_ohlcv_by_date(ymd(start), ymd(end), "2001")
        out = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
        return SourceFrame("pykrx KOSDAQ", normalize_ohlcv(out, "pykrx KOSDAQ"))
    except Exception as e:
        return SourceFrame("pykrx KOSDAQ", pd.DataFrame(), f"{type(e).__name__}: {e}")


def load_krx_index_by_market(market: str, start: date, end: date) -> SourceFrame:
    index_symbol = domestic_index_symbol(market)
    src = load_fdr(index_symbol, start, end)
    if market_index_frame_is_valid(market, src.data):
        src.name = f"FinanceDataReader {index_symbol}"
        src.note = index_symbol
        return src
    try:
        with suppress_external_output():
            from pykrx import stock

            index_code = "1001" if market == "KOSPI" else "2001"
            df = stock.get_index_ohlcv_by_date(ymd(start), ymd(end), index_code)
        out = df.rename(columns={"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"})
        name = "pykrx KOSPI" if market == "KOSPI" else "pykrx KOSDAQ"
        normalized = normalize_ohlcv(out, name)
        if market_index_frame_is_valid(market, normalized):
            return SourceFrame(name, normalized, index_code)
        return SourceFrame(f"{market} index", pd.DataFrame(), "시장 지수 값이 정상 범위를 벗어남")
    except Exception as e:
        return SourceFrame(f"{market} index", pd.DataFrame(), f"{type(e).__name__}: {e}")


def load_market_index(market: str, start: date, end: date) -> SourceFrame:
    if market in {"KOSPI", "KOSDAQ"}:
        return load_krx_index_by_market(market, start, end)
    src = load_yfinance("^IXIC", start, end, "yfinance Nasdaq")
    if not src.data.empty:
        return src
    return load_yfinance("^GSPC", start, end, "yfinance S&P 500")


def us_sector_etf_for_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker in {"NVDA", "AMD", "AVGO", "QCOM", "INTC", "MU", "TSM"}:
        return "SMH"
    if ticker in {"TSLA", "AMZN", "NKE", "SBUX"}:
        return "XLY"
    if ticker in {"META", "GOOGL", "GOOG", "NFLX", "DIS"}:
        return "XLC"
    if ticker in {"JPM", "BAC", "GS", "MS", "WFC"}:
        return "XLF"
    if ticker in {"XOM", "CVX", "COP"}:
        return "XLE"
    if ticker in {"UNH", "LLY", "PFE", "MRK", "JNJ"}:
        return "XLV"
    return "QQQ"


def load_us_market_refs(ticker: str, stock_df: pd.DataFrame, start: date, end: date) -> dict[str, dict[str, float]]:
    refs = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        f"섹터 ETF {us_sector_etf_for_ticker(ticker)}": us_sector_etf_for_ticker(ticker),
    }
    result: dict[str, dict[str, float]] = {}
    for name, symbol in refs.items():
        src = load_yfinance(symbol, start, end, f"yfinance {name}")
        vals = relative_returns(stock_df, src.data)
        if vals:
            result[name] = vals
    return result


def load_peer_returns(start: date, end: date, code: str | None = None, stock_name: str = "") -> dict[str, dict[str, float]]:
    normalized_code = str(code or "").zfill(6)
    if normalized_code == "403870" or "HPSP" in str(stock_name).upper():
        peers = {
            "테스 095610": "095610",
            "원익IPS 240810": "240810",
            "주성엔지니어링 036930": "036930",
            "피에스케이 319660": "319660",
        }
    elif normalized_code == "010140" or "중공업" in str(stock_name) or "조선" in str(stock_name):
        peers = {
            "HD현대중공업 329180": "329180",
            "한화오션 042660": "042660",
            "HD한국조선해양 009540": "009540",
            "HD현대미포 010620": "010620",
        }
    elif normalized_code == "009150":
        peers = {
            "LG이노텍 011070": "011070",
            "대덕전자 353200": "353200",
            "해성디에스 195870": "195870",
            "삼성전자 005930": "005930",
        }
    elif normalized_code == "033100" or "전기" in str(stock_name) or "일렉트릭" in str(stock_name):
        peers = {
            "HD현대일렉트릭 267260": "267260",
            "LS ELECTRIC 010120": "010120",
            "효성중공업 298040": "298040",
            "LS 006260": "006260",
        }
    else:
        peers = {}
    result = {}
    for name, code in peers.items():
        src = load_fdr(code, start, end)
        if src.data.empty:
            continue
        s = src.data["Close"].dropna()
        vals = {}
        for p in [20, 60]:
            if len(s) > p:
                vals[f"{p}일"] = pct(s.iloc[-1], s.iloc[-p - 1])
        result[name] = vals
    return result


def format_peer_returns(peer_returns: dict[str, dict[str, float]]) -> str:
    if not peer_returns:
        return "동종업종 비교 데이터 부족"
    parts = []
    for name, vals in peer_returns.items():
        v20 = fpct(vals.get("20일"))
        v60 = fpct(vals.get("60일"))
        parts.append(f"{name}: 20일 {v20}, 60일 {v60}")
    return "; ".join(parts)


def format_relative_ref(vals: dict[str, float] | None) -> str:
    if not vals:
        return "데이터 부족"
    return (
        f"20일 지수 {fpct(vals.get('20일 지수'))}, 종목 {fpct(vals.get('20일 종목'))}, 초과 {fpct(vals.get('20일 초과'))}; "
        f"60일 지수 {fpct(vals.get('60일 지수'))}, 종목 {fpct(vals.get('60일 종목'))}, 초과 {fpct(vals.get('60일 초과'))}"
    )


def decision_logic(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    levels: dict[str, Any],
    reliability: str,
    stop_precision: bool,
    market_rel: dict[str, float],
    validation_note: str = "",
    price_label: str = "",
    volume_label: str = "",
    supply_status: str = "",
) -> dict[str, Any]:
    row = daily.iloc[-1]
    prev = daily.iloc[-2] if len(daily) > 1 else row
    basis = float(row["Close"])
    target1 = levels["target1"]
    target2 = levels["target2"]
    defense = levels["defense"]
    warning = levels["warning"]
    pull_low = levels["pull_low"]
    pull_high = levels["pull_high"]
    breakout = levels["breakout"]
    pull_high = separate_buy_high_from_breakout(pull_high, breakout)
    if pull_low > pull_high:
        pull_low = pull_high
    rebreak_line = round_to_tick(max(levels.get("recent5_high", breakout), basis), "nearest")
    price_context = classify_price_context(basis, pull_low, pull_high, rebreak_line, breakout, target1, defense)

    risk = downside_risk_pct(basis, defense)
    reward1 = pct(target1, basis)
    reward2 = pct(target2, basis)
    rr1 = reward1 / risk if risk and np.isfinite(risk) and risk > 0 else np.nan
    rr2 = reward2 / risk if risk and np.isfinite(risk) and risk > 0 else np.nan

    ma_pos = ma_alignment(row, [5, 10, 20, 60, 120, 240])
    daily_cloud = ichimoku_position(row)
    weekly_cloud = ichimoku_position(weekly.iloc[-1]) if not weekly.empty else "데이터 부족"
    monthly_cloud = ichimoku_position(monthly.iloc[-1]) if not monthly.empty else "데이터 부족"
    rsi = last_valid(row, "RSI14")
    macd_hist = last_valid(row, "MACD히스토그램")
    macd_hist_prev = last_valid(prev, "MACD히스토그램")
    vol_ratio = last_valid(row, "거래량비율20")
    bb_upper = last_valid(row, "BB상단")
    bb_mid = last_valid(row, "BB중심")
    atr = last_valid(row, "ATR14")
    recovery_text, recovery_line = recovery_confirmation_level(
        basis,
        pull_high,
        last_valid(row, "MA20"),
        bb_mid,
        breakout,
    )
    confirm_entry_line = recovery_line if _finite_or_none(recovery_line) is not None else breakout
    confirm_reward = pct(target1, confirm_entry_line)
    confirm_risk = downside_risk_pct(confirm_entry_line, defense)
    confirm_rr = confirm_reward / confirm_risk if np.isfinite(confirm_risk) and confirm_risk > 0 else np.nan
    trade_state = build_trade_state(
        current_price=basis,
        pullback_low=pull_low,
        pullback_high=pull_high,
        recovery_line=recovery_line,
        breakout_line=breakout,
        target1=target1,
        target2=target2,
        defense_line=defense,
        short_rebreak_line=rebreak_line,
        open_price=last_valid(row, "Open"),
        high_price=last_valid(row, "High"),
        low_price=last_valid(row, "Low"),
        close_price=basis,
        volume_ratio20=vol_ratio,
        macd=last_valid(row, "MACD"),
        macd_signal=last_valid(row, "MACD신호"),
        macd_hist=macd_hist,
        rsi=rsi,
        entry_rr=confirm_rr,
        swing_rr=rr1,
        intraday_rr=None,
        price_label=price_label,
        volume_label=volume_label,
        validation_note=validation_note,
        reliability=reliability,
        supply_status=supply_status,
        stop_precision=stop_precision,
        intraday_mode=False,
        close_confirmed=True,
        completed_daily=True,
        trend_state=ma_pos,
    )
    state_actions = render_trade_state_actions(
        trade_state.final_action_state,
        {
            "recovery": money(recovery_line) if _finite_or_none(recovery_line) is not None else money(breakout),
            "pullback": format_price_range(pull_low, pull_high, "단일 지지선"),
            "target1": money(target1),
            "defense": money(defense),
        },
    )

    above_ma20 = basis > last_valid(row, "MA20") if np.isfinite(last_valid(row, "MA20")) else False
    above_ma60 = basis > last_valid(row, "MA60") if np.isfinite(last_valid(row, "MA60")) else False
    macd_improving = np.isfinite(macd_hist) and np.isfinite(macd_hist_prev) and macd_hist >= macd_hist_prev
    rsi_ok = np.isfinite(rsi) and rsi >= 50
    overheat = (np.isfinite(rsi) and rsi >= 70) or (np.isfinite(bb_upper) and basis >= bb_upper * 0.98)
    market_weak = market_rel.get("20일 초과", 0) < -3 and market_rel.get("60일 초과", 0) < -3

    if stop_precision:
        final = "데이터 불일치로 정밀 판단 중단"
        now_buy = "불가"
        reason = "일봉 가격 데이터 검증 실패"
    elif reliability == "낮음":
        final = "데이터 부족으로 분석 제한"
        now_buy = "불가"
        reason = "데이터 신뢰도가 낮아 공격 매수 판단 금지"
    elif "보조 소스" in validation_note and "지연" in validation_note:
        final = "보조 소스 지연으로 보수 판단"
        now_buy = "불가"
        reason = "pykrx/FDR 대표가격은 일치하지만 yfinance 최신거래일이 지연되어 보수 판단"
    elif basis < defense:
        final = "방어 매도 필요"
        now_buy = "불가"
        reason = "기준가가 방어선 아래"
    elif rr1 >= 1.5 and above_ma20 and above_ma60 and daily_cloud == "구름 위" and macd_improving and rsi_ok and vol_ratio >= 1.0 and not overheat:
        final = "공격 매수 가능"
        now_buy = "가능"
        reason = "추세, 구름, 모멘텀, 거래량, 손익비가 동시에 양호"
    elif rr1 < 1.5:
        final = "지금은 대기"
        now_buy = "불가"
        reason = "현재 기준 1차 목표 대비 손익비가 1.5 미만"
    elif overheat:
        final = "눌림목 대기"
        now_buy = "불가"
        reason = "단기 과열 부담으로 눌림목 회복 또는 지지 확인 필요"
    elif daily_cloud != "구름 위" and above_ma20:
        final = "돌파 재확인 대기"
        now_buy = "불가"
        reason = "주요 저항 돌파 확인 전까지 추세 신뢰도 제한"
    elif market_weak:
        final = "추가매수 금지"
        now_buy = "불가"
        reason = "시장 대비 상대강도가 약해 신규 매수 우위 낮음"
    else:
        final = "눌림목 대기"
        now_buy = "불가"
        reason = f"주요 지지 재확인 또는 {money(breakout)} 돌파 확인 필요"

    if final == "눌림목 대기" and not overheat and macd_improving and rr2 >= 1.5 and vol_ratio >= 1.2:
        reason = "상승 동력은 있으나 1차 저항이 가까워 눌림 확인 후 손익비 개선 필요"

    strategy_labels = strategy_labels_by_price(
        basis,
        pull_low,
        pull_high,
        rebreak_line,
        breakout,
        warning,
        defense,
        reliability,
    )
    primary_strategy = strategy_labels["primary_strategy"]
    secondary_strategy = breakout_sentence(price_context)
    if stop_precision:
        primary_strategy = "데이터 확인 대기"
        final = "데이터 불일치로 정밀 판단 중단"
    elif reliability == "낮음":
        final = "데이터 부족으로 분석 제한"
    elif now_buy == "가능" and final == "공격 매수 가능":
        primary_strategy = "조건 충족 시 분할 매수"
    elif now_buy == "불가" and final not in {"방어 매도 필요", "추가매수 금지", "보조 소스 지연으로 보수 판단"}:
        final = strategy_labels["final"]

    final = state_actions["final_judgment"]
    now_buy = state_actions["now_buy"]
    reason = state_actions["no_buy_reason"]
    primary_strategy = state_actions["primary_strategy"]

    return {
        "기준가": basis,
        "최종판단": final,
        "지금바로매수": now_buy,
        "주전략": primary_strategy,
        "보조전략": secondary_strategy,
        "지금매수하지않는이유": reason if now_buy == "불가" else "조건 충족",
        "눌림목": format_price_range(pull_low, pull_high, "단일 지지선"),
        "눌림하단": pull_low,
        "눌림상단": pull_high,
        "단기재돌파확인선": rebreak_line,
        "회복확인가": recovery_text,
        "회복확인선": recovery_line,
        "확인진입손익비": confirm_rr,
        "회복/돌파공통확인선": breakout if recovery_line is not None and same_price_level(recovery_line, breakout) else None,
        "눌림조건": "거래량 감소 조정 후 일봉 종가가 지지대 위에서 마감, RSI 40~55 회복",
        "돌파": money(breakout),
        "돌파가격": breakout,
        "돌파조건": "일봉 종가 기준 돌파, 거래량 20일 평균 1.2배 이상, MACD 히스토그램 증가",
        "1차목표": target1,
        "2차목표": target2,
        "주의선": warning,
        "방어선": defense,
        "전량이탈": f"일봉 종가 {money(defense)} 이탈 후 다음 거래일 회복 실패",
        "예상수익률1": reward1,
        "예상수익률2": reward2,
        "하락위험률": risk,
        "손익비1": rr1,
        "손익비2": rr2,
        "MA배열": ma_pos,
        "일봉구름": daily_cloud,
        "주봉구름": weekly_cloud,
        "월봉구름": monthly_cloud,
        "RSI": rsi,
        "MACD개선": macd_improving,
        "거래량비율": vol_ratio,
        "ATR": atr,
        "과열": overheat,
        "상태코드": trade_state_to_dict(trade_state),
        "상태코드표": state_code_report_rows(trade_state),
        "상태코드잠금": True,
        "final_action_state": trade_state.final_action_state,
        "new_buyer_action": trade_state.new_buyer_action,
        "holder_action": trade_state.holder_action,
        "add_buyer_action": trade_state.add_buyer_action,
        "stop_loss_action": trade_state.stop_loss_action,
    }


def md_table(rows: list[tuple[str, Any]], headers: tuple[str, str] = ("항목", "값")) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "| --- | --- |"]
    for a, b in rows:
        lines.append(f"| {a} | {b} |")
    return "\n".join(lines)


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "데이터 부족"
    return df.to_markdown(index=False)


def compact_validation_md(validation: pd.DataFrame) -> str:
    out = validation.copy()
    for col in ["시가", "고가", "저가", "종가", "거래량"]:
        out[col] = out[col].apply(lambda x: f"{int(x):,}" if str(x).strip() and pd.notna(x) else "")
    return out.to_markdown(index=False)


def make_charts(
    out_dir: Path,
    stock_name: str,
    code: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    levels: dict[str, Any],
    intraday60: pd.DataFrame,
    intraday15: pd.DataFrame,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    paths: list[Path] = []

    def save(fig, filename: str) -> None:
        path = out_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    def plot_ichimoku(df: pd.DataFrame, title: str, filename: str) -> None:
        tail = df.tail(80)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(tail.index, tail["Close"], label="종가", color="#111827", linewidth=1.6)
        for c, label, color in [("전환선", "전환선", "#2563eb"), ("기준선", "기준선", "#dc2626")]:
            if c in tail:
                ax.plot(tail.index, tail[c], label=label, color=color, linewidth=1)
        if "선행스팬1" in tail and "선행스팬2" in tail:
            ax.fill_between(
                tail.index,
                tail["선행스팬1"].astype(float).values,
                tail["선행스팬2"].astype(float).values,
                color="#9ca3af",
                alpha=0.25,
                label="구름",
            )
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        save(fig, filename)

    plot_ichimoku(monthly, f"{stock_name} {code} 월봉 일목", f"01_{stock_name}_{code}_월봉_일목.png")
    plot_ichimoku(weekly, f"{stock_name} {code} 주봉 일목", f"02_{stock_name}_{code}_주봉_일목.png")

    tail = daily.tail(160)
    fig, (ax, av) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(tail.index, tail["Close"], label="종가", color="#111827", linewidth=1.6)
    for p, color in [(5, "#0284c7"), (10, "#16a34a"), (20, "#f59e0b"), (60, "#7c3aed"), (120, "#64748b"), (240, "#0f766e")]:
        col = f"MA{p}"
        if col in tail:
            ax.plot(tail.index, tail[col], label=f"{p}일선", linewidth=0.95, color=color)
    for label, price, color in [
        ("기준가", levels.get("basis"), "#111827"),
        ("주의선", levels["warning"], "#f97316"),
        ("방어선", levels["defense"], "#ef4444"),
        ("1차 목표", levels["target1"], "#059669"),
        ("2차 목표", levels["target2"], "#0d9488"),
    ]:
        if price:
            ax.axhline(price, linestyle="--", linewidth=0.9, color=color, alpha=0.8)
    av.bar(tail.index, tail["Volume"], color=np.where(tail["Close"].diff() >= 0, "#ef4444", "#2563eb"), alpha=0.65)
    av.plot(tail.index, tail["거래량20평균"], color="#111827", linewidth=1, label="20일 평균 거래량")
    ax.set_title(f"{stock_name} {code} 일봉 이동평균/거래량")
    ax.grid(True, alpha=0.25)
    av.grid(True, alpha=0.2)
    ax.legend(ncol=4, fontsize=8)
    av.legend(fontsize=8)
    save(fig, f"03_{stock_name}_{code}_일봉_이동평균_거래량.png")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    ax1.plot(tail.index, tail["Close"], color="#111827", label="종가")
    ax1.plot(tail.index, tail["BB상단"], color="#dc2626", linewidth=0.8, label="볼린저 상단")
    ax1.plot(tail.index, tail["BB중심"], color="#64748b", linewidth=0.8, label="볼린저 중심")
    ax1.plot(tail.index, tail["BB하단"], color="#2563eb", linewidth=0.8, label="볼린저 하단")
    ax2.plot(tail.index, tail["MACD"], color="#2563eb", label="MACD")
    ax2.plot(tail.index, tail["MACD신호"], color="#dc2626", label="신호")
    ax2.bar(tail.index, tail["MACD히스토그램"], color=np.where(tail["MACD히스토그램"] >= 0, "#ef4444", "#2563eb"), alpha=0.6)
    ax3.plot(tail.index, tail["RSI14"], color="#7c3aed", label="RSI")
    ax3.axhline(70, color="#ef4444", linestyle="--", linewidth=0.8)
    ax3.axhline(50, color="#64748b", linestyle="--", linewidth=0.8)
    ax3.axhline(30, color="#2563eb", linestyle="--", linewidth=0.8)
    ax1.set_title(f"{stock_name} {code} 일봉 MACD/RSI/거래량")
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    save(fig, f"04_{stock_name}_{code}_일봉_MACD_RSI_거래량.png")

    profile = levels["profile"].head(12).sort_values("중심")
    fig, ax = plt.subplots(figsize=(9, 7))
    if not profile.empty:
        ax.barh(profile["중심"], profile["거래량"], height=(profile["상단"] - profile["하단"]) * 0.82, color="#94a3b8")
        ax.axhline(float(daily.iloc[-1]["Close"]), color="#111827", linestyle="--", label="기준가")
    ax.set_title(f"{stock_name} {code} 일봉 매물대")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, f"05_{stock_name}_{code}_일봉_매물대.png")

    def plot_intraday(df: pd.DataFrame, interval_name: str, filename: str) -> None:
        if df.empty:
            return
        t = add_indicators(df.copy(), [5, 10, 20, 60]).tail(140)
        fig, (ax, av) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(t.index, t["Close"], label="종가", color="#111827")
        for p, color in [(5, "#0284c7"), (20, "#f59e0b"), (60, "#7c3aed")]:
            col = f"MA{p}"
            if col in t:
                ax.plot(t.index, t[col], label=f"{p}봉선", color=color, linewidth=0.9)
        try:
            vwap = (t["Close"] * t["Volume"]).cumsum() / t["Volume"].cumsum().replace(0, np.nan)
            ax.plot(t.index, vwap, color="#0f766e", linewidth=1, label="VWAP")
        except Exception:
            pass
        av.bar(t.index, t["Volume"], color="#94a3b8")
        ax.set_title(f"{stock_name} {code} {interval_name} 타이밍")
        ax.grid(True, alpha=0.25)
        av.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        save(fig, filename)

    plot_intraday(intraday60, "60분봉", f"06_{stock_name}_{code}_60분봉_타이밍.png")
    plot_intraday(intraday15, "15분봉", f"07_{stock_name}_{code}_15분봉_타이밍.png")
    return paths


def html_from_markdown(markdown_text: str, title: str) -> str:
    try:
        import markdown

        body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code", "toc"])
    except Exception:
        body = "<pre>" + markdown_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre>"
    css = """
body{font-family:Malgun Gothic,Apple SD Gothic Neo,Arial,sans-serif;line-height:1.62;color:#111827;max-width:1180px;margin:32px auto;padding:0 24px;background:#fff}
h1,h2,h3{line-height:1.25} table{border-collapse:collapse;width:100%;margin:14px 0 22px} th,td{border:1px solid #d1d5db;padding:8px 10px;text-align:left;vertical-align:top} th{background:#f3f4f6} code{background:#f3f4f6;padding:2px 4px;border-radius:4px} img{max-width:100%;height:auto;border:1px solid #e5e7eb;margin:10px 0 24px}
"""
    return f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>{body}</body></html>"


def build_report(
    stock_name: str,
    code: str,
    market: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    validation: pd.DataFrame,
    reliability: str,
    price_label: str,
    vol_label: str,
    validation_note: str,
    intraday60: pd.DataFrame,
    intraday15: pd.DataFrame,
    levels: dict[str, Any],
    decision: dict[str, Any],
    market_rel: dict[str, float],
    peer_returns: dict[str, dict[str, float]],
    naver_investor: dict[str, Any],
    chart_paths: list[Path],
) -> str:
    row = daily.iloc[-1]
    prev20 = daily.tail(20)
    prev60 = daily.tail(60)
    rise_vol = prev20.loc[prev20["Close"].diff() > 0, "Volume"].sum()
    fall_vol = prev20.loc[prev20["Close"].diff() < 0, "Volume"].sum()
    basis = decision["기준가"]
    user_avg = "미제공"
    data_date = iso(daily.index[-1])
    holder_candidate = levels.get("nearest_resistance", decision["1차목표"])
    holder_reward = pct(holder_candidate, basis)
    target1 = decision["1차목표"]
    target2 = decision["2차목표"]
    defense = decision["방어선"]

    intraday_status = "사용 불가"
    if not intraday60.empty or not intraday15.empty:
        intraday_status = "참고 가능"

    source1 = validation.iloc[0]["소스"] if len(validation) > 0 else "데이터 부족"
    source2 = validation.iloc[1]["소스"] if len(validation) > 1 else "데이터 부족"
    close = basis
    ma_summary = ", ".join(
        [f"{p}일 {money(last_valid(row, f'MA{p}'))}" for p in [5, 10, 20, 60, 120, 240] if np.isfinite(last_valid(row, f"MA{p}"))]
    )
    bb_state = (
        f"중심 {money(last_valid(row, 'BB중심'))}, 상단 {money(last_valid(row, 'BB상단'))}, 하단 {money(last_valid(row, 'BB하단'))}"
    )
    sector_label = infer_sector_label(code, stock_name)
    macd_value = last_valid(row, "MACD")
    macd_signal = last_valid(row, "MACD신호")
    macd_hist = last_valid(row, "MACD히스토그램")
    rsi_value = last_valid(row, "RSI14")
    bb_mid = last_valid(row, "BB중심")
    bb_upper = last_valid(row, "BB상단")
    bb_lower = last_valid(row, "BB하단")
    macd_comment_text = macd_comment(macd_value, macd_signal, macd_hist, close, levels.get("nearest_resistance"), decision.get("돌파가격"))
    rsi_comment_text = rsi_comment(rsi_value)
    bollinger_comment_text = bollinger_comment(close, bb_mid, bb_upper, bb_lower, levels.get("nearest_resistance"), decision.get("돌파가격"))
    volume_context = assess_volume_candle(
        last_valid(row, "Open"),
        last_valid(row, "High"),
        last_valid(row, "Low"),
        close,
        decision.get("거래량비율"),
    )
    major_support_text = format_price_level_list([levels["recent60_low"], levels["low52"]])
    major_resistance_text = format_price_level_list([levels["recent60_high"], levels["high52"]])
    mfi = last_valid(row, "MFI14")
    cmf = last_valid(row, "CMF20")
    obv_now = last_valid(row, "OBV")
    obv_prev20 = daily["OBV"].tail(20).iloc[0] if "OBV" in daily and len(daily) >= 20 else np.nan
    obv_state = "상승" if np.isfinite(obv_now) and np.isfinite(obv_prev20) and obv_now > obv_prev20 else "둔화 또는 하락"
    index_label = "Nasdaq" if market == "US" else market
    market_summary = (
        f"{index_label} 대비 20일 초과수익 {fpct(market_rel.get('20일 초과'))}, 60일 초과수익 {fpct(market_rel.get('60일 초과'))}"
        if market_rel
        else "시장 상대강도 데이터 부족"
    )
    profile_text = format_nearby_profile(levels["profile"], close)
    ma_comment = moving_average_comment(close, row)
    chart_md = "\n".join([f"![{p.name}]({p.name})" for p in chart_paths])
    pull = decision["눌림목"]
    breakout = decision["돌파"]
    pull_low_value = _finite_or_none(decision.get("눌림하단"))
    pull_confirm_text = (
        f"{pull} 회복 후 지지 확인"
        if pull_low_value is not None and close < pull_low_value
        else f"{pull} 지지 확인"
    )
    pull_alert_meaning = "회복 후 지지 확인 가격" if pull_low_value is not None and close < pull_low_value else "지지 확인 가격"
    rebreak_line = _finite_or_none(decision.get("단기재돌파확인선")) or _finite_or_none(levels.get("recent5_high")) or decision["돌파가격"]
    price_context = classify_price_context(
        close,
        _finite_or_none(decision.get("눌림하단")) or close,
        _finite_or_none(decision.get("눌림상단")) or close,
        rebreak_line,
        decision["돌파가격"],
        target1,
        defense,
    )
    buy_context_text = buy_zone_sentence(price_context)
    rebreak_context_text = rebreak_sentence(price_context)
    breakout_context_text = breakout_sentence(price_context)
    target_context_text = target_sentence(price_context)
    buy_action_text = buy_zone_action(price_context)
    rebreak_action_text = rebreak_action(price_context)
    breakout_action_text = breakout_action(price_context)
    yahoo_symbol = code.upper() if market == "US" else f"{code}{'.KS' if market == 'KOSPI' else '.KQ'}"
    yahoo_url = f"https://finance.yahoo.com/quote/{yahoo_symbol}/"
    if market == "US":
        sector_key = next((k for k in peer_returns if k.startswith("섹터 ETF")), "섹터 ETF")
        market_environment_section = f"""미국 주식:

* S&P 500: {format_relative_ref(peer_returns.get('S&P 500'))}
* Nasdaq: {format_relative_ref(peer_returns.get('Nasdaq'))}
* 섹터 ETF: {sector_key} - {format_relative_ref(peer_returns.get(sector_key))}
* 거래량: {shares(row['Volume'])}, 20일 평균 대비 {fratio(decision['거래량비율'])}
* 최근 확인된 재료: 공개 데이터 자동 수집 범위 밖이므로 데이터 부족
* 판단: 미국 지수와 섹터 ETF 대비 상대강도는 보조 확인이며, 최종 매수는 {money(decision['주의선'])} 지지 또는 {breakout} 돌파가 핵심입니다."""
        source_lines = f"""* yfinance: Yahoo Finance `{yahoo_symbol}` 일봉 및 분봉 보조 확인에 사용
* Stooq: 미국 주식 일봉 교차검증에 사용
* Yahoo Finance: {yahoo_url}"""
    else:
        market_environment_section = f"""국내 주식:

* 외국인: {naver_investor.get('status', '데이터 부족')}
* 기관: 공개 데이터 한계로 정밀 집계 불가
* 개인: 공개 데이터 한계로 정밀 집계 불가
* 신용잔고: 데이터 부족
* 공매도/대차잔고: 데이터 부족
* KOSPI/KOSDAQ: {market_summary}
* 관련주/동종업종: {format_peer_returns(peer_returns)}
* 판단: 시장 대비 상대강도와 {sector_label} 관련 흐름은 보조 확인이며, 최종 매수는 {money(decision['주의선'])} 지지 또는 {breakout} 돌파가 핵심입니다."""
        source_lines = f"""* pykrx: KRX/NAVER 기반 국내 일봉 데이터 수집에 사용
* FinanceDataReader: 국내 일봉 교차검증에 사용
* yfinance: Yahoo Finance `{yahoo_symbol}` 일봉 및 분봉 보조 확인에 사용
* 네이버 증권: https://finance.naver.com/item/main.naver?code={code}
* Yahoo Finance: {yahoo_url}"""

    score_context = {
        "supply_failed": "데이터 부족" in naver_investor.get("status", "") or "실패" in naver_investor.get("status", ""),
        "market_rel_strong": bool(market_rel and market_rel.get("20일 초과", 0) > 0),
    }
    trading_scores = calculate_trading_scores(
        decision,
        {"rr1": decision.get("손익비1"), "rr2": decision.get("손익비2"), "reward1": decision.get("예상수익률1"), "current_price": close},
        score_context,
    )
    trading_total = float(trading_scores["총점"])
    pull_high_value = _finite_or_none(decision.get("눌림상단"))
    entry_label = "회복 확인가" if pull_low_value is not None and close < pull_low_value else "눌림목 지지가"
    if pull_low_value is not None and close < pull_low_value:
        recovery_price_text = pull
        support_price_text = "해당 없음"
    elif pull_high_value is not None and close > pull_high_value and close < decision["돌파가격"]:
        recovery_price_text = decision.get("회복확인가") or "해당 없음"
        support_price_text = pull
    else:
        recovery_price_text = "해당 없음"
        support_price_text = pull
    common_confirmation_active = (
        _finite_or_none(decision.get("회복확인선")) is not None
        and same_price_level(decision.get("회복확인선"), decision.get("돌파가격"))
    )
    if common_confirmation_active:
        recovery_price_text = breakout
    recovery_action_price = recovery_price_text if recovery_price_text != "해당 없음" else breakout
    rebreak_display_text, rebreak_duplicate_action, rebreak_merged = display_rebreak_line(rebreak_line, target1, target2)
    rebreak_label = rebreak_display_label(rebreak_line, target1, target2)
    rebreak_display_action = rebreak_duplicate_action if rebreak_merged else rebreak_action_text
    if rebreak_merged:
        rebreak_context_text = "단기 재돌파선은 목표가 또는 강한 저항과 중복되어 별도 진입선으로 쓰지 않습니다."
    holder_text = (
        f"{recovery_action_price} 회복 실패 시 추가매수 보류, "
        f"{money(pull_high_value) if pull_high_value is not None else pull} 재이탈 시 단기 비중 축소 검토, "
        f"{money(target1)} 접근 시 1차 익절 검토, {money(defense)} 이탈 시 방어/손절합니다."
    )
    data_action_needed = (
        any(word in str(validation_note) for word in ["지연", "stale", "불일치", "경고", "실패", "제외"])
        and not (price_label == "통과" and vol_label == "통과")
    )
    data_status_prefix = "데이터 상태: 보조 소스 지연/불일치 여부 확인; " if data_action_needed else ""
    today_action_prices = (
        f"{data_status_prefix}"
        f"{money(pull_high_value) if pull_high_value is not None else pull} 재지지 확인, "
        f"{recovery_action_price} 종가 회복 확인, "
        f"{money(target1)} 접근 시 보유자 일부 익절, "
        f"{money(defense)} 이탈 시 방어"
    )
    if decision["지금바로매수"] == "가능":
        current_zone = "매수 구간"
        today_action = today_action_prices
        avoid_action = "한 번에 전액 진입하지 않습니다."
    elif close <= defense:
        current_zone = "방어 구간"
        today_action = today_action_prices
        avoid_action = "방어선 아래에서 물타기하지 않습니다."
    elif "대기" in decision["최종판단"] or "재확인" in decision["최종판단"]:
        current_zone = "관망 구간"
        today_action = today_action_prices
        avoid_action = "조건 확인 전 추격매수와 물타기를 하지 않습니다."
    else:
        current_zone = "추격 금지 구간"
        today_action = today_action_prices
        target_gap = pct(target1, close)
        avoid_action = (
            "목표가 여유와 별개로 손익비가 맞지 않는 추격매수를 하지 않습니다."
            if np.isfinite(target_gap) and target_gap >= 5
            else "근접 저항 바로 아래에서 추격매수하지 않습니다."
        )
    entry_confirm = f"{buy_context_text} {breakout_context_text} {breakout_volume_condition_comment(volume_context)} 시장은 급락이 아니어야 합니다."
    failure_condition = f"{money(decision['주의선'])} 이탈, {money(defense)} 일봉 종가 이탈, {breakout} 돌파 실패 후 거래량 동반 하락이 나오면 분석 전제를 낮춥니다."
    priority_text = f"1순위 {entry_label} 확인, 2순위 일봉 돌파 확인가 종가 유지/안착, 3순위 거래량 1.2배 이상 확인"
    today_first_action = (
        "지금 바로 매수 조건이 유지되는지 확인하되 계획 비중만 분할 적용합니다."
        if decision["지금바로매수"] == "가능"
        else "지금 바로 매수하지 않는 원칙을 먼저 확인합니다."
    )
    now_buy_weight = "0%" if decision["지금바로매수"] == "불가" else ("20~30%" if trading_total >= 70 else "10% 이하")
    pullback_weight = "20~30%" if trading_total >= 70 else "10~20%"
    breakout_weight = "20~30%" if trading_total >= 70 else "10~20%"
    if pull_low_value is not None and close < pull_low_value:
        confirm_entry_line = pull_low_value
    elif recovery_price_text != "해당 없음" and _finite_or_none(decision.get("회복확인선")) is not None:
        confirm_entry_line = decision["회복확인선"]
    else:
        confirm_entry_line = decision["돌파가격"]
    confirm_reward = pct(target1, confirm_entry_line)
    confirm_risk = downside_risk_pct(confirm_entry_line, defense)
    confirm_rr = confirm_reward / confirm_risk if np.isfinite(confirm_risk) and confirm_risk > 0 else np.nan
    decision["확인진입손익비"] = confirm_rr
    decision["회복확인가표시"] = recovery_price_text
    decision["눌림목지지가표시"] = support_price_text
    reliability_parts = reliability_breakdown(
        price_label,
        vol_label,
        naver_investor.get("status", "데이터 부족"),
        "해당 없음",
        True,
        validation_note,
    )
    rr_condition_unmet = (
        decision["지금바로매수"] == "불가"
        and np.isfinite(decision["손익비1"])
        and decision["손익비1"] >= 1.5
        and (entry_label == "회복 확인가" or close < decision["돌파가격"])
    )
    rr_caution_text = (
        f"현재가 기준 단순 손익비는 {fratio(decision['손익비1'])}로 양호하지만, 돌파 유지 실패와 데이터 신뢰도 {reliability} 상태로 인해 "
        "회복 확인가 또는 돌파 확인가 조건 미충족으로 지금 바로 신규매수 조건은 충족하지 못했습니다."
        if rr_condition_unmet
        else "손익비와 매수 가능 여부는 조건 충족 여부를 분리해서 해석합니다."
    )
    low_rr_summary = compressed_low_rr_warning(decision.get("손익비1"), confirm_rr)
    if low_rr_summary:
        rr_caution_text = low_rr_summary
    deep_pullback_note = (
        "현재가는 깊은 눌림목 구간 안 또는 회복 확인 전 구간이므로 반등 확인 후 신규매수를 검토합니다."
        if (
            (pull_high_value is not None and close <= pull_high_value)
            or str((decision.get("상태코드") or {}).get("final_action_state", "")) == "NO_BUY_BELOW_RECOVERY"
            or str((decision.get("상태코드") or {}).get("price_position_state", "")) == "BELOW_PULLBACK"
        )
        else ""
    )
    volume_momentum_conflict = assess_volume_momentum_conflict(
        close,
        pull_high_value,
        _finite_or_none(decision.get("회복확인선")) or _finite_or_none(decision.get("돌파가격")),
        decision.get("거래량비율"),
        rsi_value,
        macd_value,
        macd_signal,
    )
    volume_momentum_note = ""
    if volume_momentum_conflict["applies"]:
        volume_momentum_note = (
            f"{volume_momentum_conflict['state']}. "
            f"{volume_momentum_conflict['template']} "
            f"{volume_momentum_conflict['final']}."
        )
        if volume_momentum_conflict["primary_strategy"] and volume_momentum_conflict["primary_strategy"] not in str(decision.get("주전략", "")):
            decision["주전략"] = f"{decision.get('주전략', '')}; {volume_momentum_conflict['primary_strategy']}".strip("; ")
        if volume_momentum_conflict["final"] and volume_momentum_conflict["final"] not in str(decision.get("최종판단", "")):
            decision["최종판단"] = f"{decision.get('최종판단', '')}·{volume_momentum_conflict['final']}".strip("·")
    state_locked = bool(decision.get("상태코드잠금"))
    if not state_locked:
        if np.isfinite(confirm_rr) and confirm_rr < 0.5:
            decision["최종판단"] = "신규매수 금지, 돌파 추격 전략 성립 불가"
        elif any(np.isfinite(v) and v < 0.8 for v in [decision["손익비1"], confirm_rr]):
            decision["최종판단"] = "신규매수 금지에 가까움·스윙/돌파 손익비 부족"
        elif any(np.isfinite(v) and v < 1.2 for v in [decision["손익비1"], confirm_rr]) and "손익비" not in str(decision["최종판단"]):
            decision["최종판단"] = f"{decision['최종판단']}·스윙/돌파 손익비 부족"
        if volume_context.get("bearish_high_volume") and decision["지금바로매수"] == "불가":
            decision["주전략"] = f"고거래량 음봉 경고, {breakout} 회복 전 신규매수 금지"
            if "신규매수 금지" not in str(decision["최종판단"]):
                decision["최종판단"] = f"고거래량 음봉 경고·{decision['최종판단']}"

    report = f"""# 주식 매매타점 분석 보고서

## 프로 트레이더 판단

| 항목 | 내용 |
| --- | --- |
| 현재 구간 정의 | {current_zone} |
| 주 전략 | {decision.get('주전략', '눌림목 대기')} |
| 보조 전략 | {decision.get('보조전략', f'{breakout} 돌파 조건 재확인')} |
| 오늘 할 행동 | {today_action} |
| 오늘 하지 말아야 할 행동 | {avoid_action} |
| 진입 전 확인 조건 | {entry_confirm} |
| 실패 조건 | {failure_condition} |
| 우선순위 | {priority_text} |

## 오늘 할 행동

1. {today_first_action}
2. {pull_confirm_text} 여부를 확인합니다.
3. {breakout} 일봉 종가 돌파 여부를 확인합니다.
4. 거래량이 20일 평균 1.2배 이상 붙는지 확인합니다.
5. {money(defense)} 이탈 시 스윙 관점은 접고 방어 판단으로 전환합니다.

## 알림 설정 가격

| 알림 가격 | 의미 | 행동 |
| ---: | --- | --- |
| {pull} | {pull_alert_meaning} | 분할매수 검토 |
| {breakout} | 일봉 돌파 확인 가격 | 종가 확인 |
| {money(decision['주의선'])} | 스윙 주의 가격 | 관찰 강화 |
| {money(defense)} | 스윙 최종 방어 가격 | 손절/비중 축소 |

## 매매 시나리오

| 시나리오 | 조건 | 행동 | 진입 비중 | 손절/방어 | 목표 |
| --- | --- | --- | --- | --- | --- |
| A. 지금 매수 | {decision['지금매수하지않는이유'] if decision['지금바로매수'] == "불가" else "조건 충족"} | {"매수 금지" if decision['지금바로매수'] == "불가" else "분할 매수"} | {now_buy_weight} | {"해당 없음" if decision["지금바로매수"] == "불가" and now_buy_weight == "0%" else money(defense)} | {"해당 없음" if decision["지금바로매수"] == "불가" and now_buy_weight == "0%" else money(target1)} |
| B. 눌림목 매수 | {pull_confirm_text} | 분할매수 | {pullback_weight} | {money(defense)} | {money(target1)} |
| C. 돌파 매수 | {breakout} 일봉 종가 돌파 + 거래량 1.2배 이상 | 매수 검토 | {breakout_weight} | {breakout} 재이탈 | {money(target2)} |
| D. 관망 접기 | {money(defense)} 일봉 종가 이탈 또는 돌파 실패 후 거래량 동반 하락 | 관망/정리 | 0% | {money(defense)} | 없음 |

## 투자자 상태별 대응

| 투자자 상태 | 대응 |
| --- | --- |
| 신규매수자 | 지금 진입 가능 여부는 {decision['지금바로매수']}입니다. {pull_confirm_text} 또는 {breakout} 돌파 확인 전까지 기다립니다. |
| 기존 보유자 | 평단가 미제공으로 개인별 손익률 판단은 제외합니다. {money(holder_candidate)} 근접 저항에서는 일부 익절을 검토하고 {money(defense)} 이탈 시 방어합니다. |
| 추가매수자 | 불타기와 물타기는 모두 보류하고, {breakout} 종가 돌파 또는 {pull_confirm_text} 뒤에만 검토합니다. |
| 손실 보유자 | 평단가 미제공으로 손실률 판단은 제외합니다. {money(defense)} 일봉 종가 이탈 시 반등 기대보다 방어를 우선합니다. |

## 트레이딩 점수

| 항목 | 점수 | 기준 |
| --- | ---: | --- |
| 추세 점수 | {trading_scores['추세 점수']} / 20 | 이동평균선, 일목, 고점/저점 구조 |
| 모멘텀 점수 | {trading_scores['모멘텀 점수']} / 15 | MACD, RSI, Stochastic |
| 거래량 점수 | {trading_scores['거래량 점수']} / 15 | 20일 평균 대비 거래량, OBV/MFI/CMF |
| 현재가 기준 손익비 점수 | {trading_scores['현재가 기준 손익비 점수']} / 20 | 현재 위치에서 바로 진입할 경우의 손익비 |
| 눌림목 진입 기준 손익비 점수 | {trading_scores['눌림목 진입 기준 손익비 점수']} / 20 | 눌림목 가격에서 진입할 경우의 손익비 |
| 돌파 진입 기준 손익비 점수 | {trading_scores['돌파 진입 기준 손익비 점수']} / 20 | 돌파 확인 후 진입할 경우의 손익비 |
| 손익비 점수 | {trading_scores['손익비 점수']} / 20 | 세부 손익비 점수 가중 평균 |
| 시장/섹터 점수 | {trading_scores['시장/섹터 점수']} / 10 | 시장 상대강도와 관련주 흐름 |
| 수급 점수 | {trading_scores['수급 점수']} / 10 | 외국인/기관/개인 수급, 데이터 부족 시 중립 이하 |
| 위치 점수 | {trading_scores['위치 점수']} / 10 | 현재가가 매수하기 좋은 위치인지 |
| 총점 | {trading_scores['총점']}점 | {trading_scores['판정']} |

## 이 분석이 틀렸다고 보는 조건

* 핵심 지지선인 {money(decision['주의선'])} 이탈 후 회복하지 못하는 경우
* {breakout} 돌파 시도 후 거래량 동반 하락이 나오는 경우
* 스윙 최종 방어선 {money(defense)} 아래로 일봉 종가가 마감하는 경우
* 시장 지수가 급락하고 {sector_label} 관련 흐름이 동시에 꺾이는 경우
* 수급 데이터 부족 또는 수집 실패로 수급 판단 신뢰도가 낮아지는 경우
* 데이터 신뢰도가 낮음으로 떨어지는 경우

## 1. 최종 매매 의사결정표

| 항목 | 판단 |
| --- | --- |
| 지금 바로 매수 | {decision['지금바로매수']} |
| 지금 매수하지 않는 이유 | {decision['지금매수하지않는이유']} |
| 주 전략 | {decision.get('주전략', '눌림목 대기')} |
| 보조 전략 | {decision.get('보조전략', f'{breakout} 돌파 조건 재확인')} |
| 회복 확인가 | {recovery_price_text} |
| 눌림목 지지가 | {support_price_text} |
| 눌림목/회복 확인 조건 | {decision['눌림조건']} |
| {rebreak_label} | {money(rebreak_line)} |
| 일봉 돌파 확인가 | {breakout} |
| 돌파 매수 조건 | {decision['돌파조건']} |
| 근접 저항/보유자 일부 익절 후보 | {money(holder_candidate)} |
| 근접 저항/보유자 일부 익절 후보 조건 | 전고점/매물대 접근 시 보유자 일부 익절, 거래량 둔화 또는 윗꼬리 확인 |
| 신규매수 기준 1차 목표 | {money(target1)} |
| 신규매수 기준 1차 목표 조건 | 돌파 후 거래량 유지 시 일부 실현 검토 |
| 신규매수 기준 2차 목표 | {money(target2)} |
| 신규매수 기준 2차 목표 조건 | 돌파 후 거래량 유지 시 추가 실현, RSI 70 이상 과열 시 비중 축소 |
| 스윙 주의선 | {money(decision['주의선'])} |
| 스윙 최종 방어선 | {money(defense)} |
| 전량 이탈 조건 | {decision['전량이탈']} |
| 최종 판단 | {decision['최종판단']} |

## 2. 기본 정보

| 항목 | 값 |
| --- | --- |
| 종목명 | {stock_name} |
| 종목코드/티커 | {code} |
| 시장 | {market} |
| 기준가 | {money(close)} |
| 기준가 산정 기준 | 최신 완료 일봉 종가 |
| 사용자 평단가 | {user_avg} |
| 데이터 기준일 | {data_date} |
| 매매 스타일 | 스윙 |
| 데이터 방식 | API 없는 공개 데이터 모드 |

## 3. 데이터 신뢰도 점검

| 항목 | 결과 |
| --- | --- |
| 1차 데이터 소스 | {source1} |
| 2차 데이터 소스 | {source2} |
| 최신 거래일 | {data_date} |
| 가격 검증 | {price_label} |
| 거래량 검증 | {vol_label} |
| 조정주가 사용 여부 | 아니오 |
| 분봉 데이터 신뢰도 | {intraday_status} |
| 가격 신뢰도 | {reliability_parts['가격 신뢰도']} |
| 거래량 신뢰도 | {reliability_parts['거래량 신뢰도']} |
| 수급 신뢰도 | {reliability_parts['수급 신뢰도']} |
| 장중 가격 신뢰도 | {reliability_parts['장중 가격 신뢰도']} |
| 해석 완전성 | {reliability_parts['해석 완전성']} |
| 최종 데이터 신뢰도 | {reliability} |

검증 세부:

{compact_validation_md(validation)}

검증 메모: {validation_note}

## 4. 핵심 요약

* 현재 주가는 {money(close)}이며, 최신 완료 일봉 기준으로 최근 급등 후 {format_price_range(levels['nearest_resistance'], levels['target1'], '단일 저항선')}을 앞두고 있습니다.
* 지금 바로 매수 판단은 `{decision['지금바로매수']}`입니다. 핵심 이유는 {decision['지금매수하지않는이유']}입니다.
* 주 전략은 {decision.get('주전략', '눌림목 대기')}입니다.
* 보조 전략은 {decision.get('보조전략', f'{breakout} 돌파 조건 재확인')}입니다.
* {entry_label}는 {pull}이며, 거래량이 줄어든 조정 후 해당 가격 회복 또는 지지 확인이 필요합니다.
* 일봉 돌파 확인가는 {breakout}이며, 20일 평균 거래량의 1.2배 이상이 필요합니다.
* 근접 저항/보유자 일부 익절 후보는 {money(holder_candidate)}입니다.
* 신규매수 기준 1차 목표는 {money(target1)}으로 기준가 대비 {fpct(decision['예상수익률1'])}, 신규매수 기준 2차 목표는 {money(target2)}으로 {fpct(decision['예상수익률2'])}입니다.
* 스윙 최종 방어선은 {money(defense)}이며, 기준가 대비 하락 위험률은 {fpct(decision['하락위험률'])}입니다.
* 최종 판단은 `{decision['최종판단']}`입니다.

## 5. 핵심 가격표

| 가격대 | 의미 | 대응 |
| --: | --- | --- |
| {money(close)} | 기준가 | 최신 완료 일봉 종가 기준 판단 |
| {recovery_price_text} | 회복 확인가 | 현재가보다 위에 있는 재진입 확인 가격 |
| {support_price_text} | 눌림목 지지가 | 현재가 부근 또는 아래의 지지 확인 가격 |
| {money(rebreak_line)} | {rebreak_label} | 단기 탄력 회복 확인 |
| {breakout} | 일봉 돌파 확인가 | 거래량 동반 돌파 시 매수 후보 |
| {money(holder_candidate)} | 근접 저항/보유자 일부 익절 후보 | 보유자 일부 익절 후보 |
| {money(target1)} | 신규매수 기준 1차 목표 | 신규매수 손익비 산정용 |
| {money(target2)} | 신규매수 기준 2차 목표 | 추가 익절 후보 |
| {money(decision['주의선'])} | 스윙 주의선 | 이탈 시 매수 관점 약화 |
| {money(defense)} | 스윙 최종 방어선 | 이탈 시 손절/비중 축소 |
| {money(defense)} 종가 이탈 후 회복 실패 | 전량 이탈 조건 | 추세 훼손 |

## 6. 예상 수익률과 하락 위험

| 시나리오 | 가격 | 기준가 대비 수익률 | 평단 대비 수익률 | 판단 |
| --- | --: | --: | --: | --- |
| 근접 저항/보유자 일부 익절 후보 | {money(holder_candidate)} | {fpct(holder_reward)} | 미제공 | 보유자 일부 익절 후보 |
| 신규매수 기준 1차 목표 | {money(target1)} | {fpct(decision['예상수익률1'])} | 미제공 | 신규매수 손익비 산정 기준 |
| 신규매수 기준 2차 목표 | {money(target2)} | {fpct(decision['예상수익률2'])} | 미제공 | 추가 익절 후보 |
| 손절/방어 | {money(defense)} | -{fpct(decision['하락위험률'])} | 미제공 | 종가 이탈 시 방어 |

* 하락 위험률: {fpct(decision['하락위험률'])}
* 장중 방어선 기준 손익비: 해당 없음
* 스윙 손절선 기준 손익비: {fratio(decision['손익비1'])}
* 회복 확인가 또는 돌파 확인가 진입 기준 손익비: {fratio(confirm_rr)}
* 신규매수 기준 1차 목표 손익비: {fratio(decision['손익비1'])}
* 신규매수 기준 2차 목표 손익비: {fratio(decision['손익비2'])}
* 매수 매력도: {"높음" if decision['손익비1'] >= 1.5 and decision['지금바로매수'] == "가능" else "눌림 또는 돌파 확인 전까지 제한적"}

## 7. 보조지표 종합 점검표

| 지표 | 현재 상태 | 해석 |
| --- | --- | --- |
| 이동평균선 | {decision['MA배열']}; {ma_summary} | {ma_comment} |
| 일목균형표 | 일봉 {decision['일봉구름']}, 주봉 {decision['주봉구름']}, 월봉 {decision['월봉구름']} | 구름 위는 추세 우위, 구름 안/아래는 돌파 확인 필요입니다. |
| MACD | MACD {macd_value:.2f}, 신호 {macd_signal:.2f}, 히스토그램 {macd_hist:.2f} | {macd_comment_text} |
| RSI | {rsi_value:.2f} | {rsi_comment_text} |
| Stochastic | K {last_valid(row, 'StochK'):.2f}, D {last_valid(row, 'StochD'):.2f} | 단기 과열/둔화 확인용입니다. |
| 볼린저밴드 | {bb_state} | {bollinger_comment_text} |
| ATR | {money(decision['ATR'])} | 손절선과 목표가 간격 산정 기준입니다. |
| OBV | {obv_state} | 20거래일 기준 누적 거래량 방향입니다. |
| MFI/CMF | MFI {mfi:.2f}, CMF {cmf:.3f} | 자금 유입 강도와 종가 위치 기반 매수 압력을 함께 봅니다. |
| 거래량 | {shares(row['Volume'])}, 20일 평균 대비 {fratio(decision['거래량비율'])} | 돌파 매수는 1.2배 이상이 필요합니다. |
| 매물대 | {profile_text} | 현재가 주변 상단 매물대와 하단 지지 매물대를 분리해서 봅니다. |
| 수급 | {naver_investor.get('status', '데이터 부족')} | 상세 기관/개인 수급은 공개 데이터 한계로 보수적으로 해석합니다. |
| 시장/섹터 | {market_summary} | 종목이 지수보다 강하면 눌림 매수 신뢰도가 올라갑니다. |

## 8. 차트 분석

### 월봉/주봉

* 추세: 월봉/주봉 기준 종가는 장기 평균선과 일목 위치를 함께 보면 중기 반등 흐름입니다.
* 일목균형표: 월봉 {decision['월봉구름']}, 주봉 {decision['주봉구름']}입니다.
* 주요 지지: {major_support_text}
* 주요 저항: {major_resistance_text}
* 판단: 장기 추세가 완전히 훼손된 구간은 아니지만, 최근 급등 직후라 신규 진입은 지지 확인이 유리합니다.

### 일봉

* 이동평균선: {ma_summary}
* 일목균형표: {decision['일봉구름']}
* 거래량: 최근 20일 상승일 거래량 합계 {shares(rise_vol)}, 하락일 거래량 합계 {shares(fall_vol)}
* MACD: {last_valid(row, 'MACD'):.2f}, 신호선 {last_valid(row, 'MACD신호'):.2f}, 히스토그램 {last_valid(row, 'MACD히스토그램'):.2f}
* RSI: {decision['RSI']:.2f}
* 볼린저밴드: {bb_state}
* ATR: {money(decision['ATR'])}
* 매물대: {profile_text}
* 판단: {breakout} 돌파 전까지는 추격보다 눌림 확인이 우선입니다.

### 분봉

* 60분봉: {"참고 가능" if not intraday60.empty else "데이터 사용 불가"}
* 30분봉 또는 15분봉: {"참고 가능" if not intraday15.empty else "데이터 사용 불가"}
* 단기 타이밍 판단: {"분봉은 보조로만 사용하고 일봉 기준 판단 우선" if not intraday60.empty or not intraday15.empty else "분봉 데이터 불안정으로 일봉 기준 판단 우선"}

## 9. 수급 및 시장 환경

{market_environment_section}

## 10. 매매 계획

### 매수 계획

* 지금 바로 매수: {decision['지금바로매수']} - {decision['지금매수하지않는이유']}
* 눌림목 매수: {pull}에서 거래량 감소 조정과 일봉 종가 지지 확인 후 분할 접근
* 돌파 매수: {breakout} 이상 일봉 종가 마감, 거래량 20일 평균 1.2배 이상 확인
* 매수 금지 조건: 일봉 종가 {money(defense)} 이탈, 거래량 없는 반등, MACD 재둔화, RSI 40 아래 재하락, 시장 급락 동반

### 익절 계획

* 근접 저항/보유자 일부 익절 후보: {money(holder_candidate)} 도달 시 보유자 일부 익절
* 신규매수 기준 1차 목표: {money(target1)} 도달 시 일부 실현 검토
* 신규매수 기준 2차 목표: {money(target2)} 도달 시 추가 실현 검토
* 전량 익절 후보: 목표가 도달 후 거래량 둔화, RSI 70 이상 과열, 장대양봉 뒤 윗꼬리 발생

### 손절/방어 계획

* 스윙 주의선: {money(decision['주의선'])}
* 스윙 최종 방어선: {money(defense)}
* 전량 이탈 조건: {decision['전량이탈']}

## 11. 최종 판단

{decision['최종판단']}

기준가에서 신규매수 기준 1차 목표까지의 손익비는 {fratio(decision['손익비1'])}입니다.
{rr_caution_text}
최근 급등 후 전고점 저항이 가까워 신규 진입은 눌림목 회복/지지 확인 또는 거래량 동반 돌파 확인이 필요합니다.
데이터 신뢰도는 {reliability}으로 가격/거래량 검증은 {price_label}/{vol_label}입니다.

## 12. 최종 한 문단 판단

{stock_name} {code}은 기준가 {money(close)}에서 바로 추격 매수하기보다 {pull_confirm_text} 또는 {breakout} 돌파를 기다리는 전략이 유리합니다. {volume_momentum_note} 근접 저항/보유자 일부 익절 후보는 {money(holder_candidate)}이고, 신규매수 기준 1차 목표는 {money(target1)}으로 예상 수익률 {fpct(decision['예상수익률1'])}, 신규매수 기준 2차 목표는 {money(target2)}으로 {fpct(decision['예상수익률2'])}이며, 스윙 최종 방어선은 {money(defense)}입니다. 가장 중요한 가격은 {breakout}이며, 이 가격을 거래량으로 돌파하지 못하면 단기 매물 소화 구간으로 봅니다.

## 부록. 차트

{chart_md}

## 부록. 데이터 출처

{source_lines}
"""
    monthly_state = practical_state_from_text(decision.get("월봉구름"))
    weekly_state = practical_state_from_text(decision.get("주봉구름"))
    daily_state = daily_trend_state_from_values(
        close,
        row,
        macd_value,
        macd_signal,
        rsi_value,
        practical_state_from_text(decision.get("일봉구름"), decision.get("MA배열"), ma_comment),
    )
    minute_state = "중립" if intraday_status == "참고 가능" else "데이터 부족"
    ma_grade = practical_grade_from_text(decision.get("MA배열"), ma_comment)
    cloud_grade = practical_grade_from_text(decision.get("일봉구름"), decision.get("주봉구름"))
    macd_grade = macd_grade_from_values(macd_value, macd_signal, macd_hist, close, decision.get("돌파가격"))
    rsi_grade = rsi_grade_from_value(rsi_value)
    volume_grade = volume_context["status"]
    profile_grade = practical_grade_from_text(profile_text)
    supply_grade = practical_grade_from_text(naver_investor.get("status", "데이터 부족"))
    precision_limited = "정밀 판단 중단" in str(decision.get("최종판단", ""))
    def point_value(text: Any) -> str:
        value = str(text)
        if precision_limited and value not in {"해당 없음", ""} and not value.startswith("참고 "):
            return f"참고 {value}"
        return value
    state_code_section = decision.get("상태코드표", "| 상태 항목 | 상태코드 |\n|---|---|\n| 상태코드 | 데이터 부족 |")
    if common_confirmation_active:
        conclusion_confirmation_rows = f"| 회복/돌파 공통 확인가 | {point_value(f'{breakout} - 종가 안착 + 거래량 유지 확인')} |"
        trading_confirmation_rows = f"| 회복/돌파 공통 확인가 | {point_value(breakout)} | 종가 안착 + 거래량 유지 확인 |"
    else:
        conclusion_confirmation_rows = (
            f"| 회복 확인가 | {point_value(recovery_price_text)} |\n"
            f"| 일봉 돌파 확인가 | {point_value(breakout)} |"
        )
        trading_confirmation_rows = (
            f"| 회복 확인가 | {point_value(recovery_price_text)} | 현재가보다 위에 있는 재진입 확인 가격 |\n"
            f"| 일봉 돌파 확인가 | {point_value(breakout)} | {breakout_action_text} |"
        )
    report = f"""# {stock_name} {code} 실전 매매 판단 리포트

## 1. 최종 결론

| 항목 | 판단 |
|---|---|
| 현재가 | {money(close)} |
| 지금 매수 | {decision['지금바로매수']} |
| 주 전략 | {decision.get('주전략', '눌림목 대기')} |
{conclusion_confirmation_rows}
| 눌림목 지지가 | {point_value(support_price_text)} |
| {rebreak_label} | {point_value(rebreak_display_text)} |
| 1차 목표 | {point_value(money(target1))} |
| 2차 목표 | {point_value(money(target2))} |
| 손절/방어 | {point_value(money(defense))} |
| 스윙 손절선 기준 손익비 | {fratio(decision['손익비1'])} |
| 회복/돌파 진입 기준 손익비 | {fratio(confirm_rr)} |
| 데이터 검증 메모 | {one_line(validation_note)} |
| 최종 판단 | {decision['최종판단']} |

## 1-1. 상태코드 기반 판단

{state_code_section}

## 2. 프로 트레이더 관점

- 지금 할 행동: {today_action}
- 지금 하지 말아야 할 행동: {avoid_action}
- 신규매수자: {decision.get('new_buyer_action', f'{buy_context_text} {breakout_context_text}')}
- 보유자: {decision.get('holder_action', holder_text)}
- 추가매수자: {decision.get('add_buyer_action', breakout_context_text)}
- 손실보유자: {decision.get('stop_loss_action', f'{money(defense)} 아래에서는 물타기보다 비중 축소를 우선합니다.')}

## 3. 차트 분석

| 구분 | 상태 | 판단 |
|---|---|---|
| 월봉 | {monthly_state} | {one_line(monthly_chart_comment(decision['월봉구름']))} |
| 주봉 | {weekly_state} | {decision['주봉구름']}이며, 중기 저항 돌파 확인이 필요합니다. |
| 일봉 | {daily_state} | {one_line(ma_comment)} |
| 분봉 | {minute_state} | {intraday_status} 상태이므로 매수 타이밍 보조로만 봅니다. |

## 4. 보조지표 판단

| 지표 | 상태 | 매매 판단 |
|---|---|---|
| 이동평균선 | {ma_grade} | {one_line(ma_comment)} |
| 일목균형표 | {cloud_grade} | 일봉 {decision['일봉구름']}, 주봉 {decision['주봉구름']}입니다. |
| MACD | {macd_grade} | {one_line(macd_comment_text)} |
| RSI | {rsi_grade} | {one_line(rsi_comment_text)} |
| 거래량 | {volume_grade} | {one_line(volume_context['comment'])} |
| 매물대 | {profile_grade} | {one_line(profile_text)} |
| 수급 | {supply_grade} | {one_line(naver_investor.get('status', '데이터 부족'))} |

## 5. 매매 타점

| 구분 | 가격 | 행동 |
|---|---:|---|
{trading_confirmation_rows}
| 눌림목 지지가 | {point_value(support_price_text)} | 현재가 부근 또는 아래의 지지 확인 가격 |
| {rebreak_label} | {point_value(rebreak_display_text)} | {rebreak_display_action} |
| 1차 목표 | {point_value(money(target1))} | 일부 익절 후보 |
| 2차 목표 | {point_value(money(target2))} | 추가 익절 후보 |
| 장중 방어선 | 해당 없음 | 장외/일봉 기준 분석입니다. |
| 스윙 손절선 | {point_value(money(defense))} | 종가 이탈 시 방어/손절 |

## 6. 최종 한 문단

{stock_name} {code}은 현재 {money(close)} 기준으로 {deep_pullback_note} {volume_momentum_note} {buy_context_text} {rebreak_context_text} {breakout_context_text} {target_context_text} 1차 목표는 {money(target1)}, 2차 목표는 {money(target2)}이고, {money(defense)} 이탈 시 매수 관점은 낮춥니다. {rr_caution_text} 가격/거래량 신뢰도는 {reliability_parts['가격 신뢰도']}/{reliability_parts['거래량 신뢰도']}, 수급 신뢰도는 {reliability_parts['수급 신뢰도']}, 해석 완전성은 {reliability_parts['해석 완전성']}입니다."""
    return report


def save_csvs(out_dir: Path, stock_name: str, code: str, validation: pd.DataFrame, daily: pd.DataFrame) -> tuple[Path, Path]:
    validation_path = out_dir / f"{stock_name}_{code}_데이터검증.csv"
    summary_path = out_dir / f"{stock_name}_{code}_지표요약.csv"
    validation.to_csv(validation_path, index=False, encoding="utf-8-sig")
    cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "MA5",
        "MA10",
        "MA20",
        "MA60",
        "MA120",
        "MA240",
        "전환선",
        "기준선",
        "선행스팬1",
        "선행스팬2",
        "MACD",
        "MACD신호",
        "MACD히스토그램",
        "RSI14",
        "StochK",
        "StochD",
        "WilliamsR",
        "CCI20",
        "ROC12",
        "ATR14",
        "BB상단",
        "BB중심",
        "BB하단",
        "OBV",
        "MFI14",
        "CMF20",
        "거래량20평균",
        "거래량비율20",
        "거래대금",
    ]
    available = [c for c in cols if c in daily.columns]
    out = daily[available].tail(120).copy()
    out.insert(0, "Date", [iso(i) for i in out.index])
    out.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return validation_path, summary_path


def console_output(
    stock_name: str,
    code: str,
    decision: dict[str, Any],
    reliability: str,
    out_dir: Path,
    md_path: Path,
    html_path: Path,
    levels: dict[str, Any],
) -> str:
    entry_label = "회복 확인가" if _finite_or_none(decision.get("눌림하단")) is not None and decision["기준가"] < decision["눌림하단"] else "눌림목 지지가"
    return f"""[분석 완료]

종목: {stock_name} {code}
현재가: {money(decision['기준가'])}
지금 매수: {decision['지금바로매수']}
주 전략: {decision.get('주전략', '눌림목 대기')}
{entry_label}: {decision['눌림목']}
일봉 돌파 확인가: {decision['돌파']}
1차 목표: {money(decision['1차목표'])}
손절/방어: {money(decision['방어선'])}
최종 판단: {decision['최종판단']}
데이터 신뢰도: {reliability}
보고서 경로: {md_path}"""


def run(code: str, fallback_name: str | None = None) -> str:
    code = code.strip()
    if not (code.isdigit() and len(code) == 6):
        code = code.upper()
    now = today_kst()
    end_limit = latest_completed_candidate(now)
    start_daily = end_limit - timedelta(days=365 * 6)
    start_validation = end_limit - timedelta(days=365 * 2)

    stock_name, market, yf_suffix = detect_name_market(code, fallback_name, end_limit)
    safe_name = sanitize_filename(stock_name)
    out_dir = REPORTS_DIR / f"{safe_name}_{code}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if market == "US":
        src_pykrx = SourceFrame("pykrx", pd.DataFrame(), "미국 주식에서는 국내 pykrx 조회 생략")
        src_fdr = load_stooq(code, start_daily, end_limit)
        src_yf = load_yfinance(code, start_validation, end_limit, "yfinance")
        validation_sources = [src_yf, src_fdr]
    else:
        src_pykrx = load_pykrx(code, start_daily, end_limit)
        src_fdr = load_fdr(code, start_daily, end_limit)
        src_yf = load_yfinance(code + yf_suffix, start_validation, end_limit, "yfinance")
        validation_sources = [src_pykrx, src_fdr, src_yf]
    validation, reliability, stop_precision = source_validation(validation_sources, end_limit)
    price_label, vol_label, validation_note = validation_labels(validation)

    representative_source = ""
    if "대표가격사용" in validation.columns:
        reps = validation[validation["대표가격사용"] == "예"]
        if not reps.empty:
            representative_source = str(reps.iloc[0]["소스"])

    if representative_source == "pykrx" and not src_pykrx.data.empty:
        daily_base = src_pykrx.data[src_pykrx.data.index.date <= end_limit].copy()
    elif representative_source == "FinanceDataReader" and not src_fdr.data.empty:
        daily_base = src_fdr.data[src_fdr.data.index.date <= end_limit].copy()
    elif representative_source == "yfinance" and not src_yf.data.empty:
        daily_base = src_yf.data[src_yf.data.index.date <= end_limit].copy()
    elif market == "US" and not src_yf.data.empty:
        daily_base = src_yf.data[src_yf.data.index.date <= end_limit].copy()
    elif market == "US" and not src_fdr.data.empty:
        daily_base = src_fdr.data[src_fdr.data.index.date <= end_limit].copy()
    elif not src_pykrx.data.empty:
        daily_base = src_pykrx.data[src_pykrx.data.index.date <= end_limit].copy()
    elif not src_fdr.data.empty:
        daily_base = src_fdr.data[src_fdr.data.index.date <= end_limit].copy()
    elif not src_yf.data.empty:
        daily_base = src_yf.data[src_yf.data.index.date <= end_limit].copy()
    else:
        raise RuntimeError("일봉 OHLCV 데이터를 수집하지 못했습니다.")

    daily = add_indicators(daily_base, [5, 10, 20, 60, 120, 240])
    weekly = add_indicators(resample_ohlcv(daily_base, "W-FRI"), [5, 10, 20, 60, 120, 240])
    monthly = add_indicators(resample_ohlcv(daily_base, "M"), [5, 10, 20, 60, 120, 240])
    levels = nearest_levels(daily_base, daily)
    levels["basis"] = float(daily.iloc[-1]["Close"])

    yf_symbol = code if market == "US" else code + yf_suffix
    intraday60_src = load_yfinance_intraday(yf_symbol, "60m", "2mo")
    intraday15_src = load_yfinance_intraday(yf_symbol, "15m", "1mo")
    intraday60 = intraday60_src.data
    intraday15 = intraday15_src.data

    index_src = load_market_index(market, start_daily, end_limit)
    market_rel = relative_returns(daily_base, index_src.data)
    market_index_value = None
    if market in {"KOSPI", "KOSDAQ"} and not index_src.data.empty:
        market_index_value = float(index_src.data["Close"].dropna().iloc[-1])
    market_index_invalid = market in {"KOSPI", "KOSDAQ"} and not index_src.data.empty and not market_index_frame_is_valid(market, index_src.data)
    peer_returns = (
        load_us_market_refs(code, daily_base, end_limit - timedelta(days=365), end_limit)
        if market == "US"
        else load_peer_returns(end_limit - timedelta(days=160), end_limit, code, safe_name)
    )
    naver_investor = {"status": "미국 주식은 국내 수급 조회 생략"} if market == "US" else naver_investor_table(code)

    decision = decision_logic(
        daily,
        weekly,
        monthly,
        levels,
        reliability,
        stop_precision,
        market_rel,
        validation_note,
        price_label,
        vol_label,
        naver_investor.get("status", "데이터 부족"),
    )
    validation_path, summary_path = save_csvs(out_dir, safe_name, code, validation, daily)
    chart_paths = make_charts(out_dir, safe_name, code, daily, weekly, monthly, levels, intraday60, intraday15)
    report_md = build_report(
        safe_name,
        code,
        market,
        daily,
        weekly,
        monthly,
        validation,
        reliability,
        price_label,
        vol_label,
        validation_note,
        intraday60,
        intraday15,
        levels,
        decision,
        market_rel,
        peer_returns,
        naver_investor,
        chart_paths,
    )

    md_path = out_dir / f"{safe_name}_{code}_매매타점_분석보고서.md"
    html_path = out_dir / f"{safe_name}_{code}_매매타점_분석보고서.html"
    metrics = {
        "rr1": decision.get("손익비1"),
        "rr2": decision.get("손익비2"),
        "confirm_rr": decision.get("확인진입손익비"),
        "reward1": decision.get("예상수익률1"),
        "current_price": decision.get("기준가"),
        "target1": decision.get("1차목표"),
        "target2": decision.get("2차목표"),
        "recovery_line": decision.get("회복확인선") or decision.get("회복/돌파공통확인선"),
        "breakout_line": decision.get("돌파가격"),
        "rebreak_line": decision.get("단기재돌파확인선"),
        "buy_low": decision.get("눌림하단"),
        "buy_high": decision.get("눌림상단"),
        "intraday_defense_line": decision.get("주의선"),
        "swing_defense_line": decision.get("방어선"),
        "rsi": decision.get("RSI"),
        "macd": last_valid(daily.iloc[-1], "MACD"),
        "macd_signal": last_valid(daily.iloc[-1], "MACD신호"),
        "macd_hist": last_valid(daily.iloc[-1], "MACD히스토그램"),
        "ma20": last_valid(daily.iloc[-1], "MA20"),
        "ma60": last_valid(daily.iloc[-1], "MA60"),
        "open_price": last_valid(daily.iloc[-1], "Open"),
        "high_price": last_valid(daily.iloc[-1], "High"),
        "low_price": last_valid(daily.iloc[-1], "Low"),
        "close_price": decision.get("기준가"),
        "volume_ratio20": decision.get("거래량비율"),
    }
    indicators = build_indicator_snapshot(daily, levels)
    context = {
        "stock_name": safe_name,
        "code": code,
        "market": market,
        "suffix": yf_suffix,
        "current_price": decision.get("기준가"),
        "validation_note": validation_note,
        "approved_price_range_set": [(levels.get("pull_low"), levels.get("pull_high"))],
        "supply_failed": "실패" in naver_investor.get("status", "") or "데이터 부족" in naver_investor.get("status", ""),
        "market_index_source": index_src.name,
        "market_index_symbol": index_src.note if index_src.note in {"KS11", "KQ11", "1001", "2001"} else (domestic_index_symbol(market) if market in {"KOSPI", "KOSDAQ"} else ""),
        "market_index_value": market_index_value,
        "market_index_invalid": market_index_invalid,
        "trade_state": decision.get("상태코드", {}),
    }
    report_reliability = reliability_breakdown(
        price_label,
        vol_label,
        naver_investor.get("status", "데이터 부족"),
        "해당 없음",
        True,
        validation_note,
    )
    state_dict_for_qa = decision.get("상태코드") or {}
    state_blocking_count = len(state_dict_for_qa.get("blocking_errors") or state_dict_for_qa.get("qa_blocking_errors") or [])
    final_report_md = f"{report_md.rstrip()}\n\n{build_report_qa_section(reliability, validation_error_count=state_blocking_count, reliability_details=report_reliability)}\n"
    try:
        run_report_qa(final_report_md, decision, metrics, context, indicators)
    except ReportValidationError as e:
        if md_path.exists():
            md_path.unlink()
        if html_path.exists():
            html_path.unlink()
        qa_fail_path = save_qa_failure(out_dir, safe_name, code, str(e), final_report_md)
        return f"""[분석 중단: 보고서 QA 실패]

종목: {safe_name} {code}
실패 사유:
{e}
수정 필요 항목:
- {qa_fail_path} 확인"""
    qa_fail_path = out_dir / f"{safe_name}_{code}_보고서_QA실패.md"
    if qa_fail_path.exists():
        qa_fail_path.unlink()
    md_path.write_text(final_report_md, encoding="utf-8-sig")
    html_path.write_text(html_from_markdown(final_report_md, f"{safe_name} {code} 매매타점 분석보고서"), encoding="utf-8-sig")

    return console_output(safe_name, code, decision, reliability, out_dir, md_path, html_path, levels)


def main() -> int:
    parser = argparse.ArgumentParser(description="국내/미국 주식 매매타점 분석 보고서 생성")
    parser.add_argument("code", help="종목코드 또는 티커")
    parser.add_argument("name", nargs="?", default=None, help="종목명")
    args = parser.parse_args()
    try:
        print(run(args.code, args.name))
        return 0
    except Exception as e:
        print(f"분석 실패: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
