from __future__ import annotations

"""Kiwoom-backed conditional command chart analyzer.

This entrypoint generates analysis-only reports. It never sends orders and does
not expose automated trading functions.
"""

import argparse
import sys
from dataclasses import dataclass
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
from core.decision_engine import DecisionContext, DecisionLevels, DecisionResult, PriceEvidence, evaluate_decision, format_price
from core.indicators import calculate_standard_indicators, indicators_valid
from kiwoom.provider import KiwoomDataError, KiwoomDataProvider


KST = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class DailyData:
    stock_name: str
    market: str
    yf_suffix: str
    frame: pd.DataFrame
    reliability: str
    validation_note: str
    stop_precision: bool


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


def collect_daily_data(code: str, fallback_name: str | None) -> DailyData:
    now = today_kst()
    end_limit = latest_completed_candidate(now)
    start_daily = end_limit - timedelta(days=365 * 6)
    start_validation = end_limit - timedelta(days=365 * 2)
    stock_name, market, yf_suffix = detect_name_market(code, fallback_name, end_limit)

    if market == "US":
        src_primary = load_yfinance(code, start_validation, end_limit, "yfinance")
        src_secondary = load_stooq(code, start_daily, end_limit)
        sources = [src_primary, src_secondary]
    else:
        src_pykrx = load_pykrx(code, start_daily, end_limit)
        src_fdr = load_fdr(code, start_daily, end_limit)
        src_yf = load_yfinance(code + yf_suffix, start_validation, end_limit, "yfinance")
        sources = [src_pykrx, src_fdr, src_yf]

    validation, reliability, stop_precision = source_validation(sources, end_limit)
    _price_label, _volume_label, validation_note = validation_labels(validation)
    if stop_precision:
        validation_note = f"{validation_note}; 대표가격 산정 불가"

    representative = ""
    if "대표가격사용" in validation.columns:
        reps = validation[validation["대표가격사용"] == "예"]
        if not reps.empty:
            representative = str(reps.iloc[0]["소스"])

    selected = None
    for src in sources:
        if src.name == representative and src.data is not None and not src.data.empty:
            selected = src
            break
    if selected is None:
        selected = next((src for src in sources if src.data is not None and not src.data.empty), None)
    if selected is None:
        raise RuntimeError("대표 일봉 데이터를 수집하지 못했습니다.")

    frame = selected.data[selected.data.index.date <= end_limit].copy()
    if frame.empty:
        raise RuntimeError("대표 일봉 데이터가 비어 있습니다.")
    frame = _standardize_daily_frame(frame)
    return DailyData(stock_name, market, yf_suffix, frame, reliability, validation_note, stop_precision)


def analyze_command_chart(
    code: str,
    fallback_name: str | None = None,
    provider: KiwoomDataProvider | None = None,
    now: datetime | None = None,
) -> str:
    code = code.strip()
    if not (code.isdigit() and len(code) == 6):
        code = code.upper()

    provider = provider or KiwoomDataProvider()
    invalid_reasons: list[str] = []
    daily_data: DailyData | None = None
    quote = None
    minute3 = pd.DataFrame()
    minute5 = pd.DataFrame()
    safe_name = sanitize_filename(fallback_name or code)

    try:
        daily_data = collect_daily_data(code, fallback_name)
        safe_name = sanitize_filename(daily_data.stock_name)
    except Exception as exc:
        invalid_reasons.append(f"대표가격 산정 불가: {exc}")

    try:
        quote = provider.get_quote(code)
    except Exception as exc:
        invalid_reasons.append(f"키움 현재가 수집 실패: {exc}")

    try:
        minute3 = provider.get_intraday_ohlcv(code, interval_minutes=3)
        minute5 = provider.get_intraday_ohlcv(code, interval_minutes=5)
        if len(minute3) < 20 or len(minute5) < 20:
            invalid_reasons.append("키움 체결 데이터 부족 또는 분봉 생성 실패")
    except Exception as exc:
        invalid_reasons.append(f"분봉 생성 실패: {exc}")

    session_intraday = is_korea_regular_session(now)
    if daily_data is None or quote is None:
        return _stop_and_write_failure(safe_name, code, invalid_reasons or ["장중/장외 상태 판정 실패"])

    out_dir = REPORTS_DIR / f"{safe_name}_{code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    current_price = safe_round(quote.price) or 0

    try:
        daily_ind = calculate_standard_indicators(daily_data.frame)
        minute3_ind = calculate_standard_indicators(minute3)
        minute5_ind = calculate_standard_indicators(minute5)
    except Exception as exc:
        return _stop_and_write_failure(safe_name, code, invalid_reasons + [f"필수 지표 계산 실패: {exc}"])

    if not indicators_valid(daily_ind) or not indicators_valid(minute3_ind) or not indicators_valid(minute5_ind):
        invalid_reasons.append("필수 지표 계산 실패")
    if daily_data.reliability == "낮음" or daily_data.stop_precision:
        invalid_reasons.append(f"데이터 신뢰도 낮음: {daily_data.validation_note}")

    daily_close = finite(daily_ind.iloc[-1].get("Close"))
    if daily_close and current_price and abs(current_price - daily_close) / daily_close > 0.30:
        invalid_reasons.append("키움 현재가와 일봉 대표 가격의 비정상 불일치")

    levels = build_decision_levels(current_price, daily_ind, minute3_ind, minute5_ind)
    decision = evaluate_decision(
        DecisionContext(
            current_price=current_price,
            levels=levels,
            is_intraday=session_intraday,
            data_valid=not invalid_reasons,
            invalid_reasons=tuple(invalid_reasons),
            volume_ratio20=last_valid(daily_ind.iloc[-1], "거래량비율20"),
            rsi14=last_valid(daily_ind.iloc[-1], "RSI14"),
            risk_reward=_risk_reward(levels),
        )
    )

    if decision.stopped:
        return _stop_and_write_failure(safe_name, code, list(decision.blocking_errors))

    report = render_report(safe_name, code, current_price, session_intraday, daily_data, decision)
    md_path = out_dir / f"{safe_name}_{code}_조건부명령형_차트분석.md"
    html_path = out_dir / f"{safe_name}_{code}_조건부명령형_차트분석.html"
    qa_path = out_dir / f"{safe_name}_{code}_보고서_QA실패.md"
    qa_path.unlink(missing_ok=True)
    md_path.write_text(report, encoding="utf-8")
    html_path.write_text(html_from_markdown(report, f"{safe_name} {code} 조건부 명령형 차트 분석"), encoding="utf-8")

    return _console_summary(safe_name, code, current_price, decision, md_path)


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


def render_report(stock_name: str, code: str, current_price: int, is_intraday: bool, daily_data: DailyData, decision: DecisionResult) -> str:
    session_text = "장중" if is_intraday else "장마감 이후"
    evidence = "\n".join(f"* {item.summary()}" for item in decision.price_evidence)
    actions = "\n".join(f"* {line}" for line in decision.actions)
    buy = "\n".join(f"* {line}" for line in decision.buy_conditions)
    no_buy = "\n".join(f"* {line}" for line in decision.no_buy_conditions)
    sell = "\n".join(f"* {line}" for line in decision.sell_conditions)
    holder = "\n".join(f"* {line}" for line in decision.holder_conditions)
    return f"""# {stock_name} {code} 조건부 명령형 차트 분석

[최종 매매 지시]

최종 판정: {decision.verdict}
현재가: {format_price(current_price)}
분석 구분: {session_text}

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

## 내부 검증

내부 검증: 통과
가격 신뢰도: {daily_data.reliability}
거래량 신뢰도: {daily_data.reliability}
지표 신뢰도: 높음
교차검증 완전성: {daily_data.reliability}
수급 신뢰도: 낮음
해석 완전성: 높음
"""


def _stop_and_write_failure(stock_name: str, code: str, reasons: list[str]) -> str:
    safe_name = sanitize_filename(stock_name or code)
    out_dir = REPORTS_DIR / f"{safe_name}_{code}"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{safe_name}_{code}_조건부명령형_차트분석.md"
    html_path = out_dir / f"{safe_name}_{code}_조건부명령형_차트분석.html"
    qa_path = out_dir / f"{safe_name}_{code}_보고서_QA실패.md"
    md_path.unlink(missing_ok=True)
    html_path.unlink(missing_ok=True)
    body = "\n".join(f"- {reason}" for reason in reasons) if reasons else "- 원인 미상"
    qa_path.write_text(f"# {safe_name} {code} 조건부 명령형 차트 분석 QA 실패\n\n최종판정: 분석 중단\n\n## 실패 사유\n\n{body}\n", encoding="utf-8")
    return f"""[조건부 명령형 차트 분석 중단]

종목: {safe_name} {code}
최종 판정: 분석 중단
QA 실패 파일: {qa_path}"""


def _console_summary(stock_name: str, code: str, current_price: int, decision: DecisionResult, md_path: Any) -> str:
    return f"""[조건부 명령형 차트 분석 완료]

종목: {stock_name} {code}
현재가: {format_price(current_price)}
최종 판정: {decision.verdict}
지금 할 행동: {' '.join(decision.actions)}
보고서 경로: {md_path}"""


def _standardize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "DateTime" not in out.columns:
        out.insert(0, "DateTime", pd.to_datetime(out.index))
    if "TradeValue" not in out.columns:
        out["TradeValue"] = out["Close"] * out["Volume"]
    return out[["DateTime", "Open", "High", "Low", "Close", "Volume", "TradeValue"]]


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
