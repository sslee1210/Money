from __future__ import annotations

FINALIZED_STOCK_AGENT_VERSION = "2026-06-17-final"

import argparse
import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from analyze_stock import (
    REPORTS_DIR,
    ReportValidationError,
    add_indicators,
    build_indicator_snapshot,
    build_report_qa_section,
    build_trade_state,
    breakout_action,
    breakout_sentence,
    buy_zone_action,
    buy_zone_sentence,
    calculate_trading_scores,
    classify_price_context,
    compact_validation_md,
    compressed_low_rr_warning,
    daily_trend_state_from_values,
    detect_name_market,
    df_to_md,
    downside_risk_pct,
    format_nearby_profile,
    format_price_range,
    format_price_level_list,
    fpct,
    fratio,
    get_tick_unit,
    html_from_markdown,
    ichimoku_position,
    infer_sector_label,
    iso,
    one_line,
    domestic_index_symbol,
    load_fdr,
    load_market_index,
    load_pykrx,
    load_yfinance,
    load_yfinance_intraday,
    macd_grade_from_values,
    ma_alignment,
    market_index_frame_is_valid,
    make_charts,
    money,
    bollinger_comment,
    macd_comment,
    moving_average_comment,
    nearest_levels,
    naver_investor_table,
    normalize_ohlcv,
    pct,
    practical_grade_from_text,
    practical_state_from_text,
    reliability_breakdown,
    rebreak_action,
    rebreak_sentence,
    relative_returns,
    resample_ohlcv,
    round_to_tick,
    separate_buy_high_from_breakout,
    assess_intraday_overheated_breakout,
    assess_volume_candle,
    assess_volume_momentum_conflict,
    breakout_volume_condition_comment,
    same_price_level,
    run as run_completed_analysis,
    run_report_qa,
    save_qa_failure,
    sanitize_filename,
    shares,
    state_code_report_rows,
    trade_state_to_dict,
    render_trade_state_actions,
    rsi_grade_from_value,
    monthly_chart_comment,
    recovery_confirmation_level,
    display_rebreak_line,
    rebreak_display_label,
    strategy_labels_by_price,
    target_sentence,
    rsi_comment,
    source_validation,
    validation_labels,
    volume_profile,
    ymd,
)


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://finance.naver.com/",
}


def today_kst() -> datetime:
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo("Asia/Seoul"))
    except Exception:
        return datetime.now()


def parse_num(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).replace(",", "").replace("백만", "").strip()
    if text in {"", "-", "N/A", "None"}:
        return float("nan")
    try:
        return float(text)
    except Exception:
        return float("nan")


def fmt_time(value: str | None) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S KST")
    except Exception:
        return value


def is_korean_regular_market(now: datetime) -> bool:
    return now.weekday() < 5 and time(9, 0) <= now.time() <= time(15, 30)


def previous_calendar_day(now: datetime) -> date:
    d = now.date() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def fetch_json(url: str) -> Any:
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def naver_intraday_sources(code: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        js = fetch_json(f"https://api.finance.naver.com/service/itemSummary.nhn?itemcode={code}")
        rows.append(
            {
                "구분": "장중",
                "소스": "네이버금융 itemSummary",
                "조회시각": "",
                "현재가": parse_num(js.get("now")),
                "전일종가": float("nan"),
                "시가": float("nan"),
                "고가": parse_num(js.get("high")),
                "저가": parse_num(js.get("low")),
                "거래량": parse_num(js.get("quant")),
                "비고": f"등락률 {js.get('rate')}%",
            }
        )
    except Exception as e:
        rows.append({"구분": "장중", "소스": "네이버금융 itemSummary", "조회시각": "", "현재가": np.nan, "비고": f"수집 실패: {type(e).__name__}"})

    try:
        js = fetch_json(f"https://polling.finance.naver.com/api/realtime/domestic/stock/{code}?query=SERVICE_ITEM:{code}")
        data = js["datas"][0]
        prev_close = parse_num(data.get("closePrice")) - parse_num(data.get("compareToPreviousClosePrice"))
        rows.append(
            {
                "구분": "장중",
                "소스": "네이버금융 realtime",
                "조회시각": fmt_time(data.get("localTradedAt")),
                "현재가": parse_num(data.get("closePrice")),
                "전일종가": prev_close,
                "시가": parse_num(data.get("openPrice")),
                "고가": parse_num(data.get("highPrice")),
                "저가": parse_num(data.get("lowPrice")),
                "거래량": parse_num(data.get("accumulatedTradingVolume")),
                "비고": f"marketStatus={data.get('marketStatus')}",
            }
        )
        over = data.get("overMarketPriceInfo") or {}
        if over:
            rows.append(
                {
                    "구분": "장중",
                    "소스": "네이버금융 NXT/통합 참고",
                    "조회시각": fmt_time(over.get("localTradedAt")),
                    "현재가": parse_num(over.get("overPrice")),
                    "전일종가": parse_num(over.get("overPrice")) - parse_num(over.get("compareToPreviousClosePrice")),
                    "시가": parse_num(over.get("openPrice")),
                    "고가": parse_num(over.get("highPrice")),
                    "저가": parse_num(over.get("lowPrice")),
                    "거래량": parse_num(over.get("accumulatedTradingVolume")),
                    "비고": "NXT/통합 가격은 참고값",
                }
            )
    except Exception as e:
        rows.append({"구분": "장중", "소스": "네이버금융 realtime", "조회시각": "", "현재가": np.nan, "비고": f"수집 실패: {type(e).__name__}"})

    try:
        js = fetch_json(f"https://m.stock.naver.com/api/stock/{code}/basic")
        rows.append(
            {
                "구분": "장중",
                "소스": "네이버 모바일 basic",
                "조회시각": fmt_time(js.get("localTradedAt")),
                "현재가": parse_num(js.get("closePrice")),
                "전일종가": parse_num(js.get("closePrice")) - parse_num(js.get("compareToPreviousClosePrice")),
                "시가": float("nan"),
                "고가": float("nan"),
                "저가": float("nan"),
                "거래량": float("nan"),
                "비고": f"marketStatus={js.get('marketStatus')}",
            }
        )
    except Exception as e:
        rows.append({"구분": "장중", "소스": "네이버 모바일 basic", "조회시각": "", "현재가": np.nan, "비고": f"수집 실패: {type(e).__name__}"})
    return rows


def pykrx_fdr_today_sources(code: str, today: date) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        src = load_pykrx(code, today, today)
        if not src.data.empty:
            r = src.data.iloc[-1]
            rows.append(
                {
                    "구분": "장중",
                    "소스": "pykrx 장중 일봉",
                    "조회시각": iso(today),
                    "현재가": float(r["Close"]),
                    "전일종가": float("nan"),
                    "시가": float(r["Open"]),
                    "고가": float(r["High"]),
                    "저가": float(r["Low"]),
                    "거래량": float(r["Volume"]),
                    "비고": "장중 미확정 일봉",
                }
            )
    except Exception as e:
        rows.append({"구분": "장중", "소스": "pykrx 장중 일봉", "조회시각": "", "현재가": np.nan, "비고": f"수집 실패: {type(e).__name__}"})
    try:
        src = load_fdr(code, today, today)
        if not src.data.empty:
            r = src.data.iloc[-1]
            rows.append(
                {
                    "구분": "장중",
                    "소스": "FinanceDataReader 장중 일봉",
                    "조회시각": iso(today),
                    "현재가": float(r["Close"]),
                    "전일종가": float("nan"),
                    "시가": float(r["Open"]),
                    "고가": float(r["High"]),
                    "저가": float(r["Low"]),
                    "거래량": float(r["Volume"]),
                    "비고": "장중 미확정 일봉",
                }
            )
    except Exception as e:
        rows.append({"구분": "장중", "소스": "FinanceDataReader 장중 일봉", "조회시각": "", "현재가": np.nan, "비고": f"수집 실패: {type(e).__name__}"})
    return rows


def yfinance_latest_source(ticker: str) -> dict[str, Any]:
    src = load_yfinance_intraday(ticker, "1m", "1d")
    if src.data.empty:
        return {"구분": "장중", "소스": "yfinance 1분봉", "조회시각": "", "현재가": np.nan, "비고": "수집 실패 또는 데이터 없음"}
    r = src.data.iloc[-1]
    return {
        "구분": "장중",
        "소스": "yfinance 1분봉",
        "조회시각": pd.Timestamp(src.data.index[-1]).strftime("%Y-%m-%d %H:%M:%S KST"),
        "현재가": float(r["Close"]),
        "전일종가": float("nan"),
        "시가": float(r["Open"]),
        "고가": float(r["High"]),
        "저가": float(r["Low"]),
        "거래량": float(r["Volume"]),
        "비고": "분봉 지연 가능",
    }


def load_peer_returns_for_stock(code: str, stock_name: str, start: date, end: date) -> dict[str, dict[str, float]]:
    if code == "010140" or "중공업" in stock_name:
        peers = {
            "HD현대중공업 329180": "329180",
            "한화오션 042660": "042660",
            "HD한국조선해양 009540": "009540",
            "HD현대미포 010620": "010620",
        }
    elif code == "009150":
        peers = {
            "LG이노텍 011070": "011070",
            "대덕전자 353200": "353200",
            "해성디에스 195870": "195870",
            "삼성전자 005930": "005930",
        }
    elif code == "033100" or "전기" in stock_name:
        peers = {
            "HD현대일렉트릭 267260": "267260",
            "LS ELECTRIC 010120": "010120",
            "효성중공업 298040": "298040",
            "LS 006260": "006260",
        }
    elif code == "403870" or "HPSP" in stock_name.upper():
        peers = {
            "테스 095610": "095610",
            "원익IPS 240810": "240810",
            "주성엔지니어링 036930": "036930",
            "피에스케이 319660": "319660",
        }
    else:
        peers = {}
    result: dict[str, dict[str, float]] = {}
    for name, peer_code in peers.items():
        src = load_fdr(peer_code, start, end)
        if src.data.empty:
            continue
        s = src.data["Close"].dropna()
        vals: dict[str, float] = {}
        for p in [20, 60]:
            if len(s) > p:
                vals[f"{p}일"] = pct(s.iloc[-1], s.iloc[-p - 1])
        result[name] = vals
    return result


def evaluate_yfinance_minute(row: dict[str, Any], representative_price: float, now: datetime) -> tuple[bool, str]:
    price = parse_num(row.get("현재가"))
    volume = parse_num(row.get("거래량"))
    open_price = parse_num(row.get("시가"))
    high_price = parse_num(row.get("고가"))
    low_price = parse_num(row.get("저가"))
    quote_time_raw = row.get("조회시각")
    delayed = False
    try:
        quote_time = datetime.strptime(str(quote_time_raw).replace(" KST", ""), "%Y-%m-%d %H:%M:%S")
        delayed = (now.replace(tzinfo=None) - quote_time).total_seconds() > 600
    except Exception:
        delayed = True
    zero_volume = np.isfinite(volume) and volume <= 0
    flat_ohlc = all(np.isfinite(v) for v in [open_price, high_price, low_price, price]) and len({open_price, high_price, low_price, price}) == 1
    price_diff = abs(pct(price, representative_price)) if np.isfinite(price) and representative_price else np.nan
    far_price = np.isfinite(price_diff) and price_diff > 0.5
    excluded = delayed or zero_volume or flat_ohlc or far_price
    reasons = []
    if delayed:
        reasons.append("10분 이상 지연")
    if zero_volume:
        reasons.append("거래량 0")
    if flat_ohlc:
        reasons.append("OHLC 동일")
    if far_price:
        reasons.append("네이버 현재가와 0.5% 이상 차이")
    return (not excluded), "정상" if not reasons else ", ".join(reasons)


def choose_intraday_basis(rows: list[dict[str, Any]], prev_close: float, now: datetime) -> tuple[dict[str, Any], str, str, str, dict[str, Any]]:
    df = pd.DataFrame(rows)
    usable = df[pd.to_numeric(df.get("현재가"), errors="coerce").notna()].copy()
    if usable.empty:
        return {"현재가": prev_close, "전일종가": prev_close}, "낮음", "실패", "실패", {}

    realtime_sources = ["네이버금융 realtime", "네이버 모바일 basic"]
    realtime_usable = usable[usable["소스"].isin(realtime_sources)].copy()
    primary_price_usable = usable[~usable["소스"].astype(str).str.contains("NXT/통합|yfinance", na=False)].copy()
    price_usable = (
        realtime_usable
        if not realtime_usable.empty
        else primary_price_usable
        if not primary_price_usable.empty
        else usable[~usable["소스"].astype(str).str.contains("NXT/통합 참고", na=False)].copy()
    )
    prices = pd.to_numeric(price_usable["현재가"], errors="coerce")
    regular_volume_rows = usable[~usable["소스"].astype(str).str.contains("NXT/통합|yfinance", na=False)].copy()
    vols = pd.to_numeric(regular_volume_rows.get("거래량"), errors="coerce")
    diff = (prices.max() - prices.min()) / prices.mean() * 100 if prices.mean() else np.nan
    vol_diff = (vols.max() - vols.min()) / vols.mean() * 100 if vols.notna().sum() >= 2 and vols.mean() else np.nan
    price_label = "실패" if diff > 1.0 else ("경고" if diff > 0.5 else "통과")
    if price_label == "실패" and realtime_usable.empty:
        price_label = "장중 경고"
    volume_label = "경고" if np.isfinite(vol_diff) and vol_diff > 15 else "통과"
    reliability = "낮음" if price_label == "실패" else ("중간" if price_label in {"경고", "장중 경고"} or volume_label == "경고" else "높음")

    priority = ["네이버금융 realtime", "네이버금융 itemSummary", "네이버 모바일 basic", "pykrx 장중 일봉", "FinanceDataReader 장중 일봉"]
    selected = None
    for name in priority:
        m = usable[usable["소스"] == name]
        if not m.empty:
            selected = m.iloc[0].to_dict()
            break
    if selected is None:
        selected = usable.iloc[0].to_dict()
    selected["전일종가"] = selected.get("전일종가") if np.isfinite(parse_num(selected.get("전일종가"))) else prev_close
    nxt_rows = usable[usable["소스"].astype(str).str.contains("NXT/통합 참고", na=False)]
    yfinance_rows = df[df["소스"].astype(str).str.contains("yfinance", na=False)]
    yf_ok = True
    yf_note = "해당 없음"
    if not yfinance_rows.empty:
        yf_row = yfinance_rows.iloc[0].to_dict()
        yf_price = parse_num(yf_row.get("현재가"))
        yf_note_raw = str(yf_row.get("비고") or "")
        if not np.isfinite(yf_price):
            yf_ok = False
            yf_note = yf_note_raw or "수집 실패 또는 데이터 없음"
        else:
            yf_ok, yf_note = evaluate_yfinance_minute(yf_row, float(selected["현재가"]), now)
    price_range = {
        "min": float(prices.min()) if len(prices) else np.nan,
        "max": float(prices.max()) if len(prices) else np.nan,
        "diff_pct": diff,
        "regular_volume": float(pd.to_numeric(regular_volume_rows.get("거래량"), errors="coerce").dropna().iloc[0])
        if pd.to_numeric(regular_volume_rows.get("거래량"), errors="coerce").dropna().size
        else np.nan,
        "nxt_volume": float(pd.to_numeric(nxt_rows.get("거래량"), errors="coerce").dropna().iloc[0])
        if not nxt_rows.empty and pd.to_numeric(nxt_rows.get("거래량"), errors="coerce").dropna().size
        else np.nan,
        "yfinance_minute_ok": yf_ok,
        "yfinance_note": yf_note,
    }
    return selected, reliability, price_label, volume_label, price_range


def validation_table_for_report(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in ["현재가", "전일종가", "시가", "고가", "저가", "거래량"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{int(round(float(x))):,}" if pd.notna(x) and np.isfinite(float(x)) else "")
    return out.to_markdown(index=False)


def source_rows_to_csv(daily_validation: pd.DataFrame, intraday_validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in daily_validation.iterrows():
        rows.append(
            {
                "구분": "완료일봉",
                "소스": r.get("소스"),
                "조회시각/최신거래일": r.get("최신거래일"),
                "현재가/종가": r.get("종가"),
                "전일종가": "",
                "시가": r.get("시가"),
                "고가": r.get("고가"),
                "저가": r.get("저가"),
                "거래량": r.get("거래량"),
                "비고": r.get("비고"),
            }
        )
    for _, r in intraday_validation.iterrows():
        rows.append(
            {
                "구분": "장중",
                "소스": r.get("소스"),
                "조회시각/최신거래일": r.get("조회시각"),
                "현재가/종가": r.get("현재가"),
                "전일종가": r.get("전일종가"),
                "시가": r.get("시가"),
                "고가": r.get("고가"),
                "저가": r.get("저가"),
                "거래량": r.get("거래량"),
                "비고": r.get("비고"),
            }
        )
    return pd.DataFrame(rows)


def last_valid(row: pd.Series, key: str) -> float:
    try:
        v = row[key]
        return float(v) if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def make_report(
    stock_name: str,
    code: str,
    market: str,
    suffix: str,
    analysis_time: datetime,
    current: dict[str, Any],
    price_meta: dict[str, Any],
    intraday_validation: pd.DataFrame,
    intraday_reliability: str,
    intraday_price_label: str,
    intraday_volume_label: str,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    daily_validation: pd.DataFrame,
    daily_reliability: str,
    daily_price_label: str,
    daily_volume_label: str,
    daily_note: str,
    levels: dict[str, Any],
    intraday60: pd.DataFrame,
    intraday15: pd.DataFrame,
    market_rel: dict[str, float],
    peer_returns: dict[str, dict[str, float]],
    investor: dict[str, Any],
    chart_paths: list[Path],
) -> tuple[str, dict[str, Any]]:
    row = daily.iloc[-1]
    is_kr_stock = market in {"KOSPI", "KOSDAQ", "KONEX"} or suffix in {".KS", ".KQ"}

    def normalize_display_price(value: Any, direction: str = "nearest") -> float:
        parsed = parse_num(value)
        if not np.isfinite(parsed):
            return parsed
        return float(round_to_tick(parsed, direction)) if is_kr_stock else float(parsed)

    current_price = normalize_display_price(current["현재가"])
    prev_close = normalize_display_price(current.get("전일종가", daily.iloc[-1]["Close"]))
    open_price = normalize_display_price(current.get("시가"))
    high_price = normalize_display_price(current.get("고가"))
    low_price = normalize_display_price(current.get("저가"))
    intraday_volume = parse_num(current.get("거래량"))
    avg20_volume = float(daily["Volume"].tail(20).mean())
    volume_ratio = intraday_volume / avg20_volume if np.isfinite(intraday_volume) and avg20_volume else np.nan
    market_start = analysis_time.replace(hour=9, minute=0, second=0, microsecond=0)
    market_end = analysis_time.replace(hour=15, minute=30, second=0, microsecond=0)
    elapsed = max(0.0, min((analysis_time - market_start).total_seconds(), (market_end - market_start).total_seconds()))
    progress = elapsed / (market_end - market_start).total_seconds() if (market_end - market_start).total_seconds() else np.nan
    weighted_volume = intraday_volume / progress if np.isfinite(intraday_volume) and progress > 0 else np.nan
    weighted_volume_ratio = weighted_volume / avg20_volume if np.isfinite(weighted_volume) and avg20_volume else np.nan

    deep_pull_low = float(levels["pull_low"])
    deep_pull_high = float(levels["pull_high"])
    breakout_price = float(levels["breakout"])
    target1 = float(levels["target1"])
    target2 = float(levels["target2"])
    warning = float(levels["warning"])
    defense = float(levels["defense"])
    shallow_low = max(warning, prev_close, current_price - max(last_valid(row, "ATR14") * 0.45, current_price * 0.035))
    shallow_high = min(current_price - 1, breakout_price, max(prev_close, current_price - max(last_valid(row, "ATR14") * 0.15, current_price * 0.012)))
    if shallow_low > shallow_high:
        shallow_low, shallow_high = min(prev_close, warning), max(prev_close, warning)
    shallow_low = round_to_tick(shallow_low, "down")
    shallow_high = round_to_tick(shallow_high, "up")
    close_confirm_line = float(levels["breakout"])
    if np.isfinite(close_confirm_line):
        shallow_high = separate_buy_high_from_breakout(shallow_high, close_confirm_line)
    if shallow_low > shallow_high:
        shallow_low = shallow_high
    shallow_pull = format_price_range(shallow_low, shallow_high, "단일 지지선")
    deep_pull = format_price_range(deep_pull_low, deep_pull_high, "단일 지지선")
    pull = f"얕은 눌림 {shallow_pull} / 깊은 눌림 {deep_pull}"
    today_close_check = breakout_price
    rebreak_line = round_to_tick((high_price if np.isfinite(high_price) else breakout_price) + 1, "up")
    psychological_seed = max(current_price, high_price if np.isfinite(high_price) else current_price, close_confirm_line)
    psychological_line = round_to_tick(math.ceil(psychological_seed / 1000) * 1000, "up")
    breakout_entry_line = max(rebreak_line, psychological_line if psychological_line > current_price else rebreak_line)
    intraday_warning_line = max(
        warning,
        prev_close if np.isfinite(prev_close) else warning,
        low_price if np.isfinite(low_price) else warning,
    )
    intraday_defense_line = min(
        warning,
        prev_close if np.isfinite(prev_close) else warning,
        low_price if np.isfinite(low_price) else warning,
    )
    if intraday_warning_line <= intraday_defense_line:
        intraday_warning_line = round_to_tick(intraday_defense_line + 1, "up")

    reward1 = pct(target1, current_price)
    reward2 = pct(target2, current_price)
    risk_defense = downside_risk_pct(current_price, defense)
    risk_warning = downside_risk_pct(current_price, warning)
    rr1 = reward1 / risk_defense if risk_defense > 0 else np.nan
    rr2 = reward2 / risk_defense if risk_defense > 0 else np.nan
    target1_is_near = (np.isfinite(reward1) and reward1 < 3) or (np.isfinite(rr1) and rr1 < 1.5)
    target1_role = "근접 저항/보유자 일부 익절 후보" if target1_is_near else "신규매수 기준 1차 목표"
    entry_target1 = target2 if target1_is_near else target1
    entry_target2 = round_to_tick(max(target2, entry_target1 * 1.08), "up") if target1_is_near else target2
    entry_reward1 = pct(entry_target1, current_price)
    entry_reward2 = pct(entry_target2, current_price)
    entry_rr1 = entry_reward1 / risk_defense if risk_defense > 0 else np.nan
    entry_rr2 = entry_reward2 / risk_defense if risk_defense > 0 else np.nan
    rsi_value = last_valid(row, "RSI14")
    macd_value = last_valid(row, "MACD")
    macd_signal = last_valid(row, "MACD신호")
    macd_hist = last_valid(row, "MACD히스토그램")

    broke_and_failed = np.isfinite(high_price) and high_price >= breakout_price and current_price < breakout_price
    current_under_breakout = current_price < close_confirm_line
    poor_rr = not np.isfinite(rr1) or rr1 < 1.5
    close_target = np.isfinite(reward1) and reward1 < 3
    breakout_disabled = breakout_entry_line >= target1
    insufficient_volume = np.isfinite(volume_ratio) and volume_ratio < 0.8
    now_buy = "불가"
    sell_needed = "아니오"
    if current_under_breakout:
        if np.isfinite(high_price) and high_price < close_confirm_line:
            no_buy_reason = (
                f"현재가는 {money(close_confirm_line)} 종가 유지 확인선 아래이며, "
                f"{money(high_price)} 장중 고점 부근 재돌파 확인이 필요합니다."
            )
        else:
            no_buy_reason = f"장중 {money(close_confirm_line)} 돌파 시도 후 안착하지 못했습니다."
    elif poor_rr:
        no_buy_reason = "근접 저항까지 손익비가 1.5 미만"
    elif close_target:
        no_buy_reason = "근접 저항이 현재가에서 3% 이내라 신규매수 여유가 부족"
    elif breakout_disabled:
        no_buy_reason = "단기 재돌파 확인선이 근접 저항 이상이라 돌파 매수 비활성화"
    elif insufficient_volume:
        no_buy_reason = "돌파 구간 대비 장중 거래량 확인이 아직 부족"
    else:
        no_buy_reason = "장중 돌파 시도 구간이며 오늘 종가 유지 확인 전 신규매수 보류"

    if intraday_reliability == "낮음" or intraday_price_label == "실패":
        final = "데이터 불일치로 정밀 판단 중단"
    elif broke_and_failed:
        final = "돌파 재확인 대기"
    elif poor_rr or close_target or breakout_disabled:
        final = "돌파 재확인 대기" if current_price >= close_confirm_line else "눌림목 대기"
    elif current_price >= close_confirm_line:
        final = "종가 유지 전 신규매수 보류"
    elif deep_pull_low <= current_price <= deep_pull_high and max(rr1, rr2) >= 1.5:
        final = "눌림목 대기"
    else:
        final = "눌림목 대기"

    primary_strategy = "종가 유지 확인, 신규매수 보류" if current_price >= close_confirm_line else "회복 확인 대기"
    secondary_strategy = f"{money(close_confirm_line)} 이상 종가 유지 + 거래량 1.2배 이상은 장중 돌파 시도일 뿐, 신규매수는 종가 확정 후 재판단"

    near_high_rebreak = round_to_tick(high_price if np.isfinite(high_price) else max(current_price, close_confirm_line), "nearest")
    price_context = classify_price_context(
        current_price,
        shallow_low,
        shallow_high,
        near_high_rebreak,
        close_confirm_line,
        entry_target1,
        intraday_defense_line,
    )
    buy_context_text = buy_zone_sentence(price_context)
    rebreak_context_text = rebreak_sentence(price_context)
    breakout_context_text = breakout_sentence(price_context)
    target_context_text = target_sentence(price_context)
    buy_action_text = buy_zone_action(price_context)
    rebreak_action_text = rebreak_action(price_context)
    breakout_action_text = breakout_action(price_context)
    strategy_labels = strategy_labels_by_price(
        current_price,
        shallow_low,
        shallow_high,
        near_high_rebreak,
        close_confirm_line,
        intraday_defense_line,
        defense,
        intraday_reliability,
    )
    if intraday_reliability == "낮음" or intraday_price_label == "실패":
        primary_strategy = "데이터 확인 대기"
        final = "데이터 불일치로 정밀 판단 중단"
    elif current_price >= close_confirm_line:
        primary_strategy = "종가 유지 확인, 신규매수 보류"
        final = "종가 유지 전 신규매수 보류"
    else:
        primary_strategy = strategy_labels["primary_strategy"]
        final = strategy_labels["final"]
    if current_price >= close_confirm_line:
        intraday_position_text = (
            f"{breakout_context_text} {target_context_text}"
        )
        intraday_timing_text = "돌파 기준 위에 있으나 근접 저항이 가까워 신규 매수는 종가 유지 확인이 우선입니다."
        final_action_text = f"오늘 종가가 {money(today_close_check)} 위에서 유지되는지 확인해야 합니다."
    elif broke_and_failed:
        intraday_position_text = (
            f"장중 고가는 {money(high_price)}로 {money(close_confirm_line)} 돌파선을 넘었지만 "
            "현재가는 돌파선 아래라 돌파 유지가 확인되지 않았습니다."
        )
        intraday_timing_text = f"장중 고점 {money(high_price)} 이후 현재가가 밀려 돌파 유지 실패 가능성이 있습니다."
        final_action_text = f"오늘 종가가 {money(today_close_check)} 위에서 마감하는지 확인해야 합니다."
    else:
        intraday_position_text = breakout_context_text
        intraday_timing_text = "단기 반등은 있으나 돌파 기준과 눌림목 회복 확인 전까지 신규 매수는 보수적으로 봅니다."
        final_action_text = f"오늘 종가가 {money(today_close_check)} 위로 회복되는지 확인해야 합니다."

    yfinance_note = str(price_meta.get("yfinance_note") or "수집 실패 또는 데이터 없음")
    yfinance_status = "정상" if price_meta.get("yfinance_minute_ok") else yfinance_note
    minute_reliability = "통과" if price_meta.get("yfinance_minute_ok") else ("부분 통과" if intraday_reliability == "높음" else "경고")
    regular_volume = price_meta.get("regular_volume", np.nan)
    nxt_volume = price_meta.get("nxt_volume", np.nan)
    price_range_low = normalize_display_price(price_meta.get("min"))
    price_range_high = normalize_display_price(price_meta.get("max"))
    price_range_text = format_price_range(price_range_low, price_range_high, "단일 확인가")
    early_volume_note = (
        " 장초반 30분 이내 시간가중 환산 거래량은 과장될 수 있으므로 참고값으로만 봅니다."
        if analysis_time.time() < time(9, 30)
        else ""
    )
    volume_comment = (
        f"장중 누적 거래량은 20일 평균의 {fratio(volume_ratio)}이나, "
        f"{analysis_time.strftime('%H:%M')} 기준 시간가중 환산 거래량은 20일 평균의 {fratio(weighted_volume_ratio)}입니다. "
        f"{early_volume_note} 거래량 판단은 돌파 지속 여부를 함께 확인합니다."
    )
    volume_context = assess_volume_candle(
        open_price,
        high_price,
        low_price,
        current_price,
        weighted_volume_ratio if np.isfinite(weighted_volume_ratio) else volume_ratio,
    )
    if volume_context.get("bearish_high_volume"):
        volume_comment = volume_context["comment"]
    breakout_rule_text = f"{rebreak_context_text} 신규 돌파 매수는 {breakout_context_text}"
    if volume_context.get("bearish_high_volume"):
        breakout_rule_text = breakout_volume_condition_comment(volume_context)
    breakout_entry_text = (
        f"비활성화: {money(breakout_entry_line)}이 근접 저항 {money(target1)} 이상"
        if breakout_disabled
        else f"{money(breakout_entry_line)}"
    )
    short_rebreak_text = f"{money(near_high_rebreak)}"
    daily_breakout_text = f"{money(close_confirm_line)}"
    investor_status = investor.get("status", "데이터 부족")
    if "실패" in investor_status or "데이터 부족" in investor_status:
        investor_judgment = "수급 데이터 부족으로 수급 판단 보류"
        investor_detail = "외국인/기관/개인 수급 데이터 수집 실패로 수급 기반 판단은 보류합니다. 따라서 최종 판단은 가격, 거래량, 지표, 시장/섹터 상대강도 중심으로 제한합니다."
    else:
        investor_judgment = "수급 데이터는 보조 확인"
        investor_detail = "수집 가능한 공개 수급 데이터는 방향성 보조로만 사용합니다."
    entry_label = "회복 확인가" if current_price < shallow_low else "눌림목 지지가"
    if current_price < shallow_low:
        recovery_price_text = shallow_pull
        recovery_line = shallow_low
        support_price_text = "해당 없음"
    elif current_price < close_confirm_line:
        recovery_price_text, recovery_line = recovery_confirmation_level(
            current_price,
            shallow_high,
            last_valid(row, "MA20"),
            last_valid(row, "BB중심"),
            close_confirm_line,
            allow_inside_support=True,
        )
        support_price_text = shallow_pull
    else:
        recovery_price_text = "이미 회복, 종가 유지 확인으로 대체" if current_price >= close_confirm_line else "해당 없음"
        recovery_line = None
        support_price_text = shallow_pull
    common_confirmation_active = recovery_line is not None and same_price_level(recovery_line, close_confirm_line)
    if common_confirmation_active:
        recovery_price_text = money(close_confirm_line)
    rebreak_display_text, rebreak_duplicate_action, rebreak_merged = display_rebreak_line(near_high_rebreak, entry_target1, entry_target2)
    rebreak_label = rebreak_display_label(near_high_rebreak, entry_target1, entry_target2)
    rebreak_display_action = rebreak_duplicate_action if rebreak_merged else rebreak_action_text
    if abs(current_price - near_high_rebreak) <= get_tick_unit(current_price):
        rebreak_label = "장중 현재가 유지 기준"
        rebreak_display_text = f"장중 현재가 유지 기준 {money(current_price)}"
        rebreak_display_action = "현재가 부근 위 유지 시 단기 탄력 유지, 이탈 시 장중 돌파 실패 가능성 확인"
        short_rebreak_text = rebreak_display_text
        rebreak_context_text = rebreak_display_action
    if rebreak_merged:
        rebreak_context_text = "단기 재돌파선은 1차 목표/강한 저항과 중복되어 별도 진입선으로 쓰지 않습니다."
    intraday_defense_risk = downside_risk_pct(current_price, intraday_defense_line)
    intraday_defense_rr = entry_reward1 / intraday_defense_risk if np.isfinite(intraday_defense_risk) and intraday_defense_risk > 0 else np.nan
    confirm_entry_line = recovery_line if recovery_line is not None else close_confirm_line
    confirm_defense_line = intraday_defense_line if entry_label == "회복 확인가" else defense
    confirm_reward = pct(entry_target1, confirm_entry_line)
    confirm_risk = downside_risk_pct(confirm_entry_line, confirm_defense_line)
    confirm_rr = confirm_reward / confirm_risk if np.isfinite(confirm_risk) and confirm_risk > 0 else np.nan
    overheat_breakout_state = assess_intraday_overheated_breakout(
        current_price,
        close_confirm_line,
        weighted_volume_ratio,
        rsi_value,
        intraday_defense_rr,
        confirm_rr,
        entry_rr1,
    )
    if overheat_breakout_state["applies"]:
        now_buy = "불가"
        primary_strategy = overheat_breakout_state["primary_strategy"]
        final = overheat_breakout_state["final"]
        no_buy_reason = overheat_breakout_state["no_buy_reason"]
        secondary_strategy = "보유자는 종가 유지 확인, 신규자는 눌림 또는 종가 확정 후 재판단"
    volume_momentum_conflict = assess_volume_momentum_conflict(
        current_price,
        shallow_high,
        recovery_line if recovery_line is not None else close_confirm_line,
        weighted_volume_ratio,
        rsi_value,
        macd_value,
        macd_signal,
    )
    if volume_momentum_conflict["applies"] and not overheat_breakout_state["applies"]:
        now_buy = "불가"
        primary_strategy = volume_momentum_conflict["primary_strategy"]
        final = volume_momentum_conflict["final"]
        no_buy_reason = volume_momentum_conflict["template"]
        secondary_strategy = f"{money(close_confirm_line)} 종가 안착 또는 {money(shallow_high)} 재지지 전까지 보류"
    near_intraday_defense = (
        np.isfinite(current_price)
        and np.isfinite(intraday_defense_line)
        and current_price > 0
        and abs(current_price - intraday_defense_line) / current_price <= 0.005
    )
    near_defense_warning_text = (
        "장중 방어선이 가까워 신규 진입은 손익비보다 실패 확인 리스크가 더 큼"
        if near_intraday_defense
        else ""
    )
    reliability_parts = reliability_breakdown(
        intraday_price_label,
        intraday_volume_label,
        investor_judgment,
        intraday_reliability,
        True,
        daily_note,
    )
    intraday_location_line = (
        f"현재가는 {money(current_price)}이며 종가 유지 확인선 {money(close_confirm_line)} 위입니다."
        if current_price >= close_confirm_line
        else f"현재가는 {money(current_price)}이며 종가 유지 확인선 {money(close_confirm_line)} 아래입니다."
    )
    shallow_confirm_text = buy_context_text
    shallow_alert_meaning = "회복 후 지지 확인 가격" if current_price < shallow_low else "지지 확인 가격"
    in_shallow_pull = shallow_low <= current_price <= shallow_high
    in_deep_pull = deep_pull_low <= current_price <= deep_pull_high
    new_buyer_text = (
        f"지금 매수 불가. {buy_context_text} {rebreak_context_text} {breakout_context_text}"
        if now_buy == "불가"
        else f"{money(close_confirm_line)} 위 유지와 거래량 확인 시 분할 매수 가능."
    )
    if now_buy == "불가" and not overheat_breakout_state["applies"]:
        no_buy_reason = one_line(f"{buy_context_text} {breakout_context_text} {target_context_text}")
    elif overheat_breakout_state["applies"]:
        no_buy_reason = overheat_breakout_state["no_buy_reason"]
    if in_shallow_pull and in_deep_pull:
        pullback_status_text = (
            f"현재가는 얕은 눌림목과 깊은 눌림목이 겹치는 구간에 있으나, 전일 종가와 장중 주의선 아래에 있어 "
            f"신규매수는 {shallow_confirm_text} 또는 {money(intraday_defense_line)} 이탈 방어 확인 등 반등 확인 후에만 검토합니다."
        )
    elif in_shallow_pull:
        pullback_status_text = f"현재가는 얕은 눌림목 구간 안에 있으므로 {shallow_confirm_text} 후에만 신규매수를 검토합니다."
    elif in_deep_pull:
        pullback_status_text = (
            f"현재가는 깊은 눌림목 구간 안에 있으나, 장중 방어선과 가까워 신규매수는 {shallow_confirm_text} 또는 "
            f"{money(intraday_defense_line)} 이탈 방어 확인 등 반등 확인 후에만 검토합니다."
        )
    else:
        pullback_status_text = "현재가는 눌림목 구간 밖이므로 주요 지지와 돌파 확인을 우선합니다."
    recovery_action_price = recovery_price_text if recovery_price_text != "해당 없음" else money(close_confirm_line)
    recovery_action_text = (
        "신규 진입 기준은 오늘 종가 유지 확인으로 대체"
        if "이미 회복" in recovery_price_text or "종가 유지 확인" in recovery_price_text
        else "현재가보다 위에 있는 재진입 확인 가격"
    )
    if recovery_price_text == "해당 없음":
        recovery_action_text = "회복 확인가 미형성, 일봉 돌파 확인가 종가 회복 확인"
    if common_confirmation_active:
        recovery_action_price = money(close_confirm_line)
        recovery_action_text = "종가 안착 + 거래량 유지 확인"
    holder_text = (
        f"{money(close_confirm_line)} 종가 회복 확인 전 추가매수 보류. "
        f"{money(shallow_high)} 재이탈 시 단기 비중 축소 검토. "
        f"{money(entry_target1)} 접근 시 일부 익절 검토. "
        f"{money(intraday_defense_line)} 이탈 시 장중 방어, {money(defense)} 일봉 종가 이탈 시 방어/손절합니다."
    )
    if overheat_breakout_state["applies"]:
        new_buyer_text = "지금 추격 금지. 눌림목 지지가 재지지 또는 종가 확정 후 다음 거래일 눌림 확인 시 검토합니다."
        additional_buyer_text = "추가매수는 신규매수보다 더 엄격하게 보류하고, 과열 해소와 종가 확정 후 눌림 재지지를 재확인합니다."
    elif volume_momentum_conflict["applies"]:
        new_buyer_text = f"지금 매수 불가. {money(close_confirm_line)} 종가 안착 또는 {money(shallow_high)} 재지지 확인 전까지 보류합니다."
        additional_buyer_text = f"추가매수는 {money(close_confirm_line)} 종가 안착과 RSI/MACD 회복 확인 전까지 보류합니다."
    else:
        additional_buyer_text = f"{money(close_confirm_line)} 종가 유지와 다음 거래일 눌림 확인 전까지 추가매수는 보류합니다."
    rr_caution_text = (
        f"현재가 기준 단순 손익비는 {fratio(entry_rr1)}로 양호하지만, 돌파 유지 실패와 데이터 신뢰도 {intraday_reliability} 상태로 인해 "
        "회복 확인가 또는 돌파 확인가 조건 미충족으로 지금 바로 신규매수 조건은 충족하지 못했습니다."
        if now_buy == "불가" and np.isfinite(entry_rr1) and entry_rr1 >= 1.5
        else "현재가 기준 손익비와 매수 가능 여부는 조건 충족 여부를 분리해서 해석합니다."
    )
    rr_low_warnings: list[str] = []
    if np.isfinite(intraday_defense_rr) and intraday_defense_rr < 1.2:
        rr_low_warnings.append("장중 방어선 기준 손익비 1.2 미만으로 장중 신규 진입 매력 낮음")
    if np.isfinite(entry_rr1) and entry_rr1 < 1.2:
        rr_low_warnings.append("스윙 손절선 기준 손익비 부족으로 신규매수 매력 낮음")
    if np.isfinite(entry_rr1) and entry_rr1 < 1.0:
        rr_low_warnings.append("스윙 손절선 기준 손익비 1.0 미만으로 스윙 신규매수 부적합")
    if np.isfinite(confirm_rr) and confirm_rr < 1.0:
        rr_low_warnings.append("회복/돌파 진입 기준 손익비 1.0 미만으로 돌파 추격매수 부적합")
    low_swing_or_breakout_rr = any(np.isfinite(v) and v < 1.2 for v in [entry_rr1, confirm_rr])
    if low_swing_or_breakout_rr:
        rr_low_warnings.append("스윙/돌파 신규매수 매력 낮음")
        if np.isfinite(intraday_defense_rr) and intraday_defense_rr >= 1.2:
            rr_low_warnings.append("단기 트레이딩 손익비는 가능하나 스윙/돌파 추격 손익비는 부족")
        if "손익비" not in final:
            final = f"{final}·스윙/돌파 손익비 부족"
    if np.isfinite(rsi_value) and rsi_value >= 70 and any(np.isfinite(v) and v < 1.2 for v in [intraday_defense_rr, confirm_rr, entry_rr1]):
        rr_low_warnings.append("과열 추격 금지")
    low_rr_summary = compressed_low_rr_warning(entry_rr1, confirm_rr)
    if low_rr_summary:
        rr_caution_text = low_rr_summary
        if np.isfinite(entry_rr1) and entry_rr1 >= 1.5 and now_buy == "불가":
            rr_caution_text = (
                f"스윙 손익비는 {fratio(entry_rr1)}로 양호하지만, "
                f"회복/돌파 진입 손익비 {fratio(confirm_rr)}로 낮아 회복 확인가 또는 돌파 확인가 조건 미충족 상태에서는 "
                f"지금 바로 신규매수 조건은 충족하지 못했습니다. {low_rr_summary}"
            )
    if rr_low_warnings:
        extra_rr_warnings = []
        for warning in rr_low_warnings:
            if "장중 신규 진입 매력 낮음" in warning and "장중 신규 진입 매력 낮음" not in rr_caution_text:
                extra_rr_warnings.append(warning)
            elif "과열 추격 금지" in warning and "과열 추격 금지" not in rr_caution_text:
                extra_rr_warnings.append(warning)
        if extra_rr_warnings:
            rr_caution_text = one_line(f"{rr_caution_text} {'; '.join(dict.fromkeys(extra_rr_warnings))}.")
    if (
        np.isfinite(rsi_value)
        and rsi_value >= 70
        and np.isfinite(confirm_rr)
        and confirm_rr < 1.0
        and "신규 추격매수 부적합" not in rr_caution_text
    ):
        rr_caution_text = one_line(f"{rr_caution_text} 과열 추격 금지·신규 추격매수 부적합.")
    if near_defense_warning_text:
        rr_caution_text = one_line(f"{rr_caution_text} {near_defense_warning_text}.")

    trade_state = build_trade_state(
        current_price=current_price,
        pullback_low=shallow_low,
        pullback_high=shallow_high,
        recovery_line=recovery_line,
        breakout_line=close_confirm_line,
        target1=entry_target1,
        target2=entry_target2,
        defense_line=intraday_defense_line,
        short_rebreak_line=near_high_rebreak,
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=current_price,
        volume_ratio20=weighted_volume_ratio,
        macd=macd_value,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        rsi=rsi_value,
        entry_rr=confirm_rr,
        swing_rr=entry_rr1,
        intraday_rr=intraday_defense_rr,
        price_label=intraday_price_label,
        volume_label=intraday_volume_label,
        validation_note=daily_note,
        reliability=intraday_reliability,
        supply_status=investor_judgment,
        stop_precision=intraday_reliability == "낮음" or intraday_price_label == "실패",
        intraday_mode=True,
        close_confirmed=False,
        completed_daily=False,
        trend_state=ma_alignment(row, [5, 10, 20, 60, 120, 240]),
    )
    state_actions = render_trade_state_actions(
        trade_state.final_action_state,
        {
            "recovery": recovery_action_price,
            "pullback": shallow_pull,
            "target1": money(entry_target1),
            "defense": money(intraday_defense_line),
        },
    )
    now_buy = state_actions["now_buy"]
    primary_strategy = state_actions["primary_strategy"]
    final = state_actions["final_judgment"]
    no_buy_reason = state_actions["no_buy_reason"]
    new_buyer_text = state_actions["new_buyer_action"]
    holder_text = state_actions["holder_action"]
    additional_buyer_text = state_actions["add_buyer_action"]
    stop_loss_action_text = state_actions["stop_loss_action"]
    if volume_momentum_conflict["applies"] and volume_momentum_conflict["primary_strategy"] not in primary_strategy:
        primary_strategy = f"{primary_strategy}; {volume_momentum_conflict['primary_strategy']}"
    deep_pullback_note = (
        pullback_status_text
        if "깊은 눌림목 구간 안" in pullback_status_text or "겹치는 구간" in pullback_status_text
        else "현재가는 깊은 눌림목 구간 안 또는 회복 확인 전 구간이므로 반등 확인 후 신규매수를 검토합니다."
        if trade_state.final_action_state == "NO_BUY_BELOW_RECOVERY" or trade_state.price_position_state == "BELOW_PULLBACK"
        else ""
    )
    if current_price < intraday_warning_line and "방어 관찰이 우선" not in holder_text:
        holder_text = f"현재가가 장중 주의선 {money(intraday_warning_line)} 아래라 보유자는 방어 관찰이 우선입니다. {holder_text}"
    state_code_table = state_code_report_rows(trade_state)

    ma_summary = ", ".join(
        [
            f"{p}일 {money(last_valid(row, f'MA{p}'))}"
            for p in [5, 10, 20, 60, 120, 240]
            if np.isfinite(last_valid(row, f"MA{p}"))
        ]
    )
    daily_cloud = ichimoku_position(row)
    weekly_cloud = ichimoku_position(weekly.iloc[-1]) if not weekly.empty else "데이터 부족"
    monthly_cloud = ichimoku_position(monthly.iloc[-1]) if not monthly.empty else "데이터 부족"
    ma_state = ma_alignment(row, [5, 10, 20, 60, 120, 240])
    profile_text = format_nearby_profile(levels["profile"], current_price)
    ma_comment = moving_average_comment(current_price, row)
    sector_label = infer_sector_label(code, stock_name)
    bb_mid = last_valid(row, "BB중심")
    bb_upper = last_valid(row, "BB상단")
    bb_lower = last_valid(row, "BB하단")
    macd_comment_text = macd_comment(macd_value, macd_signal, macd_hist, current_price, near_high_rebreak, close_confirm_line)
    rsi_comment_text = rsi_comment(rsi_value)
    bollinger_comment_text = bollinger_comment(current_price, bb_mid, bb_upper, bb_lower, near_high_rebreak, close_confirm_line)
    major_support_text = format_price_level_list([levels["recent60_low"], levels["low52"]])
    major_resistance_text = format_price_level_list([levels["recent10_high"], levels["recent20_high"]])
    rise_vol = daily.tail(20).loc[daily["Close"].diff().tail(20) > 0, "Volume"].sum()
    fall_vol = daily.tail(20).loc[daily["Close"].diff().tail(20) < 0, "Volume"].sum()
    market_summary = (
        f"{market} 대비 20일 초과수익 {fpct(market_rel.get('20일 초과'))}, 60일 초과수익 {fpct(market_rel.get('60일 초과'))}"
        if market_rel
        else "시장 상대강도 데이터 부족"
    )
    peer_text = "; ".join(
        [
            f"{name}: 20일 {fpct(vals.get('20일'))}, 60일 {fpct(vals.get('60일'))}"
            for name, vals in peer_returns.items()
        ]
    ) or "동종업종 비교 데이터 부족"
    chart_md = "\n".join([f"![{p.name}]({p.name})" for p in chart_paths])
    current_source = current.get("소스", "데이터 부족")
    yahoo_symbol = f"{code}{suffix}"
    yahoo_url = f"https://finance.yahoo.com/quote/{yahoo_symbol}/"
    intraday_msg = (
        "본 보고서는 분석 실행 시각 기준 장중 현재가를 우선 반영했습니다.\n"
        "단, 장중 캔들은 확정 일봉이 아니므로 최종 매수/매도 확정은 오늘 종가 기준으로 재확인해야 합니다."
    )
    def status_with_prefix(prefix: str, label: str) -> str:
        clean = str(label or "데이터 부족").strip()
        for repeated in [f"{prefix} {prefix} ", f"{prefix} "]:
            if clean.startswith(repeated):
                clean = clean[len(repeated) :].strip()
                break
        return f"{prefix} {clean}"

    decision = {
        "현재가": current_price,
        "전일종가": prev_close,
        "시가": open_price,
        "고가": high_price,
        "저가": low_price,
        "거래량": intraday_volume,
        "데이터신뢰도": intraday_reliability,
        "최종판단": final,
        "지금바로매수": now_buy,
        "주전략": primary_strategy,
        "보조전략": secondary_strategy,
        "진입표시명": entry_label,
        "진입표시값": shallow_pull,
        "회복확인가": recovery_price_text,
        "회복확인선": recovery_line,
        "회복/돌파공통확인선": close_confirm_line if common_confirmation_active else None,
        "눌림목지지가": support_price_text,
        "이유": no_buy_reason,
        "눌림": pull,
        "얕은눌림": shallow_pull,
        "깊은눌림": deep_pull,
        "얕은눌림하단": shallow_low,
        "얕은눌림상단": shallow_high,
        "깊은눌림하단": deep_pull_low,
        "깊은눌림상단": deep_pull_high,
        "돌파": breakout_price,
        "종가유지확인선": close_confirm_line,
        "재돌파가격": breakout_entry_line,
        "단기재돌파확인선": near_high_rebreak,
        "일봉돌파확인선": close_confirm_line,
        "당일고가재돌파": near_high_rebreak,
        "심리저항": psychological_line,
        "재돌파문구": f"단기 재돌파 확인선 {short_rebreak_text}, 일봉 돌파 확인선 {daily_breakout_text}",
        "단기재돌파표시": rebreak_display_text,
        "1차목표": target1,
        "2차목표": target2,
        "신규1차목표": entry_target1,
        "신규2차목표": entry_target2,
        "목표역할": target1_role,
        "주의선": warning,
        "방어선": defense,
        "장중주의선": intraday_warning_line,
        "장중방어선": intraday_defense_line,
        "종가확인": today_close_check,
        "중요가격": breakout_price,
        "현재가범위하단": price_range_low,
        "현재가범위상단": price_range_high,
        "예상수익률1": reward1,
        "예상수익률2": reward2,
        "신규예상수익률1": entry_reward1,
        "신규예상수익률2": entry_reward2,
        "하락위험률": risk_defense,
        "장중방어손익비": intraday_defense_rr,
        "스윙손절손익비": entry_rr1,
        "확인진입손익비": confirm_rr,
        "과열돌파손익비부족": bool(overheat_breakout_state["applies"]),
        "손익비부족경고": ", ".join(list(overheat_breakout_state["warnings"]) + rr_low_warnings),
        "장중방어근접경고": near_defense_warning_text,
        "손익비1": entry_rr1,
        "손익비2": entry_rr2,
        "상태코드": trade_state_to_dict(trade_state),
        "상태코드표": state_code_table,
        "상태코드잠금": True,
        "final_action_state": trade_state.final_action_state,
        "new_buyer_action": trade_state.new_buyer_action,
        "holder_action": trade_state.holder_action,
        "add_buyer_action": trade_state.add_buyer_action,
        "stop_loss_action": trade_state.stop_loss_action,
        "RSI": last_valid(row, "RSI14"),
        "MACD개선": last_valid(row, "MACD히스토그램") >= last_valid(daily.iloc[-2], "MACD히스토그램") if len(daily) > 1 else False,
        "장중거래량비율": weighted_volume_ratio if np.isfinite(weighted_volume_ratio) else volume_ratio,
    }
    score_context = {
        "supply_failed": "실패" in investor_status or "데이터 부족" in investor_status,
        "market_rel_strong": bool(market_rel and market_rel.get("20일 초과", 0) > 0),
    }
    trading_scores = calculate_trading_scores(
        decision,
        {
            "rr1": entry_rr1,
            "rr2": entry_rr2,
            "reward1": entry_reward1,
            "current_price": current_price,
            "target1": entry_target1,
            "target2": entry_target2,
            "pullback_low": shallow_low,
            "pullback_entry": shallow_low,
            "pullback_defense": intraday_defense_line,
            "breakout_entry": close_confirm_line,
            "breakout_defense": close_confirm_line,
            "breakout_line": close_confirm_line,
            "rebreak_line": near_high_rebreak,
            "intraday_high": high_price,
        },
        score_context,
    )
    trading_total = trading_scores["총점"]
    validation_status_text = " ".join([str(intraday_price_label), str(intraday_volume_label), str(daily_price_label), str(daily_volume_label)])
    yfinance_delay_text = str(yfinance_note)
    data_action_needed = any(word in validation_status_text for word in ["지연", "stale", "불일치", "경고", "실패"])
    if not (
        intraday_price_label == "통과"
        and intraday_volume_label == "통과"
        and daily_price_label == "통과"
        and daily_volume_label == "통과"
    ):
        data_action_needed = data_action_needed or any(
            word in yfinance_delay_text for word in ["지연", "stale", "최신거래일 불일치", "가격 불일치", "거래량 불일치"]
        )
    data_action_prefix = "데이터 상태: 장중/완료 일봉 소스 지연 여부 확인; " if data_action_needed else ""
    today_action_prices = (
        f"{data_action_prefix}"
        f"가격 상태: {money(shallow_high)} 재지지, {recovery_action_price} 종가 회복, "
        f"{money(near_high_rebreak)} 장중 재돌파 확인; "
        f"위험 상태: {money(intraday_defense_line)} 이탈 시 장중 방어, {money(defense)} 이탈 시 스윙 방어"
    )
    if now_buy == "가능":
        current_zone = "매수 구간"
        today_action = today_action_prices
    elif current_price <= defense:
        current_zone = "방어 구간"
        today_action = today_action_prices
    elif current_price < close_confirm_line:
        current_zone = "관망 구간"
        today_action = today_action_prices
    else:
        current_zone = "추격 금지 구간"
        today_action = today_action_prices
    avoid_parts: list[str] = []
    if np.isfinite(rsi_value) and rsi_value >= 70:
        avoid_parts.append("과열권에서 장중 급등 가격을 추격매수하지 않습니다.")
    if (np.isfinite(confirm_rr) and confirm_rr < 1.0) or (np.isfinite(entry_rr1) and entry_rr1 < 1.0):
        avoid_parts.append("손익비가 맞지 않는 돌파 추격매수를 하지 않습니다.")
    target_gap_pct = pct(entry_target1, current_price)
    if current_price <= defense:
        avoid_parts.append("방어선 아래에서 물타기하지 않습니다.")
    elif current_price < close_confirm_line:
        avoid_parts.append("종가 확인 전 추격매수와 물타기를 하지 않습니다.")
    elif not avoid_parts:
        avoid_parts.append("종가 확정 전 장중 돌파 가격을 추격매수하지 않습니다." if np.isfinite(target_gap_pct) and target_gap_pct >= 5 else "근접 저항 바로 아래에서 추격매수하지 않습니다.")
    avoid_action = " ".join(dict.fromkeys(avoid_parts))
    entry_confirm = (
        f"가격은 {shallow_confirm_text} 또는 {money(close_confirm_line)} 종가 안착, "
        "거래량은 20일 평균 1.2배 이상, 캔들은 윗꼬리 축소, 시장은 급락이 아니어야 합니다."
    )
    failure_condition = (
        f"{money(intraday_defense_line)} 장중 방어선 이탈, {money(defense)} 일봉 종가 이탈, "
        f"{money(close_confirm_line)} 돌파 실패 후 거래량 동반 하락, 시장 급락이 동시에 나오면 분석 전제를 낮춥니다."
    )
    priority_text = (
        f"1순위 {shallow_confirm_text}, 2순위 {money(close_confirm_line)} 종가 안착, "
        f"3순위 {money(near_high_rebreak)} 재돌파와 거래량 동반 여부 확인"
    )
    today_first_action = (
        "지금 바로 매수 조건이 유지되는지 확인하되 계획 비중만 분할 적용합니다."
        if now_buy == "가능"
        else "지금 바로 매수하지 않는 원칙을 먼저 확인합니다."
    )
    now_buy_weight = "0%" if now_buy == "불가" else ("20~30%" if float(trading_total) >= 70 else "10% 이하")
    pullback_weight = "20~30%" if float(trading_total) >= 70 else "10~20%"
    breakout_weight = "20~30%" if float(trading_total) >= 70 else "10~20%"

    md = f"""# 주식 매매타점 분석 보고서

## 0. 최우선 요약

| 항목 | 판단 |
| --- | --- |
| 지금 바로 매수 가능 여부 | {now_buy} |
| 지금 매수 불가 사유 | {no_buy_reason} |
| 주 전략 | {primary_strategy} |
| 보조 전략 | {secondary_strategy} |
| 신규매수자 기준 대응 | {new_buyer_text} |
| 보유자 기준 대응 | {holder_text} |
| 얕은 눌림목 대기 가격 | {shallow_pull} |
| 깊은 눌림목 대기 가격 | {deep_pull} |
| 단기 재돌파 확인선 | {short_rebreak_text} |
| 일봉 돌파 확인선 | {daily_breakout_text} |
| 근접 저항/보유자 일부 익절 후보 | {money(target1)} - {target1_role} |
| 신규매수 기준 1차 목표 | {money(entry_target1)} |
| 신규매수 기준 2차 목표 | {money(entry_target2)} |
| 장중 주의선 | {money(intraday_warning_line)} |
| 장중 방어선 | {money(intraday_defense_line)} |
| 스윙 최종 방어선 | {money(defense)} |
| 전량 이탈 조건 | {money(defense)} 일봉 종가 이탈 후 다음 거래일 회복 실패 |
| 최종 판단 | {final} |
| 데이터 신뢰도 | {intraday_reliability} |

## 프로 트레이더 판단

| 항목 | 내용 |
| --- | --- |
| 현재 구간 정의 | {current_zone} |
| 눌림목 상태 | {pullback_status_text} |
| 주 전략 | {primary_strategy} |
| 보조 전략 | {secondary_strategy} |
| 오늘 할 행동 | {today_action} |
| 오늘 하지 말아야 할 행동 | {avoid_action} |
| 진입 전 확인 조건 | {entry_confirm} |
| 실패 조건 | {failure_condition} |
| 우선순위 | {priority_text} |

## 오늘 할 행동

1. {today_first_action}
2. {shallow_confirm_text} 여부를 확인합니다.
3. {money(close_confirm_line)} 종가 안착 여부를 확인합니다.
4. {money(near_high_rebreak)} 재돌파 시 거래량 동반 여부를 확인합니다.
5. {pullback_status_text}
6. {money(defense)} 이탈 시 스윙 관점은 접고 방어 판단으로 전환합니다.

## 알림 설정 가격

| 알림 가격 | 의미 | 행동 |
| ---: | --- | --- |
| {shallow_pull} | {shallow_alert_meaning} | 분할매수 검토 |
| {money(near_high_rebreak)} | 단기 재돌파 가격 | 관찰 강화 |
| {money(close_confirm_line)} | 일봉 돌파 확인 가격 | 종가 확인 |
| {money(intraday_defense_line)} | 장중 방어 가격 | 신규매수 금지 |
| {money(defense)} | 스윙 최종 방어 가격 | 손절/비중 축소 |

## 매매 시나리오

| 시나리오 | 조건 | 행동 | 진입 비중 | 손절/방어 | 목표 |
| --- | --- | --- | --- | --- | --- |
| A. 지금 매수 | {no_buy_reason if now_buy == "불가" else "조건 충족"} | {"매수 금지" if now_buy == "불가" else "분할 매수"} | {now_buy_weight} | {"해당 없음" if now_buy == "불가" and now_buy_weight == "0%" else f"{money(intraday_defense_line)} / {money(defense)}"} | {"해당 없음" if now_buy == "불가" and now_buy_weight == "0%" else money(entry_target1)} |
| B. 눌림목 매수 | {shallow_confirm_text} | 분할매수 | {pullback_weight} | {money(intraday_defense_line)} | {money(entry_target1)} |
| C. 돌파 매수 | {money(close_confirm_line)} 종가 안착 + 거래량 1.2배 이상 | 매수 검토 | {breakout_weight} | {money(close_confirm_line)} 재이탈 | {money(entry_target2)} |
| D. 관망 접기 | {money(defense)} 일봉 종가 이탈 또는 돌파 실패 후 거래량 동반 하락 | 관망/정리 | 0% | {money(defense)} | 없음 |

## 투자자 상태별 대응

| 투자자 상태 | 대응 |
| --- | --- |
| 신규매수자 | {new_buyer_text} |
| 기존 보유자 | {holder_text} |
| 추가매수자 | 불타기와 물타기는 모두 보류하고, {money(close_confirm_line)} 종가 안착 또는 {shallow_confirm_text} 뒤에만 검토합니다. |
| 손실 보유자 | 평단가 미제공으로 개인별 손익률 판단은 제외합니다. {money(defense)} 일봉 종가 이탈 시 반등 기대보다 방어를 우선합니다. |

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

* 핵심 지지선인 {money(intraday_defense_line)} 이탈 후 회복하지 못하는 경우
* {money(close_confirm_line)} 돌파 시도 후 거래량 동반 하락이 나오는 경우
* 스윙 최종 방어선 {money(defense)} 아래로 일봉 종가가 마감하는 경우
* 시장 지수가 급락하고 {sector_label} 관련주 흐름이 동시에 꺾이는 경우
* 수급 데이터 부족 또는 수집 실패로 수급 판단 신뢰도가 낮아지는 경우
* 장중 데이터 신뢰도가 낮음으로 떨어지는 경우

## 1. 장중 매매 판단

| 항목 | 판단 |
| --- | --- |
| 현재 시각 | {analysis_time.strftime('%Y-%m-%d %H:%M KST')} |
| 현재가 | {money(current_price)} |
| 장중 현재가 범위 | {price_range_text} |
| 대표 기준가 | {money(current_price)} |
| 장중 가격 신뢰도 | {intraday_reliability} |
| 전일 종가 | {money(prev_close)} |
| 당일 고가 / 저가 | {money(high_price)} / {money(low_price)} |
| 장중 누적 거래량 | {shares(intraday_volume)} |
| 정규장 거래량 | {shares(regular_volume)} |
| NXT/통합 참고 거래량 | {shares(nxt_volume)} |
| 장중 누적 거래량 / 20일 평균 | {fratio(volume_ratio)} |
| 시간가중 환산 거래량 / 20일 평균 | {fratio(weighted_volume_ratio)} |
| 지금 바로 매수 | {now_buy} |
| 지금 매수하지 않는 이유 | {no_buy_reason} |
| 지금 매도 필요 | {sell_needed} |
| 단기 재돌파 확인선 | {short_rebreak_text} |
| 일봉 돌파 확인선 | {daily_breakout_text} |
| 장중 눌림 가격 | 얕은 눌림 {shallow_pull}, 깊은 눌림 {deep_pull} |
| 장중 주의선 | {money(intraday_warning_line)} |
| 장중 방어선 | {money(intraday_defense_line)} |
| 스윙 최종 방어선 | {money(defense)} |
| 전량 이탈 조건 | {money(defense)} 일봉 종가 이탈 후 다음 거래일 회복 실패 |
| 오늘 종가 확인 필요 가격 | {money(today_close_check)} |
| 오늘 가장 중요한 가격 | {money(breakout_price)} |

{intraday_msg}

## 2. 최종 매매 의사결정표

| 항목 | 판단 |
| --- | --- |
| 지금 바로 매수 | {now_buy} |
| 지금 매수하지 않는 이유 | {no_buy_reason} |
| 주 전략 | {primary_strategy} |
| 보조 전략 | {secondary_strategy} |
| 눌림목 매수 가격 | {pull} |
| 눌림목 매수 조건 | 얕은 눌림은 단기 반등 유지 시만 유효, 깊은 눌림은 돌파 실패 후 재지지 확인용 |
| 단기 재돌파 확인선 | {short_rebreak_text} |
| 일봉 돌파 확인선 | {daily_breakout_text} |
| 돌파 매수 조건 | {breakout_rule_text} |
| 근접 저항/보유자 일부 익절 후보 | {money(target1)} |
| 근접 저항/보유자 일부 익절 후보 조건 | {money(target1)} 접근 후 거래량 둔화 또는 윗꼬리 발생 시 보유자 일부 익절 |
| 신규매수 기준 1차 목표 | {money(entry_target1)} |
| 신규매수 기준 1차 목표 조건 | 종가 돌파 후 거래량 유지 시 일부 실현 검토 |
| 신규매수 기준 2차 목표 | {money(entry_target2)} |
| 신규매수 기준 2차 목표 조건 | 돌파 후 거래량 유지 시 추가 실현, RSI 70 이상 과열 시 비중 축소 |
| 장중 주의선 | {money(intraday_warning_line)} |
| 장중 방어선 | {money(intraday_defense_line)} |
| 스윙 최종 방어선 | {money(defense)} |
| 전량 이탈 조건 | 일봉 종가 {money(defense)} 이탈 후 다음 거래일 회복 실패 |
| 최종 판단 | {final} |

## 3. 기본 정보

| 항목 | 값 |
| --- | --- |
| 종목명 | {stock_name} |
| 종목코드/티커 | {code} |
| 시장 | {market} |
| 분석 실행 시각 | {analysis_time.strftime('%Y-%m-%d %H:%M KST')} |
| 기준가 | {money(current_price)} |
| 기준가 산정 기준 | 장중 현재가 |
| 장중 현재가 범위 | {price_range_text} |
| 대표 기준가 | {money(current_price)} |
| 전일 종가 | {money(prev_close)} |
| 당일 시가 | {money(open_price)} |
| 당일 고가 | {money(high_price)} |
| 당일 저가 | {money(low_price)} |
| 장중 누적 거래량 | {shares(intraday_volume)} |
| 정규장 거래량 | {shares(regular_volume)} |
| NXT/통합 참고 거래량 | {shares(nxt_volume)} |
| 20일 평균 거래량 대비 | 누적 {fratio(volume_ratio)}, 시간가중 {fratio(weighted_volume_ratio)} |
| 사용자 평단가 | 미제공 |
| 데이터 기준일 | 완료 일봉 {iso(daily.index[-1])}, 장중 {analysis_time.strftime('%Y-%m-%d')} |
| 매매 스타일 | 스윙 |
| 데이터 방식 | API 없는 공개 데이터 모드 |

## 4. 데이터 신뢰도 점검

| 항목 | 결과 |
| --- | --- |
| 1차 데이터 소스 | {current_source} |
| 2차 데이터 소스 | pykrx / FinanceDataReader 장중 일봉 |
| 최신 거래일 / 조회 시각 | {current.get('조회시각') or analysis_time.strftime('%Y-%m-%d %H:%M KST')} |
| 가격 검증 | {status_with_prefix("장중", intraday_price_label)}, {status_with_prefix("완료 일봉", daily_price_label)} |
| 거래량 검증 | {status_with_prefix("장중", intraday_volume_label)}, {status_with_prefix("완료 일봉", daily_volume_label)} |
| 조정주가 사용 여부 | 아니오 |
| 분봉 데이터 신뢰도 | {minute_reliability} |
| 장중 데이터 신뢰도 | {intraday_reliability} |
| 최종 데이터 신뢰도 | {daily_reliability if intraday_reliability == '높음' else intraday_reliability} |

장중 검증 세부:

{validation_table_for_report(intraday_validation)}

완료 일봉 검증 세부:

{compact_validation_md(daily_validation)}

## 5. 핵심 요약

* 현재가는 {money(current_price)}이며 전일 종가 {money(prev_close)} 대비 {fpct(pct(current_price, prev_close))}입니다.
* 장중 현재가 범위는 {price_range_text}, 대표 기준가는 {money(current_price)}, 장중 가격 신뢰도는 {intraday_reliability}입니다.
* {intraday_position_text}
* 지금 바로 매수 판단은 `{now_buy}`입니다. 이유는 {no_buy_reason}입니다.
* 주 전략은 {primary_strategy}입니다.
* 보조 전략은 {secondary_strategy}입니다.
* 얕은 눌림목은 {shallow_pull}, 깊은 눌림목은 {deep_pull}, 일봉 돌파 확인선은 {daily_breakout_text}입니다.
* 단기 재돌파 확인선 {short_rebreak_text} 회복은 분봉상 단기 회복 신호이고, 신규 돌파 매수는 {daily_breakout_text} 종가 안착과 거래량 1.2배 이상일 때만 검토합니다.
* 근접 저항/보유자 일부 익절 후보는 {money(target1)}입니다.
* 신규매수 기준 1차 목표는 {money(entry_target1)}으로 장중 기준 수익률 {fpct(entry_reward1)}, 신규매수 기준 2차 목표는 {money(entry_target2)}으로 {fpct(entry_reward2)}입니다.
* 스윙 최종 방어선은 {money(defense)}이며, 현재가 기준 하락 위험률은 {fpct(risk_defense)}입니다.
* 오늘 종가 확인 필요 가격은 {money(today_close_check)}입니다.
* 최종 판단은 `{final}`입니다.

## 6. 핵심 가격표

| 가격대 | 의미 | 대응 |
| --: | --- | --- |
| {money(current_price)} | 기준가 | 장중 현재가 기준 판단 |
| {shallow_pull} | 얕은 눌림목 매수 | 단기 반등 유지 시만 유효 |
| {deep_pull} | 깊은 눌림목 매수 | 돌파 실패 후 재지지 확인용 |
| {short_rebreak_text} | 단기 재돌파 확인선 | 분봉상 단기 회복 신호 |
| {daily_breakout_text} | 일봉 돌파 확인선 | 신규 돌파 매수 검토 조건 |
| {money(target1)} | 근접 저항/보유자 일부 익절 후보 | 보유자 일부 익절 후보 |
| {money(entry_target1)} | 신규매수 기준 1차 목표 | 신규매수 손익비 산정용 |
| {money(entry_target2)} | 신규매수 기준 2차 목표 | 추가 익절 후보 |
| {money(intraday_warning_line)} | 장중 주의선 | 이탈 시 장중 매수 관점 약화 |
| {money(intraday_defense_line)} | 장중 방어선 | 이탈 시 장중 방어 후보 |
| {money(defense)} | 스윙 최종 방어선 | 이탈 시 손절/비중 축소 |
| {money(defense)} 종가 이탈 후 회복 실패 | 전량 이탈 조건 | 추세 훼손 |

## 7. 예상 수익률과 하락 위험

| 시나리오 | 가격 | 기준가 대비 수익률 | 평단 대비 수익률 | 판단 |
| --- | --: | --: | --: | --- |
| 근접 저항/보유자 일부 익절 후보 | {money(target1)} | {fpct(reward1)} | 미제공 | 보유자 일부 익절 후보 |
| 신규매수 기준 1차 목표 | {money(entry_target1)} | {fpct(entry_reward1)} | 미제공 | 신규매수 손익비 산정 기준 |
| 신규매수 기준 2차 목표 | {money(entry_target2)} | {fpct(entry_reward2)} | 미제공 | 추가 익절 후보 |
| 스윙 최종 방어선 | {money(defense)} | -{fpct(risk_defense)} | 미제공 | 종가 이탈 시 방어 |

* 하락 위험률: {fpct(risk_defense)}
* 신규매수 기준 1차 목표 손익비: {fratio(entry_rr1)}
* 신규매수 기준 2차 목표 손익비: {fratio(entry_rr2)}
* 매수 매력도: 현재가 추격 매수 매력 낮음, 눌림 또는 종가 돌파 확인 필요

장중 거래량 해석:

* 장중 진행률: {fpct(progress * 100)}
* 장중 누적 거래량 / 20일 평균 거래량: {fratio(volume_ratio)}
* 시간가중 환산 거래량 / 20일 평균 거래량: {fratio(weighted_volume_ratio)}
* 판단: {volume_comment}

## 8. 보조지표 종합 점검표

| 지표 | 현재 상태 | 해석 |
| --- | --- | --- |
| 이동평균선 | {ma_state}; {ma_summary} | {ma_comment} |
| 일목균형표 | 일봉 {daily_cloud}, 주봉 {weekly_cloud}, 월봉 {monthly_cloud} | 장중 가격은 보조이고 구름 위치는 완료 일봉 기준으로 봅니다. |
| MACD | MACD {macd_value:.2f}, 신호 {macd_signal:.2f}, 히스토그램 {macd_hist:.2f} | {macd_comment_text} |
| RSI | {rsi_value:.2f} | {rsi_comment_text} |
| Stochastic | K {last_valid(row, 'StochK'):.2f}, D {last_valid(row, 'StochD'):.2f} | 단기 반등 탄력 확인용입니다. |
| 볼린저밴드 | 중심 {money(bb_mid)}, 상단 {money(bb_upper)}, 하단 {money(bb_lower)} | {bollinger_comment_text} |
| ATR | {money(last_valid(row, 'ATR14'))} | 변동성이 커서 손절 폭이 넓습니다. |
| OBV | 20거래일 기준 누적 거래량 반등 시도 | 거래량 동반 종가 돌파가 필요합니다. |
| MFI/CMF | MFI {last_valid(row, 'MFI14'):.2f}, CMF {last_valid(row, 'CMF20'):.3f} | 자금 유입은 아직 강한 추세 전환으로 보기 어렵습니다. |
| 거래량 | 장중 {shares(intraday_volume)}, 누적 {fratio(volume_ratio)}, 시간가중 {fratio(weighted_volume_ratio)} | 단순 누적보다 시간가중 거래량과 돌파 유지 여부를 함께 봅니다. |
| 매물대 | {profile_text} | 현재가 주변 상단 매물대와 하단 지지 매물대를 분리해서 봅니다. |
| 수급 | {investor_status} | {investor_judgment} |
| 시장/섹터 | {market_summary} | 시장보다 강한 종목 흐름인지 확인이 필요합니다. |

## 9. 차트 분석

### 월봉/주봉

* 추세: 장기 급등 후 조정, 중기 반등 시도 구간입니다.
* 일목균형표: 월봉 {monthly_cloud}, 주봉 {weekly_cloud}입니다.
* 주요 지지: {major_support_text}
* 주요 저항: {major_resistance_text}
* 판단: 장기 추세 훼손은 아니지만 단기 저항을 거래량으로 넘겨야 합니다.

### 일봉

* 이동평균선: {ma_summary}
* 일목균형표: {daily_cloud}
* 거래량: 최근 20일 상승일 거래량 합계 {shares(rise_vol)}, 하락일 거래량 합계 {shares(fall_vol)}
* MACD: {last_valid(row, 'MACD'):.2f}, 신호선 {last_valid(row, 'MACD신호'):.2f}, 히스토그램 {last_valid(row, 'MACD히스토그램'):.2f}
* RSI: {last_valid(row, 'RSI14'):.2f}
* 볼린저밴드: 중심 {money(last_valid(row, 'BB중심'))}, 상단 {money(last_valid(row, 'BB상단'))}, 하단 {money(last_valid(row, 'BB하단'))}
* ATR: {money(last_valid(row, 'ATR14'))}
* 매물대: {profile_text}
* 판단: {money(breakout_price)} 위 종가 유지 전까지 돌파 매수 확정은 어렵습니다.

### 장중/분봉

* 장중 현재가 위치: {intraday_location_line}
* 60분봉: {"참고 가능" if not intraday60.empty else "데이터 사용 불가"}
* 30분봉 또는 15분봉: {"참고 가능" if not intraday15.empty else "데이터 사용 불가"}
* yfinance 분봉: {yfinance_status}
* 분봉 차트: 생성 가능 시 보조 참고만 적용
* 분봉 데이터 신뢰도: {minute_reliability}
* 단기 타이밍 판단: {intraday_timing_text}
* 오늘 종가 확인 필요 가격: {money(today_close_check)}

## 10. 수급 및 시장 환경

국내 주식:

* 외국인: {investor_status}
* 기관: {investor_judgment}
* 개인: 공개 데이터 한계로 정밀 집계 불가
* 신용잔고: 데이터 부족
* 공매도/대차잔고: 데이터 부족
* KOSPI/KOSDAQ: {market_summary}
* 관련주/동종업종: {peer_text}
* 판단: {investor_detail}

## 11. 매매 계획

### 매수 계획

* 지금 바로 매수: {now_buy} - {no_buy_reason}
* 얕은 눌림목 매수: {shallow_confirm_text}과 거래량 감소 확인
* 깊은 눌림목 매수: {deep_pull}에서 돌파 실패 후 재지지 확인
* 돌파 매수: {breakout_rule_text}
* 매수 금지 조건: 일봉 종가 {money(defense)} 이탈, 단기 재돌파 실패 후 거래량 동반 하락, RSI 40 아래 재하락

### 익절 계획

* 근접 저항/보유자 일부 익절 후보: {money(target1)} 도달 시 보유자 일부 익절
* 신규매수 기준 1차 목표: {money(entry_target1)} 도달 시 일부 실현 검토
* 신규매수 기준 2차 목표: {money(entry_target2)} 도달 시 추가 실현 검토
* 전량 익절 후보: 목표가 도달 후 거래량 둔화, RSI 70 이상 과열, 장대양봉 뒤 윗꼬리 발생

### 손절/방어 계획

* 장중 주의선: {money(intraday_warning_line)}
* 장중 방어선: {money(intraday_defense_line)}
* 스윙 최종 방어선: {money(defense)}
* 전량 이탈 조건: 일봉 종가 {money(defense)} 이탈 후 다음 거래일 회복 실패
* 평단 방어선: 사용자 평단가 미제공
* 장중 방어 필요 조건: {money(intraday_defense_line)} 이탈 후 거래량 동반 하락 지속
* 오늘 종가 확인 필요 가격: {money(today_close_check)}

### 신규매수자 / 보유자 기준

신규매수자:
{new_buyer_text}

보유자:
{holder_text}

## 12. 최종 판단

{final}

{intraday_position_text}
현재가 기준 신규매수 1차 목표 손익비는 {fratio(entry_rr1)}입니다.
{rr_caution_text}
{pullback_status_text}
{final_action_text}

신규매수 기준:
{new_buyer_text}

보유자 기준:
{holder_text}

## 13. 최종 한 문단 판단

{stock_name} {code}은 장중 현재가 {money(current_price)} 기준으로 바로 사기보다 {shallow_confirm_text} 또는 오늘 종가 {money(today_close_check)} 이상 유지 확인을 기다리는 편이 낫습니다. 근접 저항/보유자 일부 익절 후보는 {money(target1)}이고, 신규매수 기준 1차 목표는 {money(entry_target1)}으로 예상 수익률 {fpct(entry_reward1)}, 신규매수 기준 2차 목표는 {money(entry_target2)}으로 {fpct(entry_reward2)}입니다. 스윙 최종 방어선은 {money(defense)}이며, 장중 분석에서 가장 중요한 가격은 {money(breakout_price)}입니다.

## 부록. 차트

{chart_md}

## 부록. 데이터 출처

* 네이버 증권: https://finance.naver.com/item/main.naver?code={code}
* 네이버 모바일 증권 API: https://m.stock.naver.com/api/stock/{code}/basic
* 네이버 실시간 증권 API: https://polling.finance.naver.com/api/realtime/domestic/stock/{code}?query=SERVICE_ITEM:{code}
* Yahoo Finance: {yahoo_url}
* pykrx, FinanceDataReader: 완료 일봉과 장중 미확정 일봉 교차검증에 사용
"""
    monthly_state = practical_state_from_text(monthly_cloud)
    weekly_state = practical_state_from_text(weekly_cloud)
    daily_state = daily_trend_state_from_values(
        current_price,
        row,
        macd_value,
        macd_signal,
        rsi_value,
        practical_state_from_text(daily_cloud, ma_state, ma_comment),
    )
    minute_state = "중립" if minute_reliability != "사용 불가" else "참고 제한"
    ma_grade = practical_grade_from_text(ma_state, ma_comment)
    cloud_grade = practical_grade_from_text(daily_cloud, weekly_cloud)
    macd_grade = macd_grade_from_values(macd_value, macd_signal, macd_hist, current_price, close_confirm_line)
    rsi_grade = rsi_grade_from_value(rsi_value)
    if volume_context.get("bearish_high_volume"):
        volume_grade = volume_context["status"]
        volume_judgment_text = volume_context["comment"]
    else:
        volume_grade = "좋음" if np.isfinite(weighted_volume_ratio) and weighted_volume_ratio >= 1.2 else "보통"
        volume_judgment_text = (
            f"{volume_momentum_conflict['state']}; 시간가중 20일 평균 대비 {fratio(weighted_volume_ratio)}입니다. 거래량만으로 돌파 매수 조건을 긍정 해석하지 않고 RSI/MACD 회복을 함께 확인합니다."
            if volume_momentum_conflict["applies"]
            else f"시간가중 20일 평균 대비 {fratio(weighted_volume_ratio)}이며 돌파 매수는 1.2배 이상이 필요합니다."
        )
    profile_grade = practical_grade_from_text(profile_text)
    supply_grade = practical_grade_from_text(investor_judgment)
    chart_intraday_note = "장중 현재가는 임시 기준이며 최종 매수/매도 판단은 오늘 종가로 재확인합니다."
    precision_limited = "정밀 판단 중단" in str(final)
    def point_value(text: Any) -> str:
        value = str(text)
        if precision_limited and value not in {"해당 없음", ""} and not value.startswith("참고 "):
            return f"참고 {value}"
        return value
    if common_confirmation_active:
        conclusion_confirmation_rows = f"| 회복/돌파 공통 확인가 | {point_value(f'{money(close_confirm_line)} - 종가 안착 + 거래량 유지 확인')} |"
        trading_confirmation_rows = f"| 회복/돌파 공통 확인가 | {point_value(money(close_confirm_line))} | 종가 안착 + 거래량 유지 확인 |"
    else:
        conclusion_confirmation_rows = (
            f"| 회복 확인가 | {point_value(recovery_price_text)} |\n"
            f"| 일봉 돌파 확인가 | {point_value(money(close_confirm_line))} |"
        )
        trading_confirmation_rows = (
            f"| 회복 확인가 | {point_value(recovery_price_text)} | {recovery_action_text} |\n"
            f"| 일봉 돌파 확인가 | {point_value(money(close_confirm_line))} | {breakout_action_text} |"
        )
    final_paragraph_text = (
        f"{overheat_breakout_state['template']} {rr_caution_text}"
        if overheat_breakout_state["applies"]
        else (
            f"{volume_momentum_conflict['state']}. {volume_momentum_conflict['template']} {volume_momentum_conflict['final']}. {rr_caution_text} 가격/거래량 신뢰도는 {reliability_parts['가격 신뢰도']}/{reliability_parts['거래량 신뢰도']}, "
            f"수급 신뢰도는 {reliability_parts['수급 신뢰도']}, 해석 완전성은 {reliability_parts['해석 완전성']}입니다."
            if volume_momentum_conflict["applies"]
            else (
                f"{stock_name} {code}은 현재 {money(current_price)} 기준으로 {deep_pullback_note} {buy_context_text} {rebreak_context_text} {breakout_context_text} "
                f"{target_context_text} 1차 목표는 {money(entry_target1)}, 2차 목표는 {money(entry_target2)}이며 {money(defense)} 이탈 시 스윙 관점은 낮춥니다. "
                f"{rr_caution_text} 가격/거래량 신뢰도는 {reliability_parts['가격 신뢰도']}/{reliability_parts['거래량 신뢰도']}, "
                f"수급 신뢰도는 {reliability_parts['수급 신뢰도']}, 해석 완전성은 {reliability_parts['해석 완전성']}입니다."
            )
        )
    )
    current_rr_score_display = float(trading_scores["현재가 기준 손익비 점수"])
    if now_buy == "불가":
        current_rr_score_display = min(current_rr_score_display, 14.0)
    md = f"""# {stock_name} {code} 실전 매매 판단 리포트

## 1. 최종 결론

| 항목 | 판단 |
|---|---|
| 현재가 | {money(current_price)} |
| 지금 매수 | {now_buy} |
| 주 전략 | {primary_strategy} |
{conclusion_confirmation_rows}
| 눌림목 지지가 | {point_value(support_price_text)} |
| {rebreak_label} | {point_value(rebreak_display_text)} |
| 1차 목표 | {point_value(money(entry_target1))} |
| 2차 목표 | {point_value(money(entry_target2))} |
| 손절/방어 | {point_value(f"장중 {money(intraday_defense_line)} / 스윙 {money(defense)}")} |
| 장중 방어선 기준 손익비 | {fratio(intraday_defense_rr)} |
| 스윙 손절선 기준 손익비 | {fratio(entry_rr1)} |
| 회복/돌파 진입 기준 손익비 | {fratio(confirm_rr)} |
| 가격 신뢰도 | {reliability_parts['가격 신뢰도']} |
| 거래량 신뢰도 | {reliability_parts['거래량 신뢰도']} |
| 수급 신뢰도 | {reliability_parts['수급 신뢰도']} |
| 장중 가격 신뢰도 | {reliability_parts['장중 가격 신뢰도']} |
| 해석 완전성 | {reliability_parts['해석 완전성']} |
| 최종 판단 | {final} |

## 1-1. 상태코드 기반 판단

{state_code_table}

## 2. 프로 트레이더 관점

- 지금 할 행동: {today_action}
- 지금 하지 말아야 할 행동: {avoid_action}
- 신규매수자: {one_line(new_buyer_text)}
- 보유자: {one_line(holder_text)}
- 추가매수자: {one_line(additional_buyer_text)}
- 손실보유자: {one_line(stop_loss_action_text)}

## 트레이딩 점수

| 항목 | 점수 | 기준 |
|---|---:|---|
| 추세 점수 | {trading_scores['추세 점수']} / 20 | 이동평균선, 일목, 고점/저점 구조 |
| 모멘텀 점수 | {trading_scores['모멘텀 점수']} / 15 | MACD, RSI, Stochastic |
| 거래량 점수 | {trading_scores['거래량 점수']} / 15 | 20일 평균 대비 거래량과 캔들 방향 |
| 현재가 기준 손익비 점수 | {current_rr_score_display:g} / 20 | 지금 바로 진입할 경우의 손익비, 매수 불가 시 14점 이하 제한 |
| 눌림목 진입 기준 손익비 점수 | {trading_scores['눌림목 진입 기준 손익비 점수']} / 20 | 눌림목 가격에서 진입할 경우의 손익비 |
| 돌파 진입 기준 손익비 점수 | {trading_scores['돌파 진입 기준 손익비 점수']} / 20 | 돌파 확인 후 진입할 경우의 손익비 |
| 손익비 점수 | {trading_scores['손익비 점수']} / 20 | 세부 손익비 점수 가중 평균 |
| 시장/섹터 점수 | {trading_scores['시장/섹터 점수']} / 10 | 시장 상대강도와 관련주 흐름 |
| 수급 점수 | {trading_scores['수급 점수']} / 10 | 수급 데이터 부족 시 중립 이하 |
| 위치 점수 | {trading_scores['위치 점수']} / 10 | 현재가가 매수하기 좋은 위치인지 |
| 총점 | {trading_scores['총점']}점 | {trading_scores['판정']} |

## 3. 차트 분석

| 구분 | 상태 | 판단 |
|---|---|---|
| 월봉 | {monthly_state} | {one_line(monthly_chart_comment(monthly_cloud))} |
| 주봉 | {weekly_state} | {weekly_cloud}이며, 중기 저항 돌파 확인이 필요합니다. |
| 일봉 | {daily_state} | {one_line(ma_comment)} |
| 분봉 | {minute_state} | 분봉은 장중 타이밍 보조로만 봅니다. |

장중 주의: {chart_intraday_note}

## 4. 보조지표 판단

| 지표 | 상태 | 매매 판단 |
|---|---|---|
| 이동평균선 | {ma_grade} | {one_line(ma_comment)} |
| 일목균형표 | {cloud_grade} | 일봉 {daily_cloud}, 주봉 {weekly_cloud}입니다. |
| MACD | {macd_grade} | {one_line(macd_comment_text)} |
| RSI | {rsi_grade} | {one_line(rsi_comment_text)} |
| 거래량 | {volume_grade} | {volume_judgment_text} |
| 매물대 | {profile_grade} | {one_line(profile_text)} |
| 수급 | {supply_grade} | {one_line(investor_judgment)} |

## 5. 매매 타점

| 구분 | 가격 | 행동 |
|---|---:|---|
{trading_confirmation_rows}
| 눌림목 지지가 | {point_value(support_price_text)} | 현재가 부근 또는 아래의 지지 확인 가격 |
| {rebreak_label} | {point_value(rebreak_display_text)} | {rebreak_display_action} |
| 1차 목표 | {point_value(money(entry_target1))} | 일부 익절 후보 |
| 2차 목표 | {point_value(money(entry_target2))} | 추가 익절 후보 |
| 장중 방어선 | {point_value(money(intraday_defense_line))} | 이탈 시 신규매수 금지 |
| 스윙 손절선 | {point_value(money(defense))} | 종가 이탈 시 방어/손절 |

## 6. 최종 한 문단

{final_paragraph_text}"""
    return md, decision


def run(code: str, fallback_name: str | None = None) -> str:
    code = code.strip()
    if not (code.isdigit() and len(code) == 6):
        code = code.upper()
    now = today_kst()
    market_open = is_korean_regular_market(now)
    completed_end = previous_calendar_day(now) if market_open else now.date()
    start_daily = completed_end - timedelta(days=365 * 6)
    start_validation = completed_end - timedelta(days=365 * 2)
    stock_name, market, suffix = detect_name_market(code, fallback_name, completed_end)
    if market == "US":
        return run_completed_analysis(code, fallback_name)
    safe_name = sanitize_filename(stock_name)
    out_dir = REPORTS_DIR / f"{safe_name}_{code}"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_pykrx = load_pykrx(code, start_daily, completed_end)
    src_fdr = load_fdr(code, start_daily, completed_end)
    src_yf = load_yfinance(code + suffix, start_validation, completed_end, "yfinance")
    daily_validation, daily_reliability, daily_stop = source_validation([src_pykrx, src_fdr, src_yf], completed_end)
    daily_price_label, daily_volume_label, daily_note = validation_labels(daily_validation)
    if not src_pykrx.data.empty:
        daily_base = src_pykrx.data[src_pykrx.data.index.date <= completed_end].copy()
    elif not src_fdr.data.empty:
        daily_base = src_fdr.data[src_fdr.data.index.date <= completed_end].copy()
    elif not src_yf.data.empty:
        daily_base = src_yf.data[src_yf.data.index.date <= completed_end].copy()
    else:
        raise RuntimeError("완료 일봉 데이터를 수집하지 못했습니다.")

    daily = add_indicators(daily_base, [5, 10, 20, 60, 120, 240])
    weekly = add_indicators(resample_ohlcv(daily_base, "W-FRI"), [5, 10, 20, 60, 120, 240])
    monthly = add_indicators(resample_ohlcv(daily_base, "M"), [5, 10, 20, 60, 120, 240])
    levels = nearest_levels(daily_base, daily)
    levels["basis"] = float(daily.iloc[-1]["Close"])

    intraday60_src = load_yfinance_intraday(code + suffix, "60m", "2mo")
    intraday15_src = load_yfinance_intraday(code + suffix, "15m", "1mo")
    intraday60 = intraday60_src.data
    intraday15 = intraday15_src.data

    if market_open:
        rows = naver_intraday_sources(code)
        rows += pykrx_fdr_today_sources(code, now.date())
        rows.append(yfinance_latest_source(code + suffix))
        intraday_validation = pd.DataFrame(rows)
        prev_close = float(daily.iloc[-1]["Close"])
        current, intraday_reliability, intraday_price_label, intraday_volume_label, price_meta = choose_intraday_basis(rows, prev_close, now)
    else:
        current = {
            "구분": "완료일봉",
            "소스": "최신 완료 일봉 종가",
            "조회시각": iso(daily.index[-1]),
            "현재가": float(daily.iloc[-1]["Close"]),
            "전일종가": float(daily.iloc[-2]["Close"]),
            "시가": float(daily.iloc[-1]["Open"]),
            "고가": float(daily.iloc[-1]["High"]),
            "저가": float(daily.iloc[-1]["Low"]),
            "거래량": float(daily.iloc[-1]["Volume"]),
            "비고": "정규장 외",
        }
        intraday_validation = pd.DataFrame([current])
        intraday_reliability = "해당 없음"
        intraday_price_label = "해당 없음"
        intraday_volume_label = "해당 없음"
        price_meta = {
            "min": current["현재가"],
            "max": current["현재가"],
            "regular_volume": current["거래량"],
            "nxt_volume": np.nan,
            "yfinance_minute_ok": False,
            "yfinance_note": "정규장 외",
        }

    index_src = load_market_index(market, start_daily, now.date())
    index_df = index_src.data if hasattr(index_src, "data") else pd.DataFrame()
    market_index_value = None
    if market in {"KOSPI", "KOSDAQ"} and not index_df.empty:
        market_index_value = float(index_df["Close"].dropna().iloc[-1])
    market_index_invalid = market in {"KOSPI", "KOSDAQ"} and not index_df.empty and not market_index_frame_is_valid(market, index_df)
    market_rel = relative_returns(daily_base, index_df) if not index_df.empty else {}
    peer_returns = load_peer_returns_for_stock(code, safe_name, completed_end - timedelta(days=160), now.date())
    investor = naver_investor_table(code)
    chart_paths = make_charts(out_dir, safe_name, code, daily, weekly, monthly, levels, intraday60, intraday15)
    report_md, decision = make_report(
        safe_name,
        code,
        market,
        suffix,
        now,
        current,
        price_meta,
        intraday_validation,
        intraday_reliability,
        intraday_price_label,
        intraday_volume_label,
        daily,
        weekly,
        monthly,
        daily_validation,
        daily_reliability,
        daily_price_label,
        daily_volume_label,
        daily_note,
        levels,
        intraday60,
        intraday15,
        market_rel,
        peer_returns,
        investor,
        chart_paths,
    )

    combined_validation = source_rows_to_csv(daily_validation, intraday_validation)
    combined_validation.to_csv(out_dir / f"{safe_name}_{code}_데이터검증.csv", index=False, encoding="utf-8-sig")
    cols = [c for c in daily.columns if c in daily.columns]
    summary = daily[cols].tail(120).copy()
    summary.insert(0, "Date", [iso(i) for i in summary.index])
    summary.to_csv(out_dir / f"{safe_name}_{code}_지표요약.csv", index=False, encoding="utf-8-sig")

    md_path = out_dir / f"{safe_name}_{code}_매매타점_분석보고서.md"
    html_path = out_dir / f"{safe_name}_{code}_매매타점_분석보고서.html"
    metrics = {
        "rr1": decision.get("손익비1"),
        "rr2": decision.get("손익비2"),
        "intraday_rr": decision.get("장중방어손익비"),
        "confirm_rr": decision.get("확인진입손익비"),
        "current_rr": decision.get("손익비1"),
        "reward1": decision.get("신규예상수익률1"),
        "current_price": decision.get("현재가"),
        "target1": decision.get("신규1차목표"),
        "target2": decision.get("신규2차목표"),
        "recovery_line": decision.get("회복확인선") or decision.get("회복/돌파공통확인선"),
        "breakout_line": decision.get("종가유지확인선"),
        "rebreak_line": decision.get("단기재돌파확인선"),
        "close_confirm_line": decision.get("종가유지확인선"),
        "intraday_high": decision.get("고가"),
        "intraday_warning_line": decision.get("장중주의선"),
        "intraday_defense_line": decision.get("장중방어선"),
        "shallow_pull_low": decision.get("얕은눌림하단"),
        "shallow_pull_high": decision.get("얕은눌림상단"),
        "deep_pull_low": decision.get("깊은눌림하단"),
        "deep_pull_high": decision.get("깊은눌림상단"),
        "rsi": last_valid(daily.iloc[-1], "RSI14"),
        "macd": last_valid(daily.iloc[-1], "MACD"),
        "macd_signal": last_valid(daily.iloc[-1], "MACD신호"),
        "macd_hist": last_valid(daily.iloc[-1], "MACD히스토그램"),
        "ma20": last_valid(daily.iloc[-1], "MA20"),
        "ma60": last_valid(daily.iloc[-1], "MA60"),
        "open_price": decision.get("시가"),
        "high_price": decision.get("고가"),
        "low_price": decision.get("저가"),
        "close_price": decision.get("현재가"),
        "volume_ratio20": decision.get("장중거래량비율"),
        "weighted_volume_ratio": decision.get("장중거래량비율"),
        "bb_mid": last_valid(daily.iloc[-1], "BB중심"),
        "analysis_time": now,
    }
    indicators = build_indicator_snapshot(
        daily,
        levels,
        {
            "current_price": decision.get("현재가"),
            "current_price_min": decision.get("현재가범위하단"),
            "current_price_max": decision.get("현재가범위상단"),
            "prev_close": decision.get("전일종가"),
            "open_price": decision.get("시가"),
            "high_price": decision.get("고가"),
            "low_price": decision.get("저가"),
            "shallow_pull_low": decision.get("얕은눌림하단"),
            "shallow_pull_high": decision.get("얕은눌림상단"),
            "deep_pull_low": decision.get("깊은눌림하단"),
            "deep_pull_high": decision.get("깊은눌림상단"),
            "close_confirm_line": decision.get("종가유지확인선"),
            "rebreak_line": decision.get("재돌파가격"),
            "near_high_rebreak": decision.get("당일고가재돌파"),
            "intraday_high": decision.get("고가"),
            "psychological_line": decision.get("심리저항"),
            "entry_target1": decision.get("신규1차목표"),
            "entry_target2": decision.get("신규2차목표"),
            "intraday_warning_line": decision.get("장중주의선"),
            "intraday_defense_line": decision.get("장중방어선"),
        },
    )
    context = {
        "stock_name": safe_name,
        "code": code,
        "market": market,
        "suffix": suffix,
        "analysis_time": now,
        "current_price": decision.get("현재가"),
        "current_price_min": decision.get("현재가범위하단"),
            "current_price_max": decision.get("현재가범위상단"),
            "high_price": decision.get("고가"),
        "deep_pull_low": decision.get("깊은눌림하단"),
        "deep_pull_high": decision.get("깊은눌림상단"),
        "intraday_warning_line": decision.get("장중주의선"),
        "sector_label": infer_sector_label(code, safe_name),
        "approved_price_range_set": [
            (decision.get("현재가범위하단"), decision.get("현재가범위상단")),
            (decision.get("얕은눌림하단"), decision.get("얕은눌림상단")),
            (decision.get("깊은눌림하단"), decision.get("깊은눌림상단")),
        ],
        "supply_failed": "실패" in investor.get("status", "") or "데이터 부족" in investor.get("status", ""),
        "market_index_source": index_src.name,
        "market_index_symbol": index_src.note if index_src.note in {"KS11", "KQ11", "1001", "2001"} else domestic_index_symbol(market),
        "market_index_value": market_index_value,
        "market_index_invalid": market_index_invalid,
        "trade_state": decision.get("상태코드", {}),
    }
    report_reliability = reliability_breakdown(
        intraday_price_label,
        intraday_volume_label,
        investor.get("status", "데이터 부족"),
        intraday_reliability,
        True,
    )
    state_blocking_count = len((decision.get("상태코드") or {}).get("qa_blocking_errors") or [])
    final_report_md = f"{report_md.rstrip()}\n\n{build_report_qa_section(decision['데이터신뢰도'], validation_error_count=state_blocking_count, reliability_details=report_reliability)}\n"
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
    md_path.write_text(final_report_md, encoding="utf-8")
    html_path.write_text(html_from_markdown(final_report_md, f"{safe_name} {code} 매매타점 분석보고서"), encoding="utf-8")

    if market_open:
        if decision["현재가"] >= decision["종가유지확인선"]:
            core_judgement = "돌파선 위지만 근접 저항이 가까워 지금 추격보다 종가 유지 확인이 우선입니다."
        else:
            core_judgement = "현재가는 일봉 돌파 확인선 아래입니다. 단기 재돌파 확인선 회복 또는 일봉 돌파 확인선 종가 안착 전까지는 추격보다 눌림 지지가 우선입니다."
        return f"""[분석 완료]

종목: {safe_name} {code}
현재가: {money(decision['현재가'])}
지금 매수: {decision['지금바로매수']}
주 전략: {decision.get('주전략', '눌림목 대기')}
{decision.get('진입표시명', '회복/눌림 확인가')}: {decision.get('진입표시값', decision['눌림'])}
일봉 돌파 확인가: {money(decision['일봉돌파확인선'])}
1차 목표: {money(decision['신규1차목표'])}
손절/방어: 장중 {money(decision['장중방어선'])} / 스윙 {money(decision['방어선'])}
최종 판단: {decision['최종판단']}
데이터 신뢰도: {decision['데이터신뢰도']}
보고서 경로: {md_path}"""
    return f"분석 완료: {md_path}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("code")
    parser.add_argument("name", nargs="?", default=None)
    args = parser.parse_args()
    print(run(args.code, args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
