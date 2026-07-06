from __future__ import annotations

"""Hidden-gem candidate scanner.

This module is analysis-only. It does not send orders, create order signals,
or implement automated trading.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from analyze_stock import REPORTS_DIR, html_from_markdown, money, round_to_tick, sanitize_filename
from command_chart_analyzer import _standardize_daily_frame
from core.sse_indicator import SSELevels, add_sse_columns, calculate_sse_indicator, validate_sse_levels
from kiwoom.provider import KiwoomDataProvider


KST = ZoneInfo("Asia/Seoul")
DEFAULT_MAX_CANDIDATES = 80
DEFAULT_LIMIT = 20
MIN_TRADE_AMOUNT_MILLION = 3_000
MAX_SCAN_DAILY_ROWS = 420


@dataclass(frozen=True)
class RawCandidate:
    code: str
    name: str
    sector: str
    price: float
    change_rate: float
    volume: float
    trade_amount_million: float
    source: str


@dataclass(frozen=True)
class HiddenGemResult:
    code: str
    name: str
    sector: str
    recommendation: str
    current_price: float
    entry: float
    stop: float
    target1: float
    no_chase: float
    rr1: float
    risk_pct: float
    volume_ratio20: float
    trade_value_ratio20: float
    pressure: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def run_hidden_gem_scan(
    *,
    provider: KiwoomDataProvider | None = None,
    base_url: str = "http://127.0.0.1:8765",
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    limit: int = DEFAULT_LIMIT,
    save_report: bool = True,
) -> str:
    provider = provider or KiwoomDataProvider()
    raw_candidates = fetch_bridge_candidates(base_url=base_url, max_candidates=max_candidates)
    results = scan_candidates(raw_candidates, provider=provider, limit=limit)
    markdown = render_hidden_gem_report(results, raw_candidates)
    if save_report:
        save_hidden_gem_report(markdown)
    return summarize_hidden_gem_scan(results, markdown)


def fetch_bridge_candidates(base_url: str = "http://127.0.0.1:8765", max_candidates: int = DEFAULT_MAX_CANDIDATES) -> list[RawCandidate]:
    base_url = base_url.rstrip("/")
    rows: dict[str, RawCandidate] = {}
    for path, params, source in (
        ("/snapshot", {"sectorLimit": 80, "stocksPerSector": 30, "maxRealtimeCodes": max_candidates, "sort": "tradeAmount"}, "snapshot"),
        ("/rising-amount-rank", {"limit": max_candidates}, "rising-amount-rank"),
        ("/daily-amount-rank", {"limit": max_candidates}, "daily-amount-rank"),
    ):
        try:
            response = requests.get(f"{base_url}{path}", params=params, timeout=(3.0, 20.0))
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue
        for row in _candidate_rows_from_payload(payload):
            candidate = _candidate_from_row(row, source)
            if candidate is None or _excluded_name(candidate.name):
                continue
            previous = rows.get(candidate.code)
            if previous is None or candidate.trade_amount_million > previous.trade_amount_million:
                rows[candidate.code] = candidate

    return sorted(rows.values(), key=lambda item: item.trade_amount_million, reverse=True)[:max_candidates]


def scan_candidates(
    candidates: list[RawCandidate],
    *,
    provider: KiwoomDataProvider,
    limit: int = DEFAULT_LIMIT,
) -> list[HiddenGemResult]:
    results: list[HiddenGemResult] = []
    for candidate in candidates:
        if candidate.trade_amount_million < MIN_TRADE_AMOUNT_MILLION:
            continue
        result = analyze_hidden_gem_candidate(candidate, provider)
        if result is not None:
            results.append(result)

    order = {"O": 0, "△": 1, "X": 2}
    results.sort(
        key=lambda item: (
            order.get(item.recommendation, 9),
            -safe_float(item.rr1),
            safe_float(item.risk_pct),
            abs(safe_float(item.current_price) - safe_float(item.entry)) / safe_float(item.entry, 1.0),
        )
    )
    return results[:limit]


def analyze_hidden_gem_candidate(candidate: RawCandidate, provider: KiwoomDataProvider) -> HiddenGemResult | None:
    try:
        daily = _standardize_daily_frame(provider.get_daily_ohlcv(candidate.code, limit=MAX_SCAN_DAILY_ROWS))
        if len(daily) < 120:
            return None
        current_price = candidate.price
        if not np.isfinite(current_price) or current_price <= 0:
            current_price = float(daily.iloc[-1]["Close"])
        sse_frame = add_sse_columns(daily)
        sse_result = calculate_sse_indicator(daily, current_price=current_price, is_intraday=False)
        levels = sse_result.levels
        if validate_sse_levels(levels):
            return None
        row = sse_frame.iloc[-1]
        recommendation, reasons, warnings = classify_hidden_gem(candidate, levels, row, current_price)
        risk_pct = (levels.entry - levels.stop) / levels.entry if levels.entry > 0 else float("nan")
        return HiddenGemResult(
            code=candidate.code,
            name=candidate.name,
            sector=candidate.sector,
            recommendation=recommendation,
            current_price=current_price,
            entry=levels.entry,
            stop=levels.stop,
            target1=levels.target1,
            no_chase=levels.no_chase,
            rr1=levels.rr1,
            risk_pct=risk_pct,
            volume_ratio20=_max_finite(row.get("SSE_VOLUME_RATIO20"), float("nan")),
            trade_value_ratio20=_max_finite(row.get("SSE_TRADE_VALUE_RATIO20"), float("nan")),
            pressure=levels.pressure,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )
    except Exception:
        return None


def classify_hidden_gem(
    candidate: RawCandidate,
    levels: SSELevels,
    row: pd.Series,
    current_price: float,
) -> tuple[str, list[str], list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    volume_ratio = _max_finite(row.get("SSE_VOLUME_RATIO20"), row.get("SSE_TRADE_VALUE_RATIO20"))
    risk_pct = (levels.entry - levels.stop) / levels.entry if levels.entry > 0 else float("inf")
    entry_distance_pct = (levels.entry - current_price) / levels.entry if levels.entry > 0 else float("inf")
    upside_pct = (levels.target1 - current_price) / current_price if current_price > 0 else float("-inf")

    if candidate.change_rate >= 10:
        warnings.append(f"당일 등락률 {candidate.change_rate:.2f}%로 이미 급등 구간")
    if current_price >= levels.no_chase:
        warnings.append("현재가가 SSE 추격 금지선 이상")
    if current_price >= levels.target1:
        warnings.append("현재가가 1차 목표권 이상")
    if levels.pressure >= 1.2:
        warnings.append(f"SSE 압력값 {levels.pressure:.2f}: 늦은 진입 위험")
    if levels.rr1 < 1.2:
        warnings.append(f"SSE RR1 {levels.rr1:.2f}배로 손익비 부족")
    if risk_pct > 0.10:
        warnings.append(f"예상 손절폭 {risk_pct * 100:.1f}%로 짧은 손절 조건 미흡")
    if not (0.6 <= volume_ratio <= 2.0):
        warnings.append(f"거래량/거래대금 20일 대비 {volume_ratio:.2f}배로 조용한 유입 범위 밖")

    if -0.6 <= levels.pressure <= 1.0:
        reasons.append(f"SSE 압력값 {levels.pressure:.2f}: 과열 전 회복 감시 구간")
    if 0.75 <= volume_ratio <= 1.8:
        reasons.append(f"거래량/거래대금 20일 대비 {volume_ratio:.2f}배로 조용한 유입")
    if abs(entry_distance_pct) <= 0.025:
        reasons.append(f"SSE 진입가까지 거리 {entry_distance_pct * 100:.1f}%로 진입 트리거 근접")
    if levels.rr1 >= 1.2:
        reasons.append(f"SSE RR1 {levels.rr1:.2f}배로 최소 손익비 충족")
    if risk_pct <= 0.08:
        reasons.append(f"예상 손절폭 {risk_pct * 100:.1f}%로 방어 기준이 비교적 짧음")
    if upside_pct >= 0.03:
        reasons.append(f"1차 목표까지 여유 {upside_pct * 100:.1f}%")

    hard_block = (
        candidate.change_rate >= 10
        or current_price >= levels.no_chase
        or current_price >= levels.target1
        or levels.pressure >= 1.2
        or levels.rr1 < 1.2
        or risk_pct > 0.10
    )
    if hard_block:
        return "X", reasons or ["진흙 속 진주 조건 미충족"], warnings

    if (
        -0.4 <= levels.pressure <= 0.9
        and 0.75 <= volume_ratio <= 1.8
        and abs(entry_distance_pct) <= 0.02
        and levels.rr1 >= 1.4
        and risk_pct <= 0.08
        and upside_pct >= 0.04
    ):
        return "O", reasons, warnings

    return "△", reasons or ["관심 조건 일부 충족, 분봉 진입 타이밍 대기"], warnings


def render_hidden_gem_report(results: list[HiddenGemResult], raw_candidates: list[RawCandidate]) -> str:
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 진흙 속 진주 스캐너",
        "",
        f"- 실행 시각: {now}",
        f"- 원천 후보: {len(raw_candidates)}종목",
        "- 추천 표기: O=정밀 감시 후보, △=조건 대기, X=제외",
        "- 주문/자동매매 기능은 없으며, O도 즉시 매수 신호가 아닙니다.",
        "",
        "| 추천 | 종목 | 섹터 | 현재가 | 진입 트리거 | 손절 | 1차 목표 | 추격 금지 | RR1 | 손절폭 | 거래유입 | 핵심 이유 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        reason = "; ".join(item.reasons[:3]) if item.reasons else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    item.recommendation,
                    f"{item.name} {item.code}",
                    item.sector or "-",
                    _price(item.current_price),
                    _price(item.entry),
                    _price(item.stop),
                    _price(item.target1),
                    _price(item.no_chase),
                    _ratio(item.rr1),
                    f"{item.risk_pct * 100:.1f}%" if np.isfinite(item.risk_pct) else "-",
                    f"{_max_finite(item.volume_ratio20, item.trade_value_ratio20):.2f}배",
                    reason.replace("|", "/"),
                ]
            )
            + " |"
        )

    lines.extend(["", "## 후보별 주의사항", ""])
    for item in results:
        warnings = "; ".join(item.warnings) if item.warnings else "특이 경고 없음. 그래도 분봉 종가 유지 확인 전 신규매수 금지."
        lines.append(f"- {item.recommendation} {item.name} {item.code}: {warnings}")
    return "\n".join(lines).strip() + "\n"


def summarize_hidden_gem_scan(results: list[HiddenGemResult], markdown: str) -> str:
    counts = {mark: sum(1 for item in results if item.recommendation == mark) for mark in ("O", "△", "X")}
    top = [item for item in results if item.recommendation in {"O", "△"}][:5]
    lines = [
        "[진흙 속 진주 스캐너 완료]",
        f"추천 O: {counts['O']}종목 / △: {counts['△']}종목 / X: {counts['X']}종목",
        "",
    ]
    if top:
        lines.append("상위 감시 후보:")
        for item in top:
            lines.append(
                f"- {item.recommendation} {item.name} {item.code}: 현재가 {_price(item.current_price)}, "
                f"진입 트리거 {_price(item.entry)}, 손절 {_price(item.stop)}, RR1 {_ratio(item.rr1)}"
            )
    else:
        lines.append("현재 조건에 맞는 감시 후보가 없습니다.")
    report_path = latest_hidden_gem_report_path()
    if report_path:
        lines.append(f"\n보고서 경로: {report_path}")
    return "\n".join(lines)


def save_hidden_gem_report(markdown: str) -> Path:
    out_dir = REPORTS_DIR / "hidden_gems"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"[진흙 속 진주] 후보 보고서_{stamp}.md"
    html_path = md_path.with_suffix(".html")
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_from_markdown(markdown, "진흙 속 진주 후보 보고서"), encoding="utf-8")
    _write_latest_pointer(md_path)
    return md_path


def latest_hidden_gem_report_path() -> Path | None:
    pointer = REPORTS_DIR / "hidden_gems" / "latest.txt"
    if not pointer.exists():
        return None
    path = Path(pointer.read_text(encoding="utf-8").strip())
    return path if path.exists() else None


def _write_latest_pointer(path: Path) -> None:
    pointer = path.parent / "latest.txt"
    pointer.write_text(str(path), encoding="utf-8")


def _candidate_rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    rows: list[dict[str, Any]] = []
    for key in ("items", "stocks", "candidates"):
        value = payload.get(key)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, dict))
    for sector in payload.get("sectors", []) or []:
        if isinstance(sector, dict):
            sector_name = str(sector.get("name") or "")
            for stock in sector.get("stocks", []) or []:
                if isinstance(stock, dict):
                    rows.append({**stock, "sector": stock.get("sector") or sector_name})
    return rows


def _candidate_from_row(row: dict[str, Any], source: str) -> RawCandidate | None:
    code = "".join(ch for ch in str(row.get("code") or row.get("itemCode") or "") if ch.isdigit()).zfill(6)[-6:]
    name = str(row.get("name") or row.get("stockName") or "").strip()
    if not code or code == "000000" or not name:
        return None
    return RawCandidate(
        code=code,
        name=name,
        sector=str(row.get("sector") or row.get("industry") or row.get("sectorName") or "-").strip(),
        price=_num(row.get("price", row.get("closePrice", row.get("closePriceRaw")))),
        change_rate=_num(row.get("changeRate", row.get("fluctuationsRatio"))),
        volume=_num(row.get("volume", row.get("accumulatedTradingVolumeRaw"))),
        trade_amount_million=_num(row.get("tradeAmountMillion", row.get("accumulatedTradingValueRaw"))) / (1_000_000 if row.get("tradeAmountMillion") is None else 1),
        source=source,
    )


def _excluded_name(name: str) -> bool:
    upper = str(name or "").upper()
    return any(token in upper for token in ("ETF", "ETN", "ELW", "스팩", "리츠", "KODEX", "TIGER", "ACE", "RISE", "SOL", "KOSEF", "HANARO", "KBSTAR"))


def _price(value: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return "-"
    return money(round_to_tick(float(value), "nearest"))


def _ratio(value: float) -> str:
    return f"{value:.2f}배" if np.isfinite(value) else "-"


def _num(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(str(value).replace(",", "").replace("%", ""))
    except Exception:
        return 0.0


def _max_finite(*values: Any) -> float:
    finite = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if np.isfinite(number):
            finite.append(number)
    return max(finite) if finite else float("nan")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return number if np.isfinite(number) else default


def main() -> int:
    parser = argparse.ArgumentParser(description="진흙 속 진주 후보를 찾습니다. 주문/자동매매 기능은 없습니다.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    print(run_hidden_gem_scan(base_url=args.bridge_url, max_candidates=args.max_candidates, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
