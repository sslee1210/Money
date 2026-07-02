from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import analyze_stock
import analyze_stock_intraday
import command_chart_analyzer


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data" / "cache"
TICKER_CACHE = CACHE_DIR / "krx_tickers.csv"
KST = ZoneInfo("Asia/Seoul")


STOP_WORDS = {
    "분석",
    "분석해",
    "분석해줘",
    "분석해주세요",
    "해줘",
    "해주세요",
    "봐줘",
    "매매",
    "차트",
    "주식",
    "종목",
    "좀",
    "기반",
    "현재",
    "지금",
    "오늘",
    "조건부",
    "명령형",
    "키움",
    "장중",
    "실시간",
    "일봉",
    "스윙",
    "장외",
}


@dataclass(frozen=True)
class ParsedRequest:
    code: str
    name: str | None
    mode: str
    original: str


def is_korean_market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(KST)
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=9, minute=0, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end


def detect_mode(text: str) -> str:
    # Money Assistant의 자연어 입력은 하나의 통합 분석 파이프라인만 사용한다.
    # 키움 브릿지/로그인이 준비되지 않으면 정상 리포트 대신 QA 실패로 중단한다.
    return "integrated"


def normalize_query(text: str) -> str:
    normalized = re.sub(r"[\"'`.,!?()\[\]{}:;]", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def token_candidates(text: str) -> list[str]:
    normalized = normalize_query(text)
    tokens = []
    for token in normalized.split():
        stripped = token.strip()
        if not stripped or stripped in STOP_WORDS:
            continue
        if re.fullmatch(r"\d{1,3}(,\d{3})*(원)?", stripped):
            continue
        tokens.append(stripped)
    return tokens


def load_krx_tickers() -> pd.DataFrame:
    if TICKER_CACHE.exists():
        try:
            cached = pd.read_csv(TICKER_CACHE, dtype=str)
            if {"code", "name"}.issubset(cached.columns) and not cached.empty:
                return cached
        except Exception:
            pass

    rows: list[dict[str, str]] = []
    try:
        from pykrx import stock

        today = datetime.now(KST).strftime("%Y%m%d")
        for market in ("KOSPI", "KOSDAQ", "KONEX"):
            for code in stock.get_market_ticker_list(today, market=market):
                name = stock.get_market_ticker_name(code)
                if name:
                    rows.append({"code": str(code).zfill(6), "name": str(name), "market": market})
    except Exception:
        rows = []

    if not rows:
        try:
            import FinanceDataReader as fdr

            listing = fdr.StockListing("KRX")
            code_col = "Code" if "Code" in listing.columns else "Symbol"
            name_col = "Name" if "Name" in listing.columns else "Name"
            market_col = "Market" if "Market" in listing.columns else None
            for _, row in listing.iterrows():
                code = str(row.get(code_col, "")).zfill(6)
                name = str(row.get(name_col, "")).strip()
                if re.fullmatch(r"\d{6}", code) and name:
                    rows.append({"code": code, "name": name, "market": str(row.get(market_col, "")) if market_col else ""})
        except Exception:
            rows = []

    df = pd.DataFrame(rows, columns=["code", "name", "market"]).drop_duplicates("code")
    if not df.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TICKER_CACHE, index=False, encoding="utf-8")
    return df


def resolve_stock(text: str) -> tuple[str, str | None]:
    code_match = re.search(r"\b(\d{6})\b", text)
    if code_match:
        code = code_match.group(1)
        name_tokens = [t for t in token_candidates(text) if t != code and not re.fullmatch(r"\d{6}", t)]
        name = name_tokens[0] if name_tokens else None
        return code, name

    tokens = token_candidates(text)
    us_tokens = [t.upper() for t in tokens if re.fullmatch(r"[A-Za-z]{1,6}", t)]
    if us_tokens:
        return us_tokens[0], None

    if not tokens:
        raise ValueError("종목명 또는 종목코드를 찾지 못했습니다.")

    query = "".join(tokens)
    tickers = load_krx_tickers()
    if tickers.empty:
        raise ValueError("종목코드 없이 종목명을 조회할 수 없습니다. 예: 005930 삼성전자 분석해줘")

    exact = tickers[tickers["name"] == query]
    if not exact.empty:
        row = exact.iloc[0]
        return str(row["code"]).zfill(6), str(row["name"])

    contains = tickers[tickers["name"].str.contains(re.escape(query), na=False)]
    if len(contains) == 1:
        row = contains.iloc[0]
        return str(row["code"]).zfill(6), str(row["name"])
    if len(contains) > 1:
        options = ", ".join(f"{r.name}({r.code})" for r in contains.head(8).itertuples(index=False))
        raise ValueError(f"종목명이 여러 개와 일치합니다. 종목코드를 함께 입력하세요: {options}")

    compact_names = tickers.assign(_compact=tickers["name"].str.replace(" ", "", regex=False))
    compact = compact_names[compact_names["_compact"].str.contains(re.escape(query), na=False)]
    if len(compact) == 1:
        row = compact.iloc[0]
        return str(row["code"]).zfill(6), str(row["name"])
    if len(compact) > 1:
        options = ", ".join(f"{r.name}({r.code})" for r in compact.head(8).itertuples(index=False))
        raise ValueError(f"종목명이 여러 개와 일치합니다. 종목코드를 함께 입력하세요: {options}")

    raise ValueError(f"'{query}'에 해당하는 종목을 찾지 못했습니다. 종목코드를 함께 입력하세요.")


def parse_request(text: str) -> ParsedRequest:
    text = text.strip()
    if not text:
        raise ValueError("분석할 문장을 입력하세요.")
    code, name = resolve_stock(text)
    return ParsedRequest(code=code, name=name, mode=detect_mode(text), original=text)


def run_request(request: ParsedRequest) -> str:
    if request.mode == "kiwoom":
        return command_chart_analyzer.analyze_command_chart(request.code, request.name)
    if request.mode == "intraday":
        return analyze_stock_intraday.run(request.code, request.name)
    if request.mode == "daily":
        return analyze_stock.run(request.code, request.name)
    return command_chart_analyzer.analyze_integrated_chart(request.code, request.name)


def run_text(text: str) -> str:
    request = parse_request(text)
    label = "통합"
    print(f"[분석 실행] {label}: {request.code} {request.name or ''}".strip())
    return run_request(request)


def repl() -> int:
    print("Money 분석 프롬프트")
    print("예: 삼성전자 분석해줘 / 005930 삼성전자 분석해줘")
    print("키움 로그인 완료 후 공개 데이터 + 키움 실시간 보정 + SSE 통합 보고서 1개로 생성됩니다.")
    print("종료: exit 또는 quit")
    while True:
        try:
            line = input("Money> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in {"exit", "quit", "q", "종료", "나가기"}:
            return 0
        try:
            print(run_text(line))
        except Exception as exc:
            print(f"[오류] {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="자연어 입력으로 Money 분석기를 실행합니다.")
    parser.add_argument("prompt", nargs="*", help='예: "삼성전자 분석해줘"')
    args = parser.parse_args()
    if args.prompt:
        try:
            print(run_text(" ".join(args.prompt)))
            return 0
        except Exception as exc:
            print(f"[오류] {exc}", file=sys.stderr)
            return 1
    return repl()


if __name__ == "__main__":
    raise SystemExit(main())
