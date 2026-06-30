from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
TEST_OUTPUTS_DIR = ROOT / "test_outputs"


@dataclass
class TestCase:
    name: str
    code: str
    market_expected: str
    purpose: str = ""


@dataclass
class TestResult:
    case: TestCase
    passed: bool
    failures: list[str] = field(default_factory=list)
    report_md: Path | None = None
    report_html: Path | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def category_summary(self) -> dict[str, int]:
        buckets = {
            "QA 실패": 0,
            "계산 오류": 0,
            "가격 복붙 오류": 0,
            "문장 의미 오류": 0,
            "시장 suffix 오류": 0,
            "데이터 수집 실패": 0,
            "분봉 상태 문구 오류": 0,
            "목표/익절 라벨 오류": 0,
            "방어선 라벨 오류": 0,
            "장초반 거래량 해석 오류": 0,
            "프로 판단 레이어 오류": 0,
            "매매 시나리오 오류": 0,
            "트레이딩 점수 오류": 0,
            "눌림목 아래 문구 오류": 0,
            "거래량 점수 과대평가 오류": 0,
            "현재가 손익비 점수 과대평가 오류": 0,
            "알림 가격 섹션명 오류": 0,
            "A시나리오 목표/손절 오류": 0,
            "깊은 눌림목 상태 설명 오류": 0,
            "손익비 제한 설명 오류": 0,
            "중복 상태 문구 오류": 0,
            "보유자 방어 관찰 문구 오류": 0,
            "섹터 복붙 문구 오류": 0,
            "RSI 해석 오류": 0,
            "MACD 해석 오류": 0,
            "볼린저밴드 해석 오류": 0,
            "눌림목 겹침 구간 설명 오류": 0,
            "지지/저항 중복 출력 오류": 0,
        }
        joined = "\n".join(self.failures)
        if "QA" in joined or "보고서_QA실패" in joined:
            buckets["QA 실패"] = 1
        if "수익률" in joined or "손익비" in joined or "위험률" in joined:
            buckets["계산 오류"] = 1
        if "whitelist" in joined or "복붙" in joined or "승인되지 않은" in joined:
            buckets["가격 복붙 오류"] = 1
        if "문구" in joined or "돌파" in joined or "범위" in joined or "매물대" in joined or "회복 필요" in joined:
            buckets["문장 의미 오류"] = 1
        if "suffix" in joined or ".KS" in joined or ".KQ" in joined:
            buckets["시장 suffix 오류"] = 1
        if "수집" in joined or "분석 실패" in joined or "RuntimeError" in joined:
            buckets["데이터 수집 실패"] = 1
        if "yfinance 분봉" in joined or "분봉 상태" in joined:
            buckets["분봉 상태 문구 오류"] = 1
        if "목표/익절" in joined or "1차 익절 가격" in joined or "1차 목표가" in joined:
            buckets["목표/익절 라벨 오류"] = 1
        if "방어 라벨" in joined or "방어선 라벨" in joined or "모호한 방어" in joined:
            buckets["방어선 라벨 오류"] = 1
        if "장초반" in joined or "시간가중" in joined:
            buckets["장초반 거래량 해석 오류"] = 1
        if "프로 트레이더" in joined or "상태별 대응" in joined:
            buckets["프로 판단 레이어 오류"] = 1
        if "시나리오" in joined:
            buckets["매매 시나리오 오류"] = 1
        if "트레이딩 점수" in joined:
            buckets["트레이딩 점수 오류"] = 1
        if "눌림목 아래" in joined or "회복 후 지지" in joined:
            buckets["눌림목 아래 문구 오류"] = 1
        if "거래량 점수" in joined and "과도" in joined:
            buckets["거래량 점수 과대평가 오류"] = 1
        if "현재가 기준 손익비 점수" in joined:
            buckets["현재가 손익비 점수 과대평가 오류"] = 1
        if "알림 가격" in joined or "알림 설정 가격" in joined or "눌림 설정 가격" in joined:
            buckets["알림 가격 섹션명 오류"] = 1
        if "A. 지금 매수" in joined or "A시나리오" in joined:
            buckets["A시나리오 목표/손절 오류"] = 1
        if "깊은 눌림목" in joined:
            buckets["깊은 눌림목 상태 설명 오류"] = 1
        if "손익비 제한" in joined or "신규매수 조건" in joined:
            buckets["손익비 제한 설명 오류"] = 1
        if "중복 상태" in joined or "장중 장중" in joined or "분봉 분봉" in joined or "완료 일봉 완료 일봉" in joined:
            buckets["중복 상태 문구 오류"] = 1
        if "방어 관찰" in joined:
            buckets["보유자 방어 관찰 문구 오류"] = 1
        if "섹터" in joined or "전력기기/변압기" in joined:
            buckets["섹터 복붙 문구 오류"] = 1
        if "RSI" in joined:
            buckets["RSI 해석 오류"] = 1
        if "MACD" in joined:
            buckets["MACD 해석 오류"] = 1
        if "볼린저" in joined or "중심선" in joined:
            buckets["볼린저밴드 해석 오류"] = 1
        if "겹침" in joined or "겹치는 구간" in joined:
            buckets["눌림목 겹침 구간 설명 오류"] = 1
        if "주요 저항" in joined or "주요 지지" in joined:
            buckets["지지/저항 중복 출력 오류"] = 1
        return buckets


def load_test_cases(path: Path) -> list[TestCase]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(text)
    except Exception:
        raw = _minimal_yaml_load(text)
    items = raw.get("domestic_stocks", []) if isinstance(raw, dict) else []
    cases: list[TestCase] = []
    for item in items:
        cases.append(
            TestCase(
                name=str(item["name"]),
                code=str(item["code"]).zfill(6),
                market_expected=str(item["market_expected"]),
                purpose=str(item.get("purpose", "")),
            )
        )
    if not cases:
        raise RuntimeError(f"테스트 케이스를 읽지 못했습니다: {path}")
    return cases


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_domestic = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("domestic_stocks:"):
            in_domestic = True
            continue
        if in_domestic and not line.startswith((" ", "-")) and line.strip().endswith(":"):
            break
        if not in_domestic:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current:
                items.append(current)
            current = {}
            stripped = stripped[2:].strip()
            if stripped and ":" in stripped:
                k, v = stripped.split(":", 1)
                current[k.strip()] = v.strip().strip('"')
        elif current is not None and ":" in stripped:
            k, v = stripped.split(":", 1)
            current[k.strip()] = v.strip().strip('"')
    if current:
        items.append(current)
    return {"domestic_stocks": items}


def run_one(case: TestCase, script_name: str, timeout: int) -> TestResult:
    script = ROOT / script_name
    result = TestResult(case=case, passed=False)
    cmd = [sys.executable, str(script), case.code, case.name]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    result.stdout = proc.stdout
    result.stderr = proc.stderr
    if proc.returncode != 0:
        result.failures.append(f"분석 스크립트 실패(returncode={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
        return result
    if "[분석 중단: 보고서 QA 실패]" in proc.stdout:
        result.failures.append("보고서 QA 실패가 발생했습니다.")
    out_dir = REPORTS_DIR / f"{safe_name(case.name)}_{case.code}"
    md_path = out_dir / f"{safe_name(case.name)}_{case.code}_매매타점_분석보고서.md"
    html_path = out_dir / f"{safe_name(case.name)}_{case.code}_매매타점_분석보고서.html"
    qa_fail_path = out_dir / f"{safe_name(case.name)}_{case.code}_보고서_QA실패.md"
    validation_csv = out_dir / f"{safe_name(case.name)}_{case.code}_데이터검증.csv"
    summary_csv = out_dir / f"{safe_name(case.name)}_{case.code}_지표요약.csv"
    result.report_md = md_path if md_path.exists() else None
    result.report_html = html_path if html_path.exists() else None

    for required, label in [
        (md_path, "Markdown 보고서"),
        (html_path, "HTML 보고서"),
        (validation_csv, "데이터검증 CSV"),
        (summary_csv, "지표요약 CSV"),
    ]:
        if not required.exists():
            result.failures.append(f"{label}가 생성되지 않았습니다: {required}")
    if qa_fail_path.exists():
        result.failures.append(f"QA 실패 파일이 생성되었습니다: {qa_fail_path}")
    if md_path.exists():
        text = md_path.read_text(encoding="utf-8", errors="replace")
        result.failures.extend(validate_report_text(text, case))
    result.passed = not result.failures
    return result


def safe_name(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", text).strip()


def validate_report_text(text: str, case: TestCase) -> list[str]:
    failures: list[str] = []
    if "## 내부 검증" not in text:
        failures.append("내부 검증 섹션이 없습니다.")
    if "내부 검증: 통과" not in text:
        failures.append("내부 검증 통과 문구가 없습니다.")
    for label in ["가격 신뢰도:", "거래량 신뢰도:", "지표 신뢰도:", "교차검증 완전성:", "수급 신뢰도:", "해석 완전성:"]:
        if label not in text:
            failures.append(f"내부 검증 신뢰도 분리 항목이 없습니다: {label}")
    if "데이터 부족 기준으로" in text:
        failures.append("월봉 데이터 부족 금지 문구가 사용되었습니다.")
    if "눌림목 지지가" in text and "매수 관심가" in text:
        failures.append("용어 불일치: 눌림목 지지가와 매수 관심가가 혼용되었습니다.")
    if re.search(r"수급 신뢰도:\s*(낮음|데이터 부족)", text) and re.search(r"데이터 신뢰도(?:는|:)\s*높음", text):
        failures.append("수급 데이터 부족인데 전체 데이터 신뢰도만 높음으로 표시했습니다.")
    if case.market_expected == "KOSPI" and f"{case.code}.KQ" in text:
        failures.append("KOSPI 종목 보고서에 .KQ suffix가 포함되었습니다.")
    if case.market_expected == "KOSDAQ" and f"{case.code}.KS" in text:
        failures.append("KOSDAQ 종목 보고서에 .KS suffix가 포함되었습니다.")
    if any(x in text for x in ["미국 주식:", "S&P 500:", "Nasdaq:", "섹터 ETF:"]):
        failures.append("국내 주식 보고서에 미국 주식 섹션이 포함되었습니다.")
    if re.search(r"(\d[\d,]*)\s*원?\s*~\s*\1\s*원", text):
        failures.append("같은 가격을 범위로 표시했습니다.")
    if re.search(r"당일 고가\s+\d[\d,]*원\s+재돌파\s+또는\s+\d[\d,]*원\s+안착", text):
        failures.append("단기 재돌파선과 일봉 돌파 확인선을 같은 레벨로 묶었습니다.")
    if "## 눌림 설정 가격" in text:
        failures.append("알림 가격 섹션명 오류: ## 눌림 설정 가격이 남아 있습니다.")
    if "| 눌림 가격 |" in text:
        failures.append("알림 가격 섹션명 오류: 표 헤더 '눌림 가격'이 남아 있습니다.")
    if "20일/60일선 회복 확인" in text:
        failures.append("이동평균선 해석에 구식/모호 문구가 남아 있습니다.")
    if re.search(r"\|\s*매수 관심가\s*\|", text):
        current = table_first_price(text, ("현재가", "기준가"))
        recovery = table_first_price(text, ("회복 확인가",))
        if current is not None and recovery is not None and current < recovery:
            failures.append("현재가보다 위의 가격을 매수 관심가 단일 명칭으로 표시했습니다.")
    current = table_first_price(text, ("현재가", "기준가"))
    support = table_first_price(text, ("눌림목 지지가",))
    recovery_text = table_cell_text(text, ("회복 확인가",))
    breakout = table_first_price(text, ("일봉 돌파 확인가", "돌파 확인가"))
    if current is not None and support is not None and breakout is not None and support < current < breakout:
        if "해당 없음" in recovery_text or not recovery_text:
            failures.append("회복 확인가가 필요한 구조인데 해당 없음으로 표시되었습니다.")
    rebreak = table_first_price(text, ("단기 재돌파선",))
    target1 = table_first_price(text, ("1차 목표", "신규매수 기준 1차 목표"))
    if rebreak is not None and target1 is not None and abs(rebreak - target1) <= max(1, target1 * 0.001):
        failures.append("단기 재돌파선과 1차 목표가 같은 가격으로 표시되었습니다.")
    action_text = table_cell_text(text, ("오늘 할 행동",))
    action_lines = [line for line in text.splitlines() if line.strip().startswith("- 지금 할 행동:")]
    action_text = " ".join([action_text, *action_lines])
    if action_text and not extract_prices(action_text):
        failures.append("지금 할 행동에 구체 가격이 없습니다.")
    yfinance_failed = any(
        "yfinance 1분봉" in line and any(phrase in line for phrase in ["수집 실패", "데이터 없음"])
        for line in text.splitlines()
    )
    if yfinance_failed:
        if "yfinance 분봉: 정상" in text:
            failures.append("yfinance 1분봉 수집 실패 상태에서 yfinance 분봉 정상 문구가 사용되었습니다.")
        if "| 분봉 데이터 신뢰도 | 통과 |" in text:
            failures.append("yfinance 1분봉 수집 실패 상태에서 분봉 데이터 신뢰도를 통과로 표시했습니다.")
    for label in ["1차 익절 가격", "2차 익절 가격", "1차 목표가", "2차 목표가"]:
        if label in text:
            failures.append(f"목표/익절 라벨이 표준 표현과 어긋났습니다: {label}")
    if "## 1. 장중 매매 판단" in text:
        for label in ["장중 주의선", "장중 방어선", "스윙 최종 방어선", "전량 이탈 조건"]:
            if label not in text:
                failures.append(f"장중 보고서에 필수 방어 라벨이 없습니다: {label}")
        for label in ["| 주의선 |", "| 방어선 |"]:
            if label in text:
                failures.append(f"장중 보고서에서 모호한 방어 라벨을 사용했습니다: {label}")
    analysis_time = extract_analysis_time(text)
    if analysis_time is not None and analysis_time.time() < time(9, 30):
        for phrase in ["거래량 급증 확정", "강한 거래량 확인", "거래량 동반 확정"]:
            if phrase in text:
                failures.append(f"장초반 시간가중 거래량을 확정 신호로 표현했습니다: {phrase}")
        if "시간가중 환산 거래량" in text and "장초반 30분 이내 시간가중 환산 거래량은 과장될 수 있으므로 참고값으로만 봅니다." not in text:
                failures.append("장초반 시간가중 환산 거래량 참고값 안내 문구가 누락되었습니다.")
    failures.extend(validate_pro_trader_layer(text))
    failures.extend(validate_new_qa_items(text))
    if "수급 데이터 부족" in text or "수급 판단 보류" in text or "수집 실패" in text:
        for phrase in ["매도 흐름", "동반 순매수", "동반 순매도", "수급 우호", "수급 악화"]:
            if phrase in text:
                failures.append(f"수급 실패/부족 상태에서 수급 방향을 단정했습니다: {phrase}")
    return failures


def extract_prices(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"(\d{1,3}(?:,\d{3})+|\d{4,7})\s*원", text):
        try:
            values.append(float(match.group(1).replace(",", "")))
        except ValueError:
            pass
    return values


def table_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


def table_cell_text(text: str, labels: tuple[str, ...]) -> str:
    for cells in table_rows(text):
        if len(cells) >= 2 and cells[0] in labels:
            return " | ".join(cells[1:])
    return ""


def table_first_price(text: str, labels: tuple[str, ...]) -> float | None:
    prices = extract_prices(table_cell_text(text, labels))
    return prices[0] if prices else None


def extract_first_ratio(text: str, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def report_section(text: str, heading: str) -> str:
    match = re.search(rf"^##\s+{re.escape(heading)}\s*$", text, re.MULTILINE)
    if not match:
        return ""
    rest = text[match.end() :]
    next_heading = re.search(r"\n##\s+", rest)
    return rest[: next_heading.start()] if next_heading else rest


def extract_trading_score(text: str) -> float | None:
    section = report_section(text, "트레이딩 점수")
    match = re.search(r"\|\s*총점\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*점\s*\|", section)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_trading_score_item(text: str, label: str) -> float | None:
    section = report_section(text, "트레이딩 점수")
    for cells in table_rows(section):
        if len(cells) >= 2 and cells[0] == label:
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", cells[1])
            if match:
                return float(match.group(1))
    return None


def scenario_row(text: str, scenario_name: str) -> list[str] | None:
    section = report_section(text, "매매 시나리오")
    for cells in table_rows(section):
        if cells and cells[0] == scenario_name:
            return cells
    return None


def find_row(text: str, label: str) -> list[str] | None:
    for cells in table_rows(text):
        if cells and cells[0] == label:
            return cells
    return None


def validate_new_qa_items(text: str) -> list[str]:
    failures: list[str] = []
    for phrase in ["장중 장중", "분봉 분봉", "완료 일봉 완료 일봉"]:
        if phrase in text:
            failures.append(f"중복 상태 문구 오류: {phrase} 중복 표현이 있습니다.")
    if re.search(r"주요 저항:\s*([0-9,]+원),\s*\1", text):
        failures.append("지지/저항 중복 출력 오류: 주요 저항 가격이 중복 출력되었습니다.")
    if re.search(r"주요 지지:\s*([0-9,]+원),\s*\1", text):
        failures.append("지지/저항 중복 출력 오류: 주요 지지 가격이 중복 출력되었습니다.")

    current = table_first_price(text, ("현재가", "기준가", "대표 기준가"))
    pullback_text = table_cell_text(text, ("얕은 눌림목 대기 가격", "눌림목 매수 가격"))
    pullback_prices = extract_prices(pullback_text)
    pullback_low = min(pullback_prices) if pullback_prices else None
    if current is not None and pullback_low is not None and current < pullback_low:
        has_recovery_wording = "회복 후 지지 확인" in text or "재진입" in text
        if not has_recovery_wording and any(phrase in text for phrase in ["눌림목 지지 중", "지지 확인 전까지 대기"]):
            failures.append("눌림목 아래 문구 오류: 현재가가 눌림목 아래인데 회복 후 지지 확인 표현이 없습니다.")

    intraday_high = table_first_price(text, ("당일 고가", "당일 고가 / 저가"))
    breakout_line = table_first_price(text, ("일봉 돌파 확인선", "오늘 종가 확인 필요 가격"))
    volume_score = extract_trading_score_item(text, "거래량 점수")
    if intraday_high is not None and current is not None and breakout_line is not None:
        if intraday_high > breakout_line and current < breakout_line and volume_score is not None and volume_score > 12:
            failures.append("거래량 점수 과대평가 오류: 돌파 유지 실패 상태에서 거래량 점수가 12점을 초과했습니다.")

    current_rr_score = extract_trading_score_item(text, "현재가 기준 손익비 점수")
    now_buy_unavailable = "| 지금 매수 | 불가 |" in text or "| 지금 바로 매수 | 불가 |" in text or "| 지금 바로 매수 가능 여부 | 불가 |" in text
    if now_buy_unavailable:
        if current_rr_score is None:
            failures.append("현재가 손익비 점수 과대평가 오류: 현재가 기준 손익비 점수 행이 없습니다.")
        elif current_rr_score > 14:
            failures.append("현재가 손익비 점수 과대평가 오류: 지금 바로 매수 불가인데 현재가 기준 손익비 점수가 14점을 초과했습니다.")
        a_row = scenario_row(text, "A. 지금 매수")
        if a_row and "매수 금지" in a_row and any("0%" in cell for cell in a_row):
            if len(a_row) < 6 or "해당 없음" not in a_row[4] or "해당 없음" not in a_row[5]:
                failures.append("A시나리오 목표/손절 오류: 지금 매수 금지/0%인데 목표 또는 손절 가격이 표시되었습니다.")

    deep_text = table_cell_text(text, ("깊은 눌림목 대기 가격",))
    if not deep_text:
        deep_text = table_cell_text(text, ("장중 눌림 가격", "눌림목 매수 가격"))
    deep_prices = extract_prices(deep_text)
    if current is not None and len(deep_prices) >= 2:
        deep_low = min(deep_prices)
        deep_high = max(deep_prices)
        in_deep = deep_low <= current <= deep_high
        in_shallow = pullback_low is not None and pullback_prices and min(pullback_prices) <= current <= max(pullback_prices)
        if in_shallow and in_deep and "겹치는 구간" not in text:
            failures.append("눌림목 겹침 구간 설명 오류: 현재가가 얕은/깊은 눌림목에 동시에 포함되는데 겹침 구간 설명이 없습니다.")
        if deep_low <= current <= deep_high:
            if not in_shallow and ("깊은 눌림목 구간 안" not in text or "반등 확인 후" not in text):
                failures.append("깊은 눌림목 상태 설명 오류: 현재가가 깊은 눌림목 구간 안인데 설명이 부족합니다.")

    current_rr = extract_first_ratio(
        text,
        (
            r"현재가 기준 신규매수 1차 목표 손익비는\s*([0-9]+(?:\.[0-9]+)?)배",
            r"신규매수 기준 1차 목표 손익비:\s*([0-9]+(?:\.[0-9]+)?)배",
            r"기준가에서 신규매수 기준 1차 목표까지의 손익비는\s*([0-9]+(?:\.[0-9]+)?)배",
        ),
    )
    if now_buy_unavailable and current_rr is not None and current_rr >= 1.5:
        if "손익비는" not in text or "지금 바로 신규매수 조건은 충족하지 못했습니다" not in text:
            failures.append("손익비 제한 설명 오류: 매수 불가 상태에서 양호한 손익비 제한 설명이 없습니다.")

    intraday_warning = table_first_price(text, ("장중 주의선",))
    if current is not None and intraday_warning is not None and current < intraday_warning:
        if "방어 관찰이 우선" not in text:
            failures.append("보유자 방어 관찰 문구 오류: 현재가가 장중 주의선 아래인데 방어 관찰 우선 문구가 없습니다.")

    code_text = table_cell_text(text, ("종목코드/티커",))
    code_match = re.search(r"\d{6}", code_text)
    code = code_match.group(0) if code_match else ""
    if code != "033100" and "전력기기/변압기" in text:
        failures.append("섹터 복붙 문구 오류: 비해당 종목 보고서에 전력기기/변압기 문구가 포함되었습니다.")

    rsi_row = find_row(text, "RSI")
    if rsi_row and len(rsi_row) >= 3:
        rsi_vals = re.findall(r"[-+]?\d+(?:\.\d+)?", rsi_row[1])
        if rsi_vals:
            rsi = float(rsi_vals[0])
            rsi_comment_text = " | ".join(rsi_row[2:])
            if rsi >= 50 and "50 아래" in rsi_comment_text:
                failures.append("RSI 해석 오류: RSI가 50 이상인데 50 아래 문구가 사용되었습니다.")
            if rsi < 50 and "50선을 회복한 상태" in rsi_comment_text:
                failures.append("RSI 해석 오류: RSI가 50 미만인데 50선 회복 문구가 사용되었습니다.")

    macd_row = find_row(text, "MACD")
    if macd_row and len(macd_row) >= 3:
        macd_nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", macd_row[1])[:3]]
        if len(macd_nums) >= 3:
            macd, signal, hist = macd_nums[:3]
            macd_comment_text = " | ".join(macd_row[2:])
            if macd > signal and hist > 0 and any(phrase in macd_comment_text for phrase in ["히스토그램 개선 확인 전", "MACD 개선 확인 전"]):
                failures.append("MACD 해석 오류: MACD가 신호선 위이고 히스토그램 양수인데 개선 대기 문구가 사용되었습니다.")

    bb_row = find_row(text, "볼린저밴드")
    if bb_row and len(bb_row) >= 3 and current is not None:
        bb_prices = extract_prices(bb_row[1])
        if bb_prices:
            bb_mid = bb_prices[0]
            bb_comment_text = " | ".join(bb_row[2:])
            if current > bb_mid * 1.02 and "중심선 근처" in bb_comment_text:
                failures.append("볼린저밴드 해석 오류: 현재가가 중심선보다 충분히 위인데 중심선 근처 문구가 사용되었습니다.")
            if current < bb_mid * 0.98 and "중심선 위" in bb_comment_text:
                failures.append("볼린저밴드 해석 오류: 현재가가 중심선 아래인데 중심선 위 문구가 사용되었습니다.")
    return failures


def validate_pro_trader_layer(text: str) -> list[str]:
    failures: list[str] = []
    for heading in [
        "1. 최종 결론",
        "2. 프로 트레이더 관점",
        "3. 차트 분석",
        "4. 보조지표 판단",
        "5. 매매 타점",
        "6. 최종 한 문단",
        "내부 검증",
    ]:
        if f"## {heading}" not in text:
            failures.append(f"프로 판단 레이어 필수 섹션이 없습니다: {heading}")

    common_confirmation = "회복/돌파 공통 확인가" in text
    for label in ["회복 확인가", "눌림목 지지가", "단기 재돌파선", "일봉 돌파 확인가", "장중 방어선", "스윙 손절선"]:
        if label in {"회복 확인가", "일봉 돌파 확인가"} and common_confirmation:
            continue
        if label not in text:
            failures.append(f"가격 명칭 분리 필수 항목이 없습니다: {label}")

    now_buy_unavailable = "| 지금 매수 | 불가 |" in text or "| 지금 바로 매수 | 불가 |" in text or "| 지금 바로 매수 가능 여부 | 불가 |" in text
    if now_buy_unavailable and "| 주 전략 | 눌림목 대기 |" in text:
        failures.append("지금 매수 불가 보고서에서 주 전략이 구식 눌림목 대기로 표시되었습니다.")

    if "| 지금 매수 | 가능 |" in text and "공격 매수 가능" in text:
        if "회복 확인가 또는 돌파 확인가 조건 미충족" in text:
            failures.append("지금 매수 가능인데 조건 미충족 문구가 함께 사용되었습니다.")

    supply_section = text
    if any(phrase in text for phrase in ["수급 데이터 부족", "수급 판단 보류", "수급 데이터 수집 실패"]):
        match = re.search(r"\|\s*수급 점수\s*\|\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10\s*\|", supply_section)
        if match and float(match.group(1)) > 6:
            failures.append("수급 데이터 부족 상태에서 수급 점수를 과도하게 높게 부여했습니다.")
    return failures


def extract_analysis_time(text: str) -> datetime | None:
    match = re.search(r"\|\s*(?:분석 실행 시각|현재 시각)\s*\|\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})(?::\d{2})?\s*KST\s*\|", text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M")
    except ValueError:
        return None


def write_summary(results: list[TestResult]) -> Path:
    TEST_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    category_totals = {
        "QA 실패": 0,
        "계산 오류": 0,
        "가격 복붙 오류": 0,
        "문장 의미 오류": 0,
        "시장 suffix 오류": 0,
        "데이터 수집 실패": 0,
        "분봉 상태 문구 오류": 0,
        "목표/익절 라벨 오류": 0,
        "방어선 라벨 오류": 0,
        "장초반 거래량 해석 오류": 0,
        "프로 판단 레이어 오류": 0,
        "매매 시나리오 오류": 0,
        "트레이딩 점수 오류": 0,
        "눌림목 아래 문구 오류": 0,
        "거래량 점수 과대평가 오류": 0,
        "현재가 손익비 점수 과대평가 오류": 0,
        "알림 가격 섹션명 오류": 0,
        "A시나리오 목표/손절 오류": 0,
        "깊은 눌림목 상태 설명 오류": 0,
        "손익비 제한 설명 오류": 0,
        "중복 상태 문구 오류": 0,
        "보유자 방어 관찰 문구 오류": 0,
        "섹터 복붙 문구 오류": 0,
        "RSI 해석 오류": 0,
        "MACD 해석 오류": 0,
        "볼린저밴드 해석 오류": 0,
        "눌림목 겹침 구간 설명 오류": 0,
        "지지/저항 중복 출력 오류": 0,
    }
    for r in results:
        for k, v in r.category_summary.items():
            category_totals[k] += v
    final = "국내 주식 분석 엔진 사용 가능" if passed == total else "수정 필요"
    lines = [
        "# 국내 주식 분석 엔진 회귀 테스트 결과",
        "",
        "## 요약",
        "",
        f"총 테스트 종목: {total}개",
        f"정상 통과: {passed}개",
        f"QA 실패: {category_totals['QA 실패']}개",
        f"계산 오류: {category_totals['계산 오류']}개",
        f"가격 복붙 오류: {category_totals['가격 복붙 오류']}개",
        f"문장 의미 오류: {category_totals['문장 의미 오류']}개",
        f"시장 suffix 오류: {category_totals['시장 suffix 오류']}개",
        f"데이터 수집 실패: {category_totals['데이터 수집 실패']}개",
        f"분봉 상태 문구 오류: {category_totals['분봉 상태 문구 오류']}개",
        f"목표/익절 라벨 오류: {category_totals['목표/익절 라벨 오류']}개",
        f"방어선 라벨 오류: {category_totals['방어선 라벨 오류']}개",
        f"장초반 거래량 해석 오류: {category_totals['장초반 거래량 해석 오류']}개",
        f"프로 판단 레이어 오류: {category_totals['프로 판단 레이어 오류']}개",
        f"매매 시나리오 오류: {category_totals['매매 시나리오 오류']}개",
        f"트레이딩 점수 오류: {category_totals['트레이딩 점수 오류']}개",
        f"눌림목 아래 문구 오류: {category_totals['눌림목 아래 문구 오류']}개",
        f"거래량 점수 과대평가 오류: {category_totals['거래량 점수 과대평가 오류']}개",
        f"현재가 손익비 점수 과대평가 오류: {category_totals['현재가 손익비 점수 과대평가 오류']}개",
        f"알림 가격 섹션명 오류: {category_totals['알림 가격 섹션명 오류']}개",
        f"A시나리오 목표/손절 오류: {category_totals['A시나리오 목표/손절 오류']}개",
        f"깊은 눌림목 상태 설명 오류: {category_totals['깊은 눌림목 상태 설명 오류']}개",
        f"손익비 제한 설명 오류: {category_totals['손익비 제한 설명 오류']}개",
        f"중복 상태 문구 오류: {category_totals['중복 상태 문구 오류']}개",
        f"보유자 방어 관찰 문구 오류: {category_totals['보유자 방어 관찰 문구 오류']}개",
        f"섹터 복붙 문구 오류: {category_totals['섹터 복붙 문구 오류']}개",
        f"RSI 해석 오류: {category_totals['RSI 해석 오류']}개",
        f"MACD 해석 오류: {category_totals['MACD 해석 오류']}개",
        f"볼린저밴드 해석 오류: {category_totals['볼린저밴드 해석 오류']}개",
        f"눌림목 겹침 구간 설명 오류: {category_totals['눌림목 겹침 구간 설명 오류']}개",
        f"지지/저항 중복 출력 오류: {category_totals['지지/저항 중복 출력 오류']}개",
        "",
        f"최종 판정: {final}",
        "",
        "## 종목별 결과",
        "",
        "| 종목 | 코드 | 시장 | 결과 | 실패 사유 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        failure = "-" if r.passed else "<br>".join(escape_md(x) for x in r.failures)
        lines.append(f"| {r.case.name} | {r.case.code} | {r.case.market_expected} | {'PASS' if r.passed else 'FAIL'} | {failure} |")
    path = TEST_OUTPUTS_DIR / "regression_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def run_static_fixture_checks() -> list[str]:
    failures: list[str] = []
    from analyze_stock import (
        SourceFrame,
        assess_volume_candle,
        assess_intraday_overheated_breakout,
        assess_volume_momentum_conflict,
        breakout_volume_condition_comment,
        build_trade_state,
        build_report_qa_section,
        compressed_low_rr_warning,
        daily_trend_state_from_values,
        display_rebreak_line,
        render_trade_state_actions,
        state_code_report_rows,
        trade_state_to_dict,
        macd_grade_from_values,
        macd_comment,
        monthly_chart_comment,
        recovery_confirmation_level,
        reliability_breakdown,
        rebreak_display_label,
        rsi_grade_from_value,
        source_validation,
        strategy_labels_by_price,
        validation_labels,
        validate_holder_action_split,
        validate_clean_data_action_wording,
        validate_common_confirmation_price_qa,
        validate_daily_trend_label_qa,
        validate_intraday_overheated_breakout_qa,
        validate_low_rr_wording,
        validate_low_rr_priority_qa,
        validate_macd_zone_wording_qa,
        validate_markdown_html_table_rendering_qa,
        validate_near_intraday_defense_warning,
        validate_precision_limited_points,
        validate_rebreak_target_duplicate_qa,
        validate_rebreak_target_order_qa,
        validate_recovery_confirmation_qa,
        validate_rr_warning_dedup_qa,
        validate_stale_yfinance_explanation,
        validate_strategy_labeling_qa,
        validate_today_action_has_price,
        validate_trade_state_qa,
        validate_volume_candle_qa,
        validate_volume_momentum_conflict_qa,
    )
    import pandas as pd
    from core.decision_engine import DecisionContext, DecisionLevels, PriceEvidence, evaluate_decision

    current_price = 5100.0
    buy_low = 5200.0
    buy_high = 5250.0
    intraday_defense_line = 5080.0
    breakout_line = 5400.0
    labels = strategy_labels_by_price(
        current_price,
        buy_low,
        buy_high,
        5250.0,
        breakout_line,
        intraday_defense_line,
        5000.0,
        "중간",
    )
    if labels["primary_strategy"] not in {"회복 확인 대기", "신규매수 금지, 장중 방어 우선"}:
        failures.append(f"fixture 전략 라벨 오류: {labels['primary_strategy']}")
    if rsi_grade_from_value(50.43) != "중립 회복":
        failures.append("fixture RSI 등급 오류: RSI 50~55는 중립 회복이어야 합니다.")
    if macd_grade_from_values(10, 5, 1, current_price, breakout_line) != "혼조/개선 중":
        failures.append("fixture MACD 등급 오류: 돌파 전 양의 MACD는 혼조/개선 중이어야 합니다.")
    monthly_comment = monthly_chart_comment("데이터 부족")
    if "데이터 부족 기준으로" in monthly_comment or "계산 표본 부족" not in monthly_comment:
        failures.append("fixture 월봉 데이터 부족 문구 오류")
    qa_section = build_report_qa_section(
        "중간",
        validation_error_count=0,
        reliability_details={
            "가격 신뢰도": "높음",
            "거래량 신뢰도": "높음",
            "지표 신뢰도": "높음",
            "교차검증 완전성": "중간",
            "수급 신뢰도": "낮음",
            "해석 완전성": "중간",
        },
    )
    if "내부 검증: 통과" not in qa_section or "수급 신뢰도: 낮음" not in qa_section or "교차검증 완전성: 중간" not in qa_section:
        failures.append("fixture 내부 검증 통과 표시 오류")

    bad_report = "| 항목 | 판단 |\n|---|---|\n| 매수 관심가 | 5,200원 |\n"
    bad_decision = {"주전략": "눌림목 대기", "장중방어선": intraday_defense_line}
    bad_metrics = {"current_price": current_price, "buy_low": buy_low}
    strategy_failures = validate_strategy_labeling_qa(bad_report, bad_decision, bad_metrics, {})
    if not strategy_failures:
        failures.append("fixture 전략 QA 오류: 현재가가 매수 관심가 아래인데 눌림목 대기를 허용했습니다.")

    good_report = "장중 방어선이 가까워 신규 진입은 손익비보다 실패 확인 리스크가 더 큼"
    near_failures = validate_near_intraday_defense_warning(
        good_report,
        {"장중방어선": intraday_defense_line},
        {"current_price": current_price},
    )
    if near_failures:
        failures.append("fixture 장중 방어선 근접 경고 QA 오류")

    jeryong_current = 54600.0
    jeryong_support_high = 54100.0
    jeryong_breakout = 57200.0
    recovery_text, recovery_line = recovery_confirmation_level(
        jeryong_current,
        jeryong_support_high,
        57100.0,
        57080.0,
        jeryong_breakout,
    )
    if recovery_text == "해당 없음" or recovery_line is None:
        failures.append("fixture 제룡전기형 회복 확인가 산정 오류")
    rebreak_text, _, rebreak_merged = display_rebreak_line(60000.0, 60000.0)
    if not rebreak_merged or "별도 표시 생략" not in rebreak_text:
        failures.append("fixture 단기 재돌파선/1차 목표 중복 제거 오류")
    jeryong_rebreak_text, jeryong_rebreak_action, jeryong_rebreak_merged = display_rebreak_line(60000.0, 55700.0, 60600.0)
    jeryong_rebreak_label = rebreak_display_label(60000.0, 55700.0, 60600.0)
    if (
        not jeryong_rebreak_merged
        or jeryong_rebreak_label != "강한 저항/2차 목표 전 확인선"
        or "보유자 익절/비중관리" not in jeryong_rebreak_action
        or "강한 저항/2차 목표 전 확인선" not in jeryong_rebreak_text
    ):
        failures.append("fixture 제룡전기형 단기 재돌파선 강한 저항 재분류 오류")
    bad_recovery_report = """# fixture
| 항목 | 판단 |
|---|---|
| 현재가 | 54,600원 |
| 회복 확인가 | 해당 없음 |
| 눌림목 지지가 | 51,900원~54,100원 |
| 일봉 돌파 확인가 | 57,200원 |
"""
    recovery_failures = validate_recovery_confirmation_qa(
        bad_recovery_report,
        {},
        {"current_price": jeryong_current, "buy_high": jeryong_support_high, "breakout_line": jeryong_breakout},
    )
    if not recovery_failures:
        failures.append("fixture 회복 확인가 누락 QA 오류")
    bad_duplicate_report = """# fixture
| 구분 | 가격 | 행동 |
|---|---:|---|
| 단기 재돌파선 | 60,000원 | 확인 |
| 1차 목표 | 60,000원 | 익절 |
"""
    duplicate_failures = validate_rebreak_target_duplicate_qa(bad_duplicate_report, {}, {})
    if not duplicate_failures:
        failures.append("fixture 단기 재돌파선/1차 목표 중복 QA 오류")
    bad_rebreak_order_report = """# fixture
## 5. 매매 타점

| 구분 | 가격 | 행동 |
|---|---:|---|
| 단기 재돌파선 | 60,000원 | 확인 |
| 1차 목표 | 55,700원 | 익절 |
| 2차 목표 | 60,600원 | 추가 익절 |
| 장중 방어선 | 49,400원 | 방어 |
| 스윙 손절선 | 45,850원 | 손절 |
"""
    order_failures = validate_rebreak_target_order_qa(
        bad_rebreak_order_report,
        {"1차목표": 55700.0, "2차목표": 60600.0},
        {"rebreak_line": 60000.0, "target1": 55700.0, "target2": 60600.0},
    )
    if not order_failures:
        failures.append("fixture 단기 재돌파선/목표권 역할 재분류 QA 오류")
    bad_table_report = """# fixture
## 5. 매매 타점

| 구분 | 가격 | 행동 |
|---|---:|---|
| 회복 확인가 | 53,300원 | 확인 |
| 눌림목 지지가 | 49,100원~51,200원 | 지지 |
| 강한 저항/2차 목표 전 확인선 | 60,000원 | 보유자 익절/비중관리 |

| 1차 목표 | 55,700원 | 익절 |
| 2차 목표 | 60,600원 | 추가 익절 |
| 장중 방어선 | 해당 없음 | 장외 |
| 스윙 손절선 | 45,850원 | 손절 |
"""
    if not validate_markdown_html_table_rendering_qa(bad_table_report):
        failures.append("fixture 매매 타점 표 렌더링 QA 오류: 필수 행 분리를 잡지 못했습니다.")
    good_table_report = """# fixture
## 5. 매매 타점

| 구분 | 가격 | 행동 |
|---|---:|---|
| 회복 확인가 | 53,300원 | 확인 |
| 눌림목 지지가 | 49,100원~51,200원 | 지지 |
| 강한 저항/2차 목표 전 확인선 | 60,000원 | 보유자 익절/비중관리 |
| 1차 목표 | 55,700원 | 익절 |
| 2차 목표 | 60,600원 | 추가 익절 |
| 장중 방어선 | 해당 없음 | 장외 |
| 스윙 손절선 | 45,850원 | 손절 |
"""
    if validate_markdown_html_table_rendering_qa(good_table_report):
        failures.append("fixture 매매 타점 표 렌더링 QA 오류: 정상 표를 실패 처리했습니다.")
    no_price_action = "- 지금 할 행동: 새 지지선이 만들어지는지 확인합니다."
    if not validate_today_action_has_price(no_price_action):
        failures.append("fixture 지금 할 행동 가격 누락 QA 오류")

    invalid_target_state = build_trade_state(
        current_price=10000,
        pullback_low=9500,
        pullback_high=9900,
        recovery_line=10100,
        breakout_line=10200,
        target1=11000,
        target2=11000,
        defense_line=9000,
        short_rebreak_line=10100,
        entry_rr=1.5,
        swing_rr=1.5,
    )
    if not invalid_target_state.blocking_errors:
        failures.append("fixture blocking QA 오류: target1 >= target2를 blocking error로 만들지 못했습니다.")
    invalid_rebreak_state = build_trade_state(
        current_price=295000,
        pullback_low=277500,
        pullback_high=294500,
        recovery_line=305000,
        breakout_line=305000,
        target1=314000,
        target2=341500,
        defense_line=252500,
        short_rebreak_line=368500,
        entry_rr=1.5,
        swing_rr=1.5,
    )
    if not invalid_rebreak_state.blocking_errors:
        failures.append("fixture blocking QA 오류: short_rebreak_line > target2를 blocking error로 만들지 못했습니다.")
    blocked_report = "## 내부 검증\n\n내부 검증: 통과\n"
    blocked_decision = {"상태코드": trade_state_to_dict(invalid_target_state), "최종판단": render_trade_state_actions(invalid_target_state.final_action_state)["final_judgment"]}
    if not validate_trade_state_qa(blocked_report, blocked_decision, {}, {"trade_state": blocked_decision["상태코드"]}):
        failures.append("fixture blocking QA 오류: blocking error가 있는데 내부 검증 통과 보고서를 허용했습니다.")

    dates_core = pd.to_datetime(["2026-06-18"])
    core_frame = pd.DataFrame(
        {"Open": [27600.0], "High": [28150.0], "Low": [27450.0], "Close": [27850.0], "Volume": [12345678.0]},
        index=dates_core,
    )
    yf_stale_frame = pd.DataFrame(
        {"Open": [27050.0], "High": [27600.0], "Low": [26900.0], "Close": [27400.0], "Volume": [9876543.0]},
        index=pd.to_datetime(["2026-06-17"]),
    )
    validation, reliability, stop_precision = source_validation(
        [
            SourceFrame("pykrx", core_frame.copy(), ""),
            SourceFrame("FinanceDataReader", core_frame.copy(), ""),
            SourceFrame("yfinance", yf_stale_frame.copy(), ""),
        ],
        pd.Timestamp("2026-06-18").date(),
    )
    price_label, volume_label, validation_note = validation_labels(validation)
    if stop_precision:
        failures.append("fixture 보조 소스 지연 오류: pykrx/FDR 일치 + yfinance stale인데 정밀 판단을 중단했습니다.")
    if reliability == "낮음" or price_label == "실패" or volume_label == "실패":
        failures.append("fixture 보조 소스 지연 오류: 대표 가격 신뢰도를 과도하게 낮췄습니다.")
    yf_row = validation[validation["소스"] == "yfinance"].iloc[0]
    if yf_row["검증유형"] != "보조 소스 최신거래일 지연" or yf_row["대표가격사용"] != "제외":
        failures.append("fixture 보조 소스 지연 오류: yfinance stale을 대표 가격에서 제외하지 않았습니다.")
    if "yfinance 보조 소스 최신거래일 지연" not in validation_note:
        failures.append("fixture 보조 소스 지연 오류: 검증 메모에 yfinance 지연 설명이 없습니다.")
    rel_parts = reliability_breakdown(price_label, volume_label, "수급 데이터 부족", "해당 없음", True, validation_note)
    if rel_parts["가격 신뢰도"] == "낮음" or rel_parts["거래량 신뢰도"] == "낮음" or rel_parts["교차검증 완전성"] != "중간" or rel_parts["수급 신뢰도"] != "낮음":
        failures.append("fixture 신뢰도 분리 오류: 보조 소스 지연/수급 부족 신뢰도 산정이 기대와 다릅니다.")

    precision_report = """# fixture
| 항목 | 판단 |
|---|---|
| 최종 판단 | 데이터 불일치로 정밀 판단 중단 |
| 회복 확인가 | 27,850원 |
| 눌림목 지지가 | 참고 27,600원 |
| 일봉 돌파 확인가 | 참고 28,950원 |
| 1차 목표 | 참고 30,000원 |
| 2차 목표 | 참고 31,500원 |
| 스윙 손절선 | 참고 25,600원 |
"""
    if not validate_precision_limited_points(precision_report, {"최종판단": "데이터 불일치로 정밀 판단 중단"}):
        failures.append("fixture 정밀 판단 중단 QA 오류: 확정 타점 출력을 허용했습니다.")

    stale_report_missing = "가격 신뢰도: 높음\n거래량 신뢰도: 높음\n"
    if not validate_stale_yfinance_explanation(stale_report_missing, {"validation_note": validation_note}):
        failures.append("fixture yfinance stale 설명 QA 오류: 원인 설명 누락을 잡지 못했습니다.")
    stale_report_good = "검증 메모: yfinance 보조 소스 최신거래일 지연으로 대표 가격 산정에서 제외\n가격 신뢰도: 높음\n"
    if validate_stale_yfinance_explanation(stale_report_good, {"validation_note": validation_note}):
        failures.append("fixture yfinance stale 설명 QA 오류: 정상 설명을 실패 처리했습니다.")

    low_rr_report = "회복/돌파 진입 기준 손익비 1.0 미만으로 돌파 추격매수 부적합"
    if not validate_low_rr_wording(low_rr_report, {"rr1": 1.05, "confirm_rr": 0.95}, {"최종판단": "지금은 대기"}):
        failures.append("fixture 손익비 부족 QA 오류: 1.2 미만 신규매수 매력 낮음 누락을 잡지 못했습니다.")
    good_rr_report = "스윙 손절선 기준 손익비 부족으로 신규매수 매력 낮음. 스윙/돌파 신규매수 매력 낮음. 회복/돌파 진입 기준 손익비 1.0 미만으로 돌파 추격매수 부적합"
    if validate_low_rr_wording(good_rr_report, {"rr1": 1.05, "confirm_rr": 0.95}, {"최종판단": "지금은 대기"}):
        failures.append("fixture 손익비 부족 QA 오류: 정상 경고 문구를 실패 처리했습니다.")

    holder_good = "- 보유자: 27,850원 회복 실패 시 추가매수 보류, 27,600원 재이탈 시 단기 비중 축소 검토, 25,600원 이탈 시 방어/손절합니다."
    if validate_holder_action_split(holder_good):
        failures.append("fixture 보유자 대응 QA 오류: 분리 대응 문장을 실패 처리했습니다.")
    holder_bad = "- 보유자: 27,850원 회복 실패 시 일부 비중 축소 검토, 25,600원 이탈 시 방어합니다."
    if not validate_holder_action_split(holder_bad):
        failures.append("fixture 보유자 대응 QA 오류: 회복 확인가/눌림목 지지가 대응 혼동을 잡지 못했습니다.")

    semco_state = assess_intraday_overheated_breakout(
        current_price=2371000,
        daily_breakout_line=2278000,
        weighted_volume_ratio=1.96,
        rsi=70.69,
        intraday_rr=0.98,
        entry_rr=0.60,
        swing_rr=0.36,
    )
    if not semco_state["applies"]:
        failures.append("fixture 삼성전기형 과열 돌파/손익비 부족 판정이 적용되지 않았습니다.")
    if semco_state["primary_strategy"] != "과열·손익비 부족으로 신규 추격매수 부적합":
        failures.append("fixture 삼성전기형 주 전략 오류")
    if "과열 추격 금지" not in semco_state["final"] or "신규 추격매수 부적합" not in semco_state["final"]:
        failures.append("fixture 삼성전기형 최종 판단 오류")
    semco_good_report = """# fixture
| 항목 | 판단 |
|---|---|
| 현재가 | 2,371,000원 |
| 지금 매수 | 불가 |
| 주 전략 | 과열·손익비 부족으로 신규 추격매수 부적합 |
| 회복 확인가 | 이미 회복, 종가 유지 확인으로 대체 |
| 단기 재돌파선 | 장중 현재가 유지 기준 2,371,000원 |
| 일봉 돌파 확인가 | 2,278,000원 |
| 1차 목표 | 2,588,000원 |
| 최종 판단 | 과열 추격 금지·손익비 부족으로 신규 추격매수 부적합 |
| 신규매수자 | 눌림목 재지지 또는 종가 확정 후 다음 거래일 눌림 확인 시 검토 |
| 보유자 | 종가 유지 확인, 1차 목표 접근 시 일부 익절, 장중 방어선 이탈 시 단기 방어 |
| 추가매수자 | 신규매수자보다 더 엄격하게 보류 |

- 지금 하지 말아야 할 행동: 과열권에서 장중 급등 가격을 추격매수하지 않습니다. 손익비가 맞지 않는 돌파 추격매수를 하지 않습니다.
장중 돌파와 거래량은 긍정적이나 RSI 과열과 손익비 부족으로 신규 추격매수는 부적합합니다. 보유자는 종가 유지 여부를 확인하고, 신규자는 눌림 또는 종가 확정 후 재판단합니다.
장중 신규 진입 매력 낮음; 돌파 추격매수 부적합; 스윙 신규매수 부적합; 손익비 부족; 과열 추격 금지.
"""
    semco_metrics = {
        "current_price": 2371000,
        "breakout_line": 2278000,
        "rebreak_line": 2371000,
        "target1": 2588000,
        "intraday_rr": 0.98,
        "confirm_rr": 0.60,
        "rr1": 0.36,
        "rsi": 70.69,
    }
    semco_decision = {
        "현재가": 2371000,
        "지금바로매수": "불가",
        "최종판단": "과열 추격 금지·손익비 부족으로 신규 추격매수 부적합",
        "일봉돌파확인선": 2278000,
        "단기재돌파확인선": 2371000,
        "RSI": 70.69,
        "확인진입손익비": 0.60,
        "스윙손절손익비": 0.36,
        "장중방어손익비": 0.98,
    }
    if validate_intraday_overheated_breakout_qa(semco_good_report, semco_metrics, semco_decision, {"current_price": 2371000}):
        failures.append("fixture 삼성전기형 QA 오류: 정상 과열 돌파 문구를 실패 처리했습니다.")
    if validate_low_rr_wording(semco_good_report, semco_metrics, semco_decision):
        failures.append("fixture 삼성전기형 손익비 QA 오류: 정상 낮은 손익비 경고를 실패 처리했습니다.")
    semco_bad_report = """# fixture
| 항목 | 판단 |
|---|---|
| 현재가 | 2,371,000원 |
| 지금 매수 | 불가 |
| 회복 확인가 | 해당 없음 | 현재가보다 위에 있는 재진입 확인 가격 |
| 단기 재돌파선 | 2,371,000원 |
| 일봉 돌파 확인가 | 2,278,000원 |
| 1차 목표 | 2,588,000원 |
| 최종 판단 | 돌파 유지 확인 |
- 지금 하지 말아야 할 행동: 근접 저항 바로 아래에서 추격매수하지 않습니다.
현재가는 단기 재돌파선에 걸쳐 있으므로, 이 가격 위에서 유지되는지 확인합니다.
"""
    if not validate_intraday_overheated_breakout_qa(semco_bad_report, semco_metrics, {"지금바로매수": "불가", "최종판단": "돌파 유지 확인"}, {"current_price": 2371000}):
        failures.append("fixture 삼성전기형 QA 오류: 긍정/기계적 문구 오류를 잡지 못했습니다.")

    hyundai_row = pd.Series({"MA20": 624000.0, "MA60": 580000.0})
    hyundai_daily_state = daily_trend_state_from_values(619000, hyundai_row, -1200, -800, 44.27, "상승")
    if hyundai_daily_state != "중기 상승 속 단기 조정":
        failures.append("fixture 현대차형 일봉 상태 오류: 20일선 아래·60일선 위는 중기 상승 속 단기 조정이어야 합니다.")
    hyundai_conflict = assess_volume_momentum_conflict(
        current_price=619000,
        pullback_high=612000,
        recovery_line=624000,
        weighted_volume_ratio=1.87,
        rsi=44.27,
        macd=-1200,
        signal=-800,
    )
    if not hyundai_conflict["applies"]:
        failures.append("fixture 현대차형 거래량 강세/모멘텀 미회복 판정이 적용되지 않았습니다.")
    if hyundai_conflict["primary_strategy"] != "거래량 강하지만 모멘텀 미회복, 회복 확인 전 추격 금지":
        failures.append("fixture 현대차형 주 전략 오류")
    hyundai_good_report = """# fixture
| 항목 | 판단 |
|---|---|
| 현재가 | 619,000원 |
| 지금 매수 | 불가 |
| 주 전략 | 거래량 강하지만 모멘텀 미회복, 회복 확인 전 추격 금지 |
| 회복/돌파 공통 확인가 | 624,000원 | 종가 안착 + 거래량 유지 확인 |
| 눌림목 지지가 | 601,000원~612,000원 |
| 단기 재돌파선 | 631,000원 |
| 1차 목표 | 709,000원 |
| 장중 방어선 | 599,000원 |
| 스윙 손절선 | 540,000원 |
| 최종 판단 | 모멘텀 확인 전 추격 금지·스윙/돌파 손익비 부족 |

| 구분 | 상태 | 판단 |
|---|---|---|
| 일봉 | 중기 상승 속 단기 조정 | 20일선 아래에 있어 단기 추세 회복 확인이 필요합니다. |

| 지표 | 상태 | 매매 판단 |
|---|---|---|
| 거래량 | 좋음 | 거래량 동반 반등 시도이나 모멘텀 미회복; 시간가중 20일 평균 대비 1.87배입니다. 거래량만으로 돌파 매수 조건을 긍정 해석하지 않고 RSI/MACD 회복을 함께 확인합니다. |

- 지금 할 행동: 가격 상태: 612,000원 재지지, 624,000원 종가 회복, 631,000원 장중 재돌파 확인; 위험 상태: 599,000원 이탈 시 장중 방어
- 신규매수자: 지금 매수 불가. 624,000원 종가 안착 또는 612,000원 재지지 확인 전까지 보류합니다.
- 보유자: 709,000원 접근 시 일부 익절, 599,000원 이탈 시 장중 방어.
거래량은 강하지만 가격은 회복 확인가 아래이고 RSI/MACD가 아직 약합니다. 신규매수자는 회복 확인가 종가 안착 또는 눌림목 재지지 전까지 추격하지 않습니다.
스윙 손절선 기준 손익비 부족으로 신규매수 매력 낮음. 스윙/돌파 신규매수 매력 낮음. 단기 트레이딩 손익비는 가능하나 스윙/돌파 추격 손익비는 부족.

내부 검증: 통과
가격 신뢰도: 높음
거래량 신뢰도: 높음
교차검증 완전성: 높음
수급 신뢰도: 낮음
"""
    hyundai_metrics = {
        "current_price": 619000,
        "pullback_high": 612000,
        "shallow_pull_high": 612000,
        "recovery_line": 624000,
        "breakout_line": 624000,
        "rebreak_line": 631000,
        "target1": 709000,
        "intraday_rr": 4.50,
        "rr1": 1.14,
        "confirm_rr": 1.01,
        "weighted_volume_ratio": 1.87,
        "rsi": 44.27,
        "macd": -1200,
        "macd_signal": -800,
        "ma20": 624000,
        "ma60": 580000,
    }
    hyundai_decision = {
        "현재가": 619000,
        "지금바로매수": "불가",
        "주전략": "거래량 강하지만 모멘텀 미회복, 회복 확인 전 추격 금지",
        "최종판단": "모멘텀 확인 전 추격 금지·스윙/돌파 손익비 부족",
        "회복확인선": 624000,
        "회복/돌파공통확인선": 624000,
        "일봉돌파확인선": 624000,
        "종가유지확인선": 624000,
        "단기재돌파확인선": 631000,
        "스윙손절손익비": 1.14,
        "확인진입손익비": 1.01,
        "장중방어손익비": 4.50,
    }
    if validate_daily_trend_label_qa(hyundai_good_report, hyundai_metrics, hyundai_decision, {"MA20": 624000, "MA60": 580000}):
        failures.append("fixture 현대차형 일봉 상태 QA 오류: 정상 라벨을 실패 처리했습니다.")
    if validate_common_confirmation_price_qa(hyundai_good_report, hyundai_decision, hyundai_metrics):
        failures.append("fixture 현대차형 공통 확인가 QA 오류: 정상 통합 행을 실패 처리했습니다.")
    if validate_volume_momentum_conflict_qa(hyundai_good_report, hyundai_decision, hyundai_metrics):
        failures.append("fixture 현대차형 거래량/모멘텀 QA 오류: 정상 문구를 실패 처리했습니다.")
    if validate_clean_data_action_wording(hyundai_good_report):
        failures.append("fixture 현대차형 데이터 상태 QA 오류: 정상 가격 행동 문구를 실패 처리했습니다.")
    if validate_low_rr_wording(hyundai_good_report, hyundai_metrics, hyundai_decision) or validate_low_rr_priority_qa(hyundai_good_report, hyundai_metrics, hyundai_decision):
        failures.append("fixture 현대차형 손익비 QA 오류: 정상 낮은 손익비 경고를 실패 처리했습니다.")
    hyundai_bad_report = """# fixture
| 항목 | 판단 |
|---|---|
| 현재가 | 619,000원 |
| 지금 매수 | 불가 |
| 회복 확인가 | 624,000원 |
| 일봉 돌파 확인가 | 624,000원 |
| 최종 판단 | 돌파 전 추격매수 제한 |

| 구분 | 상태 | 판단 |
|---|---|---|
| 일봉 | 상승 | 20일선 아래에 있어 단기 추세 회복 확인이 필요합니다. |

| 지표 | 상태 | 매매 판단 |
|---|---|---|
| 거래량 | 좋음 | 시간가중 20일 평균 대비 1.87배이며 돌파 매수는 1.2배 이상이 필요합니다. |

- 지금 할 행동: 데이터 상태: 장중/완료 일봉 소스 지연 여부 확인; 가격 상태: 612,000원 재지지
1차 목표까지 여유가 남아도 진입을 검토합니다.
내부 검증: 통과
가격 신뢰도: 높음
거래량 신뢰도: 높음
교차검증 완전성: 높음
"""
    if not validate_daily_trend_label_qa(hyundai_bad_report, hyundai_metrics, hyundai_decision, {"MA20": 624000, "MA60": 580000}):
        failures.append("fixture 현대차형 QA 오류: 일봉 상승/20일선 아래 충돌을 잡지 못했습니다.")
    if not validate_common_confirmation_price_qa(hyundai_bad_report, hyundai_decision, hyundai_metrics):
        failures.append("fixture 현대차형 QA 오류: 회복/돌파 가격 중복을 잡지 못했습니다.")
    if not validate_volume_momentum_conflict_qa(hyundai_bad_report, hyundai_decision, hyundai_metrics):
        failures.append("fixture 현대차형 QA 오류: 거래량 강세/모멘텀 미회복 누락을 잡지 못했습니다.")
    if not validate_clean_data_action_wording(hyundai_bad_report):
        failures.append("fixture 현대차형 QA 오류: 정상 검증 상태의 소스 지연 확인 문구를 잡지 못했습니다.")
    if not validate_low_rr_priority_qa(hyundai_bad_report, hyundai_metrics, hyundai_decision):
        failures.append("fixture 현대차형 QA 오류: 낮은 스윙/돌파 손익비 해석 누락을 잡지 못했습니다.")

    hanmi_rebreak_text, _, hanmi_rebreak_merged = display_rebreak_line(368500, 314000, 341500)
    if not hanmi_rebreak_merged or "이전 고점/강한 저항" not in hanmi_rebreak_text:
        failures.append("fixture 한미반도체형 단기 재돌파선/목표가 역전 재분류 오류")
    hanmi_macd_comment = macd_comment(-2359.56, -4702.57, 2343.00, 295000, 368500, 305000)
    if "음수권에서 신호선 위로 회복 시도" not in hanmi_macd_comment or "양수권" in hanmi_macd_comment:
        failures.append("fixture 한미반도체형 MACD 음수권 개선 문구 오류")
    hanmi_volume_context = assess_volume_candle(345000, 345500, 287000, 295000, 2.60)
    if hanmi_volume_context["status"] != "고거래량 음봉/매물 출회 경고" or not hanmi_volume_context["strong_distribution"]:
        failures.append("fixture 한미반도체형 고거래량 음봉/매물 출회 판정 오류")
    if "매물 소화 확인" not in breakout_volume_condition_comment(hanmi_volume_context):
        failures.append("fixture 한미반도체형 장대음봉 돌파매수 제외 문구 오류")
    hanmi_rr_warning = compressed_low_rr_warning(0.45, 0.18)
    if "손익비 부족" not in hanmi_rr_warning or "돌파 매수 전략 성립 불가" not in hanmi_rr_warning:
        failures.append("fixture 한미반도체형 낮은 손익비 압축 문구 오류")
    hanmi_good_report = f"""# fixture
| 항목 | 판단 |
|---|---|
| 현재가 | 295,000원 |
| 지금 매수 | 불가 |
| 주 전략 | 고거래량 음봉 경고, 305,000원 회복 전 신규매수 금지 |
| 회복 확인가 | 305,000원 |
| 눌림목 지지가 | 277,500원~294,500원 |
| 강한 저항/목표권 확인선 | 368,500원 | 보유자 익절/비중관리 |
| 1차 목표 | 314,000원 |
| 2차 목표 | 341,500원 |
| 스윙 손절선 | 252,500원 |
| 최종 판단 | 신규매수 금지, 돌파 추격 전략 성립 불가 |

| 지표 | 상태 | 매매 판단 |
|---|---|---|
| 거래량 | 고거래량 음봉/매물 출회 경고 | 20일 평균 대비 2.60배지만 음봉이고 고가 대비 크게 밀려 강한 매물 출회 경고입니다. 거래량은 돌파 매수 근거가 아니라 매물 소화 확인 대상입니다. |
| MACD | 혼조/개선 중 | MACD는 음수권에서 신호선 위로 회복 시도 중이고 히스토그램도 양수입니다. |

- 지금 할 행동: 294,500원 재지지 확인, 305,000원 종가 회복 확인, 314,000원 접근 시 보유자 일부 익절, 252,500원 이탈 시 방어.
{hanmi_rr_warning}
내부 검증: 통과
"""
    hanmi_metrics = {
        "current_price": 295000,
        "open_price": 345000,
        "high_price": 345500,
        "low_price": 287000,
        "close_price": 295000,
        "volume_ratio20": 2.60,
        "recovery_line": 305000,
        "buy_low": 277500,
        "buy_high": 294500,
        "rebreak_line": 368500,
        "target1": 314000,
        "target2": 341500,
        "rr1": 0.45,
        "confirm_rr": 0.18,
        "macd": -2359.56,
        "macd_signal": -4702.57,
        "macd_hist": 2343.00,
        "rsi": 46.44,
    }
    hanmi_decision = {
        "지금바로매수": "불가",
        "주전략": "고거래량 음봉 경고, 305,000원 회복 전 신규매수 금지",
        "최종판단": "신규매수 금지, 돌파 추격 전략 성립 불가",
        "단기재돌파확인선": 368500,
        "1차목표": 314000,
        "2차목표": 341500,
        "손익비1": 0.45,
        "확인진입손익비": 0.18,
    }
    if validate_rebreak_target_order_qa(hanmi_good_report, hanmi_decision, hanmi_metrics):
        failures.append("fixture 한미반도체형 단기 재돌파선/목표가 순서 QA 오류: 정상 재분류를 실패 처리했습니다.")
    if validate_macd_zone_wording_qa(hanmi_good_report, hanmi_metrics):
        failures.append("fixture 한미반도체형 MACD QA 오류: 정상 음수권 개선 문구를 실패 처리했습니다.")
    if validate_volume_candle_qa(hanmi_good_report, hanmi_metrics):
        failures.append("fixture 한미반도체형 거래량 QA 오류: 정상 고거래량 음봉 경고를 실패 처리했습니다.")
    if validate_low_rr_priority_qa(hanmi_good_report, hanmi_metrics, hanmi_decision) or validate_low_rr_wording(hanmi_good_report, hanmi_metrics, hanmi_decision):
        failures.append("fixture 한미반도체형 손익비 QA 오류: 정상 낮은 손익비 경고를 실패 처리했습니다.")
    if validate_rr_warning_dedup_qa(hanmi_good_report):
        failures.append("fixture 한미반도체형 손익비 경고 중복 QA 오류: 압축 문구를 반복으로 처리했습니다.")
    if validate_clean_data_action_wording(hanmi_good_report):
        failures.append("fixture 한미반도체형 데이터 상태 QA 오류: 정상 가격 행동 문구를 실패 처리했습니다.")

    hanmi_bad_report = """# fixture
| 항목 | 판단 |
|---|---|
| 현재가 | 295,000원 |
| 지금 매수 | 불가 |
| 주 전략 | 눌림목 대기 |
| 단기 재돌파선 | 368,500원 |
| 1차 목표 | 314,000원 |
| 2차 목표 | 341,500원 |
| 최종 판단 | 조건 충족 시 매수 가능 |

| 지표 | 상태 | 매매 판단 |
|---|---|---|
| 거래량 | 좋음 | 20일 평균 대비 2.60배이며 돌파 매수는 1.2배 이상이 필요합니다. |
| MACD | 좋음 | MACD는 양수권에서 모멘텀 유지 중입니다. |

- 지금 할 행동: 대표 가격 소스 일치 여부 확인, 새 지지선이 만들어지는지 확인합니다.
스윙 손절선 기준 손익비 부족. 스윙 손절선 기준 손익비 1.0 미만. 회복/돌파 진입 기준 손익비 1.0 미만. 돌파 추격매수 부적합.
"""
    bad_checks = (
        validate_rebreak_target_order_qa(hanmi_bad_report, hanmi_decision, hanmi_metrics)
        + validate_macd_zone_wording_qa(hanmi_bad_report, hanmi_metrics)
        + validate_volume_candle_qa(hanmi_bad_report, hanmi_metrics)
        + validate_low_rr_priority_qa(hanmi_bad_report, hanmi_metrics, {"최종판단": "조건 충족 시 매수 가능", "손익비1": 0.45, "확인진입손익비": 0.18})
        + validate_rr_warning_dedup_qa(hanmi_bad_report)
    )
    if not bad_checks:
        failures.append("fixture 한미반도체형 QA 오류: 역전 재돌파선/MACD/고거래량 음봉/손익비 오류를 잡지 못했습니다.")

    state_fixtures = [
        (
            "경동제약",
            build_trade_state(
                current_price=5100,
                pullback_low=5200,
                pullback_high=5250,
                recovery_line=5200,
                breakout_line=5400,
                target1=5800,
                target2=6200,
                defense_line=5080,
                short_rebreak_line=5250,
                open_price=5120,
                high_price=5180,
                low_price=5080,
                close_price=5100,
                volume_ratio20=0.7,
                macd=10,
                macd_signal=5,
                macd_hist=1,
                rsi=50.43,
                entry_rr=1.10,
                swing_rr=1.10,
            ),
            {"price_position_state": "BELOW_PULLBACK", "rsi_state": "RSI_NEUTRAL_RECOVERY", "final_action_state": "NO_BUY_BELOW_RECOVERY"},
        ),
        (
            "제룡전기",
            build_trade_state(
                current_price=54600,
                pullback_low=51900,
                pullback_high=54100,
                recovery_line=57100,
                breakout_line=57200,
                target1=60000,
                target2=65000,
                defense_line=49400,
                short_rebreak_line=60000,
                open_price=54000,
                high_price=55200,
                low_price=53500,
                close_price=54600,
                volume_ratio20=0.9,
                macd=100,
                macd_signal=80,
                macd_hist=20,
                rsi=46,
                entry_rr=1.25,
                swing_rr=1.30,
            ),
            {"price_position_state": "ABOVE_PULLBACK_BELOW_RECOVERY", "final_action_state": "WAIT_RECOVERY_CLOSE"},
        ),
        (
            "삼성중공업",
            build_trade_state(
                current_price=27850,
                pullback_low=27600,
                pullback_high=27800,
                recovery_line=27850,
                breakout_line=28950,
                target1=30000,
                target2=31500,
                defense_line=25600,
                short_rebreak_line=28150,
                open_price=27600,
                high_price=28150,
                low_price=27450,
                close_price=27850,
                volume_ratio20=1.0,
                macd=1,
                macd_signal=0,
                macd_hist=1,
                rsi=52,
                entry_rr=1.15,
                swing_rr=1.05,
                validation_note="yfinance 보조 소스 최신거래일 지연으로 대표 가격 산정에서 제외",
                supply_status="수급 데이터 부족",
            ),
            {
                "price_data_state": "PRICE_DATA_STALE_SECONDARY",
                "volume_data_state": "VOLUME_DATA_STALE_SECONDARY",
                "supply_state": "SUPPLY_MISSING",
                "cross_validation_state": "CROSS_VALIDATION_STALE_SECONDARY",
            },
        ),
        (
            "삼성전기",
            build_trade_state(
                current_price=2371000,
                pullback_low=2150000,
                pullback_high=2276000,
                recovery_line=2278000,
                breakout_line=2278000,
                target1=2588000,
                target2=2800000,
                defense_line=2150000,
                short_rebreak_line=2371000,
                open_price=2200000,
                high_price=2390000,
                low_price=2180000,
                close_price=2371000,
                volume_ratio20=1.96,
                macd=5000,
                macd_signal=3000,
                macd_hist=2000,
                rsi=70.69,
                entry_rr=0.60,
                swing_rr=0.36,
                intraday_rr=0.98,
                intraday_mode=True,
                close_confirmed=False,
                completed_daily=False,
            ),
            {"rsi_state": "RSI_OVERHEATED", "risk_reward_state": "RR_STRATEGY_INVALID", "final_action_state": "NO_BUY_STRATEGY_INVALID"},
        ),
        (
            "현대차",
            build_trade_state(
                current_price=619000,
                pullback_low=601000,
                pullback_high=612000,
                recovery_line=624000,
                breakout_line=624000,
                target1=709000,
                target2=766000,
                defense_line=599000,
                short_rebreak_line=631000,
                open_price=601000,
                high_price=631000,
                low_price=599000,
                close_price=619000,
                volume_ratio20=1.87,
                macd=-1200,
                macd_signal=-800,
                macd_hist=-400,
                rsi=44.27,
                entry_rr=1.01,
                swing_rr=1.14,
                intraday_rr=4.50,
            ),
            {"price_position_state": "ABOVE_PULLBACK_BELOW_RECOVERY", "risk_reward_state": "RR_ACCEPTABLE_INTRADAY_ONLY", "final_action_state": "WAIT_RECOVERY_CLOSE"},
        ),
        (
            "한미반도체",
            build_trade_state(
                current_price=295000,
                pullback_low=277500,
                pullback_high=294500,
                recovery_line=305000,
                breakout_line=305000,
                target1=314000,
                target2=341500,
                defense_line=252500,
                short_rebreak_line=305000,
                open_price=345000,
                high_price=345500,
                low_price=287000,
                close_price=295000,
                volume_ratio20=2.60,
                macd=-2359.56,
                macd_signal=-4702.57,
                macd_hist=2343.00,
                rsi=46.44,
                entry_rr=0.18,
                swing_rr=0.45,
            ),
            {"volume_state": "HIGH_VOLUME_BEARISH_REVERSAL", "macd_state": "MACD_NEGATIVE_RECOVERY", "final_action_state": "NO_BUY_STRATEGY_INVALID"},
        ),
    ]
    for fixture_name, trade_state, expected in state_fixtures:
        state_dict = trade_state_to_dict(trade_state)
        rendered = render_trade_state_actions(trade_state.final_action_state)
        table = state_code_report_rows(trade_state)
        if trade_state.final_action_state not in table or not rendered["final_judgment"]:
            failures.append(f"fixture {fixture_name} 상태코드 렌더링 오류")
        if state_dict.get("blocking_errors"):
            failures.append(f"fixture {fixture_name} blocking error가 남아 있습니다: {state_dict.get('blocking_errors')}")
        for key, expected_value in expected.items():
            if state_dict.get(key) != expected_value:
                failures.append(f"fixture {fixture_name} 상태코드 오류: {key}={state_dict.get(key)} expected {expected_value}")

    command_levels = DecisionLevels(
        support=PriceEvidence("핵심 지지선", 49000, ("20일선", "볼린저밴드 중심선")),
        confirmation=PriceEvidence("매수 확인선", 49300, ("3분봉 20이평선", "5분봉 볼린저밴드 중심선")),
        breakout=PriceEvidence("돌파선", 52000, ("최근 20일 고점", "볼린저밴드 상단")),
        stop=PriceEvidence("손절/방어선", 48500, ("최근 20일 저점",)),
        no_chase=PriceEvidence("추격 금지선", 53200, ("볼린저밴드 상단",)),
    )
    intraday_decision = evaluate_decision(DecisionContext(current_price=52500, levels=command_levels, is_intraday=True, risk_reward=1.5))
    intraday_text = " ".join(intraday_decision.actions) + " " + intraday_decision.headline
    if intraday_decision.verdict != "조건부로 사라":
        failures.append(f"명령형 fixture 판정 오류: {intraday_decision.verdict}")
    if "확정" in intraday_text:
        failures.append("명령형 fixture 장중 돌파에 확정 문구가 포함되었습니다")
    if "3분봉 또는 5분봉 종가" not in intraday_text:
        failures.append("명령형 fixture 장중 조건에 분봉 종가 유지 조건이 없습니다")

    missing_evidence_levels = DecisionLevels(
        support=PriceEvidence("핵심 지지선", 49000, ()),
        confirmation=command_levels.confirmation,
        breakout=command_levels.breakout,
        stop=command_levels.stop,
        no_chase=command_levels.no_chase,
    )
    stopped_decision = evaluate_decision(DecisionContext(current_price=50000, levels=missing_evidence_levels, is_intraday=True))
    if stopped_decision.verdict != "분석 중단" or not stopped_decision.blocking_errors:
        failures.append("명령형 fixture 가격 근거 누락 시 분석 중단 처리 실패")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="국내 주식 분석 엔진 회귀 테스트")
    parser.add_argument("--cases", default="test_cases.yml", help="테스트 케이스 YAML/JSON 파일")
    parser.add_argument("--script", default="analyze_stock_intraday.py", help="분석 실행 스크립트")
    parser.add_argument("--timeout", type=int, default=300, help="종목당 timeout 초")
    parser.add_argument("--dry-run", action="store_true", help="실행하지 않고 테스트 대상만 출력")
    parser.add_argument("--fixtures-only", action="store_true", help="네트워크 분석 없이 고정 fixture 정적 검증만 실행")
    args = parser.parse_args()

    cases = load_test_cases(ROOT / args.cases)
    if args.fixtures_only:
        fixture_failures = run_static_fixture_checks()
        if fixture_failures:
            print("[fixture 검증] FAIL")
            for failure in fixture_failures:
                print(f"- {failure}")
            return 1
        print("[fixture 검증] PASS")
        return 0

    if args.dry_run:
        print("[회귀 테스트 대상]")
        for c in cases:
            print(f"- {c.name} {c.code} ({c.market_expected}) : {c.purpose}")
        fixture_failures = run_static_fixture_checks()
        if fixture_failures:
            print("\n[정적 fixture 검증] FAIL")
            for failure in fixture_failures:
                print(f"- {failure}")
            return 1
        print("\n[정적 fixture 검증] PASS")
        return 0

    results: list[TestResult] = []
    for case in cases:
        print(f"[테스트] {case.name} {case.code} ...", flush=True)
        try:
            result = run_one(case, args.script, args.timeout)
        except subprocess.TimeoutExpired:
            result = TestResult(case=case, passed=False, failures=[f"분석 timeout({args.timeout}s)"])
        except Exception as exc:
            result = TestResult(case=case, passed=False, failures=[f"테스트 실행 예외: {type(exc).__name__}: {exc}"])
        results.append(result)
        print("  PASS" if result.passed else "  FAIL")
        if not result.passed:
            for failure in result.failures:
                print(f"   - {failure}")
    summary = write_summary(results)
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    category_totals = {
        "QA 실패": 0,
        "계산 오류": 0,
        "가격 복붙 오류": 0,
        "문장 의미 오류": 0,
        "시장 suffix 오류": 0,
        "데이터 수집 실패": 0,
        "분봉 상태 문구 오류": 0,
        "목표/익절 라벨 오류": 0,
        "방어선 라벨 오류": 0,
        "장초반 거래량 해석 오류": 0,
        "프로 판단 레이어 오류": 0,
        "매매 시나리오 오류": 0,
        "트레이딩 점수 오류": 0,
        "눌림목 아래 문구 오류": 0,
        "거래량 점수 과대평가 오류": 0,
        "현재가 손익비 점수 과대평가 오류": 0,
        "알림 가격 섹션명 오류": 0,
        "A시나리오 목표/손절 오류": 0,
        "깊은 눌림목 상태 설명 오류": 0,
        "손익비 제한 설명 오류": 0,
        "중복 상태 문구 오류": 0,
        "보유자 방어 관찰 문구 오류": 0,
        "섹터 복붙 문구 오류": 0,
        "RSI 해석 오류": 0,
        "MACD 해석 오류": 0,
        "볼린저밴드 해석 오류": 0,
        "눌림목 겹침 구간 설명 오류": 0,
        "지지/저항 중복 출력 오류": 0,
    }
    for r in results:
        for k, v in r.category_summary.items():
            category_totals[k] += v
    print("\n[회귀 테스트 완료]")
    print(f"총 테스트 종목: {total}")
    print(f"정상 통과: {passed}")
    for key in [
        "QA 실패",
        "계산 오류",
        "가격 복붙 오류",
        "문장 의미 오류",
        "시장 suffix 오류",
        "데이터 수집 실패",
        "분봉 상태 문구 오류",
        "목표/익절 라벨 오류",
        "방어선 라벨 오류",
        "장초반 거래량 해석 오류",
        "프로 판단 레이어 오류",
        "매매 시나리오 오류",
        "트레이딩 점수 오류",
        "눌림목 아래 문구 오류",
        "거래량 점수 과대평가 오류",
        "현재가 손익비 점수 과대평가 오류",
        "알림 가격 섹션명 오류",
        "A시나리오 목표/손절 오류",
        "깊은 눌림목 상태 설명 오류",
        "손익비 제한 설명 오류",
        "중복 상태 문구 오류",
        "보유자 방어 관찰 문구 오류",
        "섹터 복붙 문구 오류",
        "RSI 해석 오류",
        "MACD 해석 오류",
        "볼린저밴드 해석 오류",
        "눌림목 겹침 구간 설명 오류",
        "지지/저항 중복 출력 오류",
    ]:
        print(f"{key}: {category_totals[key]}")
    print(f"\n최종 판정: {'국내 주식 분석 엔진 사용 가능' if passed == total else '수정 필요'}")
    print(f"결과 파일: {summary}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
