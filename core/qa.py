from __future__ import annotations

from core.decision_engine import DecisionLevels, DecisionResult, Verdict
from core.sse_indicator import SSEResult, validate_sse_levels


ALLOWED_VERDICTS: set[Verdict] = {
    "사라",
    "조건부로 사라",
    "사지 마라",
    "기다려라",
    "팔아라",
    "보유하라",
    "분석 중단",
}

INTRADAY_BANNED_CONFIRMED_PHRASES = ("확정 돌파", "일봉 돌파 확정", "돌파 확인 완료")
BUY_POSITIVE_IMPERATIVE_PHRASES = (
    "1차 매수하라",
    "1차 매수를 검토하라",
    "1차 분할 매수",
    "분할 매수 가능",
    "매수 가능",
    "1차 진입하라",
)
CONSERVATIVE_VERDICTS = {"사지 마라", "기다려라", "보유하라", "팔아라"}


def validate_command_report(
    report: str,
    decision: DecisionResult,
    levels: DecisionLevels,
    *,
    is_intraday: bool,
    data_valid: bool,
    current_price: int,
    sse_result: SSEResult | None = None,
    realtime_limited: bool = False,
) -> list[str]:
    """Return blocking QA errors for a command-style analysis report."""

    errors: list[str] = []
    if decision.verdict not in ALLOWED_VERDICTS:
        errors.append(f"허용되지 않은 최종 판정: {decision.verdict}")
    if not data_valid and decision.verdict != "분석 중단":
        errors.append("데이터 invalid 상태에서 정상 판정이 생성되었습니다")
    if decision.blocking_errors:
        errors.extend(decision.blocking_errors)
    if is_intraday and any(phrase in report for phrase in INTRADAY_BANNED_CONFIRMED_PHRASES):
        errors.append("장중 리포트에 확정 돌파 표현이 포함되었습니다")
    if decision.verdict in CONSERVATIVE_VERDICTS:
        positive_hits = [phrase for phrase in BUY_POSITIVE_IMPERATIVE_PHRASES if phrase in report]
        if positive_hits:
            errors.append(f"보수적 최종 판정({decision.verdict})과 충돌하는 매수 긍정 명령이 남아 있습니다: {', '.join(positive_hits)}")

    required_evidence = {
        "핵심 지지선": levels.support,
        "매수 확인선": levels.confirmation,
        "손절/방어선": levels.stop,
        "추격 금지선": levels.no_chase,
    }
    for label, evidence in required_evidence.items():
        if evidence is None or evidence.price <= 0 or not evidence.reasons:
            errors.append(f"{label} PriceEvidence 누락")
    if levels.breakout is not None and (levels.breakout.price <= 0 or not levels.breakout.reasons):
        errors.append("돌파선 PriceEvidence 누락")

    buy_text = "\n".join(decision.buy_conditions + decision.no_buy_conditions)
    for required_text in ["지지", "회복", "사지 마라"]:
        if required_text not in buy_text:
            errors.append(f"매수 조건에 {required_text} 조건이 없습니다")
    if levels.support is None:
        errors.append("매수 조건에 핵심 지지선이 없습니다")
    if levels.confirmation is None:
        errors.append("매수 조건에 매수 확인선이 없습니다")
    if levels.no_chase is None:
        errors.append("매수 조건에 매수 금지선이 없습니다")

    if levels.no_chase and current_price >= levels.no_chase.price and decision.verdict in {"사라", "조건부로 사라"}:
        errors.append("추격 금지선 이상에서 신규매수 긍정 판정이 생성되었습니다")
    if (
        is_intraday
        and decision.final_action_state == "WATCH_INTRADAY_BREAKOUT"
        and not all(phrase in report for phrase in ["장중 돌파 시도", "3분봉 또는 5분봉 종가", "오늘 종가 확인 필요"])
    ):
        errors.append("장중 돌파 리포트에 필수 장중 조건 문구가 없습니다")
    if "## 내부 검증" not in report:
        errors.append("내부 검증 섹션 누락")
    if "내부 검증: 통과" in report and (decision.blocking_errors or not data_valid):
        errors.append("QA 미통과 상태에서 내부 검증 통과가 출력되었습니다")
    if realtime_limited:
        if decision.verdict in {"사라", "조건부로 사라"}:
            errors.append("실시간 보정 실패 상태에서 신규매수 긍정 판정이 생성되었습니다")
        required_limit_text = "키움 실시간 데이터 미확인으로 장중 매수 지시는 제한합니다."
        if required_limit_text not in report:
            errors.append("실시간 보정 실패 제한 문구가 누락되었습니다")
    if sse_result is not None:
        errors.extend(validate_sse_report(report, decision, sse_result, is_intraday=is_intraday, current_price=current_price))
    return errors


def validate_sse_report(
    report: str,
    decision: DecisionResult,
    sse_result: SSEResult,
    *,
    is_intraday: bool,
    current_price: int,
) -> list[str]:
    errors: list[str] = []
    levels = sse_result.levels
    errors.extend(validate_sse_levels(levels))
    errors.extend(sse_result.blocking_errors)
    buy_positive = decision.verdict in {"사라", "조건부로 사라"}
    if current_price >= levels.no_chase and buy_positive:
        errors.append("SSE 추격 금지선 이상에서 신규매수 긍정 판정이 생성되었습니다")
    if levels.rr1 < 1.2 and buy_positive:
        errors.append("SSE RR1 < 1.2인데 신규매수 긍정 판정이 생성되었습니다")
    if levels.pressure >= 1.5 and buy_positive:
        errors.append("SSE 압력값 과열권인데 신규매수 긍정 판정이 생성되었습니다")
    if levels.pressure < -1.0 and buy_positive:
        errors.append("SSE 압력값 약세 이탈인데 신규매수 긍정 판정이 생성되었습니다")
    if is_intraday and any(phrase in report for phrase in INTRADAY_BANNED_CONFIRMED_PHRASES):
        errors.append("SSE 장중 리포트에 확정 돌파 표현이 포함되었습니다")
    if "## SSE Indicator 분석" not in report:
        errors.append("SSE Indicator 분석 섹션 누락")
    if "산출 근거:" not in report:
        errors.append("SSE 산출 근거 누락")

    report_required_labels = [
        "SSE 기준선",
        "SSE 상단선",
        "SSE 하단선",
        "SSE 압력값",
        "예상 진입가",
        "예상 손절가",
        "1차 익절가",
        "2차 익절가",
        "추격 금지선",
        "1차 목표 기준 손익비",
        "2차 목표 기준 손익비",
        "SSE 최종 판정",
    ]
    for label in report_required_labels:
        if label not in report:
            errors.append(f"SSE {label} 출력 누락")

    evidence_labels = {item.label for item in sse_result.evidence}
    required_evidence_labels = {"예상 진입가", "예상 손절가", "1차 익절가", "2차 익절가", "추격 금지선"}
    missing_evidence = sorted(required_evidence_labels - evidence_labels)
    if missing_evidence:
        errors.append(f"SSE 가격 산출 근거 누락: {', '.join(missing_evidence)}")
    if sse_result.evidence and not all(item.formula and item.reason for item in sse_result.evidence):
        errors.append("SSE 산출 근거의 공식 또는 사유가 비어 있습니다")
    return errors
