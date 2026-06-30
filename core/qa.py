from __future__ import annotations

from core.decision_engine import DecisionLevels, DecisionResult, Verdict


ALLOWED_VERDICTS: set[Verdict] = {
    "사라",
    "조건부로 사라",
    "사지 마라",
    "기다려라",
    "팔아라",
    "보유하라",
    "분석 중단",
}


def validate_command_report(
    report: str,
    decision: DecisionResult,
    levels: DecisionLevels,
    *,
    is_intraday: bool,
    data_valid: bool,
    current_price: int,
) -> list[str]:
    """Return blocking QA errors for a command-style analysis report."""

    errors: list[str] = []
    if decision.verdict not in ALLOWED_VERDICTS:
        errors.append(f"허용되지 않은 최종 판정: {decision.verdict}")
    if not data_valid and decision.verdict != "분석 중단":
        errors.append("데이터 invalid 상태에서 정상 판정이 생성되었습니다")
    if decision.blocking_errors:
        errors.extend(decision.blocking_errors)
    if is_intraday and any(phrase in report for phrase in ["일봉 돌파 확정", "확정 돌파"]):
        errors.append("장중 리포트에 확정 돌파 표현이 포함되었습니다")

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

    if levels.no_chase and current_price >= levels.no_chase.price and decision.verdict in {"사라", "조건부로 사라"}:
        errors.append("추격 금지선 이상에서 신규매수 긍정 판정이 생성되었습니다")
    if "## 내부 검증" not in report:
        errors.append("내부 검증 섹션 누락")
    return errors
