from __future__ import annotations

from dataclasses import dataclass


SAFETY_PRIORITY = {
    "조건부로 사라": 1,
    "보유하라": 2,
    "기다려라": 3,
    "사지 마라": 4,
    "팔아라": 5,
    "분석 중단": 6,
}


@dataclass(frozen=True)
class LayerStatus:
    name: str
    ok: bool
    warnings: tuple[str, ...] = ()
    blocking_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntegratedAnalysisResult:
    public_status: LayerStatus
    kiwoom_status: LayerStatus
    sse_status: LayerStatus
    final_verdict: str
    final_reason: str
    realtime_limited: bool


def run_integrated_analysis_pipeline(
    public_status: LayerStatus,
    kiwoom_status: LayerStatus,
    sse_status: LayerStatus,
    decision_verdict: str,
    *,
    sse_verdict: str | None = None,
    sse_required: bool = False,
) -> IntegratedAnalysisResult:
    """Combine analysis layers without forcing every optional layer to succeed."""

    if not public_status.ok:
        reason = "공개 데이터 대표 일봉 기준이 불확실하여 키움 단독 매수 판정을 금지합니다."
        return IntegratedAnalysisResult(
            public_status,
            kiwoom_status,
            sse_status,
            "분석 중단",
            reason,
            realtime_limited=True,
        )

    realtime_limited = not kiwoom_status.ok
    if sse_required and not sse_status.ok:
        return IntegratedAnalysisResult(
            public_status,
            kiwoom_status,
            sse_status,
            "분석 중단",
            "필수 SSE Indicator 계산이 실패하여 분석을 중단합니다.",
            realtime_limited=realtime_limited,
        )

    final_verdict = decision_verdict
    reason_parts: list[str] = ["공개 데이터 분석을 기본 판단으로 사용했습니다."]
    if kiwoom_status.ok:
        reason_parts.append(f"{kiwoom_status.name} 검증을 통과해 가격과 분봉 데이터를 반영했습니다.")
    else:
        reason_parts.append("키움 실시간 데이터 미확인으로 장중 매수 지시는 제한합니다.")

    if sse_status.ok and sse_verdict:
        final_verdict = safer_verdict(decision_verdict, sse_verdict)
        if final_verdict == sse_verdict and final_verdict != decision_verdict:
            reason_parts.append(f"SSE Indicator가 더 보수적인 {sse_verdict} 판정을 제시해 최종 판단에 우선 반영했습니다.")
        else:
            reason_parts.append("SSE Indicator를 안전 필터로 확인했습니다.")
    elif not sse_status.ok:
        reason_parts.append("SSE Indicator 계산 실패로 SSE 보정은 제외했습니다.")

    if realtime_limited and final_verdict in {"사라", "조건부로 사라"}:
        final_verdict = "기다려라"
        reason_parts.append("실시간 보정 실패 상태에서 신규매수 긍정 판정을 제한했습니다.")

    return IntegratedAnalysisResult(
        public_status,
        kiwoom_status,
        sse_status,
        final_verdict,
        " ".join(reason_parts),
        realtime_limited=realtime_limited,
    )


def safer_verdict(left: str, right: str) -> str:
    left_priority = SAFETY_PRIORITY.get(left, 0)
    right_priority = SAFETY_PRIORITY.get(right, 0)
    return right if right_priority > left_priority else left
