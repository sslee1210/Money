from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Verdict = Literal["사라", "조건부로 사라", "사지 마라", "기다려라", "팔아라", "보유하라", "분석 중단"]


@dataclass(frozen=True)
class PriceEvidence:
    label: str
    price: int
    reasons: tuple[str, ...]

    def summary(self) -> str:
        return f"{self.label} {format_price(self.price)}: {', '.join(self.reasons)} 근거"


@dataclass(frozen=True)
class DecisionLevels:
    support: PriceEvidence | None
    confirmation: PriceEvidence | None
    breakout: PriceEvidence | None
    stop: PriceEvidence | None
    no_chase: PriceEvidence | None
    target1: PriceEvidence | None = None
    target2: PriceEvidence | None = None


@dataclass(frozen=True)
class DecisionContext:
    current_price: int
    levels: DecisionLevels
    is_intraday: bool
    data_valid: bool = True
    invalid_reasons: tuple[str, ...] = ()
    volume_ratio20: float | None = None
    rsi14: float | None = None
    risk_reward: float | None = None


@dataclass(frozen=True)
class DecisionResult:
    verdict: Verdict
    headline: str
    actions: tuple[str, ...]
    buy_conditions: tuple[str, ...]
    no_buy_conditions: tuple[str, ...]
    sell_conditions: tuple[str, ...]
    holder_conditions: tuple[str, ...]
    price_evidence: tuple[PriceEvidence, ...]
    blocking_errors: tuple[str, ...] = ()
    final_action_state: str = ""

    @property
    def stopped(self) -> bool:
        return self.verdict == "분석 중단" or bool(self.blocking_errors)


def evaluate_decision(ctx: DecisionContext) -> DecisionResult:
    missing = _missing_required_evidence(ctx.levels)
    if not ctx.data_valid or ctx.invalid_reasons or missing:
        errors = tuple(ctx.invalid_reasons) + tuple(missing)
        return DecisionResult(
            verdict="분석 중단",
            headline="데이터 검증 실패: 명령형 매매 지시를 중단합니다",
            actions=("데이터가 다시 검증될 때까지 매수·매도 조건을 확정하지 마라.",),
            buy_conditions=("매수 조건 없음.",),
            no_buy_conditions=("정상 리포트 저장 금지.",),
            sell_conditions=("기존 보유 판단도 별도 확인 전까지 자동화하지 마라.",),
            holder_conditions=("데이터 회복 전까지 이 리포트로 추가매수하지 마라.",),
            price_evidence=_evidence_tuple(ctx.levels),
            blocking_errors=errors,
            final_action_state="NO_BUY_DATA_INVALID",
        )

    levels = ctx.levels
    assert levels.support and levels.confirmation and levels.stop and levels.no_chase
    current = ctx.current_price
    overheat = (ctx.rsi14 is not None and ctx.rsi14 >= 70) or current >= levels.no_chase.price
    weak_rr = ctx.risk_reward is not None and ctx.risk_reward < 1.0

    if current < levels.stop.price:
        verdict: Verdict = "팔아라"
        headline = "방어선 이탈: 매수 관점 폐기"
        actions = (f"{format_price(levels.stop.price)} 이탈 상태이므로 팔아라 또는 비중 축소하라.",)
        final_state = "DEFENSE_REQUIRED"
    elif current < levels.support.price:
        verdict = "사지 마라"
        headline = "핵심 지지선 아래: 회복 전 신규매수 금지"
        actions = (f"지금 사지 마라. {format_price(levels.support.price)} 회복 전까지 신규매수하지 마라.",)
        final_state = "NO_BUY_BELOW_RECOVERY"
    elif overheat and weak_rr:
        verdict = "사지 마라"
        headline = "과열·손익비 부족: 추격매수 금지"
        actions = (f"지금 바로 시장가로 사지 마라. {format_price(levels.no_chase.price)} 이상에서는 추격 매수하지 마라.",)
        final_state = "NO_BUY_OVERHEATED_BAD_RR"
    elif levels.breakout and current >= levels.breakout.price:
        if ctx.is_intraday:
            verdict = "조건부로 사라"
            headline = "장중 돌파 시도: 분봉 종가 유지가 필요"
            actions = (
                "지금 바로 시장가로 사지 마라.",
                f"{format_price(levels.breakout.price)} 이상을 거래량 동반해 3분봉 또는 5분봉 종가로 유지할 때만 1차 매수하라.",
            )
            final_state = "WATCH_INTRADAY_BREAKOUT"
        else:
            verdict = "사라"
            headline = "일봉 종가 돌파 확인: 분할 매수 가능"
            actions = (f"{format_price(levels.breakout.price)} 위 일봉 종가 돌파 확인 상태이므로 1차 분할 매수 가능하다.",)
            final_state = "BREAKOUT_CONFIRMED"
    elif current >= levels.confirmation.price and not weak_rr:
        verdict = "조건부로 사라"
        headline = "지지 후 회복 확인: 조건부 1차 매수"
        actions = (
            "지금 바로 시장가로 사지 마라.",
            f"{format_price(levels.support.price)}을 깨지 않고 {format_price(levels.confirmation.price)} 이상에서 3분봉 또는 5분봉 종가가 마감될 때만 1차 매수하라.",
        )
        final_state = "WAIT_RECOVERY_CLOSE" if ctx.is_intraday else "HOLD_AND_TRAIL"
    elif levels.support.price <= current < levels.confirmation.price:
        verdict = "기다려라"
        headline = "지지 구간 안: 회복 확인 전 대기"
        actions = (f"아직 사지 마라. {format_price(levels.confirmation.price)} 이상 회복 종가가 필요하다.",)
        final_state = "WAIT_RECOVERY_CLOSE"
    else:
        verdict = "보유하라"
        headline = "매수·매도 신호 사이: 보유자는 기준선 관리"
        actions = (f"보유자는 {format_price(levels.support.price)} 이탈 전까지 보유하라.",)
        final_state = "HOLD_AND_TRAIL"

    buy_conditions = (
        f"지지 매수: {format_price(levels.support.price)} 지지 후 {format_price(levels.confirmation.price)} 회복 시",
        _breakout_condition(levels, ctx.is_intraday),
    )
    no_buy_conditions = (
        f"{format_price(levels.support.price)} 아래에서 5분봉 종가가 마감되면 사지 마라.",
        f"{format_price(levels.no_chase.price)} 이상에서는 추격 매수하지 마라.",
    )
    sell_conditions = (f"{format_price(levels.stop.price)} 이탈 시 팔아라 또는 비중 축소하라.",)
    holder_conditions = (f"보유자는 {format_price(levels.support.price)} 이탈 시 추가매수 보류, {format_price(levels.stop.price)} 이탈 시 방어/손절하라.",)
    return DecisionResult(
        verdict=verdict,
        headline=headline,
        actions=actions,
        buy_conditions=buy_conditions,
        no_buy_conditions=no_buy_conditions,
        sell_conditions=sell_conditions,
        holder_conditions=holder_conditions,
        price_evidence=_evidence_tuple(levels),
        final_action_state=final_state,
    )


def format_price(price: int | float | None) -> str:
    if price is None:
        return "해당 없음"
    return f"{int(round(price)):,}원"


def _breakout_condition(levels: DecisionLevels, is_intraday: bool) -> str:
    if not levels.breakout:
        return "돌파 매수: 근거 있는 돌파선 없음"
    suffix = "거래량 동반해 5분봉 종가로 유지 시" if is_intraday else "일봉 종가 돌파 확인 시"
    return f"돌파 매수: {format_price(levels.breakout.price)} 이상을 {suffix}"


def _missing_required_evidence(levels: DecisionLevels) -> list[str]:
    required = {
        "핵심 지지선": levels.support,
        "매수 확인선": levels.confirmation,
        "손절/방어선": levels.stop,
        "추격 금지선": levels.no_chase,
    }
    errors: list[str] = []
    for label, evidence in required.items():
        if evidence is None or evidence.price <= 0 or not evidence.reasons:
            errors.append(f"{label} 가격 근거 부족")
    if levels.breakout is not None and not levels.breakout.reasons:
        errors.append("돌파선 가격 근거 부족")
    return errors


def _evidence_tuple(levels: DecisionLevels) -> tuple[PriceEvidence, ...]:
    return tuple(e for e in [levels.support, levels.confirmation, levels.breakout, levels.no_chase, levels.stop, levels.target1, levels.target2] if e is not None)
