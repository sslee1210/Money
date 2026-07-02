from core.analysis_pipeline import LayerStatus, run_integrated_analysis_pipeline


def _status(name: str, ok: bool, *errors: str) -> LayerStatus:
    return LayerStatus(name=name, ok=ok, blocking_errors=tuple(errors))


def test_public_ok_kiwoom_ok_keeps_integrated_verdict():
    result = run_integrated_analysis_pipeline(
        _status("공개 데이터 분석", True),
        _status("키움 실시간 보정", True),
        _status("SSE Indicator", True),
        "기다려라",
        sse_verdict="보유하라",
    )

    assert result.final_verdict == "기다려라"
    assert not result.realtime_limited


def test_public_ok_kiwoom_fail_limits_positive_buy():
    result = run_integrated_analysis_pipeline(
        _status("공개 데이터 분석", True),
        _status("키움 실시간 보정", False, "키움 현재가 수집 실패"),
        _status("SSE Indicator", True),
        "조건부로 사라",
        sse_verdict="조건부로 사라",
    )

    assert result.final_verdict == "기다려라"
    assert result.realtime_limited
    assert "장중 매수 지시는 제한" in result.final_reason


def test_public_fail_kiwoom_ok_blocks_kiwoom_only_buy():
    result = run_integrated_analysis_pipeline(
        _status("공개 데이터 분석", False, "대표가격 산정 불가"),
        _status("키움 실시간 보정", True),
        _status("SSE Indicator", True),
        "조건부로 사라",
        sse_verdict="조건부로 사라",
    )

    assert result.final_verdict == "분석 중단"
    assert result.realtime_limited
    assert "키움 단독 매수 판정" in result.final_reason


def test_public_fail_kiwoom_fail_stops_analysis():
    result = run_integrated_analysis_pipeline(
        _status("공개 데이터 분석", False, "대표가격 산정 불가"),
        _status("키움 실시간 보정", False, "키움 현재가 수집 실패"),
        _status("SSE Indicator", False, "SSE 계산 실패"),
        "조건부로 사라",
        sse_verdict="조건부로 사라",
    )

    assert result.final_verdict == "분석 중단"


def test_more_conservative_sse_verdict_wins():
    result = run_integrated_analysis_pipeline(
        _status("공개 데이터 분석", True),
        _status("키움 실시간 보정", True),
        _status("SSE Indicator", True),
        "조건부로 사라",
        sse_verdict="사지 마라",
    )

    assert result.final_verdict == "사지 마라"
    assert "SSE Indicator" in result.final_reason
