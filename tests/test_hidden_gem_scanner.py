from __future__ import annotations

import pandas as pd

import hidden_gem_scanner
from core.sse_indicator import SSELevels


def _levels(**overrides):
    values = {
        "base": 99.0,
        "upper": 116.0,
        "lower": 82.0,
        "pressure": 0.35,
        "entry": 100.0,
        "stop": 94.0,
        "target1": 110.0,
        "target2": 118.0,
        "no_chase": 115.0,
        "rr1": 1.67,
        "rr2": 3.0,
    }
    values.update(overrides)
    return SSELevels(**values)


def test_hidden_gem_classifies_o_without_letter_grade():
    candidate = hidden_gem_scanner.RawCandidate("005930", "삼성전자", "반도체", 100.0, 1.2, 100000, 5000, "test")
    row = pd.Series({"SSE_VOLUME_RATIO20": 1.2, "SSE_TRADE_VALUE_RATIO20": 1.25})

    recommendation, reasons, warnings = hidden_gem_scanner.classify_hidden_gem(candidate, _levels(), row, 100.0)

    assert recommendation == "O"
    assert reasons
    assert not warnings


def test_hidden_gem_blocks_no_chase_as_x():
    candidate = hidden_gem_scanner.RawCandidate("005930", "삼성전자", "반도체", 116.0, 2.0, 100000, 5000, "test")
    row = pd.Series({"SSE_VOLUME_RATIO20": 1.1, "SSE_TRADE_VALUE_RATIO20": 1.1})

    recommendation, _reasons, warnings = hidden_gem_scanner.classify_hidden_gem(candidate, _levels(), row, 116.0)

    assert recommendation == "X"
    assert any("추격 금지선" in warning for warning in warnings)


def test_hidden_gem_report_uses_only_o_triangle_x_labels():
    result = hidden_gem_scanner.HiddenGemResult(
        code="005930",
        name="삼성전자",
        sector="반도체",
        recommendation="△",
        current_price=99.0,
        entry=100.0,
        stop=94.0,
        target1=110.0,
        no_chase=115.0,
        rr1=1.67,
        risk_pct=0.06,
        volume_ratio20=1.2,
        trade_value_ratio20=1.25,
        pressure=0.2,
        reasons=("진입 트리거 근접",),
        warnings=(),
    )

    report = hidden_gem_scanner.render_hidden_gem_report([result], [])

    assert "추천" in report
    assert "△" in report
    assert "B+" not in report
    assert "A급" not in report
    assert "B급" not in report
    assert "등급" not in report


def test_hidden_gem_report_save_writes_markdown_and_html(tmp_path, monkeypatch):
    monkeypatch.setattr(hidden_gem_scanner, "REPORTS_DIR", tmp_path)

    path = hidden_gem_scanner.save_hidden_gem_report("# 진흙 속 진주 스캐너\n")

    assert path.exists()
    assert path.with_suffix(".html").exists()
    assert (tmp_path / "hidden_gems" / "latest.txt").exists()
