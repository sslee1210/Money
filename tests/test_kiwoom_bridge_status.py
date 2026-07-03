from __future__ import annotations

import kiwoom_bridge_status


def test_analysis_compatible_accepts_money_bridge_paths(monkeypatch):
    monkeypatch.setattr(
        kiwoom_bridge_status,
        "_openapi_paths",
        lambda base_url: {"/health", "/stock/{code}", "/candles/minute", "/candles/{code}"},
    )

    compatible, missing = kiwoom_bridge_status._analysis_compatible("http://127.0.0.1:8765")

    assert compatible
    assert missing == []


def test_analysis_compatible_rejects_old_bridge_without_minutes(monkeypatch):
    monkeypatch.setattr(
        kiwoom_bridge_status,
        "_openapi_paths",
        lambda base_url: {"/health", "/stock/{code}", "/candles/{code}"},
    )

    compatible, missing = kiwoom_bridge_status._analysis_compatible("http://127.0.0.1:8765")

    assert not compatible
    assert missing == ["minute"]
