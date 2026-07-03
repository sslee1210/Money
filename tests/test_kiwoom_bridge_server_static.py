from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bridge_exposes_realtime_fid_diagnostics():
    text = (ROOT / "kiwoom_bridge_server" / "kiwoom_bridge.py").read_text(encoding="utf-8")

    assert "ensure_realtime_subscription" in text
    assert "lastRealRawEventAt" in text
    assert "lastRealRawType" in text
    assert "realtimeRegistration" in text


def test_kiwoom_only_bridge_records_ignored_real_types():
    text = (ROOT / "kiwoom_bridge_server" / "kiwoom_bridge_kiwoom_only.py").read_text(encoding="utf-8")

    assert "_record_real_event(code, str(real_type), handled=False)" in text
    assert "_record_real_event(code, str(real_type), handled=True)" in text
