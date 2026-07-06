from __future__ import annotations

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
BRIDGE_ROOT = ROOT / "kiwoom_bridge_server"
if str(BRIDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(BRIDGE_ROOT))


def test_bridge_exposes_realtime_fid_diagnostics():
    text = (ROOT / "kiwoom_bridge_server" / "kiwoom_bridge.py").read_text(encoding="utf-8")

    assert "ensure_realtime_subscription" in text
    assert "lastRealRawEventAt" in text
    assert "lastRealRawType" in text
    assert "lastRealType" in text
    assert "realtimeRegistration" in text
    assert "normalize_kiwoom_text" in text


def test_bridge_quote_exposes_realtime_time_and_candle_sanity_flag():
    text = (ROOT / "kiwoom_bridge_server" / "kiwoom_bridge.py").read_text(encoding="utf-8")

    assert "realtime_quote = controller.quotes.get(normalized_code)" in text
    assert "if realtime_quote:" in text
    assert "'time': stock.get('time')" in text
    assert "'time': realtime_quote.get('time')" in text
    assert "candleComparable" in text
    assert "candle_comparable" in text


def test_bridge_exposes_millionaire_rising_amount_rank_endpoint():
    text = (ROOT / "kiwoom_bridge_server" / "kiwoom_bridge.py").read_text(encoding="utf-8")

    assert "def rising_amount_rank" in text
    assert "@api.get('/rising-amount-rank')" in text
    assert "is_rising_rank_row" in text
    assert "Kiwoom OpenAPI+ opt10032 rising-filter" in text


def test_integrated_launcher_uses_money_bridge_for_millionaire():
    text = (ROOT / "Start_Money_All.bat").read_text(encoding="utf-8")

    assert "KIWOOM_EXTERNAL_BRIDGE_ONLY=1" in text
    assert "Money bridge" in text
    assert "npm run server" in text


def test_kiwoom_only_bridge_records_ignored_real_types():
    text = (ROOT / "kiwoom_bridge_server" / "kiwoom_bridge_kiwoom_only.py").read_text(encoding="utf-8")

    assert "base.is_stock_trade_real_type(real_type)" in text
    assert "_record_real_event(code, str(real_type), handled=False)" in text
    assert "_record_real_event(code, str(real_type), handled=True)" in text


def test_realtime_type_accepts_cp949_bytes_as_latin1_text():
    text = (ROOT / "kiwoom_bridge_server" / "kiwoom_bridge.py").read_text(encoding="utf-8")
    start = text.index("def normalize_kiwoom_text")
    end = text.index("\ndef to_number", start)
    namespace = {"re": re, "Any": object}
    exec(text[start:end], namespace)

    mojibake_real_type = "주식체결".encode("cp949").decode("latin1")

    assert namespace["normalize_kiwoom_text"](mojibake_real_type) == "주식체결"
    assert namespace["is_stock_trade_real_type"](mojibake_real_type)
