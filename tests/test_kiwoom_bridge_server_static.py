from __future__ import annotations

from pathlib import Path
import importlib
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
    assert "GetMasterCodeName(QString)" in text
    assert "row[field] = normalize_kiwoom_text(value)" in text


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
    bridge_launcher = (ROOT / "Start_Kiwoom_Bridge.bat").read_text(encoding="utf-8")

    assert "KIWOOM_EXTERNAL_BRIDGE_ONLY=1" in text
    assert "Money bridge" in text
    assert "%~dp0dashboard" in text
    assert "npm run server" in text
    assert "MONEY_RESTART_BRIDGE=1" in text
    assert 'findstr ":8765"' in text
    assert "set ALLOW_NAVER_SECTOR=1" in bridge_launcher
    assert "set NAVER_SECTOR_MAX_LOOKUPS=" in bridge_launcher


def test_embedded_dashboard_uses_external_money_bridge_by_default():
    server = (ROOT / "dashboard" / "server.js").read_text(encoding="utf-8")
    index = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")

    assert "process.env.KIWOOM_EXTERNAL_BRIDGE_ONLY || '1'" in server
    assert "Start Money bridge first" in server
    assert "repairKiwoomText" in server
    assert "new TextDecoder('windows-949')" in server
    assert "Money Dashboard - Kiwoom Sector Board" in index
    assert "화면을 불러오는 중입니다" in index
    assert (ROOT / "dashboard" / "package.json").exists()
    assert not (ROOT / "dashboard" / "bridge").exists()


def test_kiwoom_only_bridge_records_ignored_real_types():
    text = (ROOT / "kiwoom_bridge_server" / "kiwoom_bridge_kiwoom_only.py").read_text(encoding="utf-8")

    assert "base.is_stock_trade_real_type(real_type)" in text
    assert "_record_real_event(code, str(real_type), handled=False)" in text
    assert "_record_real_event(code, str(real_type), handled=True)" in text


def test_realtime_type_accepts_cp949_bytes_as_latin1_text():
    text = (ROOT / "kiwoom_bridge_server" / "kiwoom_bridge.py").read_text(encoding="utf-8")
    start = text.index("def normalize_kiwoom_text")
    end = text.index("\ndef to_number", start)
    namespace = {
        "re": re,
        "Any": object,
        "HANGUL_RE": re.compile(r"[가-힣]"),
        "MOJIBAKE_RE": re.compile(r"[À-ÿ�]"),
    }
    exec(text[start:end], namespace)

    mojibake_real_type = "주식체결".encode("cp949").decode("latin1")
    mojibake_stock_name = "삼성전자".encode("cp949").decode("latin1")

    assert namespace["normalize_kiwoom_text"](mojibake_real_type) == "주식체결"
    assert namespace["normalize_kiwoom_text"](mojibake_stock_name) == "삼성전자"
    assert namespace["is_stock_trade_real_type"](mojibake_real_type)


def test_naver_upjong_sector_fallback(monkeypatch):
    sector_map = importlib.import_module("kiwoom_sector_map")

    class FakeResponse:
        def read(self):
            return (
                '<h4><span>동종업종비교</span>'
                '<em>(업종명 : <a href="/sise/sise_group_detail.naver?type=upjong&no=278">'
                '반도체와반도체장비</a>)</em></h4>'
            ).encode("utf-8")

    monkeypatch.setattr(sector_map, "NAVER_SECTOR_ENABLED", True)
    monkeypatch.setattr(sector_map, "NAVER_SECTOR_MAX_LOOKUPS", 10)
    monkeypatch.setattr(sector_map, "NAVER_SECTOR_LOOKUPS", 0)
    sector_map.NAVER_SECTOR_CACHE.clear()
    monkeypatch.setattr(sector_map, "urlopen", lambda request, timeout: FakeResponse())

    result = sector_map.pick_sector("", "삼성전자", [], "005930")

    assert result["sector"] == "반도체와반도체장비"
    assert result["sectorSource"] == "naver-upjong"
