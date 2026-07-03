from __future__ import annotations

import requests

from kiwoom.client import KiwoomBridgeClient


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


def test_quote_falls_back_to_millionaire_stock_endpoint(monkeypatch):
    calls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if url.endswith("/quote"):
            return FakeResponse({"error": "not found"}, status_code=404)
        assert url.endswith("/stock/005930")
        return FakeResponse(
            {
                "ok": True,
                "updatedAt": "2026-06-30T10:00:00+09:00",
                "stock": {
                    "code": "005930",
                    "name": "삼성전자",
                    "price": 78500,
                    "volume": 123456,
                    "tradeAmountMillion": 968000,
                },
            }
        )

    monkeypatch.setattr("kiwoom.client.requests.get", fake_get)

    quote = KiwoomBridgeClient("http://127.0.0.1:8765").get_quote("005930")

    assert quote["price"] == 78500
    assert quote["timestamp"] == "2026-06-30T10:00:00+09:00"
    assert quote["trade_value"] == 968000000000
    assert calls == ["http://127.0.0.1:8765/quote", "http://127.0.0.1:8765/stock/005930"]


def test_client_defaults_to_local_bridge_url(monkeypatch):
    monkeypatch.delenv("KIWOOM_BRIDGE_URL", raising=False)

    client = KiwoomBridgeClient()

    assert client.base_url == "http://127.0.0.1:8765"


def test_quote_falls_back_when_bridge_returns_error_payload(monkeypatch):
    calls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if url.endswith("/quote"):
            return FakeResponse({"error": "unsupported endpoint"}, status_code=200)
        assert url.endswith("/stock/005930")
        return FakeResponse(
            {
                "ok": True,
                "updatedAt": "2026-06-30T10:00:00+09:00",
                "stock": {"code": "005930", "name": "삼성전자", "price": 78500},
            }
        )

    monkeypatch.setattr("kiwoom.client.requests.get", fake_get)

    quote = KiwoomBridgeClient("http://127.0.0.1:8765").get_quote("005930")

    assert quote["price"] == 78500
    assert quote["timestamp"] == "2026-06-30T10:00:00+09:00"
    assert calls == ["http://127.0.0.1:8765/quote", "http://127.0.0.1:8765/stock/005930"]


def test_daily_candles_fall_back_to_millionaire_candles_endpoint(monkeypatch):
    calls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if url.endswith("/candles/daily"):
            return FakeResponse({"error": "not found"}, status_code=404)
        assert url.endswith("/candles/005930")
        assert params == {"days": 400}
        return FakeResponse({"ok": True, "candles": [{"date": "2026-06-29", "open": 78000, "high": 79000, "low": 77000, "close": 78500, "volume": 123}]})

    monkeypatch.setattr("kiwoom.client.requests.get", fake_get)

    candles = KiwoomBridgeClient("http://127.0.0.1:8765").get_daily_candles("005930", limit=400)

    assert candles[0]["close"] == 78500
    assert calls == ["http://127.0.0.1:8765/candles/daily", "http://127.0.0.1:8765/candles/005930"]


def test_daily_candles_fall_back_when_bridge_returns_error_payload(monkeypatch):
    calls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if url.endswith("/candles/daily"):
            return FakeResponse({"error": "unsupported endpoint"}, status_code=200)
        assert url.endswith("/candles/005930")
        return FakeResponse({"ok": True, "candles": [{"date": "2026-06-29", "close": 78500}]})

    monkeypatch.setattr("kiwoom.client.requests.get", fake_get)

    candles = KiwoomBridgeClient("http://127.0.0.1:8765").get_daily_candles("005930")

    assert candles == [{"date": "2026-06-29", "close": 78500}]
    assert calls == ["http://127.0.0.1:8765/candles/daily", "http://127.0.0.1:8765/candles/005930"]


def test_daily_candles_fall_back_to_stock_detail_candles(monkeypatch):
    calls: list[str] = []

    def fake_get(url, params=None, timeout=None):
        calls.append(url)
        if url.endswith("/candles/daily") or url.endswith("/candles/005930"):
            return FakeResponse({"error": "CommRqData failed: opt10081 result=-200"}, status_code=200)
        assert url.endswith("/stock/005930")
        return FakeResponse({"ok": True, "candles": [{"date": "2026-06-29", "close": 78500}]})

    monkeypatch.setattr("kiwoom.client.requests.get", fake_get)
    monkeypatch.setattr("kiwoom.client.time.sleep", lambda seconds: None)

    candles = KiwoomBridgeClient("http://127.0.0.1:8765").get_daily_candles("005930")

    assert candles == [{"date": "2026-06-29", "close": 78500}]
    assert calls.count("http://127.0.0.1:8765/candles/daily") == 3
    assert calls.count("http://127.0.0.1:8765/candles/005930") == 3
    assert calls[-1] == "http://127.0.0.1:8765/stock/005930"


def test_unsupported_endpoint_error_mentions_path(monkeypatch):
    def fake_get(url, params=None, timeout=None):
        return FakeResponse({"detail": "Not Found"}, status_code=404)

    monkeypatch.setattr("kiwoom.client.requests.get", fake_get)

    client = KiwoomBridgeClient("http://127.0.0.1:8765")
    try:
        client.get_minute_candles("005930", interval=5)
    except Exception as exc:
        assert "/candles/minute" in str(exc)
        assert "endpoint 미지원" in str(exc)
    else:
        raise AssertionError("expected unsupported endpoint failure")
