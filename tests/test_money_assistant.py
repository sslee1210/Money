import pandas as pd

import money_assistant


def fake_tickers():
    return pd.DataFrame(
        [
            {"code": "005930", "name": "삼성전자", "market": "KOSPI"},
            {"code": "000660", "name": "SK하이닉스", "market": "KOSPI"},
        ]
    )


def test_parse_name_request(monkeypatch):
    monkeypatch.setattr(money_assistant, "load_krx_tickers", fake_tickers)
    monkeypatch.setattr(money_assistant, "is_korean_market_open", lambda: False)

    request = money_assistant.parse_request("삼성전자 분석해줘")

    assert request.code == "005930"
    assert request.name == "삼성전자"
    assert request.mode == "integrated"


def test_parse_code_intraday_request_defaults_to_integrated():
    request = money_assistant.parse_request("005930 삼성전자 장중 분석해줘")

    assert request.code == "005930"
    assert request.name == "삼성전자"
    assert request.mode == "integrated"


def test_parse_kiwoom_request_defaults_to_integrated(monkeypatch):
    monkeypatch.setattr(money_assistant, "load_krx_tickers", fake_tickers)

    request = money_assistant.parse_request("삼성전자 키움 조건부 분석해줘")

    assert request.code == "005930"
    assert request.name == "삼성전자"
    assert request.mode == "integrated"


def test_known_ticker_fallback_resolves_without_external_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(money_assistant, "TICKER_CACHE", tmp_path / "missing.csv")

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in {"pykrx", "FinanceDataReader"}:
            raise RuntimeError("external listing unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    request = money_assistant.parse_request("한화에어로스페이스 분석해줘")

    assert request.code == "012450"
    assert request.name == "한화에어로스페이스"
    assert request.mode == "integrated"


def test_run_request_dispatches_without_order_functions(monkeypatch):
    request = money_assistant.ParsedRequest("005930", "삼성전자", "integrated", "삼성전자 분석해줘")
    monkeypatch.setattr(money_assistant.command_chart_analyzer, "analyze_integrated_chart", lambda code, name: f"integrated {code} {name}")

    assert money_assistant.run_request(request) == "integrated 005930 삼성전자"


def test_hidden_gem_request_routes_to_scanner(monkeypatch):
    request = money_assistant.parse_request("살 주식 찾아줘")
    monkeypatch.setattr(money_assistant.hidden_gem_scanner, "run_hidden_gem_scan", lambda: "hidden gems")

    assert request.mode == "hidden_gem"
    assert money_assistant.run_request(request) == "hidden gems"
