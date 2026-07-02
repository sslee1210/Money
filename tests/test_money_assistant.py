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


def test_parse_code_intraday_request():
    request = money_assistant.parse_request("005930 삼성전자 장중 분석해줘")

    assert request.code == "005930"
    assert request.name == "삼성전자"
    assert request.mode == "intraday"


def test_parse_kiwoom_request(monkeypatch):
    monkeypatch.setattr(money_assistant, "load_krx_tickers", fake_tickers)

    request = money_assistant.parse_request("삼성전자 키움 조건부 분석해줘")

    assert request.code == "005930"
    assert request.name == "삼성전자"
    assert request.mode == "kiwoom"


def test_run_request_dispatches_without_order_functions(monkeypatch):
    request = money_assistant.ParsedRequest("005930", "삼성전자", "integrated", "삼성전자 분석해줘")
    monkeypatch.setattr(money_assistant.command_chart_analyzer, "analyze_integrated_chart", lambda code, name: f"integrated {code} {name}")

    assert money_assistant.run_request(request) == "integrated 005930 삼성전자"
