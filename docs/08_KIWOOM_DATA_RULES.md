# 08_KIWOOM_DATA_RULES - 키움 데이터 브릿지 규칙

## 목적

키움증권 API는 조건부 명령형 차트 분석을 위한 데이터 수집에만 사용한다.

주문 전송, 자동매매, 시장가 주문, 조건주문 기능은 구현하지 않는다.

## 환경변수

로컬 키움 브릿지 URL은 `KIWOOM_BRIDGE_URL`로 설정한다.

cmd 예시:

```bat
set KIWOOM_BRIDGE_URL=http://127.0.0.1:8765
python command_chart_analyzer.py 005930 삼성전자
```

PowerShell 예시:

```powershell
$env:KIWOOM_BRIDGE_URL="http://127.0.0.1:8765"
python command_chart_analyzer.py 005930 삼성전자
```

`.env` 파일은 Git에 올리지 않는다.

## 필수 endpoint

브릿지는 아래 GET endpoint를 제공해야 한다.

```text
/quote
/ticks
/candles/minute
/candles/daily
```

공통 query:

```text
code=종목코드
```

## `/quote` expected JSON schema

```json
{
  "code": "005930",
  "name": "삼성전자",
  "price": 78500,
  "prev_close": 78000,
  "open": 78200,
  "high": 79000,
  "low": 77800,
  "volume": 12345678,
  "trade_value": 968000000000,
  "timestamp": "2026-06-30T14:30:00+09:00"
}
```

허용 대체 키:

- `현재가` -> `price`
- `전일종가` 또는 `기준가` -> `prev_close`
- `시가` -> `open`
- `고가` -> `high`
- `저가` -> `low`
- `거래량` -> `volume`
- `거래대금` -> `trade_value`

`price`가 없거나 0 이하이면 분석 중단이다.

`timestamp`가 없거나 파싱할 수 없으면 분석 중단이다.

장중에는 `timestamp`가 현재 시각보다 30분을 초과해 오래되면 분석 중단이다.

장마감 이후에는 timestamp 최신성 검증을 완화하되, 가격 비교 검증은 유지한다.

`prev_close`가 있으면 pykrx/FDR 최신 완료 종가와 비교한다.

- 차이 1% 초과: 경고
- 차이 3% 초과: 분석 중단

`high`와 `low`가 있으면 현재가가 장중 고가/저가 범위 안에 있어야 한다.

## `/ticks` expected JSON schema

```json
{
  "ticks": [
    {
      "timestamp": "2026-06-30T09:00:10+09:00",
      "price": 70000,
      "volume": 120,
      "trade_value": 8400000
    }
  ]
}
```

응답은 배열 자체여도 된다.

```json
[
  {
    "timestamp": "2026-06-30T09:00:10+09:00",
    "price": 70000,
    "volume": 120,
    "trade_value": 8400000
  }
]
```

체결 데이터가 부족하면 `ticks_to_ohlcv()` fallback도 실패하므로 정상 리포트를 저장하지 않는다.

## `/candles/minute` expected JSON schema

Query:

```text
code=005930
interval=3
limit=600
```

Schema:

```json
{
  "candles": [
    {
      "DateTime": "2026-06-30T09:00:00+09:00",
      "Open": 69800,
      "High": 70100,
      "Low": 69700,
      "Close": 70000,
      "Volume": 50000,
      "TradeValue": 3500000000
    }
  ]
}
```

응답은 배열 자체여도 된다.

분봉 endpoint가 비어 있으면 `/ticks`를 가져와 `ticks_to_ohlcv()`로 1분봉/3분봉/5분봉을 생성한다.

## `/candles/daily` expected JSON schema

Query:

```text
code=005930
limit=500
```

Schema:

```json
{
  "candles": [
    {
      "Date": "2026-06-29",
      "Open": 69000,
      "High": 70000,
      "Low": 68800,
      "Close": 69500,
      "Volume": 10000000,
      "TradeValue": 695000000000
    }
  ]
}
```

키움 일봉은 대표 일봉 후보로 사용할 수 있지만 단독 사용은 금지한다.

장중 `/candles/daily`에 오늘 미완성 일봉이 포함될 수 있으므로 `latest_completed_candidate()` 기준의 완료 일봉만 대표 후보로 사용한다.

오늘 미완성 일봉은 pykrx/FDR 전일 확정 일봉과 직접 비교하지 않는다.

pykrx/FDR 최신 완료 일봉과 교차검증한다.

- 가격 차이 1% 초과: 경고
- 가격 차이 3% 초과: 분석 중단

## 브릿지 미연결 시 QA 실패 이유

조건부 명령형 분석은 키움 현재가, 체결, 분봉을 핵심 데이터로 사용한다.

브릿지가 연결되지 않으면 현재가와 분봉 종가 유지 조건을 검증할 수 없다.

따라서 공개 데이터로 조용히 대체하여 `사라`, `조건부로 사라`, `팔아라` 같은 명령형 지시를 내지 않는다.

이 경우 정상 `.md/.html` 리포트는 저장하지 않고 아래 QA 실패 파일만 저장한다.

```text
reports/종목명_종목코드/[종목명, 종목코드] 분석 실패 보고서.md
```
