# 09_SSE_INDICATOR_RULES - SSE Indicator

## SSE Indicator 목적

`SSE Indicator`는 `Sungsoo Synthetic Equilibrium Indicator`의 약자다.

기존 이동평균선, 볼린저밴드, 일목균형표의 완성 결과값을 단순 가중합하지 않는다. 종가 평균, 표준편차, 기간별 고가/저가 중간값, 평균 이격, 고저 균형 이격, 원천 구름 두께, 기준선 대비 가격 압력을 하나의 합성 균형 구조로 재조합한다.

SSE는 분석과 매매타점 산출용 보조 지표다. 주문, 자동매매, 시장가 주문, 조건주문 기능으로 확장하지 않는다.

## 입력 데이터

표준 OHLCV 데이터프레임을 사용한다.

필수 컬럼:

```text
DateTime 또는 Date
Open
High
Low
Close
Volume
TradeValue
```

필수 컬럼이 없거나 SSE 기준선, 변동성, 진입가, 손절가, 목표가 계산이 불가능하면 `분석 중단`으로 처리한다.

## 구성 원리

SSE는 아래 네 가지 원천 원리를 사용한다.

1. 이동평균선 원리: 종가의 기간 평균
2. 볼린저밴드 원리: 평균과 표준편차를 이용한 변동성 범위
3. 일목균형표 원리: 기간별 고가/저가 중간값과 시간 구조
4. 추가 합성 구조: 단기/중기 평균 이격, 고저 균형 이격, 원천 구름 두께, 가격의 기준선 대비 압력

## 원천 계산값

이동평균 원리:

```python
MA20 = Close.rolling(20).mean()
MA60 = Close.rolling(60).mean()
MA120 = Close.rolling(120).mean()
MA240 = Close.rolling(240).mean()
```

표준편차 원리:

```python
MEAN20 = Close.rolling(20).mean()
STD20 = Close.rolling(20).std()
```

고저 중간값 원리:

```python
MID9 = (High.rolling(9).max() + Low.rolling(9).min()) / 2
MID26 = (High.rolling(26).max() + Low.rolling(26).min()) / 2
MID52 = (High.rolling(52).max() + Low.rolling(52).min()) / 2
```

추가 구조:

```python
MA_GAP = abs(MA20 - MA60)
BALANCE_GAP = abs(MID26 - MID52)
CLOUD_SPAN1_RAW = (MID9 + MID26) / 2
CLOUD_SPAN2_RAW = MID52
CLOUD_THICKNESS = abs(CLOUD_SPAN1_RAW - CLOUD_SPAN2_RAW)
```

`CLOUD_THICKNESS`는 기존 선행스팬 결과값을 그대로 가져오지 않고, MID9/MID26/MID52 원천값으로 재구성한 일목식 시간 구조 진단값이다.

## SSE 기준선 공식

```python
SSE_BASE =
    0.35 * MA20
  + 0.20 * MA60
  + 0.20 * MID26
  + 0.15 * MID52
  + 0.10 * MID9
```

계수 의미:

- `0.35 * MA20`: 단기 종가 평균 비중
- `0.20 * MA60`: 중기 종가 평균 비중
- `0.20 * MID26`: 중기 고저 균형값 비중
- `0.15 * MID52`: 장기 고저 균형값 비중
- `0.10 * MID9`: 단기 고저 균형값 비중

## SSE 통합 변동성 공식

```python
SSE_VOLATILITY =
    0.50 * STD20
  + 0.25 * abs(MID26 - MID52)
  + 0.25 * abs(MA20 - MA60)
```

의미:

- `STD20`: 최근 종가 변동성
- `abs(MID26 - MID52)`: 고저 균형 구조의 중장기 이격
- `abs(MA20 - MA60)`: 단기/중기 평균 추세 이격

`SSE_VOLATILITY`가 0 이하이거나 NaN이면 분석을 중단한다.

## SSE 상단선과 하단선

기본 밴드 계수:

```python
SSE_BAND_MULTIPLIER = 1.8
```

```python
SSE_UPPER = SSE_BASE + 1.8 * SSE_VOLATILITY
SSE_LOWER = SSE_BASE - 1.8 * SSE_VOLATILITY
```

향후 백테스트에서는 `1.4`, `1.6`, `1.8`, `2.0`을 비교한다.

## SSE 압력값 해석

```python
SSE_PRESSURE = (Close - SSE_BASE) / SSE_VOLATILITY
```

장중 키움 현재가 또는 별도 `current_price`가 유효하면 판정용 압력값은 현재가 기준으로 재계산한다.

```python
SSE_PRESSURE = (current_price - SSE_BASE) / SSE_VOLATILITY
```

| 압력값 | 해석 | 판정 방향 |
|---:|---|---|
| `< -1.0` | 약세 이탈 | 신규매수 금지, 손절선 아래면 팔아라 |
| `-1.0 ~ -0.3` | 하단권 | 반등 확인 전 대기 |
| `-0.3 ~ 0.3` | 기준선 근처 | 방향 확인 필요 |
| `0.3 ~ 1.0` | 매수 가능 구간 | 지지 후 회복 시 조건부 매수 가능 |
| `1.0 ~ 1.5` | 상승 우위 | 신규 진입은 손익비 확인 필요 |
| `>= 1.5` | 과열권 | 신규매수 금지, 보유자는 익절 관리 우선 |

## 진입가 산출 방식

```python
SSE_ENTRY = SSE_BASE + 0.25 * SSE_VOLATILITY
```

장중 분석에서는 3분봉 또는 5분봉 종가가 `SSE_ENTRY` 이상에서 유지될 때만 조건 충족으로 본다. 장중에는 `확정 돌파`, `일봉 돌파 확정`, `돌파 확인 완료` 같은 표현을 쓰지 않는다.

## 손절가 산출 방식

```python
SSE_STOP_RAW = SSE_BASE - 0.75 * SSE_VOLATILITY
SSE_STOP = min(SSE_STOP_RAW, MID26, MID52, 최근20일저점)
```

`SSE_STOP >= SSE_ENTRY`이면 QA 실패다.

## 익절가 산출 방식

1차 익절가:

```python
SSE_TARGET1_RAW = SSE_BASE + 1.25 * SSE_VOLATILITY
```

후보:

- `SSE_TARGET1_RAW`
- 최근 5일 고점
- 최근 20일 고점
- 기존 볼린저밴드 상단
- MA120
- MA240

현재가와 진입가 위에 있는 후보 중 가장 보수적인 가격을 사용한다.

2차 익절가:

```python
SSE_TARGET2_RAW = SSE_BASE + 1.80 * SSE_VOLATILITY
```

후보:

- `SSE_TARGET2_RAW`
- 최근 20일 고점
- 최근 60일 고점
- MA120
- MA240
- `MID52 + SSE_VOLATILITY`

1차 익절가보다 높은 후보 중 가장 보수적인 가격을 사용한다. `SSE_TARGET2 <= SSE_TARGET1`이면 QA 실패다.

## 추격 금지선 의미

```python
SSE_NO_CHASE = SSE_BASE + 1.50 * SSE_VOLATILITY
```

현재가가 `SSE_NO_CHASE` 이상이면 신규매수 긍정 판정은 금지한다.

금지 판정:

```text
사라
조건부로 사라
```

## 손익비 기준

```python
RISK = SSE_ENTRY - SSE_STOP
REWARD1 = SSE_TARGET1 - SSE_ENTRY
REWARD2 = SSE_TARGET2 - SSE_ENTRY

RR1 = REWARD1 / RISK
RR2 = REWARD2 / RISK
```

검증:

- `RISK <= 0`: 분석 중단
- `RR1 < 1.2`: 신규매수 금지
- `RR2 <= RR1`: 경고 또는 QA 실패 대상

## 장중/장마감 후 판정 차이

장중 분석:

- 키움 현재가 timestamp와 분봉 데이터가 유효해야 한다.
- 3분봉 또는 5분봉 종가가 `SSE_ENTRY` 이상에서 유지될 때만 조건부 진입을 허용한다.
- 장중 돌파만으로 일봉 돌파가 확정됐다고 쓰지 않는다.

장마감 후 분석:

- 최신 완료 일봉 기준으로 SSE를 계산한다.
- 분봉 조건은 다음 거래일 장중 확인 조건으로만 표현한다.
- 일봉 종가 기준 지지/이탈/돌파 조건을 우선한다.

## 최종 판정 우선순위

SSE와 기존 명령형 판단이 충돌하면 더 안전한 판정을 우선한다.

```text
분석 중단 > 팔아라 > 사지 마라 > 기다려라 > 보유하라 > 조건부로 사라
```

## QA 규칙

정상 리포트 저장 금지 대상:

- `SSE_STOP >= SSE_ENTRY`
- `SSE_TARGET1 <= SSE_ENTRY`
- `SSE_TARGET2 <= SSE_TARGET1`
- 현재가가 `SSE_NO_CHASE` 이상인데 신규매수 긍정 판정
- `RR1 < 1.2`인데 신규매수 긍정 판정
- `SSE_PRESSURE >= 1.5`인데 신규매수 긍정 판정
- `SSE_PRESSURE < -1.0`인데 신규매수 긍정 판정
- 장중 리포트에 `확정 돌파`, `일봉 돌파 확정`, `돌파 확인 완료` 표현 포함
- 진입가, 손절가, 익절가, 추격 금지선에 산출 근거가 없음

## 자동매매 금지 원칙

SSE Indicator는 분석 보조 지표다. 주문 전송, 자동매매, 시장가 주문, 조건주문, 자동 주문 예약 기능을 구현하지 않는다. 키움 API는 데이터 Provider로만 사용한다.
