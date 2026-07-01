# 09_SSE_INDICATOR_RULES - SSE Indicator

## 목적

`SSE Indicator`는 `Sungsoo Synthetic Equilibrium Indicator`의 약자다.

기존 이동평균선, 볼린저밴드, 일목균형표의 결과값을 단순 가중합하지 않는다. 종가 평균, 표준편차, 기간별 고가/저가 중간값이라는 원리를 하나의 합성 균형 공식으로 재조합해 분석 보조 지표로 사용한다.

SSE는 주문, 자동매매, 시장가 주문, 조건주문에 사용하지 않는다. 분석 리포트의 가격 기준과 안전 필터로만 사용한다.

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
STD20 = Close.rolling(20).std()
```

고저 중간값 원리:

```python
MID9 = (High.rolling(9).max() + Low.rolling(9).min()) / 2
MID26 = (High.rolling(26).max() + Low.rolling(26).min()) / 2
MID52 = (High.rolling(52).max() + Low.rolling(52).min()) / 2
```

## SSE 기준선

```python
SSE_BASE =
    0.35 * MA20
  + 0.20 * MA60
  + 0.20 * MID26
  + 0.15 * MID52
  + 0.10 * MID9
```

의미:

- `MA20`: 단기 종가 평균
- `MA60`: 중기 종가 평균
- `MID9`: 단기 고저 균형값
- `MID26`: 중기 고저 균형값
- `MID52`: 장기 고저 균형값

## SSE 통합 변동성

```python
SSE_VOLATILITY =
    0.50 * STD20
  + 0.25 * abs(MID26 - MID52)
  + 0.25 * abs(MA20 - MA60)
```

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

향후 백테스트에서는 `1.4`, `1.6`, `1.8`, `2.0`을 비교할 수 있다.

## SSE 압력값

```python
SSE_PRESSURE = (Close - SSE_BASE) / SSE_VOLATILITY
```

해석:

| 압력값 | 해석 |
|---:|---|
| `< -1.0` | 약세 이탈, 신규매수 금지 |
| `-1.0 ~ -0.3` | 하단권, 반등 확인 대기 |
| `-0.3 ~ 0.3` | 기준선 근처, 방향 확인 필요 |
| `0.3 ~ 1.0` | 매수 가능 구간, 지지 후 조건부 매수 |
| `1.0 ~ 1.5` | 상승 우위, 손익비 확인 필요 |
| `>= 1.5` | 과열권, 추격매수 금지 |

## 매매 가격 산출

예상 진입가:

```python
SSE_ENTRY = SSE_BASE + 0.25 * SSE_VOLATILITY
```

장중 분석에서는 3분봉 또는 5분봉 종가가 `SSE_ENTRY` 이상에서 유지될 때만 조건 충족으로 본다.

예상 손절가:

```python
SSE_STOP_RAW = SSE_BASE - 0.75 * SSE_VOLATILITY
SSE_STOP = min(SSE_STOP_RAW, MID26, MID52, 최근20일저점)
```

`SSE_STOP >= SSE_ENTRY`이면 QA 실패다.

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

추격 금지선:

```python
SSE_NO_CHASE = SSE_BASE + 1.50 * SSE_VOLATILITY
```

현재가가 `SSE_NO_CHASE` 이상이면 최종 판정은 `사지 마라` 또는 더 보수적인 판정이어야 한다.

## 손익비

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

## 최종 판정 우선순위

SSE와 기존 명령형 판단이 충돌하면 더 안전한 판정을 우선한다.

```text
분석 중단 > 팔아라 > 사지 마라 > 기다려라 > 보유하라 > 조건부로 사라
```

장중에는 `확정 돌파`, `일봉 돌파 확정`, `돌파 확인 완료` 표현을 사용하지 않는다.

## QA 규칙

아래 조건은 정상 리포트 저장 금지 대상이다.

- `SSE_STOP >= SSE_ENTRY`
- `SSE_TARGET1 <= SSE_ENTRY`
- `SSE_TARGET2 <= SSE_TARGET1`
- 현재가가 `SSE_NO_CHASE` 이상인데 신규매수 긍정 판정
- `RR1 < 1.2`인데 신규매수 긍정 판정
- `SSE_PRESSURE >= 1.5`인데 신규매수 긍정 판정
- `SSE_PRESSURE < -1.0`인데 신규매수 긍정 판정
- 가격 근거 없이 진입가, 손절가, 익절가를 출력

