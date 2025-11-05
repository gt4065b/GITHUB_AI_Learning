# yfinance 패키지 활용 가이드

## 개요

yfinance는 Yahoo Finance API를 활용하여 주식 데이터를 쉽게 가져올 수 있는 Python 오픈소스 패키지입니다.

## 설치

```bash
pip install yfinance
```

## 기본 사용법

### 1. 주식 데이터 다운로드

#### 방법 1: `yf.download()` 사용 (권장)

```python
import yfinance as yf

# 최근 3개월 데이터
df = yf.download("TSLA", period="3mo", interval="1d")

# 여러 종목 동시 다운로드
df = yf.download(["TSLA", "AAPL", "GOOGL"], period="1mo")
```

**period 옵션:**
- `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`

**interval 옵션:**
- `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo`

#### 방법 2: 날짜 범위 지정

```python
from datetime import datetime, timedelta

end_date = datetime.now()
start_date = end_date - timedelta(days=90)

df = yf.download("TSLA", start=start_date, end=end_date)
```

#### 방법 3: Ticker 객체 사용

```python
ticker = yf.Ticker("TSLA")
df = ticker.history(period="3mo")

# 기업 정보 조회
info = ticker.info
print(info['longName'])
print(info['sector'])
```

### 2. 데이터 구조

다운로드한 데이터는 pandas DataFrame 형태로 다음 컬럼을 포함합니다:

```python
# 컬럼 설명
Open        # 시가 (장 시작 시 가격)
High        # 고가 (당일 최고 가격)
Low         # 저가 (당일 최저 가격)
Close       # 종가 (장 마감 시 가격)
Adj Close   # 조정 종가 (배당, 주식분할 등 반영)
Volume      # 거래량
```

### 3. 실전 예제

#### 예제 1: 기본 데이터 조회

```python
import yfinance as yf

# 테슬라 주식 데이터
df = yf.download("TSLA", period="3mo", progress=False)

print(f"데이터 크기: {len(df)}일")
print(f"기간: {df.index[0]} ~ {df.index[-1]}")
print("\n최근 5일:")
print(df.tail())
```

#### 예제 2: 여러 종목 비교

```python
import yfinance as yf
import matplotlib.pyplot as plt

# 여러 종목 다운로드
tickers = ["TSLA", "AAPL", "GOOGL", "MSFT"]
df = yf.download(tickers, period="1y", progress=False)

# 종가 비교 그래프
df['Close'].plot(figsize=(12, 6))
plt.title('Stock Price Comparison')
plt.ylabel('Price ($)')
plt.legend(tickers)
plt.show()
```

#### 예제 3: 기술적 분석

```python
import yfinance as yf
import pandas as pd

# 데이터 다운로드
df = yf.download("TSLA", period="6mo", progress=False)

# 이동평균 계산
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# RSI 계산
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

print(df[['Close', 'MA20', 'MA50', 'RSI']].tail())
```

#### 예제 4: 기업 정보 조회

```python
import yfinance as yf

ticker = yf.Ticker("TSLA")
info = ticker.info

print(f"회사명: {info['longName']}")
print(f"섹터: {info['sector']}")
print(f"현재가: ${info['currentPrice']}")
print(f"시가총액: ${info['marketCap']:,}")
print(f"52주 최고가: ${info['fiftyTwoWeekHigh']}")
print(f"52주 최저가: ${info['fiftyTwoWeekLow']}")
```

### 4. 고급 기능

#### 옵션 데이터

```python
ticker = yf.Ticker("TSLA")

# 옵션 만료일
expirations = ticker.options

# 특정 만료일의 옵션 체인
opt = ticker.option_chain(expirations[0])
print(opt.calls)  # 콜옵션
print(opt.puts)   # 풋옵션
```

#### 배당 및 주식분할 정보

```python
ticker = yf.Ticker("AAPL")

# 배당 내역
dividends = ticker.dividends
print(dividends.tail())

# 주식분할 내역
splits = ticker.splits
print(splits)
```

#### 재무제표

```python
ticker = yf.Ticker("TSLA")

# 손익계산서
print(ticker.financials)

# 대차대조표
print(ticker.balance_sheet)

# 현금흐름표
print(ticker.cashflow)
```

## 본 프로젝트에서의 활용

### tesla_stock_analyzer.py

우리 프로젝트에서는 yfinance를 다음과 같이 활용합니다:

```python
from tesla_stock_analyzer import TeslaStockAnalyzer

# 분석기 초기화
analyzer = TeslaStockAnalyzer(ticker="TSLA", threshold=10.0)

# 데이터 수집 (3개월)
df = analyzer.fetch_stock_data(period="3mo")

# 기술적 지표 생성
df = analyzer.create_technical_indicators(df)

# 의사결정나무 학습 및 예측
X, y, df_train = analyzer.prepare_training_data(df, train_period="1mo")
analyzer.train_decision_tree(X, y)
```

### 주요 특징

1. **다중 시도 메커니즘**
   - yf.download() → Ticker.history() → 날짜 지정 방식 순으로 시도
   - 모든 방법 실패 시 시뮬레이션 데이터로 fallback

2. **에러 처리**
   - API 제한, 네트워크 오류 등 모든 예외 상황 처리
   - 사용자에게 명확한 에러 메시지 제공

3. **자동화**
   - 데이터 다운로드부터 분석까지 자동화
   - 기술적 지표 자동 계산
   - 의사결정나무 학습 및 시각화

## 실행 방법

### 1. yfinance 사용 예제

```bash
python yfinance_example.py
```

기본적인 yfinance 사용법을 학습할 수 있습니다.

### 2. 메인 분석 시스템

```bash
python tesla_stock_analyzer.py
```

전체 의사결정나무 매매 시스템을 실행합니다.

## 참고 사항

- **API 제한**: Yahoo Finance는 무료 서비스이므로 요청 제한이 있을 수 있습니다.
- **데이터 지연**: 실시간이 아닌 15-20분 지연된 데이터입니다.
- **안정성**: API가 변경되거나 차단될 수 있으므로 에러 처리가 중요합니다.
- **상업적 이용**: 상업적 목적으로 사용 시 Yahoo Finance 이용약관을 확인하세요.

## 추가 자료

- 공식 GitHub: https://github.com/ranaroussi/yfinance
- 문서: https://pypi.org/project/yfinance/
- Yahoo Finance: https://finance.yahoo.com/

## 문제 해결

### 403 에러 발생 시

```python
# User-Agent 설정 (일부 환경에서 필요)
import yfinance as yf
yf.pdr_override()

# 또는 requests_cache 사용
import requests_cache
session = requests_cache.CachedSession('yfinance.cache')
session.headers['User-agent'] = 'Mozilla/5.0'
```

### 데이터가 비어있을 때

1. 티커 심볼 확인 (예: TSLA, AAPL)
2. 날짜 범위 확인 (주말/공휴일 제외)
3. 네트워크 연결 확인
4. API 제한 확인 (잠시 후 재시도)

### 본 프로젝트의 해결책

- **자동 Fallback**: API 실패 시 자동으로 시뮬레이션 데이터 사용
- **다중 시도**: 3가지 방법으로 순차적 시도
- **명확한 로그**: 각 단계별 성공/실패 메시지 출력
