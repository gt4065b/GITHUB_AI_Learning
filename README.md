# Tesla Stock Decision Tree Trading System

테슬라 주식의 다음날 ±10달러 변동을 예측하는 의사결정나무 기반 매매 시스템입니다.

**yfinance 패키지를 활용한 실시간 주가 데이터 수집 및 분석**

## 주요 기능

### 1. 데이터 수집 (yfinance 활용)
- **실시간 데이터**: yfinance 패키지로 Yahoo Finance에서 실제 주가 데이터 수집
- **유연한 기간**: 1일부터 최대 기간까지 자유롭게 설정 가능
- **다중 시도**: 3가지 방법으로 데이터 수집 시도 (API 실패 대비)
- **자동 Fallback**: API 접근 실패 시 시뮬레이션 데이터 자동 전환

### 2. 기술적 지표 생성 (24개)
- 이동평균: MA5, MA10, MA20
- RSI (상대강도지수)
- 볼린저 밴드
- 모멘텀 지표
- 변동성 지표
- 거래량 분석

### 3. 의사결정나무 학습
- 최근 1개월 데이터로 학습
- 다음날 ±10달러 변동 예측
- 3가지 분류: 상승(1), 보합(0), 하락(-1)

### 4. 매매 시뮬레이션
- 예측 기반 자동 매매 전략
- 백테스팅 및 성과 분석
- 승률, 수익률 계산

### 5. 시각화
- 의사결정나무 구조 시각화
- 특성 중요도 그래프
- 상세 거래 결과

## 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

### 1. yfinance 기본 사용법 학습

```bash
python yfinance_example.py
```

yfinance 패키지의 기본 사용법과 다양한 활용 예제를 확인할 수 있습니다.

### 2. 메인 분석 시스템 (권장)

```bash
python tesla_stock_analyzer.py
```

개선된 버전으로, yfinance를 활용한 실시간 데이터 수집과 자동 fallback 기능을 제공합니다.

### 3. 데모 버전 (시뮬레이션 데이터)

```bash
python tesla_stock_demo.py
```

시뮬레이션 데이터로 시스템 작동을 테스트할 수 있습니다.

### 4. 기존 버전

```bash
python tesla_stock_decision_tree.py
```

초기 버전의 구현입니다.

## 파일 구조

```
GITHUB_AI_Learning/
├── tesla_stock_analyzer.py      # 메인 분석 시스템 (개선 버전, 권장)
├── yfinance_example.py          # yfinance 사용 예제
├── tesla_stock_demo.py          # 데모 버전 (시뮬레이션)
├── tesla_stock_decision_tree.py # 기존 버전
├── requirements.txt             # 필요 패키지
├── README.md                    # 이 파일
└── YFINANCE_GUIDE.md           # yfinance 상세 가이드
```

## 출력 파일

실행 후 다음 파일들이 생성됩니다:

- `tesla_decision_tree.png`: 의사결정나무 시각화
- `feature_importance.png`: 특성 중요도 그래프
- `tesla_trading_results.csv`: 상세 거래 결과
- `tesla_stock_data.csv`: 원본 주가 데이터

## yfinance 활용 방법

### 기본 사용

```python
import yfinance as yf

# 최근 3개월 데이터
df = yf.download("TSLA", period="3mo")

# 특정 기간 지정
df = yf.download("TSLA", start="2024-01-01", end="2024-12-31")

# 여러 종목
df = yf.download(["TSLA", "AAPL", "GOOGL"], period="1mo")
```

### 데이터 구조

```python
# 다운로드한 데이터는 다음 컬럼을 포함:
Open        # 시가
High        # 고가
Low         # 저가
Close       # 종가
Volume      # 거래량
```

### 본 프로젝트의 활용

```python
from tesla_stock_analyzer import TeslaStockAnalyzer

# 분석기 초기화
analyzer = TeslaStockAnalyzer(ticker="TSLA", threshold=10.0)

# 데이터 수집 (자동으로 3가지 방법 시도)
df = analyzer.fetch_stock_data(period="3mo")

# 전체 분석 파이프라인 실행
# (기술적 지표 → 학습 → 예측 → 시각화)
```

자세한 내용은 [YFINANCE_GUIDE.md](YFINANCE_GUIDE.md)를 참조하세요.

## 매매 규칙

| 예측 | 행동 | 설명 |
|------|------|------|
| 1 (상승) | BUY | 다음날 $10 이상 상승 예상 → 매수 |
| -1 (하락) | SHORT | 다음날 $10 이상 하락 예상 → 공매도 |
| 0 (보합) | HOLD | 변동 ±$10 미만 예상 → 보유 |

## 주요 특성 (24개)

### 1. 기본 가격 정보
- Open, High, Low, Close, Volume

### 2. 수익률
- Daily_Return: 일일 수익률
- Price_Change: 가격 변화

### 3. 이동평균
- MA5, MA10, MA20
- Close_MA5_Diff, Close_MA10_Diff, Close_MA20_Diff

### 4. 기술적 지표
- RSI: 상대강도지수
- BB_Width, BB_Position: 볼린저 밴드

### 5. 거래량
- Volume_MA5: 5일 평균 거래량
- Volume_Ratio: 거래량 비율

### 6. 모멘텀
- Momentum_3, Momentum_5, Momentum_10

### 7. 변동성
- Volatility_5, Volatility_10

### 8. 가격 범위
- HL_Range, HL_Range_Pct

## 시스템 특징

### ✅ 장점

1. **실시간 데이터**: yfinance로 실제 주가 데이터 수집
2. **자동화**: 데이터 수집부터 분석까지 완전 자동화
3. **에러 처리**: API 실패 시 자동 fallback
4. **다양한 지표**: 24개 기술적 지표 자동 계산
5. **시각화**: 의사결정나무와 특성 중요도 그래프
6. **확장성**: 다른 종목(AAPL, GOOGL 등)으로 쉽게 변경 가능

### ⚠️ 주의사항

1. **교육 목적**: 이 시스템은 교육 및 연구 목적입니다
2. **투자 판단**: 실제 투자 시 전문가 상담 필요
3. **과거 성과**: 과거 성과가 미래 수익을 보장하지 않음
4. **리스크 관리**: 충분한 리스크 관리 필요
5. **API 제한**: Yahoo Finance API는 무료 서비스로 제한이 있을 수 있음

## 문제 해결

### yfinance 403 에러

현재 환경에서 Yahoo Finance API 접근이 제한될 수 있습니다. 이 경우:

1. **자동 Fallback**: `tesla_stock_analyzer.py`는 자동으로 시뮬레이션 데이터로 전환
2. **네트워크 확인**: 방화벽이나 프록시 설정 확인
3. **대안 사용**: `tesla_stock_demo.py`로 시뮬레이션 데이터 사용

### 다른 종목 분석

```python
# AAPL(애플) 분석
analyzer = TeslaStockAnalyzer(ticker="AAPL", threshold=5.0)

# GOOGL(구글) 분석
analyzer = TeslaStockAnalyzer(ticker="GOOGL", threshold=20.0)
```

## 개발 환경

- Python 3.11+
- pandas, numpy
- scikit-learn
- matplotlib
- yfinance

## 참고 자료

- [yfinance 공식 GitHub](https://github.com/ranaroussi/yfinance)
- [Yahoo Finance](https://finance.yahoo.com/)
- [YFINANCE_GUIDE.md](YFINANCE_GUIDE.md) - 상세 사용 가이드

## 라이선스

교육 및 연구 목적으로 자유롭게 사용 가능합니다.
