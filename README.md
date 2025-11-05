# Tesla Stock Decision Tree Trading System

테슬라 주식의 다음날 ±10달러 변동을 예측하는 의사결정나무 기반 매매 시스템입니다.

## 기능

- **데이터 수집**: 최근 3개월간의 테슬라(TSLA) 주식 데이터 수집
- **기술적 지표**: 이동평균, RSI, 볼린저 밴드, 모멘텀, 변동성 등 24개 지표 생성
- **의사결정나무**: 최근 1개월 데이터로 학습하여 다음날 주가 변동 예측
- **매매 시뮬레이션**: 예측 기반 매매 전략 백테스팅
- **시각화**: 의사결정나무 구조 및 특성 중요도 그래프

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
python tesla_stock_decision_tree.py
```

## 출력 파일

- `tesla_decision_tree.png`: 의사결정나무 시각화
- `feature_importance.png`: 특성 중요도 그래프
- `tesla_trading_results.csv`: 상세 거래 결과

## 매매 규칙

- **예측 = 1 (상승)**: 매수 포지션 진입
- **예측 = -1 (하락)**: 공매도 포지션 진입
- **예측 = 0 (보합)**: 포지션 보유

## 주요 특성

1. 가격 지표: Open, High, Low, Close
2. 거래량: Volume, Volume_Ratio
3. 이동평균: MA5, MA10, MA20
4. 기술적 지표: RSI, Bollinger Bands
5. 모멘텀: 3일, 5일, 10일 모멘텀
6. 변동성: 5일, 10일 표준편차

## 주의사항

이 시스템은 교육 및 연구 목적으로 제작되었습니다. 실제 투자에 사용하기 전에 충분한 검증이 필요합니다.
