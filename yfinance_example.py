#!/usr/bin/env python3
"""
yfinance 패키지 사용 예제
티커별 일별 시가, 종가, 고가, 저가, 거래량 정보 조회
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("="*70)
print("yfinance 패키지 사용 예제")
print("="*70)

# 1. 기본 사용법: Ticker 객체 생성
print("\n1. 테슬라 주식 정보 가져오기")
print("-"*70)

ticker = yf.Ticker("TSLA")

# 2. 기업 정보 조회
print("\n기업 정보:")
try:
    info = ticker.info
    print(f"  회사명: {info.get('longName', 'N/A')}")
    print(f"  섹터: {info.get('sector', 'N/A')}")
    print(f"  산업: {info.get('industry', 'N/A')}")
    print(f"  현재가: ${info.get('currentPrice', 'N/A')}")
    print(f"  시가총액: ${info.get('marketCap', 'N/A'):,}")
except Exception as e:
    print(f"  기업 정보 조회 실패: {e}")

# 3. 과거 데이터 조회 - period 옵션 사용
print("\n\n2. 과거 데이터 조회 (기간 지정)")
print("-"*70)

# 방법 1: period 파라미터 사용
print("\n[방법 1] period 파라미터 사용:")
print("  - 지원 기간: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max")

# 최근 3개월 데이터
df_3mo = yf.download("TSLA", period="3mo", progress=False)
print(f"\n  최근 3개월 데이터: {len(df_3mo)}일")
print(f"  기간: {df_3mo.index[0].date()} ~ {df_3mo.index[-1].date()}")
print("\n  데이터 샘플 (최근 5일):")
print(df_3mo.tail())

# 4. 과거 데이터 조회 - start/end 날짜 지정
print("\n\n[방법 2] start/end 날짜 지정:")
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

df_dates = yf.download("TSLA", start=start_date, end=end_date, progress=False)
print(f"\n  데이터: {len(df_dates)}일")
print(f"  기간: {df_dates.index[0].date()} ~ {df_dates.index[-1].date()}")

# 5. 다양한 interval 옵션
print("\n\n3. 다양한 시간 간격 (interval)")
print("-"*70)
print("  지원 간격: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo")

# 일별 데이터 (기본값)
df_daily = yf.download("TSLA", period="1mo", interval="1d", progress=False)
print(f"\n  일별 데이터 (1개월): {len(df_daily)}일")

# 주별 데이터
df_weekly = yf.download("TSLA", period="6mo", interval="1wk", progress=False)
print(f"  주별 데이터 (6개월): {len(df_weekly)}주")

# 6. 여러 종목 동시 조회
print("\n\n4. 여러 종목 동시 조회")
print("-"*70)

tickers = ["TSLA", "AAPL", "GOOGL"]
df_multi = yf.download(tickers, period="1mo", progress=False)
print(f"\n  조회 종목: {', '.join(tickers)}")
print(f"  데이터 크기: {df_multi.shape}")
print("\n  종가 데이터:")
print(df_multi['Close'].tail())

# 7. 컬럼 설명
print("\n\n5. 데이터 컬럼 설명")
print("-"*70)
print("  Open: 시가 (장 시작 시 가격)")
print("  High: 고가 (당일 최고 가격)")
print("  Low: 저가 (당일 최저 가격)")
print("  Close: 종가 (장 마감 시 가격)")
print("  Adj Close: 조정 종가 (배당, 주식분할 등 반영)")
print("  Volume: 거래량")

# 8. 데이터 통계
print("\n\n6. 데이터 통계")
print("-"*70)
print("\n최근 3개월 통계:")
print(df_3mo[['Open', 'High', 'Low', 'Close', 'Volume']].describe())

# 9. 데이터 저장
print("\n\n7. 데이터 저장")
print("-"*70)
csv_file = "tesla_stock_data.csv"
df_3mo.to_csv(csv_file)
print(f"  데이터 저장 완료: {csv_file}")

# 10. 실전 팁
print("\n\n8. 실전 사용 팁")
print("-"*70)
print("""
  1. 일별 데이터: period="3mo", interval="1d"
  2. 주말 제외: 자동으로 거래일만 포함됨
  3. 에러 처리: try-except로 감싸서 사용
  4. 캐싱: yf.download는 자동으로 데이터를 캐시
  5. 대량 다운로드: threads=True 옵션으로 병렬 처리

  예제:
  df = yf.download("TSLA", period="3mo", interval="1d",
                   auto_adjust=True, threads=True, progress=False)
""")

print("\n\n" + "="*70)
print("yfinance 사용 예제 완료!")
print("="*70)
