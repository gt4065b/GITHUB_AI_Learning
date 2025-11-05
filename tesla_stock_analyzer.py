#!/usr/bin/env python3
"""
테슬라 주식 의사결정나무 매매 시스템 (yfinance 개선 버전)
- yfinance 패키지를 활용한 실시간 데이터 수집
- 최근 3개월 데이터 수집, 최근 1개월로 학습
- 다음날 ±10달러 변동 예측
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class TeslaStockAnalyzer:
    """테슬라 주식 의사결정나무 분석 클래스 (yfinance 활용)"""

    def __init__(self, ticker="TSLA", threshold=10.0):
        """
        Args:
            ticker: 주식 티커 심볼 (기본값: TSLA)
            threshold: 주가 변동 기준 (달러)
        """
        self.ticker = ticker
        self.threshold = threshold
        self.model = None
        self.feature_names = []
        self.data = None
        self.ticker_obj = None

    def fetch_stock_data(self, period="3mo", use_fallback=True):
        """
        yfinance로 주식 데이터 수집

        Args:
            period: 데이터 기간 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            use_fallback: API 실패 시 시뮬레이션 데이터 사용 여부
        """
        print(f"{self.ticker} 주식 데이터 수집 중 (기간: {period})...")
        print("yfinance 패키지 사용\n")

        try:
            # 방법 1: yf.download() 사용 (권장)
            print("[시도 1] yf.download() 메서드 사용...")
            df = yf.download(
                self.ticker,
                period=period,
                interval="1d",
                auto_adjust=True,  # 조정 종가 자동 적용
                threads=True,
                progress=False
            )

            if df.empty:
                raise ValueError("데이터가 비어있습니다.")

            print(f"✓ 데이터 수집 성공!")
            print(f"  기간: {df.index[0].date()} ~ {df.index[-1].date()}")
            print(f"  총 {len(df)}일 데이터")
            print(f"\n  컬럼: {', '.join(df.columns)}")

            # 데이터 미리보기
            print(f"\n최근 5일 데이터:")
            print(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail())

            self.data = df
            return df

        except Exception as e:
            print(f"✗ yf.download() 실패: {e}\n")

            # 방법 2: Ticker 객체 사용
            try:
                print("[시도 2] Ticker.history() 메서드 사용...")
                self.ticker_obj = yf.Ticker(self.ticker)
                df = self.ticker_obj.history(period=period)

                if df.empty:
                    raise ValueError("데이터가 비어있습니다.")

                print(f"✓ 데이터 수집 성공!")
                print(f"  기간: {df.index[0].date()} ~ {df.index[-1].date()}")
                print(f"  총 {len(df)}일 데이터")

                self.data = df
                return df

            except Exception as e2:
                print(f"✗ Ticker.history() 실패: {e2}\n")

                # 방법 3: 날짜 범위 지정
                try:
                    print("[시도 3] start/end 날짜 지정 방법...")

                    # period를 일수로 변환
                    period_days = {
                        "1mo": 30, "3mo": 90, "6mo": 180,
                        "1y": 365, "2y": 730, "5y": 1825
                    }
                    days = period_days.get(period, 90)

                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)

                    df = yf.download(
                        self.ticker,
                        start=start_date,
                        end=end_date,
                        progress=False
                    )

                    if df.empty:
                        raise ValueError("데이터가 비어있습니다.")

                    print(f"✓ 데이터 수집 성공!")
                    self.data = df
                    return df

                except Exception as e3:
                    print(f"✗ 날짜 지정 방법 실패: {e3}\n")

                    if use_fallback:
                        print("[Fallback] 시뮬레이션 데이터 생성...")
                        df = self._generate_sample_data(period)
                        self.data = df
                        return df
                    else:
                        raise Exception("모든 데이터 수집 방법이 실패했습니다.")

    def _generate_sample_data(self, period="3mo"):
        """API 실패 시 시뮬레이션 데이터 생성"""
        period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365}
        days = period_days.get(period, 90)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')

        np.random.seed(42)
        initial_price = 250.0
        returns = np.random.normal(0.001, 0.03, len(dates))
        prices = initial_price * np.exp(np.cumsum(returns))

        data = []
        for i, date in enumerate(dates):
            close = prices[i]
            high = close * (1 + abs(np.random.normal(0, 0.02)))
            low = close * (1 - abs(np.random.normal(0, 0.02)))
            open_price = np.random.uniform(low, high)
            volume = int(np.random.gamma(5, 20000000))

            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })

        df = pd.DataFrame(data, index=dates)
        print(f"✓ 시뮬레이션 데이터 생성 완료: {len(df)}일")
        return df

    def get_stock_info(self):
        """주식 기본 정보 조회"""
        if self.ticker_obj is None:
            self.ticker_obj = yf.Ticker(self.ticker)

        print(f"\n{self.ticker} 주식 정보:")
        print("="*60)

        try:
            info = self.ticker_obj.info
            print(f"  회사명: {info.get('longName', 'N/A')}")
            print(f"  섹터: {info.get('sector', 'N/A')}")
            print(f"  산업: {info.get('industry', 'N/A')}")
            print(f"  현재가: ${info.get('currentPrice', 'N/A')}")
            print(f"  시가총액: ${info.get('marketCap', 0):,}")
            print(f"  52주 최고가: ${info.get('fiftyTwoWeekHigh', 'N/A')}")
            print(f"  52주 최저가: ${info.get('fiftyTwoWeekLow', 'N/A')}")
            print(f"  평균 거래량: {info.get('averageVolume', 0):,}")
        except Exception as e:
            print(f"  정보 조회 실패: {e}")

    def create_technical_indicators(self, df):
        """기술적 지표 생성"""
        print("\n기술적 지표 생성 중...")

        # 가격 변동성
        df['Daily_Return'] = df['Close'].pct_change() * 100
        df['Price_Change'] = df['Close'].diff()

        # 이동평균
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()

        # 이동평균과의 차이
        df['Close_MA5_Diff'] = df['Close'] - df['MA5']
        df['Close_MA10_Diff'] = df['Close'] - df['MA10']
        df['Close_MA20_Diff'] = df['Close'] - df['MA20']

        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 볼린저 밴드
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

        # 거래량 지표
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']

        # 모멘텀
        df['Momentum_3'] = df['Close'] - df['Close'].shift(3)
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

        # 변동성
        df['Volatility_5'] = df['Close'].rolling(window=5).std()
        df['Volatility_10'] = df['Close'].rolling(window=10).std()

        # High-Low 범위
        df['HL_Range'] = df['High'] - df['Low']
        df['HL_Range_Pct'] = (df['HL_Range'] / df['Close']) * 100

        print(f"✓ 기술적 지표 생성 완료: {len(df.columns)}개 컬럼")
        return df

    def create_target_variable(self, df):
        """타겟 변수 생성"""
        print(f"\n타겟 변수 생성 중 (기준: ±${self.threshold})...")

        df['Next_Day_Change'] = df['Close'].shift(-1) - df['Close']

        df['Target'] = 0
        df.loc[df['Next_Day_Change'] >= self.threshold, 'Target'] = 1
        df.loc[df['Next_Day_Change'] <= -self.threshold, 'Target'] = -1

        target_counts = df['Target'].value_counts().sort_index()
        print(f"\n타겟 변수 분포:")
        print(f"  하락(-1): {target_counts.get(-1, 0)}일 (${self.threshold} 이상 하락)")
        print(f"  보합(0):  {target_counts.get(0, 0)}일")
        print(f"  상승(1):  {target_counts.get(1, 0)}일 (${self.threshold} 이상 상승)")

        return df

    def prepare_training_data(self, df, train_period="1mo"):
        """학습 데이터 준비"""
        print(f"\n학습 데이터 준비 중 (기간: {train_period})...")

        df_clean = df.dropna()

        # 기간별 일수 계산
        period_days = {"1mo": 30, "2mo": 60, "3mo": 90}
        days = period_days.get(train_period, 30)

        df_train = df_clean.tail(days + 1)[:-1]

        print(f"✓ 학습 데이터 기간: {df_train.index[0].date()} ~ {df_train.index[-1].date()}")
        print(f"  학습 데이터 크기: {len(df_train)}일")

        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Daily_Return', 'Price_Change',
            'MA5', 'MA10', 'MA20',
            'Close_MA5_Diff', 'Close_MA10_Diff', 'Close_MA20_Diff',
            'RSI', 'BB_Width', 'BB_Position',
            'Volume_Ratio',
            'Momentum_3', 'Momentum_5', 'Momentum_10',
            'Volatility_5', 'Volatility_10',
            'HL_Range', 'HL_Range_Pct'
        ]

        X = df_train[feature_cols]
        y = df_train['Target']
        self.feature_names = feature_cols

        return X, y, df_train

    def train_decision_tree(self, X, y):
        """의사결정나무 모델 학습"""
        print("\n의사결정나무 모델 학습 중...")

        self.model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

        self.model.fit(X, y)
        train_pred = self.model.predict(X)
        train_accuracy = accuracy_score(y, train_pred)

        print(f"✓ 학습 완료!")
        print(f"  학습 데이터 정확도: {train_accuracy:.2%}")

        return self.model

    def evaluate_model(self, X, y):
        """모델 평가"""
        print("\n" + "="*60)
        print("모델 평가")
        print("="*60)

        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"\n정확도: {accuracy:.2%}")

        print("\n혼동 행렬:")
        cm = confusion_matrix(y, y_pred)
        print(cm)

        print("\n분류 리포트:")
        unique_classes = sorted(y.unique())
        labels_map = {-1: '하락(-1)', 0: '보합(0)', 1: '상승(1)'}
        target_names = [labels_map[c] for c in unique_classes]
        print(classification_report(y, y_pred, labels=unique_classes,
                                   target_names=target_names, zero_division=0))

        print("\n특성 중요도 (상위 10개):")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:20s}: {row['importance']:.4f}")

        return feature_importance

    def visualize_tree(self, output_file='tesla_decision_tree.png'):
        """의사결정나무 시각화"""
        print(f"\n의사결정나무 시각화 중...")

        plt.figure(figsize=(25, 15))
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=['Down(-1)', 'Neutral(0)', 'Up(1)'],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title(f'{self.ticker} Stock Decision Tree\n(+/- ${self.threshold} Next Day Movement)',
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 저장 완료: {output_file}")
        plt.close()

    def visualize_feature_importance(self, feature_importance, output_file='feature_importance.png'):
        """특성 중요도 시각화"""
        print(f"특성 중요도 시각화 중...")

        top_features = feature_importance.head(15)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance', fontweight='bold')
        plt.ylabel('Feature', fontweight='bold')
        plt.title(f'{self.ticker} Feature Importance - Top 15', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 저장 완료: {output_file}")
        plt.close()

    def simulate_trading(self, df_train, X, y):
        """매매 시뮬레이션"""
        print("\n" + "="*60)
        print("매매 시뮬레이션")
        print("="*60)

        predictions = self.model.predict(X)
        actual_changes = df_train['Next_Day_Change'].values

        trades = []
        for i, (pred, actual) in enumerate(zip(predictions, actual_changes)):
            date = df_train.index[i]
            close_price = df_train['Close'].iloc[i]

            if pred == 1:
                action = "BUY"
                profit = actual
            elif pred == -1:
                action = "SHORT"
                profit = -actual
            else:
                action = "HOLD"
                profit = 0

            trades.append({
                'date': date,
                'prediction': pred,
                'actual_change': actual,
                'action': action,
                'profit': profit,
                'close_price': close_price
            })

        trades_df = pd.DataFrame(trades)

        total_trades = len(trades_df[trades_df['action'] != 'HOLD'])
        buy_trades = len(trades_df[trades_df['action'] == 'BUY'])
        short_trades = len(trades_df[trades_df['action'] == 'SHORT'])
        hold_days = len(trades_df[trades_df['action'] == 'HOLD'])

        total_profit = trades_df['profit'].sum()
        profitable_trades = len(trades_df[trades_df['profit'] > 0])
        loss_trades = len(trades_df[trades_df['profit'] < 0])
        win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0

        print(f"\n총 거래일수: {len(trades_df)}일")
        print(f"  매수(BUY): {buy_trades}일")
        print(f"  공매도(SHORT): {short_trades}일")
        print(f"  보유(HOLD): {hold_days}일")
        print(f"\n총 수익: ${total_profit:.2f}")
        print(f"수익 거래: {profitable_trades}일")
        print(f"손실 거래: {loss_trades}일")
        print(f"승률: {win_rate:.2f}%")

        if total_trades > 0:
            avg_profit = total_profit / total_trades
            print(f"거래당 평균 수익: ${avg_profit:.2f}")

        if len(trades_df) > 0:
            print("\n수익 상위 5개 거래:")
            top_trades = trades_df.nlargest(5, 'profit')
            for idx, trade in top_trades.iterrows():
                print(f"  {trade['date'].date()}: {trade['action']:6s} | "
                      f"예측={trade['prediction']:2d} | 실제=${trade['actual_change']:7.2f} | "
                      f"수익=${trade['profit']:7.2f}")

        return trades_df

    def save_results(self, df_train, trades_df, output_file='tesla_trading_results.csv'):
        """결과 저장"""
        print(f"\n결과 저장 중: {output_file}")

        result_df = df_train.copy()
        result_df['Prediction'] = self.model.predict(df_train[self.feature_names])
        result_df['Action'] = trades_df['action'].values
        result_df['Trade_Profit'] = trades_df['profit'].values

        result_df.to_csv(output_file)
        print(f"✓ 저장 완료: {output_file}")


def main():
    """메인 실행 함수"""
    print("="*70)
    print("테슬라 주식 의사결정나무 매매 시스템")
    print("yfinance 패키지 활용")
    print("="*70)

    # 분석기 초기화
    analyzer = TeslaStockAnalyzer(ticker="TSLA", threshold=10.0)

    try:
        # 1. 주식 데이터 수집 (yfinance)
        df = analyzer.fetch_stock_data(period="3mo", use_fallback=True)

        # 2. 주식 정보 조회
        analyzer.get_stock_info()

        # 3. 기술적 지표 생성
        df = analyzer.create_technical_indicators(df)

        # 4. 타겟 변수 생성
        df = analyzer.create_target_variable(df)

        # 5. 학습 데이터 준비 (최근 1개월)
        X, y, df_train = analyzer.prepare_training_data(df, train_period="1mo")

        # 6. 의사결정나무 학습
        analyzer.train_decision_tree(X, y)

        # 7. 모델 평가
        feature_importance = analyzer.evaluate_model(X, y)

        # 8. 시각화
        analyzer.visualize_tree('tesla_decision_tree.png')
        analyzer.visualize_feature_importance(feature_importance, 'feature_importance.png')

        # 9. 매매 시뮬레이션
        trades_df = analyzer.simulate_trading(df_train, X, y)

        # 10. 결과 저장
        analyzer.save_results(df_train, trades_df, 'tesla_trading_results.csv')

        print("\n" + "="*70)
        print("분석 완료!")
        print("="*70)
        print("\n생성된 파일:")
        print("  - tesla_decision_tree.png: 의사결정나무 시각화")
        print("  - feature_importance.png: 특성 중요도")
        print("  - tesla_trading_results.csv: 상세 거래 결과")
        print("  - tesla_stock_data.csv: 원본 주가 데이터")

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
