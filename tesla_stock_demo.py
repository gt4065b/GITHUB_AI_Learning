#!/usr/bin/env python3
"""
테슬라 주식 의사결정나무 매매 시스템 (데모 버전)
- 시뮬레이션 데이터로 시스템 시연
- 실제 yfinance 사용 시 데이터 수집 가능
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class TeslaStockDecisionTree:
    """테슬라 주식 의사결정나무 분석 클래스"""

    def __init__(self, threshold=10.0):
        """
        Args:
            threshold: 주가 변동 기준 (달러)
        """
        self.threshold = threshold
        self.model = None
        self.feature_names = []
        self.data = None

    def generate_sample_data(self, days=90):
        """샘플 주식 데이터 생성 (시뮬레이션)"""
        print(f"샘플 주식 데이터 생성 중 ({days}일)...")

        # 기준일: 어제
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days)

        # 날짜 범위 생성 (주말 제외)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' = business days

        # 초기 가격 (2024년 11월 테슬라 평균 가격 기준)
        initial_price = 250.0

        # 가격 시뮬레이션 (랜덤 워크 + 트렌드)
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.03, len(dates))  # 평균 0.1%, 표준편차 3%
        prices = initial_price * np.exp(np.cumsum(returns))

        # 거래량 (평균 100M)
        volumes = np.random.gamma(5, 20000000, len(dates))

        # OHLC 데이터 생성
        data = []
        for i, date in enumerate(dates):
            close = prices[i]
            high = close * (1 + abs(np.random.normal(0, 0.02)))
            low = close * (1 - abs(np.random.normal(0, 0.02)))
            open_price = np.random.uniform(low, high)
            volume = volumes[i]

            data.append({
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': int(volume)
            })

        df = pd.DataFrame(data, index=dates)

        print(f"데이터 생성 완료: {len(df)}일 ({df.index[0].date()} ~ {df.index[-1].date()})")
        print(f"가격 범위: ${df['Close'].min():.2f} ~ ${df['Close'].max():.2f}")
        print(f"평균 가격: ${df['Close'].mean():.2f}")

        self.data = df
        return df

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

        # RSI
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

        # 가격 모멘텀
        df['Momentum_3'] = df['Close'] - df['Close'].shift(3)
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)

        # 변동성
        df['Volatility_5'] = df['Close'].rolling(window=5).std()
        df['Volatility_10'] = df['Close'].rolling(window=10).std()

        # High-Low 범위
        df['HL_Range'] = df['High'] - df['Low']
        df['HL_Range_Pct'] = (df['HL_Range'] / df['Close']) * 100

        print(f"기술적 지표 생성 완료: {len(df.columns)}개 컬럼")
        return df

    def create_target_variable(self, df):
        """타겟 변수 생성: 다음날 ±10달러 변동"""
        print(f"\n타겟 변수 생성 중 (기준: ±${self.threshold})...")

        # 다음날 종가 변화
        df['Next_Day_Change'] = df['Close'].shift(-1) - df['Close']

        # 타겟: 1 (상승), -1 (하락), 0 (보합)
        df['Target'] = 0
        df.loc[df['Next_Day_Change'] >= self.threshold, 'Target'] = 1
        df.loc[df['Next_Day_Change'] <= -self.threshold, 'Target'] = -1

        # 타겟 분포
        target_counts = df['Target'].value_counts().sort_index()
        print("\n타겟 변수 분포:")
        print(f"  하락(-1): {target_counts.get(-1, 0)}일 (${self.threshold} 이상 하락)")
        print(f"  보합(0):  {target_counts.get(0, 0)}일")
        print(f"  상승(1):  {target_counts.get(1, 0)}일 (${self.threshold} 이상 상승)")

        return df

    def prepare_training_data(self, df, train_days=30):
        """학습 데이터 준비 (최근 30일)"""
        print(f"\n학습 데이터 준비 중 (최근 {train_days}일)...")

        # NaN 제거
        df_clean = df.dropna()

        # 최근 30일 데이터만 사용
        df_train = df_clean.tail(train_days + 1)  # +1 for target

        # 마지막 행 제거
        df_train = df_train[:-1]

        print(f"학습 데이터 기간: {df_train.index[0].date()} ~ {df_train.index[-1].date()}")
        print(f"학습 데이터 크기: {len(df_train)}일")

        # 특성 선택
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

        print(f"학습 완료!")
        print(f"학습 데이터 정확도: {train_accuracy:.2%}")

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
        # 실제 존재하는 클래스만 사용
        unique_classes = sorted(y.unique())
        labels_map = {-1: '하락(-1)', 0: '보합(0)', 1: '상승(1)'}
        target_names = [labels_map[c] for c in unique_classes]
        print(classification_report(y, y_pred, labels=unique_classes, target_names=target_names, zero_division=0))

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
        plt.title('Tesla Stock Decision Tree\n(+/- $10 Next Day Movement)',
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"의사결정나무 저장: {output_file}")
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
        plt.title('Feature Importance - Top 15', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"특성 중요도 저장: {output_file}")
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

            if pred == 1:  # 상승 예측
                action = "BUY"
                profit = actual
            elif pred == -1:  # 하락 예측
                action = "SHORT"
                profit = -actual
            else:  # 보합 예측
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

        # 통계
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

        # 상위/하위 거래
        if len(trades_df) > 0:
            print("\n수익 상위 5개 거래:")
            top_trades = trades_df.nlargest(5, 'profit')
            for idx, trade in top_trades.iterrows():
                print(f"  {trade['date'].date()}: {trade['action']:6s} | "
                      f"예측={trade['prediction']:2d} | 실제=${trade['actual_change']:7.2f} | "
                      f"수익=${trade['profit']:7.2f}")

            print("\n손실 상위 5개 거래:")
            worst_trades = trades_df.nsmallest(5, 'profit')
            for idx, trade in worst_trades.iterrows():
                print(f"  {trade['date'].date()}: {trade['action']:6s} | "
                      f"예측={trade['prediction']:2d} | 실제=${trade['actual_change']:7.2f} | "
                      f"수익=${trade['profit']:7.2f}")

        return trades_df

    def generate_trading_rules(self):
        """매매 규칙 요약"""
        print("\n" + "="*60)
        print("매매 규칙")
        print("="*60)

        print("\n의사결정나무 기반 매매 전략:")
        print("1. 예측 = 1 (상승) → 매수(BUY)")
        print("   - 다음날 주가가 $10 이상 오를 것으로 예측")
        print("   - 현재가에 매수 → 다음날 종가에 매도")

        print("\n2. 예측 = -1 (하락) → 공매도(SHORT)")
        print("   - 다음날 주가가 $10 이상 내릴 것으로 예측")
        print("   - 현재가에 공매도 → 다음날 종가에 청산")

        print("\n3. 예측 = 0 (보합) → 보유(HOLD)")
        print("   - 다음날 주가 변동이 ±$10 미만으로 예측")
        print("   - 거래하지 않음")

        print("\n주요 판단 기준:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, row in feature_importance.head(5).iterrows():
            print(f"  - {row['feature']}: {row['importance']*100:.1f}%")

    def save_results(self, df_train, trades_df, output_file='tesla_trading_results.csv'):
        """결과 저장"""
        print(f"\n결과 저장 중: {output_file}")

        result_df = df_train.copy()
        result_df['Prediction'] = self.model.predict(df_train[self.feature_names])
        result_df['Action'] = trades_df['action'].values
        result_df['Trade_Profit'] = trades_df['profit'].values

        result_df.to_csv(output_file)
        print(f"결과 저장 완료: {output_file}")


def main():
    """메인 실행 함수"""
    print("="*70)
    print("테슬라 주식 의사결정나무 매매 시스템 (데모)")
    print("="*70)
    print("\n주의: 실제 API 접근 제한으로 시뮬레이션 데이터를 사용합니다.")
    print("실제 환경에서는 yfinance로 실시간 데이터 수집이 가능합니다.\n")

    system = TeslaStockDecisionTree(threshold=10.0)

    try:
        # 1. 샘플 데이터 생성 (90일)
        df = system.generate_sample_data(days=90)

        # 2. 기술적 지표 생성
        df = system.create_technical_indicators(df)

        # 3. 타겟 변수 생성
        df = system.create_target_variable(df)

        # 4. 학습 데이터 준비 (최근 30일)
        X, y, df_train = system.prepare_training_data(df, train_days=30)

        # 5. 의사결정나무 학습
        system.train_decision_tree(X, y)

        # 6. 모델 평가
        feature_importance = system.evaluate_model(X, y)

        # 7. 의사결정나무 시각화
        system.visualize_tree('tesla_decision_tree.png')

        # 8. 특성 중요도 시각화
        system.visualize_feature_importance(feature_importance, 'feature_importance.png')

        # 9. 매매 시뮬레이션
        trades_df = system.simulate_trading(df_train, X, y)

        # 10. 매매 규칙
        system.generate_trading_rules()

        # 11. 결과 저장
        system.save_results(df_train, trades_df, 'tesla_trading_results.csv')

        print("\n" + "="*70)
        print("분석 완료!")
        print("="*70)
        print("\n생성된 파일:")
        print("  - tesla_decision_tree.png: 의사결정나무 시각화")
        print("  - feature_importance.png: 특성 중요도 그래프")
        print("  - tesla_trading_results.csv: 상세 거래 결과")

        print("\n\n실제 데이터 사용 방법:")
        print("  yfinance가 정상 작동하는 환경에서는")
        print("  tesla_stock_decision_tree.py 를 실행하세요.")

    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
