import pandas as pd
import numpy as np

class TechnicalAnalysis:
    @staticmethod
    def get_indicators(df):
        """Tüm temel teknik göstergeleri hesaplar."""
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bantları
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['StdDev'] = df['Close'].rolling(window=20).std()
        df['Upper_BB'] = df['SMA20'] + (df['StdDev'] * 2)
        df['Lower_BB'] = df['SMA20'] - (df['StdDev'] * 2)

        return df
