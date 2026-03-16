import pandas as pd
import numpy as np
from src.strategy.base import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    """
    Chiến lược giao cắt đường trung bình động (Golden Cross / Death Cross).
    """
    def __init__(self, short_window: int = 20, long_window: int = 50):
        super().__init__(name=f"MA_Crossover_{short_window}_{long_window}")
        self.short_col = f"sma_{short_window}"
        self.long_col = f"sma_{long_window}"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)
        df["signal"] = 0
        
        # Xác định vị thế: 1 nếu ngắn > dài, ngược lại 0
        df["position"] = (df[self.short_col] > df[self.long_col]).astype(int)
        
        # Tín hiệu là sự thay đổi của vị thế giữa 2 phiên liên tiếp
        # 1: Ngắn cắt lên Dài (BUY), -1: Ngắn cắt xuống Dài (SELL)
        df["signal"] = df.groupby("symbol", observed=True)["position"].diff().fillna(0)
        
        return df.drop(columns=["position"])

class RSIMomentumStrategy(BaseStrategy):
    """
    Chiến lược Momentum dựa trên chỉ số RSI.
    """
    def __init__(self, period: int = 14, low: int = 30, high: int = 70):
        super().__init__(name=f"RSI_Momentum_{low}_{high}")
        self.rsi_col = f"rsi_{period}"
        self.low = low
        self.high = high

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df["signal"] = 0
        
        # Mua khi quá bán
        df.loc[df[self.rsi_col] < self.low, "signal"] = 1
        # Bán khi quá mua
        df.loc[df[self.rsi_col] > self.high, "signal"] = -1
        
        return df

class BreakoutStrategy(BaseStrategy):
    """
    Chiến lược Breakout dựa trên kênh giá (Donchian Channel).
    """
    def __init__(self, window: int = 20):
        super().__init__(name=f"Breakout_{window}")
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)
        df["signal"] = 0
        
        # Tính High/Low cao nhất/thấp nhất trong window phiên
        group = df.groupby("symbol", observed=True)["close"]
        df["rolling_high"] = group.transform(lambda x: x.shift(1).rolling(window=self.window).max())
        df["rolling_low"] = group.transform(lambda x: x.shift(1).rolling(window=self.window).min())
        
        # Mua khi phá vỡ đỉnh, bán khi phá vỡ đáy
        df.loc[df["close"] > df["rolling_high"], "signal"] = 1
        df.loc[df["close"] < df["rolling_low"], "signal"] = -1
        
        return df.drop(columns=["rolling_high", "rolling_low"])

class SupportResistanceStrategy(BaseStrategy):
    """
    Chiến lược Hỗ trợ & Kháng cự (Range Trading).
    Mua tại hỗ trợ, bán tại kháng cự.
    """
    def __init__(self, window: int = 20):
        super().__init__(name=f"Support_Resistance_{window}")
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)
        df["signal"] = 0
        
        # Tính Support (Min) và Resistance (Max)
        group = df.groupby("symbol", observed=True)["close"]
        df["support"] = group.transform(lambda x: x.shift(1).rolling(window=self.window).min())
        df["resistance"] = group.transform(lambda x: x.shift(1).rolling(window=self.window).max())
        
        # Mua khi giá chạm hoặc thấp hơn hỗ trợ (Mean Reversion)
        df.loc[df["close"] <= df["support"], "signal"] = 1
        # Bán khi giá chạm hoặc cao hơn kháng cự
        df.loc[df["close"] >= df["resistance"], "signal"] = -1
        
        return df.drop(columns=["support", "resistance"])
