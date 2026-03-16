import pandas as pd
import numpy as np
from src.strategy.base import BaseStrategy

class BuyAndHoldStrategy(BaseStrategy):
    """
    Chiến lược mua vào ngày đầu tiên và nắm giữ đến cuối cùng.
    """
    def __init__(self):
        super().__init__(name="Buy_and_Hold")

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gán tín hiệu 1 (BUY) cho ngày đầu tiên của mỗi mã cổ phiếu, 
        còn lại là 0 (HOLD).
        """
        # Sắp xếp theo symbol và date
        df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)
        
        # Khởi tạo cột signal = 0 (HOLD)
        df["signal"] = 0
        
        # Lấy index của các dòng đầu tiên của mỗi nhóm symbol
        first_indices = df.groupby("symbol", observed=True).head(1).index
        df.loc[first_indices, "signal"] = 1 # BUY
        
        return df

class DCAStrategy(BaseStrategy):
    """
    Chiến lược đầu tư định kỳ (Dollar Cost Averaging).
    Ví dụ: Cứ sau một khoảng thời gian (buy_interval) sẽ thực hiện mua thêm.
    """
    def __init__(self, buy_interval: int = 20):
        super().__init__(name="DCA")
        self.buy_interval = buy_interval

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sau mỗi buy_interval dòng (phiên giao dịch), gán tín hiệu 1 (BUY).
        """
        df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)
        df["signal"] = 0
        
        # Dùng phép chia lấy dư trên số thứ tự dòng của từng nhóm symbol
        # Cumcount bắt đầu từ 0
        group_counts = df.groupby("symbol", observed=True).cumcount()
        
        # Phát tín hiệu BUY mỗi khi số phiên giao dịch chia hết cho buy_interval
        df.loc[group_counts % self.buy_interval == 0, "signal"] = 1
        
        return df
