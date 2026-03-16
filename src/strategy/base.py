import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict

class BaseStrategy(ABC):
    """
    Lớp cơ sở cho tất cả các chiến lược giao dịch.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nhận vào DataFrame dữ liệu (đã có features) và trả về DataFrame kèm cột 'signal'.
        Signal: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        pass
