import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("trading_system.features")

class TechnicalIndicators:
    """
    Lớp cung cấp các phương thức tính toán chỉ số kỹ thuật với cơ chế caching.
    """

    @staticmethod
    def _is_computed(df: pd.DataFrame, col_name: str) -> bool:
        """Kiểm tra xem chỉ số đã được tính toán và có giá trị hợp lệ chưa."""
        if col_name in df.columns:
            # Nếu cột đã tồn tại và không phải toàn NaN
            if not df[col_name].isna().all():
                logger.info(f"Chỉ số '{col_name}' đã tồn tại. Bỏ qua tính toán.")
                return True
        return False

    @staticmethod
    def add_sma(df: pd.DataFrame, window: int, name: str = None) -> pd.DataFrame:
        """Tính toán Simple Moving Average (SMA)."""
        col_name = name if name else f"sma_{window}"
        
        if TechnicalIndicators._is_computed(df, col_name):
            return df

        logger.debug(f"Đang tính toán SMA {window}...")
        df[col_name] = df.groupby("symbol", observed=True)["close"].transform(
            lambda x: x.rolling(window=window).mean()
        )
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Tính toán Relative Strength Index (RSI)."""
        col_name = f"rsi_{window}"
        
        if TechnicalIndicators._is_computed(df, col_name):
            return df

        logger.debug(f"Đang tính toán RSI {window}...")

        def _rsi(series):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))

        df[col_name] = df.groupby("symbol", observed=True)["close"].transform(_rsi)
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Tính toán MACD (Moving Average Convergence Divergence)."""
        # Kiểm tra cột chính, nếu có 'macd' thì coi như đã tính bộ này
        if TechnicalIndicators._is_computed(df, "macd"):
            return df

        logger.debug(f"Đang tính toán MACD (fast={fast}, slow={slow}, signal={signal})...")

        def _macd(series):
            exp1 = series.ewm(span=fast, adjust=False).mean()
            exp2 = series.ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            return pd.DataFrame({
                "macd": macd_line,
                "macd_signal": signal_line,
                "macd_hist": histogram
            })

        macd_results = df.groupby("symbol", observed=True)["close"].apply(_macd).reset_index(level=0, drop=True)
        
        df["macd"] = macd_results["macd"]
        df["macd_signal"] = macd_results["macd_signal"]
        df["macd_hist"] = macd_results["macd_hist"]
        
        return df

def apply_all_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Hàm tiện ích áp dụng tất cả các đặc trưng dựa trên cấu hình config.yaml.
    """
    ti = TechnicalIndicators()
    f_config = config.get("features", {})
    
    # 1. SMA
    df = ti.add_sma(df, f_config.get("sma_short", 20))
    df = ti.add_sma(df, f_config.get("sma_long", 50))
    
    # 2. RSI
    df = ti.add_rsi(df, f_config.get("rsi_period", 14))
    
    # 3. MACD
    df = ti.add_macd(
        df, 
        fast=f_config.get("macd_fast", 12),
        slow=f_config.get("macd_slow", 26),
        signal=f_config.get("macd_signal", 9)
    )
    
    return df
