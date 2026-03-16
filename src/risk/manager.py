import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("trading_system.risk")

from utils.config_loader import CONFIG

class RiskManager:
    """
    Quản trị rủi ro: Stop-loss, Position Sizing và Portfolio Metrics.
    """
    def __init__(self, 
                 stop_loss_pct: float = None, 
                 max_capital_per_trade: float = None,
                 risk_per_trade_pct: float = None,
                 trailing_stop_pct: float = None):
        
        risk_cfg = CONFIG.get("risk", {})
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else risk_cfg.get("stop_loss_pct", 0.05)
        self.max_capital_per_trade = max_capital_per_trade if max_capital_per_trade is not None else risk_cfg.get("max_capital_per_trade", 0.1)
        self.risk_per_trade_pct = risk_per_trade_pct if risk_per_trade_pct is not None else risk_cfg.get("risk_per_trade_pct", 0.01)
        self.trailing_stop_pct = trailing_stop_pct if trailing_stop_pct is not None else risk_cfg.get("trailing_stop_pct", 0.03)
        self.var_confidence = risk_cfg.get("var_confidence_level", 0.95)

    def calculate_position_size(self, current_balance: float, price: float) -> float:
        """
        Tính số lượng cổ phiếu có thể mua dựa trên quy tắc quản lý vốn.
        Sử dụng Fixed Fractional Position Sizing.
        """
        # Giới hạn số tiền đầu tư cho một mã cổ phiếu
        max_investment = current_balance * self.max_capital_per_trade
        
        # Số lượng cổ phiếu = Vốn đầu tư / Giá
        num_shares = max_investment / price
        return num_shares

    def check_stop_loss(self, entry_price: float, current_price: float, peak_price: float = 0) -> bool:
        """
        Kiểm tra cắt lỗ. Hỗ trợ cả Stop Loss cố định và Trailing Stop (từ đỉnh giá).
        """
        if entry_price == 0: return False
        
        # 1. Cắt lỗ cố định (từ giá mua)
        fixed_loss = (entry_price - current_price) / entry_price
        if fixed_loss >= self.stop_loss_pct:
            return True
            
        # 2. Trailing Stop (từ giá đỉnh cao nhất đạt được sau khi vào lệnh)
        if peak_price > entry_price:
            trailing_loss = (peak_price - current_price) / peak_price
            if trailing_loss >= self.trailing_stop_pct:
                return True
                
        return False

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Tính Value at Risk (VaR) theo phương pháp lịch sử.
        Ví dụ: VaR 95% = Mức lỗ mà 95% trường hợp thực tế không vượt quá.
        """
        if returns.empty: return 0
        return np.percentile(returns, (1 - confidence_level) * 100)

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Tính chỉ số Sharpe Ratio (thường niên)."""
        if returns.std() == 0: return 0
        mean_return = returns.mean() * 252 # Giả định 252 ngày giao dịch/năm
        std_return = returns.std() * np.sqrt(252)
        return (mean_return - risk_free_rate) / std_return

    @staticmethod
    def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
        """Tính độ sụt giảm tài sản lớn nhất (Max Drawdown)."""
        peak = portfolio_values.cummax()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()
