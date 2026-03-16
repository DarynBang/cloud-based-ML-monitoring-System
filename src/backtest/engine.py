import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("trading_system.backtest")

from src.risk.manager import RiskManager
from utils.config_loader import CONFIG

class VectorizedBacktester:
    """
    Hệ thống Backtest có tích hợp Quản trị rủi ro.
    """
    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_manager: RiskManager = None,
        fee_pct: float = None,
        slippage_bps: float = None,
    ):
        self.initial_balance = initial_balance
        self.risk_manager = risk_manager if risk_manager else RiskManager()
        tc = CONFIG.get("trading_costs", {}) or {}
        self.fee_pct = float(tc.get("fee_pct", 0.0)) if fee_pct is None else float(fee_pct)
        self.slippage_bps = float(tc.get("slippage_bps", 0.0)) if slippage_bps is None else float(slippage_bps)
        if self.fee_pct < 0:
            self.fee_pct = 0.0
        if self.slippage_bps < 0:
            self.slippage_bps = 0.0

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thực hiện backtest với quản lý vị thế và cắt lỗ.
        """
        logger.info(f"Bắt đầu Backtest với quản trị rủi ro (Stop Loss: {self.risk_manager.stop_loss_pct*100}%)...")
        
        df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)

        def _backtest_group(group):
            cash = np.zeros(len(group))
            shares = np.zeros(len(group))
            portfolio_value = np.zeros(len(group))
            
            curr_cash = self.initial_balance
            curr_shares = 0
            entry_price = 0
            peak_price = 0 # Giá cao nhất đạt được kể từ khi vào vị thế
            
            prices = group["close"].values
            signals = group["signal"].values
            fee_pct = self.fee_pct
            slippage = self.slippage_bps / 10000.0
            
            for i in range(len(group)):
                price = prices[i]
                signal = signals[i]
                
                # Cập nhật peak_price nếu đang nắm giữ cổ phiếu
                if curr_shares > 0:
                    peak_price = max(peak_price, price)

                # 1. Kiểm tra Stop Loss / Trailing Stop
                if curr_shares > 0 and self.risk_manager.check_stop_loss(entry_price, price, peak_price):
                    exec_price = price * (1 - slippage)
                    curr_cash += curr_shares * exec_price * (1 - fee_pct)
                    curr_shares = 0
                    entry_price = 0
                    peak_price = 0

                # 2. Thực thi tín hiệu
                if signal == 1 and curr_cash > 0: # BUY
                    exec_price = price * (1 + slippage)
                    all_in_cost_per_share = exec_price * (1 + fee_pct)
                    ideal_shares = self.risk_manager.calculate_position_size(
                        curr_cash + (curr_shares * price),
                        all_in_cost_per_share,
                    )
                    max_affordable_shares = curr_cash / all_in_cost_per_share if all_in_cost_per_share > 0 else 0
                    buy_shares = min(ideal_shares, max_affordable_shares)
                    
                    if buy_shares > 0:
                        # Tính giá entry trung bình (giả định mua thêm)
                        total_cost = (curr_shares * entry_price) + (buy_shares * all_in_cost_per_share)
                        curr_shares += buy_shares
                        curr_cash -= buy_shares * all_in_cost_per_share
                        entry_price = total_cost / curr_shares
                        peak_price = max(peak_price, price)
                
                elif signal == -1 and curr_shares > 0: # SELL
                    exec_price = price * (1 - slippage)
                    curr_cash += curr_shares * exec_price * (1 - fee_pct)
                    curr_shares = 0
                    entry_price = 0
                    peak_price = 0
                
                cash[i] = curr_cash
                shares[i] = curr_shares
                portfolio_value[i] = curr_cash + (curr_shares * price)
                
            return pd.DataFrame({
                "cash": cash,
                "shares": shares,
                "total": portfolio_value
            }, index=group.index)

        results = df.groupby("symbol", observed=True).apply(_backtest_group).reset_index(level=0, drop=True)
        df = pd.concat([df, results], axis=1)
        return df

    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Tính toán các chỉ số hiệu quả nâng cao."""
        if df.empty or "total" not in df.columns:
            return {
            "initial_balance": None,
            "avg_final_value": None,
            "total_roi_pct": None,
            "max_drawdown": None,
            "sharpe_ratio": None,
            "var": None,
            "total_trades": 0,
            "best_symbol": None,
            "worst_symbol": None
        }

        final_values = df.groupby("symbol", observed=True)["total"].last()
        shares_delta = df.groupby("symbol", observed=True)["shares"].diff().fillna(df["shares"])
        total_trades = int((shares_delta.abs() > 1e-12).sum())
        
        # Lợi nhuận hàng ngày để tính Sharpe/VaR
        daily_returns = df.groupby('date')['total'].mean().pct_change().fillna(0)
        
        metrics = {
            "initial_balance": self.initial_balance,
            "avg_final_value": final_values.mean(),
            "total_roi_pct": ((final_values.mean() / self.initial_balance) - 1) * 100,
            "max_drawdown": self.risk_manager.calculate_max_drawdown(df.groupby('date')['total'].mean()),
            "sharpe_ratio": self.risk_manager.calculate_sharpe_ratio(daily_returns),
            "var": self.risk_manager.calculate_var(daily_returns, self.risk_manager.var_confidence),
            "total_trades": total_trades,
            "best_symbol": final_values.idxmax(),
            "worst_symbol": final_values.idxmin()
        }
        return metrics
