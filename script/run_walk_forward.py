import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import random
from tabulate import tabulate

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import setup_logger
from utils.config_loader import CONFIG
from src.models.ml_pipeline import get_ml_config, load_artifact
from src.models.continual_learning import get_time_windows, ModelManager, run_window_training
from src.strategy.ml_strategies import MLStrategy
from src.backtest.engine import VectorizedBacktester

# Thiết lập logger
logger = setup_logger(name="trading_system.walk_forward", level=CONFIG.get("logging", {}).get("level", "INFO"))

def main():
    print("
" + "="*60)
    print(f"{'WALK-FORWARD CONTINUAL LEARNING SIMULATION':^60}")
    print("="*60)

    # 1. Cấu hình Continual Learning
    cl_cfg = CONFIG.get("continual_learning", {})
    train_months = int(cl_cfg.get("train_window_months", 12))
    test_months = int(cl_cfg.get("test_window_months", 3))
    step_months = int(cl_cfg.get("step_months", 3))
    
    ml_cfg = get_ml_config(CONFIG)
    ml_policy = ml_cfg.get("policy", {})

    # 2. Tải dữ liệu
    feature_file = CONFIG["paths"]["feature_file"]
    if not os.path.exists(feature_file):
        logger.error(f"Feature file not found: {feature_file}")
        return

    logger.info(f"Loading features from {feature_file}...")
    df_full = pd.read_parquet(feature_file)
    df_full["date"] = pd.to_datetime(df_full["date"])
    
    # Giới hạn số lượng symbols để tăng tốc mô phỏng
    sample_size = int(cl_cfg.get("symbols_sample_size", 50))
    all_symbols = list(df_full["symbol"].unique())
    random.seed(CONFIG.get("backtest", {}).get("random_seed", 42))
    sample_symbols = random.sample(all_symbols, min(sample_size, len(all_symbols)))
    df_sample = df_full[df_full["symbol"].isin(sample_symbols)].sort_values(["symbol", "date"]).copy()

    # 3. Chia cửa sổ thời gian
    start_date = df_sample["date"].min()
    end_date = df_sample["date"].max()
    
    windows = get_time_windows(
        start_date, end_date, 
        train_months=train_months, 
        test_months=test_months, 
        step_months=step_months
    )
    
    if not windows:
        logger.error("No windows found. Check your date range and window sizes.")
        return

    logger.info(f"Total windows to simulate: {len(windows)}")
    
    model_manager = ModelManager(output_root="models/walk_forward")
    all_test_signals = []

    # 4. Vòng lặp Walk-Forward
    for i, window in enumerate(windows):
        win_id = f"Window {i+1}/{len(windows)}"
        logger.info(f"[*] {win_id}: Testing from {window['test_start'].date()} to {window['test_end'].date()}")
        
        # A. Huấn luyện model Challenger cho cửa sổ này
        artifact_dir, metadata = run_window_training(df_sample, window, ml_cfg, model_manager)
        
        if not artifact_dir:
            logger.warning(f"Skipping {win_id} due to training failure.")
            continue
            
        # B. Dự báo (Backtest) trên cửa sổ Test kế tiếp
        test_df = df_sample[
            (df_sample["date"] >= window["test_start"]) & 
            (df_sample["date"] <= window["test_end"])
        ].copy()
        
        if test_df.empty:
            logger.warning(f"No test data for {win_id}")
            continue
            
        # Tạo strategy sử dụng model vừa train
        strategy = MLStrategy(
            artifact_dir=artifact_dir,
            p_buy=float(ml_policy.get("p_buy", 0.55)),
            p_sell=float(ml_policy.get("p_sell", 0.55)),
            min_conf_gap=float(ml_policy.get("min_conf_gap", 0.05)),
            name=f"WF_Model_{i}"
        )
        
        # C. Sinh tín hiệu
        df_win_signals = strategy.generate_signals(test_df)
        all_test_signals.append(df_win_signals)
        
        logger.info(f"   -> Accuracy on Test window: {metadata['metrics'].get('accuracy', 0):.2%}")

    if not all_test_signals:
        logger.error("No signals generated.")
        return

    # 5. Gộp tín hiệu và chạy Backtest tổng thể
    df_final_signals = pd.concat(all_test_signals).sort_values(["symbol", "date"])
    
    initial_balance = CONFIG.get("risk", {}).get("initial_balance", 10000)
    backtester = VectorizedBacktester(initial_balance=initial_balance)
    
    logger.info("Running final backtest on aggregated Walk-Forward signals...")
    results_df = backtester.run(df_final_signals)
    metrics = backtester.calculate_metrics(results_df)

    # 6. Hiển thị kết quả so sánh
    print("
" + "="*60)
    print(f"{'WALK-FORWARD PERFORMANCE SUMMARY':^60}")
    print("="*60)
    
    summary = [
        ["Total ROI %", f"{metrics['total_roi_pct']:.2f}%"],
        ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"],
        ["Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%"],
        ["Total Trades", f"{int(metrics.get('total_trades', 0)):,}"],
        ["Avg Final Value", f"${metrics['avg_final_value']:,.2f}"],
        ["Simulated Period", f"{df_final_signals['date'].min().date()} to {df_final_signals['date'].max().date()}"]
    ]
    print(tabulate(summary, tablefmt="psql"))
    print("="*60 + "
")

if __name__ == "__main__":
    main()
