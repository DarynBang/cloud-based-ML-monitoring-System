import sys
import os
import pandas as pd
import logging
import random
import questionary
from tabulate import tabulate

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import setup_logger
from utils.config_loader import CONFIG
from src.strategy.simple_strategies import BuyAndHoldStrategy, DCAStrategy
from src.strategy.technical_strategies import MACrossoverStrategy, RSIMomentumStrategy, BreakoutStrategy, SupportResistanceStrategy
from src.strategy.ml_strategies import MLStrategy, HybridMLStrategy
from src.backtest.engine import VectorizedBacktester
from src.risk.manager import RiskManager
from utils.cache_manager import BacktestCache
from datetime import datetime
import calendar

# Ensure Unicode output works on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thiết lập logger (chỉ in ra file hoặc mức WARNING để tránh làm nhiễu UI)
logger = setup_logger(name="trading_system.backtest", level=logging.WARNING)

# Khởi tạo Cache
cache = BacktestCache()

def resolve_ml_artifact_dir() -> str:
    ml_cfg = CONFIG.get("ml", {}) or {}
    direct_path = ml_cfg.get("artifact_dir")
    if direct_path and os.path.exists(direct_path):
        return direct_path

    root = ml_cfg.get("model_output_dir", "models")
    if not os.path.exists(root):
        return None
    subdirs = [
        os.path.join(root, d)
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    latest = subdirs[0]
    if os.path.exists(os.path.join(latest, "model.joblib")) and os.path.exists(os.path.join(latest, "metadata.json")):
        return latest
    return None

def parse_flexible_date(date_str: str, is_end: bool = False) -> pd.Timestamp:
    """
    Parse ngày từ các định dạng linh hoạt: DD/MM/YYYY, MM/YYYY, YYYY.
    """
    if not date_str or not date_str.strip():
        return None
    
    parts = date_str.strip().split('/')
    try:
        if len(parts) == 1: # YYYY
            year = int(parts[0])
            if is_end:
                return pd.Timestamp(year, 12, 31)
            else:
                return pd.Timestamp(year, 1, 1)
        elif len(parts) == 2: # MM/YYYY
            month, year = int(parts[0]), int(parts[1])
            if is_end:
                last_day = calendar.monthrange(year, month)[1]
                return pd.Timestamp(year, month, last_day)
            else:
                return pd.Timestamp(year, month, 1)
        elif len(parts) == 3: # DD/MM/YYYY
            day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            return pd.Timestamp(year, month, day)
    except Exception as e:
        logger.error(f"Lỗi parse ngày '{date_str}': {e}")
    return None

def run_experiment(df_sample, strategy, initial_balance, sample_symbols, interval=None):
    # Kiểm tra Cache
    cache_key = cache.generate_key(CONFIG, sample_symbols, strategy.name, interval)
    cached_result = cache.get(cache_key)
    
    if cached_result:
        return cached_result, True # Trả về kèm flag là từ cache

    # Nếu không có cache, chạy backtest mới
    df_with_signals = strategy.generate_signals(df_sample.copy())
    backtester = VectorizedBacktester(initial_balance=initial_balance)
    results_df = backtester.run(df_with_signals)
    metrics = backtester.calculate_metrics(results_df)
    
    # Lưu vào cache
    cache.set(cache_key, metrics)
    return metrics, False

def main():
    print("\n" + "="*60)
    print(f"{'TRADING SYSTEM BACKTEST INTERFACE':^60}")
    print("="*60)
    
    # Hỗ trợ chạy không tương tác qua tham số dòng lệnh
    is_all_mode = "--all" in sys.argv
    
    # Thiết lập seed để chọn mẫu ổn định (cho phép hit cache)
    random.seed(CONFIG.get("backtest", {}).get("random_seed", 42))
    
    feature_file = CONFIG["paths"]["feature_file"]
    if not os.path.exists(feature_file):
        print(f"Error: Không tìm thấy file {feature_file}. Hãy chạy extract_features.py trước.")
        return

    # 1. Danh sách các chiến lược có sẵn
    available_strategies = {
        "Buy & Hold": BuyAndHoldStrategy(),
        "DCA (Dollar Cost Averaging)": DCAStrategy(buy_interval=CONFIG.get("strategies", {}).get("dca_interval", 20)),
        "MA Crossover (SMA 20/50)": MACrossoverStrategy(20, 50),
        "RSI Momentum (30/70)": RSIMomentumStrategy(14, 30, 70),
        "Breakout (20-day High)": BreakoutStrategy(20),
        "Support & Resistance (50-day Range)": SupportResistanceStrategy(50)
    }

    ml_cfg = CONFIG.get("ml", {}) or {}
    if ml_cfg.get("enabled", True):
        ml_policy = ml_cfg.get("policy", {}) or {}
        ml_artifact_dir = resolve_ml_artifact_dir()
        if ml_artifact_dir:
            available_strategies["ML Strategy (3-class)"] = MLStrategy(
                artifact_dir=ml_artifact_dir,
                p_buy=float(ml_policy.get("p_buy", 0.55)),
                p_sell=float(ml_policy.get("p_sell", 0.55)),
                min_conf_gap=float(ml_policy.get("min_conf_gap", 0.05)),
            )
            available_strategies["Hybrid ML + MA Filter"] = HybridMLStrategy(
                artifact_dir=ml_artifact_dir,
                p_buy=float(ml_policy.get("p_buy", 0.55)),
                p_sell=float(ml_policy.get("p_sell", 0.55)),
                min_conf_gap=float(ml_policy.get("min_conf_gap", 0.05)),
                trend_col=str(ml_policy.get("trend_col", "sma_50")),
            )
            print(f"[*] ML artifact loaded: {ml_artifact_dir}")
        else:
            print("[*] ML artifact not found. Run: python script/train_ml_model.py")
    
    strategy_names = list(available_strategies.keys())
    choices = ["All Strategies"] + strategy_names
    
    # 2. Giao diện chọn chiến lược (hoặc tự động chọn nếu có --all)
    if is_all_mode:
        selected_name = "All Strategies"
        print("[*] Chế độ tự động: Đang chạy tất cả các chiến lược...")
    else:
        selected_name = questionary.select(
            "Chọn chiến lược muốn Backtest (Sử dụng phím mũi tên và Enter):",
            choices=choices
        ).ask()

    if not selected_name:
        return

    # 3. Tải dữ liệu và chọn mẫu
    print(f"\n[*] Đang nạp dữ liệu từ {feature_file}...")
    df_full = pd.read_parquet(feature_file)
    df_full["date"] = pd.to_datetime(df_full["date"])
    
    # Lấy min/max date của dữ liệu
    data_min_date = df_full["date"].min()
    data_max_date = df_full["date"].max()
    
    print(f"[*] Available interval: {data_min_date.strftime('%d/%m/%Y')} to {data_max_date.strftime('%d/%m/%Y')}")

    # 4. Giao diện chọn khoảng thời gian
    if is_all_mode:
        start_date = data_min_date
        end_date = data_max_date
    else:
        print("\nNhập khoảng thời gian (Enter để chọn Full Interval):")
        start_input = questionary.text(
            f"Start Time (DD/MM/YYYY, MM/YYYY, YYYY) [Mặc định: {data_min_date.strftime('%d/%m/%Y')}]:"
        ).ask()
        
        end_input = questionary.text(
            f"End Time (DD/MM/YYYY, MM/YYYY, YYYY) [Mặc định: {data_max_date.strftime('%d/%m/%Y')}]:"
        ).ask()

        start_date = parse_flexible_date(start_input, is_end=False) or data_min_date
        end_date = parse_flexible_date(end_input, is_end=True) or data_max_date

    # 5. Lọc dữ liệu theo thời gian và chọn mẫu
    all_symbols = list(df_full["symbol"].unique())
    sample_size = CONFIG.get("backtest", {}).get("sample_size", 50)
    sample_symbols = random.sample(all_symbols, min(sample_size, len(all_symbols)))
    
    # Lọc mẫu mã và khoảng thời gian
    df_sample = df_full[
        (df_full["symbol"].isin(sample_symbols)) & 
        (df_full["date"] >= start_date) & 
        (df_full["date"] <= end_date)
    ].copy()
    
    print(f"[*] Đã chọn {len(sample_symbols)} mã. Khoảng thời gian thực tế: {start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')}")

    # 6. Quyết định danh sách chiến lược chạy
    to_run = []
    if selected_name == "All Strategies":
        to_run = list(available_strategies.values())
    else:
        to_run = [available_strategies[selected_name]]

    # 7. Hiển thị thông tin cấu hình
    risk_m = RiskManager()
    initial_balance = CONFIG.get("risk", {}).get("initial_balance", 10000)
    
    tc = CONFIG.get("trading_costs", {}) or {}
    fee_pct = float(tc.get("fee_pct", 0.0))
    slippage_bps = float(tc.get("slippage_bps", 0.0))

    config_info = [
        ["Initial Balance", f"${initial_balance:,.2f}"],
        ["Stop Loss", f"{risk_m.stop_loss_pct*100}%"],
        ["Trailing Stop", f"{risk_m.trailing_stop_pct*100}%"],
        ["VaR Confidence", f"{risk_m.var_confidence*100}%"],
        ["Fee", f"{fee_pct*100:.2f}%"],
        ["Slippage", f"{slippage_bps:g} bps"],
        ["Interval", f"{start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}"],
        ["Max Capital/Trade", f"{risk_m.max_capital_per_trade*100}%"],
        ["Sample Size", f"{len(sample_symbols)} symbols"],
        ["DCA Interval", f"{CONFIG.get('strategies', {}).get('dca_interval', 20)} days"]
    ]
    print("\n" + "-"*40)
    print(f"{'SYSTEM CONFIGURATION':^40}")
    print("-"*40)
    print(tabulate(config_info, tablefmt="plain"))
    print("-"*40 + "\n")

    # 8. Thực thi
    summary = []
    actual_interval = (start_date, end_date)
    
    for strat in to_run:
        metrics, is_cached = run_experiment(df_sample, strat, initial_balance, sample_symbols, actual_interval)
        status = "[Cached]" if is_cached else "[New Run]"
        print(f"[*] {status} Thực thi: {strat.name}...")
        
        summary.append({
            "Strategy": strat.name,
            "ROI %": f"{metrics['total_roi_pct']:.2f}%",
            "Sharpe": f"{metrics['sharpe_ratio']:.2f}",
            "MaxDD": f"{metrics['max_drawdown']*100:.2f}%",
            "VaR": f"{metrics['var']*100:.2f}%",
            "Trades": f"{int(metrics.get('total_trades', 0)):,}",
            "Final Value": f"${metrics['avg_final_value']:,.2f}",
            "Cache": "YES" if is_cached else "NO",
            "_roi_sort": metrics['total_roi_pct']  # Trường ẩn để sắp xếp
        })

    # Sắp xếp theo ROI giảm dần
    summary.sort(key=lambda x: x["_roi_sort"], reverse=True)
    
    # Xóa trường ẩn trước khi hiển thị
    for item in summary:
        del item["_roi_sort"]

    # 9. Hiển thị kết quả
    print("\n" + "="*80)
    print(f"{'KẾT QUẢ BACKTEST CHI TIẾT':^80}")
    print("="*80)
    
    # Sử dụng tabulate để in bảng đẹp
    print(tabulate(summary, headers="keys", tablefmt="psql", numalign="right", stralign="center"))
    
    # 10. Chú thích các chỉ số
    print("\n[ CHÚ THÍCH CHỈ SỐ ]")
    glossary = [
        ["ROI %", "Tỷ suất lợi nhuận trên vốn đầu tư (Return on Investment)."],
        ["Sharpe", "Chỉ số Sharpe: Đo lường lợi nhuận trên một đơn vị rủi ro (Càng cao càng tốt)."],
        ["MaxDD", "Độ sụt giảm tài sản lớn nhất (Maximum Drawdown): Đo lường mức rủi ro tệ nhất."],
        ["VaR", f"Value at Risk (Độ tin cậy {risk_m.var_confidence*100}%): Mức lỗ tối đa trong điều kiện bình thường."],
        ["Trades", "Tong so giao dich duoc thuc thi (mua + ban) trong toan bo giai doan backtest."],
        ["Final Value", "Giá trị tài sản trung bình cuối kỳ của mỗi mã cổ phiếu."],
        ["Cache", "Kết quả được lấy từ bộ nhớ đệm (YES) hoặc tính toán mới (NO)."]
    ]
    print(tabulate(glossary, tablefmt="plain"))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()


