import sys
import os
import pandas as pd

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import setup_logger
from utils.config_loader import CONFIG
from src.features.indicators import apply_all_features

# Thiết lập logger
logger = setup_logger(name="trading_system.features", level=CONFIG.get("logging", {}).get("level", "INFO"))

def main():
    """
    Quy trình trích xuất đặc trưng kỹ thuật với hỗ trợ cập nhật tăng dần (Incremental).
    """
    logger.info("--- Khởi chạy quy trình trích xuất đặc trưng (Giai đoạn 2/6) ---")
    
    input_file = CONFIG["paths"]["processed_file"]
    output_file = CONFIG["paths"]["feature_file"]
    
    # Hỗ trợ Incremental Update
    is_incremental = os.path.exists(output_file)
    
    if not os.path.exists(input_file):
        logger.error(f"Không tìm thấy file dữ liệu đã làm sạch tại: {input_file}")
        return

    # 1. Tải dữ liệu mới
    df_new = pd.read_parquet(input_file)
    df_new["date"] = pd.to_datetime(df_new["date"])
    
    if is_incremental:
        logger.info(f"Phát hiện Feature Store hiện tại. Đang kiểm tra dữ liệu mới...")
        df_old = pd.read_parquet(output_file, columns=["symbol", "date"])
        df_old["date"] = pd.to_datetime(df_old["date"])
        
        # Tìm ngày cuối cùng của mỗi mã
        last_dates = df_old.groupby("symbol")["date"].max()
        
        # Chỉ lấy những dòng sau ngày cuối cùng (cần lookback để tính indicator)
        # Để SMA 50 chính xác, cần ít nhất 50 dòng lookback
        lookback_limit = 60 
        
        def filter_new_data(group):
            symbol = group.name
            if symbol in last_dates:
                last_dt = last_dates[symbol]
                # Lấy dữ liệu mới + lookback
                new_data = group[group["date"] > last_dt]
                if new_data.empty:
                    return pd.DataFrame()
                
                # Để tính đúng indicator cho dòng mới đầu tiên, cần 60 dòng trước đó
                all_data_for_symbol = group.sort_values("date")
                idx = all_data_for_symbol[all_data_for_symbol["date"] > last_dt].index[0]
                start_idx = max(0, all_data_for_symbol.index.get_loc(idx) - lookback_limit)
                return all_data_for_symbol.iloc[start_idx:]
            return group

        df_to_process = df_new.groupby("symbol", group_keys=False).apply(filter_new_data)
        
        if df_to_process.empty:
            logger.info("Không có dữ liệu mới cần cập nhật.")
            return
            
        logger.info(f"Cập nhật tăng dần cho {len(df_to_process)} dòng dữ liệu.")
        df_features = apply_all_features(df_to_process, CONFIG)
        
        # Chỉ lấy phần thực sự mới để append (tránh trùng lặp do lookback)
        def get_only_new(group):
            symbol = group.name
            if symbol in last_dates:
                return group[group["date"] > last_dates[symbol]]
            return group
            
        df_new_only = df_features.groupby("symbol", group_keys=False).apply(get_only_new)
        
        # Gộp vào file cũ
        df_final = pd.concat([pd.read_parquet(output_file), df_new_only]).drop_duplicates(["symbol", "date"])
        logger.info(f"Đã gộp thêm {len(df_new_only)} dòng mới vào Feature Store.")
    else:
        # Xử lý toàn bộ nếu chưa có file output
        logger.info("Bắt đầu tính toán toàn bộ đặc trưng (Full Refresh)...")
        df_final = apply_all_features(df_new, CONFIG)
    
    # 3. Lưu dữ liệu
    logger.info(f"Đang lưu dữ liệu tại: {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_parquet(output_file, engine='pyarrow', index=False)

    # 4. Hiển thị thông tin kiểm tra
    logger.info(f"Hoàn tất! Tổng số dòng: {len(df)}")
    
    # In một vài dòng cuối (thường có đủ dữ liệu tính toán hơn dòng đầu)
    print("\n--- 5 dòng cuối cùng của dữ liệu kèm đặc trưng ---")
    cols_to_show = ["symbol", "date", "close", "sma_20", "sma_50", "rsi_14", "macd", "macd_hist"]
    # Kiểm tra xem các cột này có tồn tại không (đề phòng config đổi tên)
    available_cols = [c for c in cols_to_show if c in df.columns]
    print(df[available_cols].tail())

if __name__ == "__main__":
    main()
