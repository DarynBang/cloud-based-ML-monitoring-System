import sys
import os
import logging

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import setup_logger
from utils.config_loader import CONFIG
from src.data.loader import StockLoader

# Thiết lập logger từ config
logger = setup_logger(name="trading_system", level=CONFIG.get("logging", {}).get("level", "INFO"))

def main():
    """
    Business logic cho việc tiền xử lý dữ liệu.
    """
    logger.info("--- Khởi chạy quy trình tiền xử lý ---")
    
    # Lấy đường dẫn từ config
    raw_dir = CONFIG["paths"]["raw_dir"]
    output_file = CONFIG["paths"]["processed_file"]

    # 1. Khởi tạo loader (Nạp từ thư mục raw được cấu hình)
    loader = StockLoader(data_dir=raw_dir)
    
    # 2. Tải dữ liệu
    df = loader.load_all()
    
    if df.empty:
        logger.error(f"Quy trình dừng lại. Hãy đảm bảo có file parquet trong: {raw_dir}")
        return

    # 3. Làm sạch dữ liệu
    df = loader.clean_data()

    # 4. Lưu dữ liệu đã xử lý
    loader.save_processed(output_file)

    # 5. Báo cáo kết quả
    mem_usage = loader.get_memory_usage()
    logger.info(f"Hoàn tất! Tổng số dòng: {len(df)}")
    logger.info(f"Mức sử dụng bộ nhớ: {mem_usage:.2f} MB")
    
    print("\n--- 5 dòng đầu tiên của dữ liệu sạch ---")
    print(df.head())

if __name__ == "__main__":
    main()
