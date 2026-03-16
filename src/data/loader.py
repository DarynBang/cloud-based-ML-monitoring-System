import pandas as pd
import glob
import os
import logging
from typing import Dict, List
from schema import STOCK_SCHEMA

# Sử dụng logger của module hoặc logger tổng
logger = logging.getLogger("trading_system.data")

class StockLoader:
    """
    Module tải và xử lý dữ liệu chứng khoán với tối ưu bộ nhớ và logging.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(self.data_dir, "*.parquet"))
        self.df = None
        self.schema = STOCK_SCHEMA

    def load_all(self) -> pd.DataFrame:
        """Tải, tối ưu kiểu dữ liệu và gộp tất cả dữ liệu."""
        if not self.files:
            logger.error(f"Không tìm thấy file parquet nào trong thư mục: {self.data_dir}")
            return pd.DataFrame()

        dataframes = []
        logger.info(f"Bắt đầu nạp {len(self.files)} tệp từ {self.data_dir}")
        
        for f in self.files:
            try:
                temp_df = pd.read_parquet(f, engine='pyarrow')
                temp_df = temp_df.astype(self.schema)
                dataframes.append(temp_df)
                logger.debug(f"Đã nạp tệp: {os.path.basename(f)}")
            except Exception as e:
                logger.error(f"Lỗi khi đọc tệp {f}: {e}")

        logger.info("Đang hợp nhất dữ liệu...")
        self.df = pd.concat(dataframes, ignore_index=True)
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """
        Thực hiện làm sạch dữ liệu: Xử lý NaN và Sắp xếp.
        """
        if self.df is None or self.df.empty:
            logger.warning("Không có dữ liệu để làm sạch.")
            return pd.DataFrame()

        logger.info("Bắt đầu quy trình làm sạch dữ liệu...")

        # 1. Sắp xếp theo Symbol và Date
        self.df = self.df.sort_values(by=["symbol", "date"]).reset_index(drop=True)
        
        # 2. Xử lý giá trị thiếu (NaN)
        initial_rows = len(self.df)
        
        # Kiểm tra NaN
        nan_counts = self.df.isna().sum()
        if nan_counts.any():
            logger.info(f"Phát hiện giá trị thiếu: \n{nan_counts[nan_counts > 0]}")
            
            # Chiến thuật: Forward fill và Backward fill theo từng mã chứng khoán
            # Sử dụng group_keys=False để tránh tạo thêm Index từ cột groupby
            self.df = self.df.groupby("symbol", observed=True, group_keys=False).apply(
                lambda x: x.ffill().bfill()
            )
            
            # Reset index để đảm bảo symbol quay lại làm cột nếu bị biến thành index
            self.df = self.df.reset_index(drop=True)
            
            # Nếu vẫn còn NaN (ví dụ mã đó toàn bộ là NaN), ta sẽ drop
            self.df = self.df.dropna()
            
            final_rows = len(self.df)
            if initial_rows != final_rows:
                logger.warning(f"Đã loại bỏ {initial_rows - final_rows} dòng không thể xử lý NaN.")

        logger.info("Hoàn tất làm sạch dữ liệu.")
        return self.df

    def save_processed(self, output_path: str):
        """Lưu dữ liệu đã tiền xử lý ra tệp parquet mới."""
        if self.df is None or self.df.empty:
            logger.error("Không có dữ liệu để lưu.")
            return

        try:
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self.df.to_parquet(output_path, engine='pyarrow', index=False)
            logger.info(f"Đã lưu dữ liệu tiền xử lý tại: {output_path}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu: {e}")

    def get_memory_usage(self):
        """Trả về mức sử dụng bộ nhớ của DataFrame hiện tại (MB)."""
        if self.df is not None:
            return self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        return 0
