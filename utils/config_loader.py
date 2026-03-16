import yaml
import os

def load_config(config_path: str = "config.yaml"):
    """
    Tải cấu hình từ tệp YAML.
    """
    if not os.path.exists(config_path):
        # Trả về giá trị mặc định nếu không tìm thấy file
        return {
            "paths": {
                "raw_dir": "dataset/raw",
                "processed_dir": "dataset/processed",
                "processed_file": "dataset/processed/cleaned_stocks.parquet"
            }
        }
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Biến toàn cục để dùng nhanh
CONFIG = load_config()
