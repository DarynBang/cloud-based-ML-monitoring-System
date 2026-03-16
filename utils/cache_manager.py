import os
import json
import hashlib
import logging

logger = logging.getLogger("trading_system.cache")

class BacktestCache:
    """
    Quản lý bộ nhớ đệm cho kết quả backtest.
    """
    def __init__(self, cache_dir: str = ".gemini/cache/backtest"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def generate_key(self, config: dict, symbols: list, strategy_name: str, interval: tuple = None) -> str:
        """
        Tạo khóa duy nhất dựa trên config, danh sách symbol, chiến lược và khoảng thời gian.
        """
        # 1. Thông tin file dữ liệu (Size + MTime)
        feature_file = config["paths"]["feature_file"]
        if os.path.exists(feature_file):
            stat = os.stat(feature_file)
            data_info = f"{stat.st_size}_{stat.st_mtime}"
        else:
            data_info = "no_data"

        # 2. Nội dung cấu hình ảnh hưởng
        relevant_config = {
            "metrics_version": 4,
            "risk": config.get("risk"),
            "strategies": config.get("strategies"),
            "features": config.get("features"),
            "ml": config.get("ml"),
            "trading_costs": config.get("trading_costs"),
            "data_info": data_info,
            "symbols": sorted(symbols),
            "strategy": strategy_name,
            "interval": [str(interval[0]), str(interval[1])] if interval else None
        }
        
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def get(self, key: str):
        """Lấy kết quả từ cache nếu tồn tại."""
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    def set(self, key: str, data: dict):
        """Lưu kết quả vào cache."""
        cache_path = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_path, "w") as f:
            json.dump(data, f)
