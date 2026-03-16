import os
import sys
import pandas as pd
import json
import numpy as np

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.config_loader import CONFIG
from utils.logger import setup_logger
from src.models.ml_pipeline import get_ml_config

logger = setup_logger(name="trading_system.baseline", level="INFO")

def main():
    logger.info("--- Generating Baseline Constraints for Cloud Monitoring ---")
    
    ml_cfg = get_ml_config(CONFIG)
    feature_cols = ml_cfg["feature_cols"]
    feature_file = CONFIG["paths"]["feature_file"]
    
    if not os.path.exists(feature_file):
        logger.error(f"Feature file not found: {feature_file}")
        return

    # 1. Load data
    df = pd.read_parquet(feature_file)
    
    # Giả sử chúng ta dùng dữ liệu huấn luyện (ví dụ: trước năm 2024) làm baseline
    # Hoặc lấy toàn bộ nếu đây là lần đầu triển khai
    logger.info(f"Calculating statistics for features: {feature_cols}")
    
    baseline_stats = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
            
        series = df[col].dropna()
        baseline_stats[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q25": float(series.quantile(0.25)),
            "q50": float(series.quantile(0.50)),
            "q75": float(series.quantile(0.75)),
            "count": int(len(series))
        }

    # 2. Save baseline constraints
    output_path = "models/baseline_constraints.json"
    os.makedirs("models", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "1.0",
            "features": baseline_stats,
            "metadata": {
                "generated_at": str(pd.Timestamp.now()),
                "source_file": feature_file
            }
        }, f, indent=2)
        
    logger.info(f"Baseline constraints saved to: {output_path}")
    print(f"
[OK] Baseline generated for {len(baseline_stats)} features.")

if __name__ == "__main__":
    main()
