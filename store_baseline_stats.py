import pandas as pd
import json

baseline = pd.read_parquet("dataset/processed/features_stocks.parquet")

features = ["sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_hist"]

stats = {}

for col in features:
    stats[col] = {
        "mean": float(baseline[col].mean()),
        "std": float(baseline[col].std())
    }

with open("dataset/baseline_stats.json", "w") as f:
    json.dump(stats, f, indent=2)