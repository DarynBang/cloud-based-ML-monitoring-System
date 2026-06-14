import json
import pandas as pd

rows = []

captured_data_path = r"C:\Users\Daryn Bang\Downloads\50-42-680-4bb3eb3f-ece9-4c6a-9129-cc90fbd56b2c.jsonl"

with open(captured_data_path, "r") as f:
    for line in f:
        obj = json.loads(line)

        raw = obj["captureData"]["endpointInput"]["data"]
        parsed = json.loads(raw)

        # Handle both single and batch cases
        if isinstance(parsed, dict):
            rows.append(parsed)
        elif isinstance(parsed, list):
            rows.extend(parsed)

df = pd.DataFrame(rows)
print(df.info())
#
# print(df.head())
# print(df.describe())


baseline = pd.read_parquet("dataset/processed/features_stocks.parquet")

# print(baseline.columns)
# match columns
features = ["sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_hist"]

print("Baseline:")
print(baseline[features].describe())

print("\nCaptured:")
print(df[features].describe())


drift_scores = {}

for col in features:
    base_mean = baseline[col].mean()
    live_mean = df[col].mean()

    diff = abs(live_mean - base_mean)

    # Normalize safely (prevents explosion)
    scale = abs(base_mean) + 1e-6

    score = diff / scale

    # Clamp to avoid insane values
    score = min(score, 1.0)

    drift_scores[col] = score

final_drift = sum(drift_scores.values()) / len(drift_scores)

print("Drift per feature:", drift_scores)
print("Final drift:", final_drift)


if final_drift > 0.3:
    print("DRIFT DETECTED")
else:
    print("NO DRIFT")
