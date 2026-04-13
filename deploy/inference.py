import os
import json
import joblib
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "sma_20",
    "sma_50",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
]
LABEL_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
# 1. model_fn  — called once at container startup
def model_fn(model_dir: str):
    """
    Load model.joblib from the SageMaker model directory.
    SageMaker extracts model.tar.gz into `model_dir` (/opt/ml/model/).
    """
    model_path = os.path.join(model_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model.joblib not found in {model_dir}. "
            f"Files present: {os.listdir(model_dir)}"
        )
    model = joblib.load(model_path)
    print(f"[inference] Model loaded from {model_path}")
    return model

# 2. input_fn  — deserialize raw request bytes into a DataFrame
def input_fn(request_body: str, content_type: str = "application/json"):
    """
    Accept JSON in two shapes:
      - Single row : {"sma_20": 150.0, "sma_50": 148.0, ...}
      - Batch      : [{"sma_20": 150.0, ...}, {"sma_20": 151.0, ...}]
    """
    if content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: {content_type}. Use application/json."
        )

    data = json.loads(request_body)

    # Normalize to list-of-dicts
    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    # Validate features
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return df[FEATURE_COLS]  # enforce column order

# 3. predict_fn  — run the model
def predict_fn(input_data: pd.DataFrame, model):
    """
    Returns a dict with predicted class indices and probabilities.
    """
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    results = []
    for pred, proba in zip(predictions, probabilities):
        results.append(
            {
                "prediction": int(pred),
                "label": LABEL_MAP[int(pred)],
                "probabilities": {
                    "DOWN": round(float(proba[0]), 4),
                    "NEUTRAL": round(float(proba[1]), 4),
                    "UP": round(float(proba[2]), 4),
                },
            }
        )
    return results


# 4. output_fn  — serialize predictions back to JSON
def output_fn(predictions: list, accept: str = "application/json"):
    """
    Serialize the prediction list to a JSON string.
    """
    if accept != "application/json":
        raise ValueError(
            f"Unsupported accept type: {accept}. Use application/json."
        )
    return json.dumps(predictions), accept