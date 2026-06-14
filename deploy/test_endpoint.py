"""
test_endpoint.py — Test SageMaker Endpoint
Reads config from .env file at project root.
Usage:
    python deploy/test_endpoint.py
Tests:
    1. Single row prediction
    2. Batch prediction (multiple rows)
    3. Missing feature validation (expects error)
"""

import os
import json
import boto3
from dotenv import load_dotenv

# Load .env
load_dotenv()
AWS_REGION  = os.getenv("AWS_REGION")
SAGEMAKER_ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT_NAME")

runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

print(" SageMaker Endpoint Test")
print(f"Endpoint: {SAGEMAKER_ENDPOINT_NAME}")
print(f"Region: {AWS_REGION}")


def invoke(payload: dict | list, label: str):
    """Send a request to the endpoint and print the result."""
    print(f"\n{'─'*60}")
    print(f"  TEST: {label}")
    print(f"  Input  : {json.dumps(payload, indent=2)}")
    try:
        response = runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode())
        print(f"  Output : {json.dumps(result, indent=2)}")
        for i, pred in enumerate(result):
            print(
                f"Row {i+1}: {pred['label']} "
                f"(DOWN={pred['probabilities']['DOWN']}, "
                f"NEUTRAL={pred['probabilities']['NEUTRAL']}, "
                f"UP={pred['probabilities']['UP']})"
            )
        print("PASSED")
    except Exception as e:
        print(f"FAILED: {e}")

# Single row (bullish-looking signal)
invoke(
    payload={
        "sma_20": 155.23,
        "sma_50": 150.10,
        "rsi_14": 62.5,
        "macd": 1.35,
        "macd_signal": 0.98,
        "macd_hist": 0.37,
    },
    label="Single row — bullish signal",
)

# Single row (bearish-looking signal)
invoke(
    payload={
        "sma_20": 142.10,
        "sma_50": 148.50,
        "rsi_14": 31.2,
        "macd": -1.80,
        "macd_signal": -1.20,
        "macd_hist": -0.60,
    },
    label="Single row — bearish signal",
)

# Batch (multiple rows)
invoke(
    payload=[
        {
            "sma_20": 155.23,
            "sma_50": 150.10,
            "rsi_14": 62.5,
            "macd": 1.35,
            "macd_signal": 0.98,
            "macd_hist": 0.37,
        },
        {
            "sma_20": 142.10,
            "sma_50": 148.50,
            "rsi_14": 31.2,
            "macd": -1.80,
            "macd_signal": -1.20,
            "macd_hist": -0.60,
        },
        {
            "sma_20": 149.00,
            "sma_50": 149.20,
            "rsi_14": 50.1,
            "macd": 0.05,
            "macd_signal": 0.03,
            "macd_hist": 0.02,
        },
    ],
    label="Batch — 3 rows (bullish, bearish, neutral)",
)

## NEW TESTCASES
import random

invoke(
    payload={
        "sma_20": random.uniform(-100, 250),
        "sma_50": random.uniform(-100, 250),
        "rsi_14": random.uniform(0, 100),
        "macd": random.uniform(-10, 10),
        "macd_signal": random.uniform(-10, 10),
        "macd_hist": random.uniform(-2, 2),
    },
    label="Drift — random noisy input 1",
)

invoke(
    payload={
        "sma_20": random.uniform(-100, 250),
        "sma_50": random.uniform(-100, 250),
        "rsi_14": random.uniform(0, 100),
        "macd": random.uniform(-10, 10),
        "macd_signal": random.uniform(-10, 10),
        "macd_hist": random.uniform(-2, 2),
    },
    label="Drift — random noisy input 2",
)

invoke(
    payload={
        "sma_20": random.uniform(-100, 250),
        "sma_50": random.uniform(-100, 250),
        "rsi_14": random.uniform(0, 100),
        "macd": random.uniform(-10, 10),
        "macd_signal": random.uniform(-10, 10),
        "macd_hist": random.uniform(-2, 2),
    },
    label="Drift — random noisy input 3",
)


print(f"\n{'='*60}")
print("  All tests complete.")
# print("  Delete the endpoint after testing!")
# print("  Run: aws sagemaker delete-endpoint --endpoint-name", SAGEMAKER_ENDPOINT_NAME)