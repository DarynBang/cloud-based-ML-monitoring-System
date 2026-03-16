import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


LABEL_DOWN = 0
LABEL_NEUTRAL = 1
LABEL_UP = 2

LABEL_NAME_MAP = {
    LABEL_DOWN: "DOWN",
    LABEL_NEUTRAL: "NEUTRAL",
    LABEL_UP: "UP",
}


def get_ml_config(config: dict) -> dict:
    ml_cfg = config.get("ml", {}) or {}
    return {
        "label_horizon_days": int(ml_cfg.get("label_horizon_days", 5)),
        "label_up_threshold": float(ml_cfg.get("label_up_threshold", 0.01)),
        "label_down_threshold": float(ml_cfg.get("label_down_threshold", -0.01)),
        "feature_shift_bars": int(ml_cfg.get("feature_shift_bars", 1)),
        "feature_cols": list(
            ml_cfg.get(
                "feature_cols",
                ["sma_20", "sma_50", "rsi_14", "macd", "macd_signal", "macd_hist"],
            )
        ),
        "train": ml_cfg.get("train", {}) or {},
        "model": ml_cfg.get("model", {}) or {},
        "policy": ml_cfg.get("policy", {}) or {},
    }


def prepare_labeled_dataset(df: pd.DataFrame, ml_cfg: dict) -> pd.DataFrame:
    work_df = df.sort_values(["symbol", "date"]).copy()
    horizon = ml_cfg["label_horizon_days"]
    up_th = ml_cfg["label_up_threshold"]
    down_th = ml_cfg["label_down_threshold"]
    shift_bars = ml_cfg["feature_shift_bars"]
    feature_cols = ml_cfg["feature_cols"]

    grouped_close = work_df.groupby("symbol", observed=True)["close"]
    work_df["ret_fwd"] = grouped_close.shift(-horizon) / work_df["close"] - 1.0

    work_df["label"] = LABEL_NEUTRAL
    work_df.loc[work_df["ret_fwd"] >= up_th, "label"] = LABEL_UP
    work_df.loc[work_df["ret_fwd"] <= down_th, "label"] = LABEL_DOWN

    for col in feature_cols:
        if col not in work_df.columns:
            raise ValueError(f"Missing feature column: {col}")
        work_df[col] = work_df.groupby("symbol", observed=True)[col].shift(shift_bars)

    required_cols = ["symbol", "date", "ret_fwd", "label"] + feature_cols
    work_df = work_df[required_cols].dropna().reset_index(drop=True)
    work_df["label"] = work_df["label"].astype(int)
    return work_df


def split_time_based(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["date"] <= train_end].copy()
    val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test_df = df[df["date"] > val_end].copy()
    return train_df, val_df, test_df


def auto_split_dates(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = np.array(sorted(df["date"].dropna().unique()))
    if len(unique_dates) < 10:
        raise ValueError("Not enough dates to split train/val/test.")
    train_idx = int(len(unique_dates) * 0.7)
    val_idx = int(len(unique_dates) * 0.85)
    train_end = pd.Timestamp(unique_dates[train_idx])
    val_end = pd.Timestamp(unique_dates[val_idx])
    return train_end, val_end


def build_model(model_type: str, random_seed: int = 42):
    if model_type == "logreg":
        return LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed,
        )
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=40,
        random_state=random_seed,
    )


def evaluate_classifier(model, x: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    if len(x) == 0:
        return {
            "rows": 0,
            "accuracy": None,
            "macro_f1": None,
            "confusion_matrix": [],
            "classification_report": {},
        }

    pred = model.predict(x)
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred, labels=[LABEL_DOWN, LABEL_NEUTRAL, LABEL_UP])
    return {
        "rows": int(len(x)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_artifact(
    model,
    output_root: str,
    model_name: str,
    metadata: dict,
) -> str:
    os.makedirs(output_root, exist_ok=True)
    run_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(output_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.joblib")
    meta_path = os.path.join(out_dir, "metadata.json")
    joblib.dump(model, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    return out_dir


def load_artifact(artifact_dir: str) -> Tuple[object, dict]:
    model_path = os.path.join(artifact_dir, "model.joblib")
    meta_path = os.path.join(artifact_dir, "metadata.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata
