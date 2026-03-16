import os
import random
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ml_pipeline import (
    LABEL_NAME_MAP,
    auto_split_dates,
    build_model,
    evaluate_classifier,
    get_ml_config,
    prepare_labeled_dataset,
    save_artifact,
    split_time_based,
)
from utils.config_loader import CONFIG
from utils.logger import setup_logger


logger = setup_logger(name="trading_system.ml", level=CONFIG.get("logging", {}).get("level", "INFO"))


def main():
    ml_cfg = get_ml_config(CONFIG)
    train_cfg = ml_cfg.get("train", {})
    model_cfg = ml_cfg.get("model", {})
    model_type = str(model_cfg.get("type", "hist_gbdt")).strip().lower()
    if model_type not in {"hist_gbdt", "logreg"}:
        model_type = "hist_gbdt"

    feature_file = CONFIG["paths"]["feature_file"]
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    logger.info(f"Loading features: {feature_file}")
    df = pd.read_parquet(feature_file)
    df["date"] = pd.to_datetime(df["date"])

    random_seed = int(train_cfg.get("random_seed", CONFIG.get("backtest", {}).get("random_seed", 42)))
    random.seed(random_seed)

    sample_size = int(train_cfg.get("symbols_sample_size", 200))
    if sample_size > 0:
        symbols = list(df["symbol"].unique())
        chosen = random.sample(symbols, min(sample_size, len(symbols)))
        df = df[df["symbol"].isin(chosen)].copy()
        logger.info(f"Training sample symbols: {len(chosen)}")

    date_start = train_cfg.get("date_start")
    date_end = train_cfg.get("date_end")
    if date_start:
        df = df[df["date"] >= pd.Timestamp(date_start)]
    if date_end:
        df = df[df["date"] <= pd.Timestamp(date_end)]

    data = prepare_labeled_dataset(df, ml_cfg)
    feature_cols = ml_cfg["feature_cols"]
    logger.info(f"Labeled rows: {len(data):,}")

    train_end = train_cfg.get("train_end")
    val_end = train_cfg.get("val_end")
    if train_end and val_end:
        train_end_ts = pd.Timestamp(train_end)
        val_end_ts = pd.Timestamp(val_end)
    else:
        train_end_ts, val_end_ts = auto_split_dates(data)
        logger.info(f"Auto split dates => train_end={train_end_ts.date()} val_end={val_end_ts.date()}")

    train_df, val_df, test_df = split_time_based(data, train_end_ts, val_end_ts)
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split is empty. Please adjust date range or split config.")

    x_train, y_train = train_df[feature_cols], train_df["label"]
    x_val, y_val = val_df[feature_cols], val_df["label"]
    x_test, y_test = test_df[feature_cols], test_df["label"]

    model = build_model(model_type=model_type, random_seed=random_seed)
    logger.info(f"Training model: {model_type}")
    model.fit(x_train, y_train)

    val_metrics = evaluate_classifier(model, x_val, y_val)
    test_metrics = evaluate_classifier(model, x_test, y_test) if len(x_test) > 0 else {"rows": 0}

    output_root = str(model_cfg.get("output_dir", "models"))
    metadata = {
        "model_type": model_type,
        "feature_cols": feature_cols,
        "feature_shift_bars": ml_cfg["feature_shift_bars"],
        "label_horizon_days": ml_cfg["label_horizon_days"],
        "label_up_threshold": ml_cfg["label_up_threshold"],
        "label_down_threshold": ml_cfg["label_down_threshold"],
        "label_name_map": LABEL_NAME_MAP,
        "split": {
            "train_end": str(train_end_ts),
            "val_end": str(val_end_ts),
        },
        "metrics": {
            "val": val_metrics,
            "test": test_metrics,
        },
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "random_seed": random_seed,
    }
    artifact_dir = save_artifact(model, output_root=output_root, model_name=model_type, metadata=metadata)

    logger.info(f"Saved model artifact: {artifact_dir}")
    logger.info(
        f"Validation macro_f1={val_metrics.get('macro_f1')} accuracy={val_metrics.get('accuracy')} rows={val_metrics.get('rows')}"
    )
    if test_metrics.get("rows", 0) > 0:
        logger.info(
            f"Test macro_f1={test_metrics.get('macro_f1')} accuracy={test_metrics.get('accuracy')} rows={test_metrics.get('rows')}"
        )
    print(f"MODEL_ARTIFACT={artifact_dir}")


if __name__ == "__main__":
    main()
