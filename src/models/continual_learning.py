import os
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import joblib

from src.models.ml_pipeline import (
    get_ml_config,
    prepare_labeled_dataset,
    build_model,
    evaluate_classifier,
    save_artifact
)

logger = logging.getLogger("trading_system.continual_learning")

def get_time_windows(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    train_months: int = 12,
    test_months: int = 1,
    step_months: int = 1
) -> List[Dict[str, pd.Timestamp]]:
    """
    Chia dữ liệu thành các cửa sổ Walk-Forward (Sliding Window).
    """
    windows = []
    current_test_start = start_date + pd.DateOffset(months=train_months)
    
    while current_test_start < end_date:
        current_test_end = current_test_start + pd.DateOffset(months=test_months)
        if current_test_end > end_date:
            current_test_end = end_date
            
        train_start = current_test_start - pd.DateOffset(months=train_months)
        train_end = current_test_start - pd.Timedelta(days=1)
        
        windows.append({
            "train_start": train_start,
            "train_end": train_end,
            "test_start": current_test_start,
            "test_end": current_test_end
        })
        
        current_test_start += pd.DateOffset(months=step_months)
        
    return windows

class ModelManager:
    """
    Quản lý việc so sánh Champion-Challenger và lưu trữ model.
    """
    def __init__(self, output_root: str = "models/continual"):
        self.output_root = output_root
        os.makedirs(output_root, exist_ok=True)
        self.champion_path = os.path.join(output_root, "champion_metadata.json")
        
    def get_champion(self) -> Dict:
        if os.path.exists(self.champion_path):
            with open(self.champion_path, "r") as f:
                return json.load(f)
        return None

    def promote_challenger(self, challenger_metadata: Dict, challenger_dir: str):
        """
        Cập nhật challenger thành champion.
        """
        champion_data = {
            "artifact_dir": challenger_dir,
            "promoted_at": str(datetime.now()),
            "metrics": challenger_metadata.get("metrics"),
            "split": challenger_metadata.get("split")
        }
        with open(self.champion_path, "w") as f:
            json.dump(champion_data, f, indent=2)
        logger.info(f"Promoted new champion: {challenger_dir}")

def run_window_training(
    df: pd.DataFrame,
    window: Dict[str, pd.Timestamp],
    ml_cfg: Dict,
    model_manager: ModelManager
) -> Tuple[str, Dict]:
    """
    Huấn luyện model cho một cửa sổ thời gian cụ thể.
    """
    # Lọc dữ liệu train
    train_df = df[(df["date"] >= window["train_start"]) & (df["date"] <= window["train_end"])].copy()
    
    # Gán nhãn (Lưu ý: nhãn dựa trên tương lai nên cần cẩn thận leakage ở cuối window)
    data = prepare_labeled_dataset(train_df, ml_cfg)
    
    if len(data) < 100:
        logger.warning(f"Not enough data for window {window['train_start']} to {window['train_end']}")
        return None, None
        
    feature_cols = ml_cfg["feature_cols"]
    x_train, y_train = data[feature_cols], data["label"]
    
    model_type = ml_cfg.get("model", {}).get("type", "hist_gbdt")
    random_seed = ml_cfg.get("train", {}).get("random_seed", 42)
    
    model = build_model(model_type=model_type, random_seed=random_seed)
    model.fit(x_train, y_train)
    
    # Đánh giá trên chính tập train (Simple sanity check)
    metrics = evaluate_classifier(model, x_train, y_train)
    
    metadata = {
        "window": {k: str(v) for k, v in window.items()},
        "metrics": metrics,
        "feature_cols": feature_cols,
        "model_type": model_type
    }
    
    artifact_dir = save_artifact(
        model, 
        output_root=model_manager.output_root, 
        model_name=f"wf_{window['test_start'].strftime('%Y%m')}", 
        metadata=metadata
    )
    
    return artifact_dir, metadata
