import numpy as np
import pandas as pd

from src.models.ml_pipeline import LABEL_DOWN, LABEL_UP, load_artifact
from src.strategy.base import BaseStrategy


class MLStrategy(BaseStrategy):
    def __init__(
        self,
        artifact_dir: str,
        p_buy: float = 0.55,
        p_sell: float = 0.55,
        min_conf_gap: float = 0.05,
    ):
        super().__init__(name="ML_3Class")
        self.model, self.metadata = load_artifact(artifact_dir)
        self.feature_cols = self.metadata.get("feature_cols", [])
        self.feature_shift_bars = int(self.metadata.get("feature_shift_bars", 1))
        self.p_buy = float(p_buy)
        self.p_sell = float(p_sell)
        self.min_conf_gap = float(min_conf_gap)

    def _predict_probabilities(self, df: pd.DataFrame) -> pd.DataFrame:
        out_df = df.copy()
        out_df = out_df.sort_values(["symbol", "date"]).reset_index(drop=True)

        for col in self.feature_cols:
            if col not in out_df.columns:
                raise ValueError(f"Missing feature column for ML strategy: {col}")
            out_df[col] = out_df.groupby("symbol", observed=True)[col].shift(self.feature_shift_bars)

        out_df["ml_prob_up"] = np.nan
        out_df["ml_prob_down"] = np.nan

        valid_mask = out_df[self.feature_cols].notna().all(axis=1)
        if valid_mask.any():
            x = out_df.loc[valid_mask, self.feature_cols]
            proba = self.model.predict_proba(x)
            classes = list(self.model.classes_)
            up_idx = classes.index(LABEL_UP)
            down_idx = classes.index(LABEL_DOWN)
            out_df.loc[valid_mask, "ml_prob_up"] = proba[:, up_idx]
            out_df.loc[valid_mask, "ml_prob_down"] = proba[:, down_idx]
        return out_df

    def _apply_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        buy_cond = (df["ml_prob_up"] >= self.p_buy) & ((df["ml_prob_up"] - df["ml_prob_down"]) >= self.min_conf_gap)
        sell_cond = (df["ml_prob_down"] >= self.p_sell) & ((df["ml_prob_down"] - df["ml_prob_up"]) >= self.min_conf_gap)

        decision = pd.Series(np.nan, index=df.index, dtype=float)
        decision.loc[sell_cond] = 0.0
        decision.loc[buy_cond] = 1.0
        df["desired_position"] = decision
        df["desired_position"] = (
            df.groupby("symbol", observed=True)["desired_position"].ffill().fillna(0.0)
        )
        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out_df = self._predict_probabilities(df)
        out_df = self._apply_policy(out_df)
        out_df["signal"] = out_df.groupby("symbol", observed=True)["desired_position"].diff().fillna(
            out_df["desired_position"]
        )
        out_df["signal"] = out_df["signal"].clip(-1, 1)
        return out_df


class HybridMLStrategy(MLStrategy):
    def __init__(
        self,
        artifact_dir: str,
        p_buy: float = 0.55,
        p_sell: float = 0.55,
        min_conf_gap: float = 0.05,
        trend_col: str = "sma_50",
    ):
        super().__init__(artifact_dir, p_buy=p_buy, p_sell=p_sell, min_conf_gap=min_conf_gap)
        self.name = "ML_Hybrid_MAFilter"
        self.trend_col = trend_col

    def _apply_policy(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.trend_col not in df.columns:
            raise ValueError(f"Missing trend filter column: {self.trend_col}")

        base_buy = (df["ml_prob_up"] >= self.p_buy) & ((df["ml_prob_up"] - df["ml_prob_down"]) >= self.min_conf_gap)
        base_sell = (df["ml_prob_down"] >= self.p_sell) & ((df["ml_prob_down"] - df["ml_prob_up"]) >= self.min_conf_gap)

        buy_cond = base_buy & (df["close"] > df[self.trend_col])
        sell_cond = base_sell | (df["close"] < df[self.trend_col])

        decision = pd.Series(np.nan, index=df.index, dtype=float)
        decision.loc[sell_cond] = 0.0
        decision.loc[buy_cond] = 1.0
        df["desired_position"] = decision
        df["desired_position"] = (
            df.groupby("symbol", observed=True)["desired_position"].ffill().fillna(0.0)
        )
        return df
