from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
from dateutil import parser as dt_parser

from src.features.utils import build_category_maps


@dataclass
class FeatureSpec:
    feature_columns: List[str]
    target_column: str = "is_fraud"


class FeatureBuilder:
    def __init__(self):
        maps = build_category_maps()
        self.category_map = maps["category_map"]
        self.device_type_map = maps["device_type_map"]
        self.channel_map = maps["channel_map"]
        self.entry_mode_map = maps["entry_mode_map"]
        self.country_map = maps["country_map"]

        self.spec = FeatureSpec(
            feature_columns=[
                "amount",
                "is_international",
                "hour",
                "is_night",
                "high_amount_flag",
                "category_code",
                "device_type_code",
                "channel_code",
                "entry_mode_code",
                "country_code",
            ],
            target_column="is_fraud",
        )

    def _parse_timestamp(self, ts: str) -> pd.Timestamp:
        return pd.to_datetime(dt_parser.parse(ts))

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Timestamp derived features
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].apply(self._parse_timestamp)

        df["hour"] = df["timestamp"].dt.hour
        df["is_night"] = df["hour"].apply(lambda h: 1 if h <= 6 or h >= 23 else 0)

        # Amount-based flags
        df["high_amount_flag"] = (df["amount"] > 5000).astype(int)

        # Encodings using fixed maps (robust for serving)
        df["category_code"] = df["category"].map(self.category_map).fillna(-1).astype(int)
        df["device_type_code"] = df["device_type"].map(self.device_type_map).fillna(-1).astype(int)
        df["channel_code"] = df["channel"].map(self.channel_map).fillna(-1).astype(int)
        df["entry_mode_code"] = df["entry_mode"].map(self.entry_mode_map).fillna(-1).astype(int)
        df["country_code"] = df["country"].map(self.country_map).fillna(-1).astype(int)

        return df

    def build_xy(
        self, df: pd.DataFrame, include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df_feat = self.add_features(df)
        X = df_feat[self.spec.feature_columns]
        y = None
        if include_target and self.spec.target_column in df_feat.columns:
            y = df_feat[self.spec.target_column].astype(int)
        return X, y

    def build_x_from_dict(self, data: dict) -> pd.DataFrame:
        """Used by API: create a single-row DataFrame and build features."""
        df = pd.DataFrame([data])
        X, _ = self.build_xy(df, include_target=False)
        return X
