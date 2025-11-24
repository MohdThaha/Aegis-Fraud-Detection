from typing import Any, Dict

import joblib
import yaml
import numpy as np

from src.features.feature_builder import FeatureBuilder
from src.serving.schema import TransactionInput
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class FraudModelService:
    def __init__(self):
        self.config = load_config()
        self.model_path = self.config["paths"]["best_model_path"]
        self._model = None
        self._fb = FeatureBuilder()

    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading model from {self.model_path}")
            self._model = joblib.load(self.model_path)
        return self._model

    def predict(self, tx: TransactionInput) -> Dict[str, Any]:
        data = tx.dict()
        X = self._fb.build_x_from_dict(data)

        proba = self.model.predict_proba(X)[:, 1][0]
        label = int(proba >= 0.5)

        # Simple explanation: show feature values
        explanation = {
            "amount": X.iloc[0]["amount"],
            "is_international": int(X.iloc[0]["is_international"]),
            "high_amount_flag": int(X.iloc[0]["high_amount_flag"]),
            "hour": int(X.iloc[0]["hour"]),
            "is_night": int(X.iloc[0]["is_night"]),
        }

        return {
            "fraud_score": float(proba),
            "fraud_label": label,
            "explanation": explanation,
        }
