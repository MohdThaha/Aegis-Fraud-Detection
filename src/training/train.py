import os
import json

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)

from src.features.feature_builder import FeatureBuilder
from src.training.model_selector import train_and_select_best
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    raw_path = config["paths"]["raw_data"]
    processed_path = config["paths"]["processed_data"]
    model_dir = config["paths"]["model_dir"]
    best_model_path = config["paths"]["best_model_path"]
    metrics_path = config["paths"]["metrics_path"]

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    fb = FeatureBuilder()
    X, y = fb.build_xy(df, include_target=True)

    logger.info(f"Saving processed features to {processed_path}")
    processed_df = X.copy()
    processed_df[config["training"]["target_col"]] = y
    processed_df.to_csv(processed_path, index=False)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y,
    )

    candidates = tuple(config["models"]["candidates"])
    model_configs = {
        "random_forest": config["models"]["random_forest"],
        "logreg": config["models"]["logreg"],
    }

    best_name, best_model, scores = train_and_select_best(
        X_train, y_train, X_val, y_val, model_configs, candidates
    )

    # Final evaluation metrics on validation set
    val_proba = best_model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, val_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, val_pred, average="binary"
    )
    acc = accuracy_score(y_val, val_pred)

    metrics = {
        "best_model_name": best_name,
        "scores_per_model": scores,
        "validation_metrics": {
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc,
        },
    }

    logger.info(f"Validation metrics: {json.dumps(metrics['validation_metrics'], indent=2)}")

    # Save model
    joblib.dump(best_model, best_model_path)
    logger.info(f"Saved best model to {best_model_path}")

    # Save metrics
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
