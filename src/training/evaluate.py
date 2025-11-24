import json

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    processed_path = config["paths"]["processed_data"]
    best_model_path = config["paths"]["best_model_path"]
    metrics_path = config["paths"]["metrics_path"]
    target_col = config["training"]["target_col"]

    logger.info(f"Loading processed feature data from {processed_path}")
    df = pd.read_csv(processed_path)

    # Split X and y directly (no feature builder here)
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    logger.info(f"Loading trained model from {best_model_path}")
    model = joblib.load(best_model_path)

    # Predictions
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary"
    )
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, output_dict=True)

    eval_metrics = {
        "full_dataset_evaluation": {
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": report,
        }
    }

    logger.info("Full dataset evaluation:")
    logger.info(json.dumps(eval_metrics["full_dataset_evaluation"], indent=2))

    # Merge with existing metrics if present
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}

    metrics.update(eval_metrics)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Updated metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
