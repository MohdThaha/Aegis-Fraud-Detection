from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.training.imbalance import get_class_weights
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_logreg(X_train, y_train, config) -> LogisticRegression:
    class_weights = get_class_weights(y_train)
    clf = LogisticRegression(
        max_iter=config["max_iter"],
        class_weight=class_weights,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_random_forest(X_train, y_train, config) -> RandomForestClassifier:
    class_weights = get_class_weights(y_train)
    clf = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        n_jobs=config["n_jobs"],
        class_weight=class_weights,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_val, y_val) -> float:
    proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)


def train_and_select_best(
    X_train,
    y_train,
    X_val,
    y_val,
    model_configs: Dict,
    candidates: Tuple[str, ...],
):
    best_model = None
    best_name = None
    best_score = -np.inf
    scores = {}

    for name in candidates:
        logger.info(f"Training model: {name}")
        if name == "logreg":
            model = train_logreg(X_train, y_train, model_configs["logreg"])
        elif name == "random_forest":
            model = train_random_forest(X_train, y_train, model_configs["random_forest"])
        else:
            logger.warning(f"Unknown model '{name}', skipping.")
            continue

        score = evaluate_model(model, X_val, y_val)
        scores[name] = score
        logger.info(f"Model {name} ROC-AUC: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    logger.info(f"Best model: {best_name} with ROC-AUC={best_score:.4f}")
    return best_name, best_model, scores
