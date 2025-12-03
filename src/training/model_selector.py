from typing import Dict, Tuple, Optional
import importlib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from src.training.imbalance import get_class_weights
from src.utils.logger import get_logger

logger = get_logger(__name__)


# -------------------------------------------------------------------------
# Helper: Safe dynamic importer
# -------------------------------------------------------------------------
def try_import(module_name: str):
    """Try importing a module dynamically. Returns module or None."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        logger.warning(f"Optional library '{module_name}' not installed. Skipping.")
        return None


# -------------------------------------------------------------------------
# Model Trainers
# -------------------------------------------------------------------------
def train_logreg(X_train, y_train, config):
    class_weights = get_class_weights(y_train)

    model = LogisticRegression(
        max_iter=config.get("max_iter", 1000),
        solver=config.get("solver", "lbfgs"),
        class_weight=class_weights,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, config):
    class_weights = get_class_weights(y_train)

    model = RandomForestClassifier(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", 12),
        n_jobs=config.get("n_jobs", -1),
        class_weight=class_weights,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, config):
    xgb = try_import("xgboost")
    if xgb is None:
        return None

    model = xgb.XGBClassifier(
        booster=config.get("booster", "gbtree"),
        max_depth=config.get("max_depth", 8),
        learning_rate=config.get("learning_rate", 0.1),
        n_estimators=config.get("n_estimators", 500),
        tree_method=config.get("tree_method", "hist"),  # REQUIRED for large datasets
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, config):
    lgb = try_import("lightgbm")
    if lgb is None:
        return None

    model = lgb.LGBMClassifier(
        num_leaves=config.get("num_leaves", 64),
        learning_rate=config.get("learning_rate", 0.05),
        n_estimators=config.get("n_estimators", 500),
        class_weight="balanced",  # LightGBM handles imbalance natively
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train, config):
    cb = try_import("catboost")
    if cb is None:
        return None

    model = cb.CatBoostClassifier(
        depth=config.get("depth", 8),
        learning_rate=config.get("learning_rate", 0.1),
        n_estimators=config.get("n_estimators", 300),
        loss_function="Logloss",
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def train_mlp(X_train, y_train, config):
    model = MLPClassifier(
        hidden_layer_sizes=tuple(config.get("hidden_layers", [128, 64, 32])),
        max_iter=config.get("epochs", 10),
        batch_size=config.get("batch_size", 4096),
        early_stopping=True,
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------------------------------------------------
# MODEL REGISTRY: Add any new model here
# -------------------------------------------------------------------------
MODEL_REGISTRY = {
    "logreg": train_logreg,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "catboost": train_catboost,
    "mlp": train_mlp,
}


# -------------------------------------------------------------------------
# Evaluation Helper
# -------------------------------------------------------------------------
def evaluate_model(model, X_val, y_val) -> float:
    """
    Returns ROC-AUC score.
    Works even for models without predict_proba (fallback to raw predictions).
    """
    try:
        proba = model.predict_proba(X_val)[:, 1]
    except Exception:
        pred = model.predict(X_val)
        proba = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)

    return roc_auc_score(y_val, proba)


# -------------------------------------------------------------------------
# Main selection loop
# -------------------------------------------------------------------------
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
        trainer = MODEL_REGISTRY.get(name)

        if trainer is None:
            logger.warning(f"Model '{name}' not found in registry. Skipping.")
            continue

        logger.info(f"----------------------------")
        logger.info(f"Training model: {name}")
        logger.info(f"----------------------------")

        config = model_configs.get(name, {})
        model = trainer(X_train, y_train, config)

        if model is None:
            logger.warning(f"Model '{name}' could not be trained (missing library).")
            continue

        score = evaluate_model(model, X_val, y_val)
        scores[name] = score

        logger.info(f"Model {name} ROC-AUC = {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    logger.info("======================================")
    logger.info(f" BEST MODEL SELECTED: {best_name}")
    logger.info(f" ROC-AUC = {best_score:.4f}")
    logger.info("======================================")

    return best_name, best_model, scores
