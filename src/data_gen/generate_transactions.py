import os
import uuid
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


CATEGORIES = ["groceries", "electronics", "travel", "restaurants", "utilities", "others"]
DEVICE_TYPES = ["mobile", "desktop", "pos_terminal"]
CHANNELS = ["online", "in_store"]
ENTRY_MODES = ["chip", "swipe", "contactless", "manual", "online"]
COUNTRIES = ["IN", "US", "GB", "AE", "SG"]  # IN as home country


def load_config(path: str = "configs/training_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_timestamps(n: int) -> list:
    """Generate random timestamps over last 30 days."""
    now = datetime.utcnow()
    start = now - timedelta(days=30)
    return [
        (start + timedelta(seconds=random.randint(0, 30 * 24 * 3600))).isoformat()
        for _ in range(n)
    ]


def generate_synthetic_transactions(n_samples: int, fraud_ratio: float, random_state: int):
    rng = np.random.default_rng(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legit amounts: mostly small to medium
    legit_amounts = rng.normal(loc=1500, scale=700, size=n_legit).clip(10, 20000)
    # Fraud amounts: larger on average
    fraud_amounts = rng.normal(loc=6000, scale=3000, size=n_fraud).clip(50, 50000)

    # User IDs
    legit_user_ids = rng.integers(1, 5000, size=n_legit)
    fraud_user_ids = rng.integers(1, 5000, size=n_fraud)

    # Countries: mostly IN for legit, more foreign for fraud
    legit_countries = rng.choice(["IN"] * 8 + ["US", "GB", "AE", "SG"] * 1, size=n_legit)
    fraud_countries = rng.choice(["IN"] * 3 + ["US", "GB", "AE", "SG"] * 7, size=n_fraud)

    # Timestamps
    legit_timestamps = generate_timestamps(n_legit)
    fraud_timestamps = generate_timestamps(n_fraud)

    def build_df(
        n,
        amounts,
        user_ids,
        countries,
        timestamps,
        is_fraud_flag: int,
    ):
        data = {
            "transaction_id": [str(uuid.uuid4()) for _ in range(n)],
            "user_id": user_ids,
            "amount": amounts.round(2),
            "merchant_id": rng.integers(1, 2000, size=n),
            "category": rng.choice(CATEGORIES, size=n, p=[0.25, 0.15, 0.10, 0.2, 0.2, 0.1]),
            "timestamp": timestamps,
            "device_type": rng.choice(DEVICE_TYPES, size=n, p=[0.6, 0.3, 0.1]),
            "channel": rng.choice(CHANNELS, size=n, p=[0.7, 0.3]),
            "country": countries,
            "city": rng.choice(
                ["Bangalore", "Mumbai", "Delhi", "Dubai", "London", "Singapore", "New York"],
                size=n,
            ),
            "entry_mode": rng.choice(ENTRY_MODES, size=n),
        }

        df = pd.DataFrame(data)
        df["is_international"] = (df["country"] != "IN").astype(int)
        df["is_fraud"] = is_fraud_flag
        return df

    legit_df = build_df(n_legit, legit_amounts, legit_user_ids, legit_countries, legit_timestamps, 0)
    fraud_df = build_df(n_fraud, fraud_amounts, fraud_user_ids, fraud_countries, fraud_timestamps, 1)

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def main():
    config = load_config()
    n_samples = config["data"]["n_samples"]
    fraud_ratio = config["data"]["fraud_ratio"]
    random_state = config["data"]["random_state"]
    raw_data_path = config["paths"]["raw_data"]

    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

    logger.info("Generating synthetic transactions...")
    df = generate_synthetic_transactions(n_samples, fraud_ratio, random_state)
    df.to_csv(raw_data_path, index=False)
    logger.info(f"Saved synthetic data to {raw_data_path} with shape {df.shape}")


if __name__ == "__main__":
    main()
