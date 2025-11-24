import requests
import uuid
from datetime import datetime

from src.data_gen.generate_transactions import CATEGORIES, DEVICE_TYPES, CHANNELS, ENTRY_MODES, COUNTRIES


def build_sample_transaction():
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": 1234,
        "amount": 7500.0,
        "merchant_id": 456,
        "category": "electronics",
        "timestamp": datetime.utcnow().isoformat(),
        "device_type": "mobile",
        "channel": "online",
        "country": "US",
        "city": "New York",
        "entry_mode": "online",
        "is_international": 1,
    }


def main():
    url = "http://127.0.0.1:8000/predict"
    sample_tx = build_sample_transaction()
    print("Sending sample transaction:\n", sample_tx)

    resp = requests.post(url, json=sample_tx)
    print("\nResponse:")
    print(resp.status_code, resp.json())


if __name__ == "__main__":
    main()
