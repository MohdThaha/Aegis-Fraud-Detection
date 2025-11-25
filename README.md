
# Aegis — Phase 1 (Core Engine) README

**Project:** Aegis — Real-time Transaction Fraud Detection  
**Phase 1:** Offline core engine — synthetic data generation, feature engineering, model training, and a simple prediction API.

---

## 1. Quick download
[Download this README](sandbox:/mnt/data/README.md)

---

## 2. What this README contains
- How to run Phase 1 (exact commands)
- Folder & file checklist 
- Deep explanation of Phase 1 components 
- Common errors and fixes
- Notes about `exploration.ipynb`

---

## 3. Prerequisites

1. Python 3.10+ (you used 3.13 — OK).
2. Virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux / macOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 4. Exact run commands 

Run these from the **project root**.

1. Generate synthetic data (creates `data/raw/transactions.csv`):
```bash
python -m src.data_gen.generate_transactions
```

2. Train models and save best model (creates `data/processed/transactions_features.csv`, `data/models/best_model.joblib`, `data/processed/metrics.json`):
```bash
python -m src.training.train
```

3. (Optional) Evaluate on the full dataset:
```bash
python -m src.training.evaluate
```
> **NOTE:** see troubleshooting below — `evaluate.py` in the provided code expects raw columns (including `timestamp`) but may attempt to read `data/processed/transactions_features.csv`. If you see `KeyError: 'timestamp'` see the Fix section.

4. Run the prediction API (model served with FastAPI):
```bash
uvicorn src.serving.main:app --reload
```

5. Test the API with the local test client:
```bash
python -m src.testing.local_test_client
# or use curl/postman:
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_tx.json
```

---

## 5. File outputs you should now see (after steps above)

```
data/
├─ raw/transactions.csv
├─ processed/transactions_features.csv
├─ processed/metrics.json
└─ models/best_model.joblib
```

---

## 6. Deep: What Phase 1 does and why

### Overview
Phase 1 establishes the **core engine**: synthetic data, feature engineering, baseline models, and a lightweight prediction API. This creates a reliable base for later adding real-time streaming (Kafka/Kinesis), model deployment, and production monitoring.

### 6.1 Data creation
**Why synthetic data?**
- Real transaction data is sensitive and often unavailable at the start.
- Synthetic data lets you design realistic distributions and fraud patterns to develop and validate your pipeline.
- It allows reproducible experiments (same random seed).

**How the generator works**
- The generator produces `n_samples` transactions with features like `transaction_id`, `user_id`, `amount`, `merchant_id`, `category`, `timestamp`, `device_type`, `channel`, `country`, `city`, `entry_mode`.
- It creates a small fraction of transactions labeled as fraud (`fraud_ratio`, e.g., 3%).
- Fraud transactions are sampled from different distributions (e.g., larger average amounts and more foreign countries) to create detectable patterns.

**Key idea:** We intentionally give the fraud class signals (high amount, international flag, unusual countries) so your models can learn meaningful patterns. In production, fraud patterns are subtler and require continuous updating.

### 6.2 Feature engineering
**Why feature engineering?**
- Raw columns (merchant names, timestamps) are rarely ideal for ML.
- Convert timestamps to hour-of-day, night flag — fraud often happens at odd hours.
- Bin or flag high amounts (`high_amount_flag`) — amount distribution matters.
- Map categorical fields to numeric codes for model input (deterministic mapping — good for serving).

**What the code does**
- `FeatureBuilder.add_features()`:
  - Parses `timestamp` to `hour` and `is_night`.
  - Adds `high_amount_flag` for amounts > 5000.
  - Encodes `category`, `device_type`, `channel`, `entry_mode`, `country` to integer codes.
- This keeps the feature set small and fast to compute in real-time.

### 6.3 Model choices
**Logistic Regression (LogReg)**
- Fast, interpretable baseline.
- Works well with good features and balanced classes.
- Gives probability output which is useful for thresholds.

**Random Forest**
- Non-linear ensemble model — handles interactions and mixed feature types well.
- Robust to noise and often achieves higher ROC-AUC on tabular data.
- Handles imbalanced datasets better when combined with class weights.

**Why start with these two?**
- They are standard, fast to train, and give strong baselines.
- Random Forest gave better ROC-AUC in Phase 1, so it becomes the selected model.

**On class imbalance**
- Fraud is rare → class imbalance.
- We compute class weights (`sklearn.utils.compute_class_weight`) and pass them to models so they give appropriate importance to the minority class.
- Alternative strategies (later): oversampling (SMOTE), anomaly detection, or cost-sensitive learning.

### 6.4 Training & testing (procedure)
1. Load raw data (`transactions.csv`).
2. Build features (FeatureBuilder).
3. Split into train/validation with stratification on the target (so both sets contain fraud examples).
4. Train candidate models on training set.
5. Validate on validation set using ROC-AUC (area under ROC) as the main metric — good for imbalanced classification because it measures ranking quality across thresholds.
6. Choose best model by ROC-AUC, save it (joblib).
7. Evaluate final model with discrete metrics (precision, recall, F1) at threshold 0.5 (these reflect a single operating point) and save metrics.

**Why stratify?**
- Fraud is rare; without stratification, a split may accidentally produce few or no fraud examples in validation or training.

**Why ROC-AUC as primary metric?**
- It measures a model's ranking ability irrespective of threshold and is stable for imbalanced classes.

---

## 7. Troubleshooting: `KeyError: 'timestamp'` when running `evaluate.py`

You observed:
```
KeyError: 'timestamp'
Traceback ...
File ... evaluate.py line X: X, y = fb.build_xy(df, include_target=True)
```

**Cause**
- `evaluate.py` was loading `data/processed/transactions_features.csv` (which — in the Phase 1 `train.py` — contains only the engineered feature columns and `is_fraud`, **not** the raw `timestamp` column).
- `FeatureBuilder.build_xy()` expects raw columns (including `timestamp`, `category`, etc.) so it can recompute features. When `evaluate.py` passes the processed CSV (without `timestamp`) into `build_xy`, the code fails.

**Two ways to fix it**

**Option A — Quick fix (recommended):**  
Make `evaluate.py` load the raw data (the same input used for feature building) instead of `processed_data`. Replace:

```py
processed_path = config["paths"]["processed_data"]
df = pd.read_csv(processed_path)
```

with:

```py
raw_path = config["paths"]["raw_data"]
df = pd.read_csv(raw_path)
```

Then proceed — `FeatureBuilder` will re-create features from raw data and evaluation will work fine.

**Option B — Use processed features directly (alternative):**  
If you want `evaluate.py` to read the processed CSV and evaluate without recomputing features, modify `evaluate.py` to:
- Read `processed/transactions_features.csv` into `df`.
- Separate X = df[feature_columns] and y = df['is_fraud'] (no `FeatureBuilder` needed).
This requires the `processed` CSV to contain exactly the feature columns used during training.

---

## 8. `exploration.ipynb` note
You asked if `exploration.ipynb` is empty — yes, I left it as a placeholder. Use it for EDA:
Suggested first cells:
```python
import pandas as pd
df = pd.read_csv("data/raw/transactions.csv")
df.head()
df.describe()
df['is_fraud'].value_counts(normalize=True)
```
Plot histograms of `amount`, boxplots for fraud vs non-fraud, and time-of-day fraud counts.

---

## 9. Suggested next steps after Phase 1 (roadmap highlights)
- Phase 2: Replace generator with Kafka or AWS Kinesis producer; add consumer that computes features and calls `/predict`.
- Phase 3: Add model versioning (MLflow) and ONNX export.
- Phase 4: Deploy to Kubernetes / ECS and add monitoring + real human feedback loop.

---

## 10. Common quick tips & gotchas
- If `train.py` gave a `ConvergenceWarning` for LogisticRegression: increase `max_iter` or scale features.
- When running API: ensure you ran `train.py` first (so `data/models/best_model.joblib` exists).
- If you change feature mapping dictionaries in `features/utils.py`, you must re-train models.

---

## 11. Useful commands summary

```bash
# setup
python -m venv .venv
.venv\\Scripts\\activate      # windows
pip install -r requirements.txt

# phase 1
python -m src.data_gen.generate_transactions
python -m src.training.train

# if evaluate fails with timestamp KeyError, either:
# EDIT src/training/evaluate.py to load raw data (see Troubleshooting), then:
python -m src.training.evaluate

# run API
uvicorn src.serving.main:app --reload

# test
python -m src.testing.local_test_client
```

---

### End — Good luck!