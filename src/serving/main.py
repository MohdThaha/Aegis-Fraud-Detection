from fastapi import FastAPI

from src.serving.predict import FraudModelService
from src.serving.schema import TransactionInput

app = FastAPI(
    title="Aegis Fraud Detection API",
    version="0.1.0",
    description="Core fraud detection engine - Phase 1",
)

model_service = FraudModelService()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_service._model is not None}


@app.post("/predict")
def predict(tx: TransactionInput):
    result = model_service.predict(tx)
    return result
