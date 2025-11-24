from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    transaction_id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    user_id: int = Field(..., example=1234)
    amount: float = Field(..., gt=0, example=1999.99)
    merchant_id: int = Field(..., example=567)
    category: str = Field(..., example="electronics")
    timestamp: str = Field(..., example="2025-01-01T10:15:30")
    device_type: str = Field(..., example="mobile")
    channel: str = Field(..., example="online")
    country: str = Field(..., example="IN")
    city: str = Field(..., example="Bangalore")
    entry_mode: str = Field(..., example="online")
    is_international: int = Field(..., ge=0, le=1, example=0)
