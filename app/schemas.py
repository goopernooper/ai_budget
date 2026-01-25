from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionBase(BaseModel):
    date: date
    description: str
    amount: float
    category: Optional[str] = None
    merchant: Optional[str] = None


class TransactionCreate(TransactionBase):
    pass


class TransactionOut(TransactionBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True


class CategoryOut(BaseModel):
    id: int
    name: str
    type: str

    class Config:
        orm_mode = True


class CategoryMapCreate(BaseModel):
    keyword: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)


class UserLabelCreate(BaseModel):
    transaction_id: int
    corrected_category: str = Field(..., min_length=1)


class BudgetRecommendationOut(BaseModel):
    id: int
    category: str
    predicted_spend: float
    recommended_budget: float
    buffer: float
    volatility: float
    explanation: str
    created_at: datetime

    class Config:
        orm_mode = True


class UploadResponse(BaseModel):
    inserted: int
    skipped: int


class SummaryResponse(BaseModel):
    total_income: float
    total_expenses: float
    net: float
    by_category: Dict[str, float]
    by_month: Dict[str, float]


class ModelTrainResponse(BaseModel):
    trained_on: int
    labels: List[str]
    accuracy: float
    version: Optional[str] = None
    dataset_hash: Optional[str] = None
    run_id: Optional[int] = None


class ModelEvalResponse(BaseModel):
    total: int
    accuracy: float
    confusion_matrix: List[List[int]]
    labels: List[str]


class InsightQuery(BaseModel):
    question: str


class InsightResponse(BaseModel):
    intent: str
    data: Dict[str, Any]
    summary: str


class UncertainTransactionOut(TransactionOut):
    predicted_category: str
    confidence: float


class ForecastItem(BaseModel):
    category: str
    month: str
    model_requested: str
    model_used: str
    yhat: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    from_cache: bool = False
    error: Optional[str] = None


class ForecastResponse(BaseModel):
    start_month: str
    horizon: int
    model_requested: str
    forecasts: List[ForecastItem]


class ForecastModelsResponse(BaseModel):
    baseline_rolling_mean: bool
    sarima: bool
    prophet: bool
    lightgbm: bool
    xgboost: bool


class BacktestRequest(BaseModel):
    models: List[str] = Field(default_factory=lambda: ["baseline_rolling_mean"])
    top_k_categories: int = 15
    min_train_months: int = 6


class BacktestResult(BaseModel):
    category: str
    model_requested: str
    model_used: str
    mae: Optional[float]
    mape: Optional[float]
    count: int
    error: Optional[str] = None


class BacktestResponse(BaseModel):
    overall: Dict[str, Dict[str, Optional[float]]]
    categories: List[BacktestResult]
    errors: List[str]


class PromoteRequest(BaseModel):
    model_family: str
    model_name: str
    version: str


class ModelRegistryOut(BaseModel):
    model_family: str
    model_name: str
    version: str
    stage: str
    promoted_at: Optional[datetime]
    created_at: datetime
    dataset_hash: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class ExperimentRunOut(BaseModel):
    id: int
    run_type: str
    model_family: str
    model_name: str
    model_version: Optional[str]
    params_json: Optional[str]
    metrics_json: Optional[str]
    dataset_hash: Optional[str]
    artifact_path: Optional[str]
    status: str
    created_at: datetime
    notes: Optional[str]


class ProductionPointerOut(BaseModel):
    model_family: str
    model_name: str
    version: str
    updated_at: datetime
