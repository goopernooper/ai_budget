from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import BudgetRecommendation, Transaction
from app.schemas import BudgetRecommendationOut
from app.services.mlops.registry import get_production_pointer
from app.services.recommender import build_recommendations

router = APIRouter()


@router.get("/recommendations", response_model=list[BudgetRecommendationOut])
def get_recommendations(
    forecast_model: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    transactions = db.query(Transaction).all()
    db.query(BudgetRecommendation).delete(synchronize_session=False)
    db.commit()
    if forecast_model is None:
        pointer = get_production_pointer(db, "forecast")
        forecast_model = pointer.model_name if pointer else "baseline_rolling_mean"
    recommendations = build_recommendations(db, transactions, forecast_model)
    db.add_all(recommendations)
    db.commit()
    return db.query(BudgetRecommendation).all()
