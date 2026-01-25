from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import InsightQuery, InsightResponse
from app.services.insights import run_insights

router = APIRouter()


@router.post("/insights/query", response_model=InsightResponse)
def query_insights(payload: InsightQuery, db: Session = Depends(get_db)):
    result = run_insights(db, payload.question)
    return InsightResponse(**result)
