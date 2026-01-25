from __future__ import annotations

from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Transaction
from app.schemas import SummaryResponse, TransactionOut

router = APIRouter()


@router.get("/transactions", response_model=List[TransactionOut])
def list_transactions(
    category: Optional[str] = None,
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Transaction)
    if category:
        query = query.filter(Transaction.category == category)
    if start_date:
        query = query.filter(Transaction.date >= start_date)
    if end_date:
        query = query.filter(Transaction.date <= end_date)
    return query.order_by(Transaction.date.desc()).all()


@router.get("/transactions/summary", response_model=SummaryResponse)
def transactions_summary(db: Session = Depends(get_db)):
    transactions = db.query(Transaction).all()
    total_income = sum(t.amount for t in transactions if t.amount > 0)
    total_expenses = -sum(t.amount for t in transactions if t.amount < 0)
    net = total_income - total_expenses

    by_category = {}
    for category, total in (
        db.query(Transaction.category, func.sum(Transaction.amount))
        .group_by(Transaction.category)
    ):
        if category is None:
            category = "Uncategorized"
        by_category[category] = round(float(total), 2)

    by_month = {}
    for month, total in (
        db.query(func.strftime("%Y-%m", Transaction.date), func.sum(Transaction.amount))
        .group_by(func.strftime("%Y-%m", Transaction.date))
    ):
        by_month[month] = round(float(total), 2)

    return SummaryResponse(
        total_income=round(total_income, 2),
        total_expenses=round(total_expenses, 2),
        net=round(net, 2),
        by_category=by_category,
        by_month=by_month,
    )
