from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Transaction
from app.schemas import TransactionOut, UncertainTransactionOut, UserLabelCreate
from app.services.categorizer import predict_with_confidence
from app.services.labels import apply_user_label

router = APIRouter()


@router.post("/labels/correct", response_model=TransactionOut)
def correct_label(payload: UserLabelCreate, db: Session = Depends(get_db)):
    try:
        transaction = apply_user_label(
            db, payload.transaction_id, payload.corrected_category
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return transaction


@router.get("/labels/uncertain", response_model=list[UncertainTransactionOut])
def uncertain_labels(limit: int = Query(10, ge=1, le=100), db: Session = Depends(get_db)):
    transactions = db.query(Transaction).order_by(Transaction.date.desc()).all()
    if not transactions:
        return []

    descriptions = [t.description for t in transactions]
    try:
        labels, confidences = predict_with_confidence(db, descriptions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    records = []
    for transaction, label, confidence in zip(transactions, labels, confidences):
        item = TransactionOut.from_orm(transaction).dict()
        item.update({"predicted_category": label, "confidence": confidence})
        records.append(item)

    records.sort(key=lambda x: x["confidence"])
    return [UncertainTransactionOut(**item) for item in records[:limit]]
