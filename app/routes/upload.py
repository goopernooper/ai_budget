from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Transaction
from app.schemas import UploadResponse
from app.services.categorizer import categorize_transactions, ensure_default_categories
from app.services.ingestion import parse_transactions_csv

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    contents = await file.read()
    try:
        records = parse_transactions_csv(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ensure_default_categories(db)

    transactions = []
    for record in records:
        transactions.append(
            Transaction(
                date=record["date"],
                description=record["description"],
                amount=record["amount"],
                category=record.get("category"),
                merchant=record.get("merchant"),
            )
        )

    db.add_all(transactions)
    db.commit()

    categorize_transactions(db, transactions, use_ml=True)

    return UploadResponse(inserted=len(transactions), skipped=0)
