from __future__ import annotations

from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Transaction
from app.schemas import ModelEvalResponse, ModelTrainResponse
from app.services.categorizer import (
    CATEGORIZER_MODEL_NAME,
    ensure_default_categories,
    evaluate_model,
    train_model_artifacts,
)
from app.services.mlops.dataset_hash import compute_hash_from_dataframe, create_dataset_snapshot
from app.services.mlops.experiment_logger import end_run_failed, end_run_success, start_run
from app.services.mlops.registry import create_registry_entry, model_dir, save_model_artifact, write_metadata
from app.services.mlops.versioning import next_version

router = APIRouter()


@router.post("/model/train", response_model=ModelTrainResponse)
def train_categorizer(db: Session = Depends(get_db)):
    ensure_default_categories(db)
    transactions = db.query(Transaction).all()
    labeled_rows = [
        {
            "transaction_id": t.id,
            "description": t.description,
            "category": t.category,
        }
        for t in transactions
        if t.category
    ]
    df = pd.DataFrame(labeled_rows)
    dataset_hash = compute_hash_from_dataframe(df, ["transaction_id", "description", "category"])
    date_values = [t.date for t in transactions if isinstance(t.date, date)]
    date_min = min(date_values) if date_values else None
    date_max = max(date_values) if date_values else None
    create_dataset_snapshot(
        db,
        name="transactions",
        dataset_hash=dataset_hash,
        row_count=len(df),
        date_min=date_min,
        date_max=date_max,
    )
    params = {"vectorizer": {"max_features": 2000, "stop_words": "english"}, "classifier": {"max_iter": 500}}
    run_id = start_run(
        db,
        run_type="categorizer_train",
        model_family="categorizer",
        model_name=CATEGORIZER_MODEL_NAME,
        params=params,
        dataset_hash=dataset_hash,
    )
    try:
        artifacts, accuracy, labels, total = train_model_artifacts(transactions)
    except ValueError as exc:
        end_run_failed(db, run_id, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        version = next_version(db, "categorizer", CATEGORIZER_MODEL_NAME)
        dest = model_dir("categorizer", CATEGORIZER_MODEL_NAME, version)
        save_model_artifact(dest, artifacts)
        metrics = {"accuracy": accuracy, "labels": labels, "trained_on": total}
        write_metadata(dest, dataset_hash, params, metrics)
        create_registry_entry(db, "categorizer", CATEGORIZER_MODEL_NAME, version, stage="staging")
        end_run_success(db, run_id, metrics, str(dest), version)
    except Exception as exc:
        end_run_failed(db, run_id, str(exc))
        raise HTTPException(status_code=500, detail="Failed to register model") from exc

    return ModelTrainResponse(
        trained_on=total,
        labels=labels,
        accuracy=accuracy,
        version=version,
        dataset_hash=dataset_hash,
        run_id=run_id,
    )


@router.post("/model/evaluate", response_model=ModelEvalResponse)
def evaluate_categorizer(db: Session = Depends(get_db)):
    transactions = db.query(Transaction).all()
    try:
        accuracy, matrix, labels, total = evaluate_model(db, transactions)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ModelEvalResponse(total=total, accuracy=accuracy, confusion_matrix=matrix, labels=labels)
