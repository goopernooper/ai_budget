from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import ExperimentRun, ModelRegistry, ProductionModelPointer
from app.schemas import (
    ExperimentRunOut,
    ModelRegistryOut,
    ProductionPointerOut,
    PromoteRequest,
)
from app.services.mlops.registry import model_dir, promote_model

router = APIRouter()


@router.post("/mlops/promote", response_model=ProductionPointerOut)
def promote(payload: PromoteRequest, db: Session = Depends(get_db)):
    try:
        pointer = promote_model(
            db, payload.model_family, payload.model_name, payload.version
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ProductionPointerOut(
        model_family=pointer.model_family,
        model_name=pointer.model_name,
        version=pointer.version,
        updated_at=pointer.updated_at,
    )


@router.get("/mlops/registry", response_model=list[ModelRegistryOut])
def list_registry(
    model_family: Optional[str] = Query(default=None),
    model_name: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(ModelRegistry)
    if model_family:
        query = query.filter(ModelRegistry.model_family == model_family)
    if model_name:
        query = query.filter(ModelRegistry.model_name == model_name)
    entries = query.order_by(ModelRegistry.created_at.desc()).all()
    results = []
    for entry in entries:
        meta_path = model_dir(entry.model_family, entry.model_name, entry.version) / "metadata.json"
        dataset_hash = None
        metrics = None
        if meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text())
                dataset_hash = payload.get("dataset_hash")
                metrics = payload.get("metrics")
            except json.JSONDecodeError:
                pass
        results.append(
            ModelRegistryOut(
                model_family=entry.model_family,
                model_name=entry.model_name,
                version=entry.version,
                stage=entry.stage,
                promoted_at=entry.promoted_at,
                created_at=entry.created_at,
                dataset_hash=dataset_hash,
                metrics=metrics,
            )
        )
    return results


@router.get("/mlops/runs", response_model=list[ExperimentRunOut])
def list_runs(
    model_family: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = db.query(ExperimentRun)
    if model_family:
        query = query.filter(ExperimentRun.model_family == model_family)
    return query.order_by(ExperimentRun.created_at.desc()).limit(limit).all()


@router.get("/mlops/production", response_model=list[ProductionPointerOut])
def production_pointers(db: Session = Depends(get_db)):
    return db.query(ProductionModelPointer).all()
