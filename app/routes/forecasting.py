from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Transaction
from app.schemas import BacktestRequest, BacktestResponse, ForecastModelsResponse, ForecastResponse
from app.services.forecasting import (
    MODEL_BASELINE,
    MODEL_LIGHTGBM,
    MODEL_PROPHET,
    MODEL_SARIMA,
    backtest_all_categories,
    build_forecast_dataset,
    forecast_all_categories,
    get_available_models,
    fit_registry_sarima,
    train_registry_forecast_model,
)
from app.services.mlops.dataset_hash import compute_hash_from_dataframe, create_dataset_snapshot
from app.services.mlops.experiment_logger import end_run_failed, end_run_success, start_run
from app.services.mlops.registry import (
    create_registry_entry,
    get_production_pointer,
    model_dir,
    save_model_artifact,
    write_metadata,
)
from app.services.mlops.versioning import next_version

router = APIRouter()


@router.get("/forecast", response_model=ForecastResponse)
def forecast(
    month: Optional[str] = Query(default=None),
    horizon: int = Query(default=1, ge=1, le=24),
    model: Optional[str] = Query(default=None),
    force: bool = Query(default=False),
    strict: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    if model is None:
        pointer = get_production_pointer(db, "forecast")
        model = pointer.model_name if pointer else MODEL_BASELINE
    if model not in {MODEL_BASELINE, MODEL_SARIMA, MODEL_PROPHET, MODEL_LIGHTGBM}:
        raise HTTPException(status_code=400, detail="Unknown model")
    transactions = db.query(Transaction).all()
    try:
        payload = forecast_all_categories(
            db,
            transactions,
            model_name=model,
            start_month_str=month,
            horizon=horizon,
            force=force,
            strict=strict,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ForecastResponse(**payload)


@router.get("/forecast/models", response_model=ForecastModelsResponse)
def list_models():
    models = get_available_models()
    return ForecastModelsResponse(**models)


@router.post("/forecast/backtest", response_model=BacktestResponse)
def backtest(payload: BacktestRequest, db: Session = Depends(get_db)):
    transactions = db.query(Transaction).all()
    result = backtest_all_categories(
        transactions,
        model_names=payload.models,
        top_k_categories=payload.top_k_categories,
        min_train_months=payload.min_train_months,
        strict=False,
    )
    dataset_df = build_forecast_dataset(transactions)
    dataset_hash = compute_hash_from_dataframe(dataset_df, ["month", "category", "y"])
    date_values = []
    if not dataset_df.empty:
        date_values = [
            pd.to_datetime(row).date() for row in dataset_df["month"].unique().tolist()
        ]
    date_min = min(date_values) if date_values else None
    date_max = max(date_values) if date_values else None
    create_dataset_snapshot(
        db,
        name="forecast_monthly",
        dataset_hash=dataset_hash,
        row_count=len(dataset_df),
        date_min=date_min,
        date_max=date_max,
    )

    for model_name in payload.models:
        params = {
            "top_k_categories": payload.top_k_categories,
            "min_train_months": payload.min_train_months,
            "requested_models": payload.models,
        }
        run_id = start_run(
            db,
            run_type="forecast_backtest",
            model_family="forecast",
            model_name=model_name,
            params=params,
            dataset_hash=dataset_hash,
        )
        try:
            version = next_version(db, "forecast", model_name)
            dest = model_dir("forecast", model_name, version)
            model_metrics = {
                "overall": result.get("overall", {}).get(model_name),
                "categories": [
                    row for row in result.get("categories", []) if row["model_requested"] == model_name
                ],
                "errors": result.get("errors", []),
            }

            if model_name == MODEL_LIGHTGBM:
                try:
                    model, features = train_registry_forecast_model(transactions)
                    save_model_artifact(dest, {"model": model, "features": features})
                except Exception as exc:
                    model_metrics["artifact_error"] = str(exc)
            elif model_name == MODEL_SARIMA:
                try:
                    monthly = (
                        dataset_df.groupby("month")["y"].sum().reset_index().sort_values("month")
                    )
                    series_df = pd.DataFrame(
                        {
                            "ds": pd.to_datetime(monthly["month"]) + pd.offsets.MonthBegin(0),
                            "y": monthly["y"],
                        }
                    )
                    fitted = fit_registry_sarima(series_df)
                    save_model_artifact(dest, fitted)
                except Exception as exc:
                    model_metrics["artifact_error"] = str(exc)

            write_metadata(dest, dataset_hash, params, model_metrics)
            create_registry_entry(db, "forecast", model_name, version, stage="staging")
            end_run_success(db, run_id, model_metrics, str(dest), version)
        except Exception as exc:
            end_run_failed(db, run_id, str(exc))

    return BacktestResponse(**result)
