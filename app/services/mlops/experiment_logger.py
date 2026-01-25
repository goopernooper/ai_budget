from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Optional

from app.models import ExperimentRun


def _mlflow_enabled() -> bool:
    return (
        os.getenv("USE_MLFLOW", "false").lower() == "true"
        and os.getenv("MLFLOW_TRACKING_URI")
        is not None
    )


def _safe_mlflow():
    try:
        import mlflow

        return mlflow
    except Exception:
        return None


def start_run(
    db,
    run_type: str,
    model_family: str,
    model_name: str,
    params: Dict[str, object],
    dataset_hash: str,
    notes: Optional[str] = None,
) -> int:
    run = ExperimentRun(
        run_type=run_type,
        model_family=model_family,
        model_name=model_name,
        params_json=json.dumps(params),
        dataset_hash=dataset_hash,
        status="started",
        notes=notes,
        created_at=datetime.utcnow(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    if _mlflow_enabled():
        mlflow = _safe_mlflow()
        if mlflow:
            mlflow.set_experiment(f"{model_family}-{run_type}")
            with mlflow.start_run(run_name=f"{model_name}-{run.id}"):
                mlflow.log_params(params)
                mlflow.set_tag("dataset_hash", dataset_hash)
                mlflow.set_tag("run_id", run.id)

    return run.id


def end_run_success(
    db,
    run_id: int,
    metrics: Dict[str, object],
    artifact_path: str,
    model_version: str,
) -> None:
    run = db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
    if not run:
        return
    run.metrics_json = json.dumps(metrics)
    run.artifact_path = artifact_path
    run.model_version = model_version
    run.status = "success"
    db.commit()

    if _mlflow_enabled():
        mlflow = _safe_mlflow()
        if mlflow:
            mlflow.set_experiment(f"{run.model_family}-{run.run_type}")
            with mlflow.start_run(run_name=f"{run.model_name}-{run.id}"):
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                mlflow.log_param("model_version", model_version)
                if artifact_path:
                    try:
                        mlflow.log_artifacts(artifact_path)
                    except Exception:
                        pass


def end_run_failed(db, run_id: int, error_message: str) -> None:
    run = db.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
    if not run:
        return
    run.metrics_json = json.dumps({"error": error_message})
    run.status = "failed"
    db.commit()
