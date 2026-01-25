from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib

from app.models import ModelRegistry, ProductionModelPointer


def registry_base_dir() -> Path:
    base = os.getenv("MODELS_DIR") or os.getenv("MLOPS_REGISTRY_DIR", "models_registry")
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir(model_family: str, model_name: str, version: str) -> Path:
    return registry_base_dir() / model_family / model_name / version


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def write_metadata(
    dest: Path,
    dataset_hash: str,
    params: Dict[str, object],
    metrics: Dict[str, object],
) -> Path:
    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "dataset_hash": dataset_hash,
        "params": params,
        "metrics": metrics,
        "code_version": _git_commit(),
        "python_version": platform.python_version(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    meta_path = dest / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    return meta_path


def save_model_artifact(dest: Path, model_obj, filename: str = "model.pkl") -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / filename
    joblib.dump(model_obj, path)
    return path


def create_registry_entry(
    db,
    model_family: str,
    model_name: str,
    version: str,
    stage: str = "staging",
) -> ModelRegistry:
    entry = ModelRegistry(
        model_family=model_family,
        model_name=model_name,
        version=version,
        stage=stage,
        created_at=datetime.utcnow(),
        promoted_at=None,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def get_production_pointer(db, model_family: str) -> Optional[ProductionModelPointer]:
    return (
        db.query(ProductionModelPointer)
        .filter(ProductionModelPointer.model_family == model_family)
        .first()
    )


def resolve_active_model(
    db, model_family: str, model_name: str
) -> Optional[Tuple[str, str]]:
    pointer = get_production_pointer(db, model_family)
    if pointer and pointer.model_name == model_name:
        return pointer.model_name, pointer.version

    latest = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_family == model_family,
            ModelRegistry.model_name == model_name,
            ModelRegistry.stage == "staging",
        )
        .order_by(ModelRegistry.created_at.desc())
        .first()
    )
    if latest:
        return latest.model_name, latest.version
    return None


def promote_model(
    db, model_family: str, model_name: str, version: str
) -> ProductionModelPointer:
    entry = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_family == model_family,
            ModelRegistry.model_name == model_name,
            ModelRegistry.version == version,
        )
        .first()
    )
    if not entry:
        raise ValueError("Model version not found")

    db.query(ModelRegistry).filter(
        ModelRegistry.model_family == model_family,
        ModelRegistry.model_name == model_name,
        ModelRegistry.stage == "production",
    ).update({"stage": "archived"}, synchronize_session=False)

    entry.stage = "production"
    entry.promoted_at = datetime.utcnow()

    pointer = get_production_pointer(db, model_family)
    if pointer:
        pointer.model_name = model_name
        pointer.version = version
        pointer.updated_at = datetime.utcnow()
    else:
        pointer = ProductionModelPointer(
            model_family=model_family,
            model_name=model_name,
            version=version,
            updated_at=datetime.utcnow(),
        )
        db.add(pointer)

    db.commit()
    db.refresh(pointer)
    return pointer
