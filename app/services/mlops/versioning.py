from __future__ import annotations

from app.models import ModelRegistry


def next_version(db, model_family: str, model_name: str) -> str:
    rows = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_family == model_family,
            ModelRegistry.model_name == model_name,
        )
        .all()
    )
    if not rows:
        return "v0001"
    numeric_versions = []
    for row in rows:
        if row.version and row.version.startswith("v"):
            try:
                numeric_versions.append(int(row.version[1:]))
            except ValueError:
                continue
    next_num = max(numeric_versions) + 1 if numeric_versions else 1
    return f"v{next_num:04d}"
