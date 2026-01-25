from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.engine import make_url

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models_registry"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))


def ensure_storage_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    url = make_url(DATABASE_URL)
    if url.drivername.startswith("sqlite") and url.database:
        if url.database != ":memory:":
            db_path = Path(url.database)
            if not db_path.is_absolute():
                db_path = Path.cwd() / db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
