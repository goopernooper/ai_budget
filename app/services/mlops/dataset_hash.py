from __future__ import annotations

import hashlib
from datetime import date
from typing import List, Optional

import pandas as pd

from app.models import DatasetSnapshot


def compute_hash_from_dataframe(df: pd.DataFrame, key_cols: List[str]) -> str:
    if df.empty:
        payload = ""
    else:
        data = df[key_cols].copy()
        data = data.sort_values(key_cols)
        payload = data.to_csv(
            index=False,
            float_format="%.6f",
            date_format="%Y-%m-%d",
        )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_dataset_snapshot(
    db,
    name: str,
    dataset_hash: str,
    row_count: int,
    date_min: Optional[date],
    date_max: Optional[date],
) -> DatasetSnapshot:
    snapshot = DatasetSnapshot(
        name=name,
        hash=dataset_hash,
        row_count=row_count,
        date_min=date_min,
        date_max=date_max,
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot
