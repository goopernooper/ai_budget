from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional

import pandas as pd

REQUIRED_COLUMNS = {"date", "description", "amount"}


def _clean_columns(columns):
    return [str(c).strip().lower() for c in columns]


def _parse_amount(value) -> Optional[float]:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        cleaned = cleaned.replace("$", "")
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_amount_sign(df: pd.DataFrame) -> pd.DataFrame:
    if "amount" not in df.columns:
        return df
    amount = df["amount"].dropna()
    if not amount.empty and (amount >= 0).all():
        type_col = None
        for candidate in ["type", "transaction_type", "txn_type"]:
            if candidate in df.columns:
                type_col = candidate
                break
        if type_col:
            type_vals = df[type_col].astype(str).str.lower()
            expense_mask = type_vals.str.contains("debit|withdraw|expense|purchase")
            df.loc[expense_mask, "amount"] = -df.loc[expense_mask, "amount"]
    return df


def _derive_merchant(description: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in description)
    tokens = [t for t in cleaned.upper().split() if t and t not in {"POS", "PURCHASE", "DEBIT", "CREDIT", "CARD"}]
    if not tokens:
        return description.strip()[:50]
    return " ".join(tokens[:2])


def parse_transactions_csv(contents: bytes) -> List[Dict[str, object]]:
    df = pd.read_csv(BytesIO(contents))
    df.columns = _clean_columns(df.columns)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["amount"] = df["amount"].apply(_parse_amount)
    df = _normalize_amount_sign(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["description"] = df["description"].fillna("").astype(str)
    if "category" in df.columns:
        df["category"] = df["category"].fillna("").astype(str)

    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        if pd.isna(row["date"]) or row["description"].strip() == "":
            continue
        amount = row["amount"]
        if amount is None or pd.isna(amount):
            continue
        description = row["description"].strip()
        record = {
            "date": row["date"],
            "description": description,
            "amount": float(amount),
            "category": row.get("category") or None,
            "merchant": _derive_merchant(description),
        }
        records.append(record)
    return records
