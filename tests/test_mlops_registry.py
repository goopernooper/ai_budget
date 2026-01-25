from datetime import date

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Transaction
from app.services.categorizer import CATEGORIZER_MODEL_NAME, categorize_transactions, train_model_artifacts
from app.services.mlops.dataset_hash import compute_hash_from_dataframe, create_dataset_snapshot
from app.services.mlops.registry import create_registry_entry, model_dir, promote_model, save_model_artifact, write_metadata
from app.services.mlops.versioning import next_version


def _db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return SessionLocal()


def test_mlops_registry_and_production_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    db = _db_session()

    transactions = []
    for idx in range(10):
        transactions.append(
            Transaction(
                date=date(2024, 1, 10),
                description=f"Blue Sky Widgets {idx}",
                amount=-10.0,
                category="Shopping",
            )
        )
        transactions.append(
            Transaction(
                date=date(2024, 1, 12),
                description=f"Green Leaf Market {idx}",
                amount=-12.0,
                category="Groceries",
            )
        )

    db.add_all(transactions)
    db.commit()

    artifacts, accuracy, labels, total = train_model_artifacts(transactions)
    df = pd.DataFrame(
        [
            {"transaction_id": t.id, "description": t.description, "category": t.category}
            for t in transactions
        ]
    )
    dataset_hash = compute_hash_from_dataframe(df, ["transaction_id", "description", "category"])
    create_dataset_snapshot(
        db,
        name="transactions",
        dataset_hash=dataset_hash,
        row_count=len(df),
        date_min=date(2024, 1, 10),
        date_max=date(2024, 1, 12),
    )

    version = next_version(db, "categorizer", CATEGORIZER_MODEL_NAME)
    dest = model_dir("categorizer", CATEGORIZER_MODEL_NAME, version)
    save_model_artifact(dest, artifacts)
    metrics = {"accuracy": accuracy, "labels": labels, "trained_on": total}
    params = {"max_features": 2000, "max_iter": 500}
    write_metadata(dest, dataset_hash, params, metrics)
    create_registry_entry(db, "categorizer", CATEGORIZER_MODEL_NAME, version, stage="staging")

    pointer = promote_model(db, "categorizer", CATEGORIZER_MODEL_NAME, version)
    assert pointer.version == version

    new_txn = Transaction(
        date=date(2024, 2, 1),
        description="Blue Sky Widgets 99",
        amount=-15.0,
    )
    db.add(new_txn)
    db.commit()

    categorize_transactions(db, [new_txn], use_ml=True)
    assert new_txn.category == "Shopping"
