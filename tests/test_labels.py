from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Transaction, UserLabel
from app.services.labels import apply_user_label


def _db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return SessionLocal()


def test_apply_user_label_updates_transaction_and_stores_label():
    db = _db_session()
    transaction = Transaction(
        date=date(2024, 1, 5),
        description="Whole Foods",
        amount=-42.5,
        category="Shopping",
    )
    db.add(transaction)
    db.commit()
    db.refresh(transaction)

    updated = apply_user_label(db, transaction.id, "Groceries")

    assert updated.category == "Groceries"
    labels = db.query(UserLabel).all()
    assert len(labels) == 1
    assert labels[0].transaction_id == transaction.id
    assert labels[0].corrected_category == "Groceries"
