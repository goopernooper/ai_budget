from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import CategoryMap, Transaction
from app.services.categorizer import categorize_transactions, ensure_default_categories


def _db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return SessionLocal()


def test_categorizer_rules_and_overrides():
    db = _db_session()
    ensure_default_categories(db)
    db.add(CategoryMap(keyword="uber", category="Rideshare"))
    db.commit()

    txn1 = Transaction(date=date(2024, 1, 1), description="Uber trip", amount=-12.5)
    txn2 = Transaction(date=date(2024, 1, 2), description="Whole Foods Market", amount=-55)
    db.add_all([txn1, txn2])
    db.commit()

    categorize_transactions(db, [txn1, txn2], use_ml=False)

    assert txn1.category == "Rideshare"
    assert txn2.category == "Groceries"
