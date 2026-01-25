from datetime import date

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.models import Category, Transaction
from app.services.recommender import build_recommendations


def _db_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return SessionLocal()


def test_recommender_builds_budget_recs():
    db = _db_session()
    db.add_all(
        [
            Category(name="Groceries", type="variable"),
            Category(name="Rent", type="fixed"),
        ]
    )
    db.commit()

    transactions = []
    for month in [1, 2, 3]:
        transactions.append(
            Transaction(
                date=date(2024, month, 5),
                description="Whole Foods",
                amount=-120.0 - month * 5,
                category="Groceries",
            )
        )
        transactions.append(
            Transaction(
                date=date(2024, month, 1),
                description="Apartment Rent",
                amount=-1200.0,
                category="Rent",
            )
        )

    db.add_all(transactions)
    db.commit()

    recs = build_recommendations(db, transactions, "baseline_rolling_mean")
    assert recs
    for rec in recs:
        assert rec.recommended_budget >= rec.predicted_spend
