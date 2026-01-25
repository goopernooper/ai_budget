from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Category, Transaction, UserLabel
from app.services.categorizer import ensure_default_categories


def apply_user_label(db: Session, transaction_id: int, corrected_category: str) -> Transaction:
    transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    if not transaction:
        raise LookupError("Transaction not found")

    category = corrected_category.strip()
    if not category:
        raise ValueError("Corrected category cannot be empty")

    ensure_default_categories(db)

    transaction.category = category
    db.add(UserLabel(transaction_id=transaction.id, corrected_category=category))

    if not db.query(Category).filter(Category.name == category).first():
        db.add(Category(name=category, type="variable"))

    db.commit()
    db.refresh(transaction)
    return transaction
