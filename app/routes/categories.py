from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Category, CategoryMap
from app.schemas import CategoryMapCreate, CategoryOut
from app.services.categorizer import ensure_default_categories

router = APIRouter()


@router.get("/categories", response_model=list[CategoryOut])
def list_categories(db: Session = Depends(get_db)):
    ensure_default_categories(db)
    return db.query(Category).order_by(Category.name).all()


@router.post("/categories/map")
def create_category_map(payload: CategoryMapCreate, db: Session = Depends(get_db)):
    ensure_default_categories(db)
    mapping = CategoryMap(keyword=payload.keyword.strip(), category=payload.category.strip())
    db.add(mapping)
    if not db.query(Category).filter(Category.name == payload.category).first():
        db.add(Category(name=payload.category.strip(), type="variable"))
    db.commit()
    return {"status": "ok", "id": mapping.id}
