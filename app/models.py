from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from app.db import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)
    description = Column(Text, nullable=False)
    amount = Column(Float, nullable=False)
    category = Column(String(100), index=True)
    merchant = Column(String(150), index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True, nullable=False)
    type = Column(String(20), default="variable", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CategoryMap(Base):
    __tablename__ = "category_maps"

    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String(100), index=True, nullable=False)
    category = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class BudgetRecommendation(Base):
    __tablename__ = "budget_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(100), index=True, nullable=False)
    predicted_spend = Column(Float, nullable=False)
    recommended_budget = Column(Float, nullable=False)
    buffer = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    explanation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class UserLabel(Base):
    __tablename__ = "user_labels"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, ForeignKey("transactions.id"), index=True, nullable=False)
    corrected_category = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ForecastCache(Base):
    __tablename__ = "forecast_cache"

    id = Column(Integer, primary_key=True, index=True)
    month = Column(String(7), index=True, nullable=False)
    category = Column(String(100), index=True, nullable=False)
    model_name = Column(String(50), index=True, nullable=False)
    yhat = Column(Float, nullable=False)
    lower = Column(Float)
    upper = Column(Float)
    metrics_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DatasetSnapshot(Base):
    __tablename__ = "dataset_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    hash = Column(String(64), index=True, nullable=False)
    row_count = Column(Integer, nullable=False)
    date_min = Column(Date)
    date_max = Column(Date)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_type = Column(String(50), nullable=False)
    model_family = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(20))
    params_json = Column(Text)
    metrics_json = Column(Text)
    dataset_hash = Column(String(64), index=True)
    artifact_path = Column(Text)
    status = Column(String(20), nullable=False, default="started")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(Text)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (
        UniqueConstraint("model_family", "model_name", "version", name="uq_model_registry"),
    )

    id = Column(Integer, primary_key=True, index=True)
    model_family = Column(String(50), nullable=False)
    model_name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    stage = Column(String(20), nullable=False, default="staging")
    promoted_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class ProductionModelPointer(Base):
    __tablename__ = "production_model_pointers"

    model_family = Column(String(50), primary_key=True)
    model_name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
