from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from app.config import ensure_storage_dirs
from app.db import Base, engine
from app.routes import (
    categories,
    forecasting,
    insights,
    labels,
    mlops,
    recommendations,
    train,
    transactions,
    upload,
)

app = FastAPI(title="AI Finance Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    ensure_storage_dirs()
    Base.metadata.create_all(bind=engine)


app.include_router(upload.router)
app.include_router(transactions.router)
app.include_router(categories.router)
app.include_router(recommendations.router)
app.include_router(insights.router)
app.include_router(train.router)
app.include_router(labels.router)
app.include_router(forecasting.router)
app.include_router(mlops.router)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
def index():
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}
