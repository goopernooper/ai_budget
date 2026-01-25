from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from app.models import Category, CategoryMap, Transaction
from app.services.mlops.registry import model_dir, resolve_active_model

CATEGORIZER_MODEL_NAME = "tfidf_logreg"
MODEL_PATH = Path("models/categorizer.joblib")

_CACHED_ARTIFACTS: Optional["ModelArtifacts"] = None
_CACHED_VERSION: Optional[str] = None

DEFAULT_RULES: Dict[str, str] = {
    "uber": "Transport",
    "lyft": "Transport",
    "shell": "Transport",
    "chevron": "Transport",
    "exxon": "Transport",
    "whole foods": "Groceries",
    "trader joe": "Groceries",
    "walmart": "Groceries",
    "target": "Shopping",
    "amazon": "Shopping",
    "netflix": "Subscriptions",
    "spotify": "Subscriptions",
    "hulu": "Subscriptions",
    "rent": "Rent",
    "mortgage": "Rent",
    "electric": "Utilities",
    "water": "Utilities",
    "comcast": "Utilities",
    "verizon": "Utilities",
    "payroll": "Income",
    "salary": "Income",
    "deposit": "Income",
    "restaurant": "Dining",
    "cafe": "Dining",
    "starbucks": "Dining",
    "gym": "Health",
    "pharmacy": "Health",
    "doctor": "Health",
    "flight": "Travel",
    "hotel": "Travel",
    "airbnb": "Travel",
    "movie": "Entertainment",
    "cinema": "Entertainment",
}

DEFAULT_CATEGORY_TYPES = {
    "Income": "fixed",
    "Rent": "fixed",
    "Utilities": "fixed",
    "Subscriptions": "fixed",
    "Groceries": "variable",
    "Transport": "variable",
    "Dining": "variable",
    "Entertainment": "variable",
    "Health": "variable",
    "Shopping": "variable",
    "Travel": "variable",
    "Education": "variable",
    "Misc": "variable",
}


@dataclass
class ModelArtifacts:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    labels: List[str]


def ensure_default_categories(db) -> None:
    existing = {c.name for c in db.query(Category).all()}
    for name, cat_type in DEFAULT_CATEGORY_TYPES.items():
        if name not in existing:
            db.add(Category(name=name, type=cat_type))
    db.commit()


def _load_user_maps(db) -> List[CategoryMap]:
    return db.query(CategoryMap).all()


def _apply_rules(description: str, user_maps: Iterable[CategoryMap]) -> Optional[str]:
    lowered = description.lower()
    for mapping in user_maps:
        if mapping.keyword.lower() in lowered:
            return mapping.category
    for keyword, category in DEFAULT_RULES.items():
        if keyword in lowered:
            return category
    return None


def _load_active_artifacts(db) -> Optional[ModelArtifacts]:
    global _CACHED_ARTIFACTS, _CACHED_VERSION
    if db is None:
        resolved = None
    else:
        resolved = resolve_active_model(db, "categorizer", CATEGORIZER_MODEL_NAME)
    if not resolved:
        if MODEL_PATH.exists():
            _CACHED_ARTIFACTS = joblib.load(MODEL_PATH)
            _CACHED_VERSION = None
            return _CACHED_ARTIFACTS
        return None
    _, version = resolved
    if _CACHED_ARTIFACTS is not None and _CACHED_VERSION == version:
        return _CACHED_ARTIFACTS
    model_path = model_dir("categorizer", CATEGORIZER_MODEL_NAME, version) / "model.pkl"
    if not model_path.exists():
        return None
    _CACHED_ARTIFACTS = joblib.load(model_path)
    _CACHED_VERSION = version
    return _CACHED_ARTIFACTS


def _predict_with_model(db, descriptions: List[str]) -> Optional[List[str]]:
    if not descriptions:
        return []
    artifacts = _load_active_artifacts(db)
    if not artifacts:
        return None
    features = artifacts.vectorizer.transform(descriptions)
    preds = artifacts.classifier.predict(features)
    return [artifacts.labels[idx] for idx in preds]


def predict_with_confidence(db, descriptions: List[str]) -> Tuple[List[str], List[float]]:
    if not descriptions:
        return [], []
    artifacts = _load_active_artifacts(db)
    if not artifacts:
        raise ValueError("Model not trained")
    if not hasattr(artifacts.classifier, "predict_proba"):
        raise ValueError("Model does not support probability predictions")
    features = artifacts.vectorizer.transform(descriptions)
    probs = artifacts.classifier.predict_proba(features)
    pred_indices = probs.argmax(axis=1)
    labels = [artifacts.labels[idx] for idx in pred_indices]
    confidences = [float(probs[i, idx]) for i, idx in enumerate(pred_indices)]
    return labels, confidences


def categorize_transactions(db, transactions: List[Transaction], use_ml: bool = False) -> None:
    user_maps = _load_user_maps(db)
    descriptions = [t.description for t in transactions]
    model_preds: Optional[List[str]] = None
    if use_ml:
        model_preds = _predict_with_model(db, descriptions)

    for idx, txn in enumerate(transactions):
        if txn.category:
            continue
        category = _apply_rules(txn.description, user_maps)
        if not category and model_preds:
            category = model_preds[idx]
        txn.category = category or "Misc"
    db.commit()


def train_model_artifacts(
    transactions: List[Transaction],
) -> Tuple[ModelArtifacts, float, List[str], int]:
    labeled = [(t.description, t.category) for t in transactions if t.category]
    if len(labeled) < 10:
        raise ValueError("Need at least 10 labeled transactions to train the model")

    texts, labels = zip(*labeled)
    label_list = sorted(set(labels))
    if len(label_list) < 2:
        raise ValueError("Need at least 2 categories to train the model")
    label_to_idx = {label: idx for idx, label in enumerate(label_list)}
    y = [label_to_idx[label] for label in labels]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(texts)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)
    acc = accuracy_score(y_test, preds)

    artifacts = ModelArtifacts(vectorizer=vectorizer, classifier=classifier, labels=label_list)
    return artifacts, acc, label_list, len(labeled)


def train_model(transactions: List[Transaction]) -> Tuple[float, List[str], int]:
    artifacts, acc, label_list, total = train_model_artifacts(transactions)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, MODEL_PATH)
    return acc, label_list, total


def evaluate_model(db, transactions: List[Transaction]) -> Tuple[float, List[List[int]], List[str], int]:
    artifacts = _load_active_artifacts(db)
    if artifacts is None and MODEL_PATH.exists():
        artifacts = joblib.load(MODEL_PATH)
    if artifacts is None:
        raise ValueError("Model not trained")
    labeled = [(t.description, t.category) for t in transactions if t.category]
    if len(labeled) < 5:
        raise ValueError("Need at least 5 labeled transactions to evaluate the model")

    texts, labels = zip(*labeled)
    label_to_idx = {label: idx for idx, label in enumerate(artifacts.labels)}

    filtered_texts = []
    y_true = []
    for text, label in zip(texts, labels):
        if label in label_to_idx:
            filtered_texts.append(text)
            y_true.append(label_to_idx[label])

    X = artifacts.vectorizer.transform(filtered_texts)
    preds = artifacts.classifier.predict(X)

    acc = accuracy_score(y_true, preds)
    matrix = confusion_matrix(y_true, preds).tolist()
    return acc, matrix, artifacts.labels, len(y_true)
