from __future__ import annotations

from typing import List

from app.models import BudgetRecommendation, Category, Transaction
from app.services.forecasting import (
    MODEL_BASELINE,
    compute_recent_spend_stats,
    forecast_all_categories,
)


def _buffer_for_category(predicted: float, volatility: float, category_type: str) -> float:
    base_rate = 0.05 if category_type == "fixed" else 0.15
    vol_factor = 0.3 if category_type == "fixed" else 0.6
    return predicted * base_rate + volatility * vol_factor


def build_recommendations(
    db, transactions: List[Transaction], forecast_model: str
) -> List[BudgetRecommendation]:
    forecast_payload = forecast_all_categories(
        db,
        transactions,
        model_name=forecast_model,
        start_month_str=None,
        horizon=1,
        force=False,
        strict=False,
    )
    forecast_map = {
        item["category"]: item for item in forecast_payload.get("forecasts", [])
    }
    stats = compute_recent_spend_stats(transactions)
    categories = {c.name: c.type for c in db.query(Category).all()}

    recommendations: List[BudgetRecommendation] = []
    for category, stat in stats.items():
        forecast_item = forecast_map.get(category)
        predicted_spend = stat["mean"]
        model_used = MODEL_BASELINE
        error_note = ""
        if forecast_item:
            predicted_spend = float(forecast_item["yhat"])
            model_used = forecast_item.get("model_used", MODEL_BASELINE)
            if forecast_item.get("error"):
                error_note = f" Fallback: {model_used}."

        category_type = categories.get(category, "variable")
        buffer_amount = _buffer_for_category(
            predicted_spend, stat["volatility"], category_type
        )
        recommended = predicted_spend + buffer_amount
        explanation = (
            f"Avg last 3 months: ${stat['mean']:.2f}. "
            f"Volatility: ${stat['volatility']:.2f}. "
            f"Model: {model_used}. Applied {category_type} buffer.{error_note}"
        )
        rec = BudgetRecommendation(
            category=category,
            predicted_spend=predicted_spend,
            recommended_budget=recommended,
            buffer=buffer_amount,
            volatility=stat["volatility"],
            explanation=explanation,
        )
        recommendations.append(rec)
    return recommendations
