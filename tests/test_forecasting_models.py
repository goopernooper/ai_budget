from datetime import date

import pytest
import warnings

from app.models import Transaction
from app.services.forecasting import (
    MODEL_BASELINE,
    MODEL_SARIMA,
    SARIMA_AVAILABLE,
    backtest_category,
    forecast_category,
)


def _make_transactions():
    txns = []
    for month in range(1, 9):
        amount = -(100 + month * 10)
        txns.append(
            Transaction(
                date=date(2024, month, 5),
                description="Test Expense",
                amount=amount,
                category="Groceries",
            )
        )
    return txns


def test_baseline_forecast_mean():
    txns = _make_transactions()
    result = forecast_category(
        "Groceries", txns, horizon_months=1, model_name=MODEL_BASELINE
    )
    yhat = float(result.forecast["yhat"].iloc[0])
    expected = (160 + 170 + 180) / 3
    assert yhat == pytest.approx(expected, rel=1e-3)


def test_sarima_forecast_runs_if_available():
    if not SARIMA_AVAILABLE:
        pytest.skip("statsmodels not available")
    txns = _make_transactions()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = forecast_category(
            "Groceries", txns, horizon_months=1, model_name=MODEL_SARIMA
        )
    assert len(result.forecast) == 1


def test_backtest_metrics_keys():
    txns = _make_transactions()
    result = backtest_category(
        "Groceries", txns, model_name=MODEL_BASELINE, min_train_months=4
    )
    assert "mae" in result
    assert "mape" in result
    assert result["count"] > 0
