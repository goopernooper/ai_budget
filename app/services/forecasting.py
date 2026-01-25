from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import json

import numpy as np
import pandas as pd

from app.models import ForecastCache, Transaction

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    SARIMA_AVAILABLE = True
except Exception:
    SARIMA_AVAILABLE = False

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import HistGradientBoostingRegressor

MODEL_BASELINE = "baseline_rolling_mean"
MODEL_SARIMA = "sarima"
MODEL_PROPHET = "prophet"
MODEL_LIGHTGBM = "lightgbm"
MODEL_XGBOOST = "xgboost"

FEATURE_COLUMNS = [
    "month_of_year",
    "year",
    "day_of_month",
    "day_of_week",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_std_3",
    "biweekly_pay_week",
    "holiday_season",
]


class ForecastError(RuntimeError):
    pass


@dataclass
class ForecastResult:
    category: str
    model_requested: str
    model_used: str
    forecast: pd.DataFrame
    error: Optional[str] = None


def get_available_models() -> Dict[str, bool]:
    return {
        MODEL_BASELINE: True,
        MODEL_SARIMA: SARIMA_AVAILABLE,
        MODEL_PROPHET: PROPHET_AVAILABLE,
        MODEL_LIGHTGBM: True,
        MODEL_XGBOOST: XGBOOST_AVAILABLE,
    }


def _transactions_to_df(transactions: List[Transaction]) -> pd.DataFrame:
    data = [
        {
            "date": t.date,
            "category": t.category or "Misc",
            "amount": t.amount,
        }
        for t in transactions
    ]
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def _build_monthly_series(df: pd.DataFrame, category: str) -> pd.DataFrame:
    if df.empty:
        return df

    cat_df = df[df["category"] == category].copy()
    if cat_df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    if category.lower() == "income":
        cat_df = cat_df[cat_df["amount"] > 0].copy()
        cat_df["y"] = cat_df["amount"].astype(float)
    else:
        cat_df = cat_df[cat_df["amount"] < 0].copy()
        cat_df["y"] = cat_df["amount"].abs().astype(float)

    if cat_df.empty:
        return pd.DataFrame(columns=["ds", "y"])

    monthly = cat_df.groupby("month")["y"].sum()
    all_months = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(all_months, fill_value=0.0)
    return pd.DataFrame({"ds": monthly.index, "y": monthly.values})


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    month_end = features["ds"] + pd.offsets.MonthEnd(0)
    features["month_of_year"] = features["ds"].dt.month
    features["year"] = features["ds"].dt.year
    features["day_of_month"] = month_end.dt.day
    features["day_of_week"] = month_end.dt.dayofweek
    features["holiday_season"] = features["month_of_year"].isin([11, 12]).astype(int)
    features["week_of_month"] = ((month_end.dt.day - 1) // 7 + 1).astype(int)
    features["biweekly_pay_week"] = features["week_of_month"].isin([1, 3]).astype(int)

    features["lag_1"] = features["y"].shift(1)
    features["lag_2"] = features["y"].shift(2)
    features["lag_3"] = features["y"].shift(3)
    features["rolling_mean_3"] = features["y"].shift(1).rolling(3).mean()
    features["rolling_std_3"] = features["y"].shift(1).rolling(3).std(ddof=0)

    return features


def make_category_timeseries_and_features(
    category: str, transactions: List[Transaction]
) -> pd.DataFrame:
    df = _transactions_to_df(transactions)
    series = _build_monthly_series(df, category)
    if series.empty:
        return series
    return _add_time_features(series)


def _month_diff(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def _parse_month(month_str: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(year=int(month_str[:4]), month=int(month_str[5:7]), day=1)
    except Exception as exc:
        raise ValueError("month must be in YYYY-MM format") from exc


def _forecast_baseline(series_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    values = series_df["y"].tolist()
    if not values:
        raise ForecastError("No data to forecast")

    preds = []
    for _ in range(horizon):
        window = values[-3:]
        mean = float(sum(window) / len(window)) if window else 0.0
        preds.append(mean)
        values.append(mean)

    future_dates = pd.date_range(
        series_df["ds"].max() + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    return pd.DataFrame({"ds": future_dates, "yhat": preds, "lower": None, "upper": None})


def _forecast_sarima(series_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not SARIMA_AVAILABLE:
        raise ForecastError("statsmodels not available")
    if len(series_df) < 6:
        raise ForecastError("Need at least 6 months for SARIMA")

    y = series_df.set_index("ds")["y"].asfreq("MS")
    seasonal_order = (0, 0, 0, 0)
    if len(series_df) >= 24:
        seasonal_order = (1, 1, 1, 12)

    model = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    forecast = result.get_forecast(steps=horizon)
    mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.2)

    future_dates = pd.date_range(
        series_df["ds"].max() + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    return pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": mean.values,
            "lower": conf_int.iloc[:, 0].values,
            "upper": conf_int.iloc[:, 1].values,
        }
    )


def _forecast_prophet(series_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not PROPHET_AVAILABLE:
        raise ForecastError("prophet not available")
    if len(series_df) < 6:
        raise ForecastError("Need at least 6 months for Prophet")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(series_df[["ds", "y"]])
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future).tail(horizon)

    return pd.DataFrame(
        {
            "ds": forecast["ds"].values,
            "yhat": forecast["yhat"].values,
            "lower": forecast["yhat_lower"].values,
            "upper": forecast["yhat_upper"].values,
        }
    )


def _build_feature_row(history: pd.DataFrame, next_ds: pd.Timestamp) -> Dict[str, float]:
    values = history["y"].tolist()
    lag_1 = values[-1] if len(values) >= 1 else 0.0
    lag_2 = values[-2] if len(values) >= 2 else lag_1
    lag_3 = values[-3] if len(values) >= 3 else lag_2
    recent = values[-3:] if values else [0.0]
    rolling_mean_3 = float(np.mean(recent)) if recent else 0.0
    rolling_std_3 = float(np.std(recent, ddof=0)) if len(recent) > 1 else 0.0

    month_end = next_ds + pd.offsets.MonthEnd(0)
    month_of_year = next_ds.month
    year = next_ds.year
    day_of_month = month_end.day
    day_of_week = month_end.dayofweek
    holiday_season = 1 if month_of_year in (11, 12) else 0
    week_of_month = int((day_of_month - 1) // 7 + 1)
    biweekly_pay_week = 1 if week_of_month in (1, 3) else 0

    return {
        "month_of_year": month_of_year,
        "year": year,
        "day_of_month": day_of_month,
        "day_of_week": day_of_week,
        "lag_1": lag_1,
        "lag_2": lag_2,
        "lag_3": lag_3,
        "rolling_mean_3": rolling_mean_3,
        "rolling_std_3": rolling_std_3,
        "biweekly_pay_week": biweekly_pay_week,
        "holiday_season": holiday_season,
    }


def _train_ml_model(train_df: pd.DataFrame):
    X = train_df[FEATURE_COLUMNS]
    y = train_df["y"]
    if LIGHTGBM_AVAILABLE:
        model = lgb.LGBMRegressor(random_state=42)
    elif XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=42,
        )
    else:
        model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model


def _train_ml_model_matrix(X: pd.DataFrame, y: pd.Series):
    if LIGHTGBM_AVAILABLE:
        model = lgb.LGBMRegressor(random_state=42)
    elif XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=42,
        )
    else:
        model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model


def _forecast_ml(features_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    train_df = features_df.dropna(subset=FEATURE_COLUMNS + ["y"])
    if len(train_df) < 6:
        raise ForecastError("Need at least 6 months for ML forecasting")

    model = _train_ml_model(train_df)
    history = features_df[["ds", "y"]].copy()
    preds = []
    future_dates = []

    for _ in range(horizon):
        next_ds = history["ds"].max() + pd.offsets.MonthBegin(1)
        row = _build_feature_row(history, next_ds)
        yhat = float(model.predict(pd.DataFrame([row]))[0])
        yhat = max(yhat, 0.0)
        preds.append(yhat)
        future_dates.append(next_ds)
        history = pd.concat(
            [history, pd.DataFrame([{"ds": next_ds, "y": yhat}])],
            ignore_index=True,
        )

    return pd.DataFrame({"ds": future_dates, "yhat": preds, "lower": None, "upper": None})


def forecast_category(
    category: str,
    transactions: List[Transaction],
    horizon_months: int,
    model_name: str,
    start_month: Optional[pd.Timestamp] = None,
    strict: bool = False,
) -> ForecastResult:
    features_df = make_category_timeseries_and_features(category, transactions)
    if features_df.empty:
        raise ForecastError("No data for category")

    series_df = features_df[["ds", "y"]].copy()
    last_month = series_df["ds"].max()
    first_forecast_month = last_month + pd.offsets.MonthBegin(1)

    if start_month is None:
        start_month = first_forecast_month
    elif start_month < first_forecast_month:
        start_month = first_forecast_month

    end_month = start_month + pd.offsets.MonthBegin(horizon_months - 1)
    total_horizon = _month_diff(first_forecast_month, end_month) + 1
    error = None
    model_used = model_name

    try:
        if model_name == MODEL_BASELINE:
            forecast_df = _forecast_baseline(series_df, total_horizon)
        elif model_name == MODEL_SARIMA:
            forecast_df = _forecast_sarima(series_df, total_horizon)
        elif model_name == MODEL_PROPHET:
            forecast_df = _forecast_prophet(series_df, total_horizon)
        elif model_name == MODEL_LIGHTGBM:
            forecast_df = _forecast_ml(features_df, total_horizon)
        else:
            raise ForecastError("Unknown model")
    except ForecastError as exc:
        if strict:
            raise
        error = str(exc)
        model_used = MODEL_BASELINE
        forecast_df = _forecast_baseline(series_df, total_horizon)

    forecast_df = forecast_df[forecast_df["ds"] >= start_month].head(horizon_months)
    return ForecastResult(
        category=category,
        model_requested=model_name,
        model_used=model_used,
        forecast=forecast_df.reset_index(drop=True),
        error=error,
    )


def _cache_key(month: str, category: str, model_name: str) -> Tuple[str, str, str]:
    return (month, category, model_name)


def _cache_meta(model_used: str, error: Optional[str]) -> str:
    return json.dumps({"model_used": model_used, "error": error})


def _read_cache_meta(metrics_json: Optional[str], fallback: str) -> Tuple[str, Optional[str]]:
    if not metrics_json:
        return fallback, None
    try:
        data = json.loads(metrics_json)
    except json.JSONDecodeError:
        return fallback, None
    return data.get("model_used", fallback), data.get("error")


def forecast_all_categories(
    db,
    transactions: List[Transaction],
    model_name: str,
    start_month_str: Optional[str],
    horizon: int,
    force: bool = False,
    strict: bool = False,
) -> Dict[str, object]:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    df = _transactions_to_df(transactions)
    if df.empty:
        return {
            "start_month": "",
            "horizon": horizon,
            "model_requested": model_name,
            "forecasts": [],
        }

    global_last_month = df["month"].max()
    default_start = global_last_month + pd.offsets.MonthBegin(1)
    if start_month_str:
        start_month = _parse_month(start_month_str)
    else:
        start_month = default_start
    if start_month < default_start:
        start_month = default_start

    end_month = start_month + pd.offsets.MonthBegin(horizon - 1)
    months = pd.date_range(start_month, end_month, freq="MS")
    month_keys = [m.strftime("%Y-%m") for m in months]

    categories = sorted(df["category"].unique())
    cached = (
        db.query(ForecastCache)
        .filter(
            ForecastCache.model_name == model_name,
            ForecastCache.category.in_(categories),
            ForecastCache.month.in_(month_keys),
        )
        .all()
    )
    cache_map = {
        _cache_key(item.month, item.category, item.model_name): item for item in cached
    }
    existing_keys = set(cache_map.keys())

    forecasts: List[Dict[str, object]] = []
    for category in categories:
        needs_forecast = False
        for month in month_keys:
            if force or _cache_key(month, category, model_name) not in cache_map:
                needs_forecast = True
                break

        if needs_forecast:
            try:
                result = forecast_category(
                    category,
                    transactions,
                    horizon_months=horizon,
                    model_name=model_name,
                    start_month=start_month,
                    strict=strict,
                )
            except ForecastError as exc:
                for month in month_keys:
                    forecasts.append(
                        {
                            "category": category,
                            "month": month,
                            "model_requested": model_name,
                            "model_used": model_name,
                            "yhat": 0.0,
                            "lower": None,
                            "upper": None,
                            "from_cache": False,
                            "error": str(exc),
                        }
                    )
                continue

            for _, row in result.forecast.iterrows():
                month_str = pd.Timestamp(row["ds"]).strftime("%Y-%m")
                cache_item = ForecastCache(
                    month=month_str,
                    category=category,
                    model_name=model_name,
                    yhat=float(row["yhat"]),
                    lower=float(row["lower"]) if not pd.isna(row["lower"]) else None,
                    upper=float(row["upper"]) if not pd.isna(row["upper"]) else None,
                    metrics_json=_cache_meta(result.model_used, result.error),
                )
                key = _cache_key(month_str, category, model_name)
                if key in cache_map:
                    existing = cache_map[key]
                    existing.yhat = cache_item.yhat
                    existing.lower = cache_item.lower
                    existing.upper = cache_item.upper
                    existing.metrics_json = cache_item.metrics_json
                else:
                    db.add(cache_item)
                    cache_map[key] = cache_item
            db.commit()

        for month in month_keys:
            cache_item = cache_map.get(_cache_key(month, category, model_name))
            if not cache_item:
                continue
            model_used, error = _read_cache_meta(cache_item.metrics_json, model_name)
            key = _cache_key(month, category, model_name)
            forecasts.append(
                {
                    "category": category,
                    "month": month,
                    "model_requested": model_name,
                    "model_used": model_used,
                    "yhat": float(cache_item.yhat),
                    "lower": cache_item.lower,
                    "upper": cache_item.upper,
                    "from_cache": (key in existing_keys) and not force,
                    "error": error,
                }
            )

    forecasts.sort(key=lambda item: (item["category"], item["month"]))
    return {
        "start_month": start_month.strftime("%Y-%m"),
        "horizon": horizon,
        "model_requested": model_name,
        "forecasts": forecasts,
    }


def compute_recent_spend_stats(
    transactions: List[Transaction], window: int = 3
) -> Dict[str, Dict[str, float]]:
    df = _transactions_to_df(transactions)
    if df.empty:
        return {}
    df = df[df["amount"] < 0].copy()
    if df.empty:
        return {}
    df["spend"] = df["amount"].abs()
    monthly = df.groupby(["category", "month"])["spend"].sum().reset_index()
    stats: Dict[str, Dict[str, float]] = {}
    for category, group in monthly.groupby("category"):
        recent = group.sort_values("month").tail(window)
        mean = float(recent["spend"].mean())
        volatility = float(recent["spend"].std(ddof=0)) if len(recent) > 1 else 0.0
        stats[category] = {"mean": mean, "volatility": volatility}
    return stats


def _forecast_one_step(
    series_df: pd.DataFrame,
    features_df: pd.DataFrame,
    model_name: str,
    strict: bool,
) -> Tuple[float, str, Optional[str]]:
    error = None
    model_used = model_name
    try:
        if model_name == MODEL_BASELINE:
            forecast_df = _forecast_baseline(series_df, 1)
        elif model_name == MODEL_SARIMA:
            forecast_df = _forecast_sarima(series_df, 1)
        elif model_name == MODEL_PROPHET:
            forecast_df = _forecast_prophet(series_df, 1)
        elif model_name == MODEL_LIGHTGBM:
            forecast_df = _forecast_ml(features_df, 1)
        else:
            raise ForecastError("Unknown model")
    except ForecastError as exc:
        if strict:
            raise
        error = str(exc)
        model_used = MODEL_BASELINE
        forecast_df = _forecast_baseline(series_df, 1)

    return float(forecast_df["yhat"].iloc[0]), model_used, error


def backtest_category(
    category: str,
    transactions: List[Transaction],
    model_name: str,
    min_train_months: int = 6,
    strict: bool = False,
) -> Dict[str, object]:
    features_df = make_category_timeseries_and_features(category, transactions)
    if features_df.empty:
        raise ForecastError("No data for category")

    series_df = features_df[["ds", "y"]].copy()
    if len(series_df) <= min_train_months:
        raise ForecastError("Not enough data for backtesting")

    errors = []
    mape_vals = []
    predictions = []
    model_used_overall = model_name
    error_msg = None

    for idx in range(min_train_months, len(series_df)):
        train_series = series_df.iloc[:idx]
        train_features = _add_time_features(train_series)
        test_row = series_df.iloc[idx]

        yhat, model_used, error = _forecast_one_step(
            train_series, train_features, model_name, strict
        )
        model_used_overall = model_used
        if error:
            error_msg = error

        mae = abs(yhat - float(test_row["y"]))
        errors.append(mae)
        if test_row["y"] != 0:
            mape_vals.append(mae / float(test_row["y"]))

        predictions.append(
            {
                "ds": test_row["ds"].strftime("%Y-%m"),
                "y": float(test_row["y"]),
                "yhat": float(yhat),
                "mae": float(mae),
            }
        )

    mae_avg = float(np.mean(errors)) if errors else None
    mape_avg = float(np.mean(mape_vals)) if mape_vals else None

    return {
        "category": category,
        "model_requested": model_name,
        "model_used": model_used_overall,
        "mae": mae_avg,
        "mape": mape_avg,
        "count": len(predictions),
        "error": error_msg,
        "predictions": predictions,
    }


def backtest_all_categories(
    transactions: List[Transaction],
    model_names: List[str],
    top_k_categories: int = 15,
    min_train_months: int = 6,
    strict: bool = False,
) -> Dict[str, object]:
    df = _transactions_to_df(transactions)
    if df.empty:
        return {"overall": {}, "categories": [], "errors": ["No transactions"]}

    df = df[df["amount"] < 0].copy()
    if df.empty:
        return {"overall": {}, "categories": [], "errors": ["No expenses"]}

    df["spend"] = df["amount"].abs()
    totals = (
        df.groupby("category")["spend"].sum().sort_values(ascending=False).head(top_k_categories)
    )
    categories = totals.index.tolist()

    results = []
    overall: Dict[str, Dict[str, Optional[float]]] = {}
    errors: List[str] = []

    for model_name in model_names:
        model_errors = []
        model_mape = []
        model_mae = []
        for category in categories:
            try:
                result = backtest_category(
                    category,
                    transactions,
                    model_name,
                    min_train_months=min_train_months,
                    strict=strict,
                )
                results.append({k: result[k] for k in result if k != "predictions"})
                if result["mae"] is not None:
                    model_mae.append(result["mae"])
                if result["mape"] is not None:
                    model_mape.append(result["mape"])
                if result["error"]:
                    model_errors.append(
                        f"{category}: {result['error']} (used {result['model_used']})"
                    )
            except ForecastError as exc:
                results.append(
                    {
                        "category": category,
                        "model_requested": model_name,
                        "model_used": MODEL_BASELINE,
                        "mae": None,
                        "mape": None,
                        "count": 0,
                        "error": str(exc),
                    }
                )
                model_errors.append(f"{category}: {exc}")

        overall[model_name] = {
            "mae": float(np.mean(model_mae)) if model_mae else None,
            "mape": float(np.mean(model_mape)) if model_mape else None,
        }
        if model_errors:
            errors.append(f"{model_name}: " + "; ".join(model_errors))

    return {"overall": overall, "categories": results, "errors": errors}


def build_forecast_dataset(transactions: list[Transaction]) -> pd.DataFrame:
    df = _transactions_to_df(transactions)
    if df.empty:
        return df
    categories = df["category"].unique().tolist()
    rows = []
    for category in categories:
        series = _build_monthly_series(df, category)
        if series.empty:
            continue
        for _, row in series.iterrows():
            rows.append(
                {
                    "month": pd.Timestamp(row["ds"]).strftime("%Y-%m"),
                    "category": category,
                    "y": float(row["y"]),
                }
            )
    return pd.DataFrame(rows)


def build_forecast_feature_dataset(transactions: list[Transaction]) -> pd.DataFrame:
    df = _transactions_to_df(transactions)
    if df.empty:
        return df
    rows = []
    for category in df["category"].unique().tolist():
        series = _build_monthly_series(df, category)
        if series.empty:
            continue
        features = _add_time_features(series)
        features["category"] = category
        rows.append(features)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def train_registry_forecast_model(transactions: list[Transaction]):
    feature_df = build_forecast_feature_dataset(transactions)
    if feature_df.empty:
        raise ForecastError("No data for ML training")
    feature_df = feature_df.dropna(subset=FEATURE_COLUMNS + ["y"]).copy()
    if len(feature_df) < 6:
        raise ForecastError("Not enough data for ML training")
    X = pd.get_dummies(
        feature_df[FEATURE_COLUMNS + ["category"]], columns=["category"], drop_first=False
    )
    y = feature_df["y"]
    model = _train_ml_model_matrix(X, y)
    return model, list(X.columns)


def fit_registry_sarima(series_df: pd.DataFrame):
    if not SARIMA_AVAILABLE:
        raise ForecastError("statsmodels not available")
    if len(series_df) < 6:
        raise ForecastError("Need at least 6 months for SARIMA")
    y = series_df.set_index("ds")["y"].asfreq("MS")
    seasonal_order = (0, 0, 0, 0)
    if len(series_df) >= 24:
        seasonal_order = (1, 1, 1, 12)
    model = SARIMAX(
        y,
        order=(1, 1, 1),
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)
