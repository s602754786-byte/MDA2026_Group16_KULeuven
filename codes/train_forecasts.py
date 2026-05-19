from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mda_matplotlib")

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "rolling_std_24h",
]


def seasonal_naive_predict(train: pd.DataFrame, target_hours: pd.Series) -> pd.Series:
    history = train[["hour", "count"]].sort_values("hour").copy()
    lookup = history.set_index("hour")["count"]
    history["hour_of_week"] = history["hour"].dt.dayofweek * 24 + history["hour"].dt.hour
    fallback = history.groupby("hour_of_week")["count"].median()
    overall = float(history["count"].median()) if not history.empty else 0.0

    predictions = []
    for hour in pd.to_datetime(target_hours):
        previous_week = hour - pd.Timedelta(days=7)
        hour_of_week = hour.dayofweek * 24 + hour.hour
        value = lookup.get(previous_week, fallback.get(hour_of_week, overall))
        predictions.append(max(float(value), 0.0))
    return pd.Series(predictions, index=target_hours.index)


def calendar_features(hours: pd.Series) -> pd.DataFrame:
    timestamps = pd.Series(pd.to_datetime(hours)).reset_index(drop=True)
    hour = timestamps.dt.hour
    weekday = timestamps.dt.dayofweek
    month = timestamps.dt.month
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "weekday_sin": np.sin(2 * np.pi * weekday / 7),
            "weekday_cos": np.cos(2 * np.pi * weekday / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "is_weekend": weekday.isin([5, 6]).astype(int),
        }
    )


def build_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        history[["hour", "count"]]
        .dropna(subset=["hour"])
        .groupby("hour", as_index=False)["count"]
        .sum()
        .sort_values("hour")
    )
    if hourly.empty:
        return pd.DataFrame(columns=["hour", "count", *FEATURE_COLUMNS])

    full_hours = pd.date_range(hourly["hour"].min(), hourly["hour"].max(), freq="h")
    frame = hourly.set_index("hour").reindex(full_hours).rename_axis("hour").reset_index()
    frame["count"] = frame["count"].astype(float)
    features = calendar_features(frame["hour"])

    counts = frame["count"]
    shifted = counts.shift(1)
    features["lag_1h"] = counts.shift(1)
    features["lag_24h"] = counts.shift(24)
    features["lag_168h"] = counts.shift(168)
    features["rolling_mean_24h"] = shifted.rolling(24, min_periods=3).mean()
    features["rolling_mean_168h"] = shifted.rolling(168, min_periods=24).mean()
    features["rolling_std_24h"] = shifted.rolling(24, min_periods=3).std()

    return pd.concat([frame[["hour", "count"]], features], axis=1)


def make_feature_models() -> dict[str, tuple[object, list[str]]]:
    return {
        "hist_gradient_boosting": (
            make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=220,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=42,
                ),
            ),
            FEATURE_COLUMNS,
        ),
    }


def feature_row(hour: pd.Timestamp, history_counts: pd.Series) -> pd.DataFrame:
    row = calendar_features(pd.Series([hour]))
    history_counts = history_counts.sort_index()
    previous = history_counts.loc[history_counts.index < hour]

    def lag(hours: int) -> float:
        value = history_counts.get(hour - pd.Timedelta(hours=hours), np.nan)
        return float(value) if pd.notna(value) else np.nan

    row["lag_1h"] = lag(1)
    row["lag_24h"] = lag(24)
    row["lag_168h"] = lag(168)
    row["rolling_mean_24h"] = previous.tail(24).mean()
    row["rolling_mean_168h"] = previous.tail(168).mean()
    row["rolling_std_24h"] = previous.tail(24).std()
    return row[FEATURE_COLUMNS]


def recursive_feature_predict(
    model: object,
    history: pd.DataFrame,
    target_hours: pd.Series,
    feature_columns: list[str],
) -> pd.Series:
    observed = (
        history[["hour", "count"]]
        .dropna(subset=["hour", "count"])
        .groupby("hour")["count"]
        .sum()
        .astype(float)
        .sort_index()
    )
    if observed.empty:
        upper_bound = 1.0
    else:
        upper_bound = max(float(observed.quantile(0.995) * 1.5), float(observed.max()), 1.0)

    predictions: dict[object, float] = {}
    for index, hour in pd.to_datetime(target_hours).items():
        row = feature_row(pd.Timestamp(hour), observed)[feature_columns]
        prediction = float(model.predict(row)[0])
        prediction = min(max(prediction, 0.0), upper_bound)
        predictions[index] = prediction
        observed = pd.concat([observed, pd.Series([prediction], index=[pd.Timestamp(hour)])]).sort_index()
    return pd.Series(predictions).reindex(target_hours.index)


def fit_prophet(train: pd.DataFrame, yearly: bool) -> Prophet:
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=yearly,
        seasonality_mode="additive",
        interval_width=0.8,
    )
    model.fit(train.rename(columns={"hour": "ds", "count": "y"})[["ds", "y"]])
    return model


def prophet_predict(model: Prophet, target_hours: pd.Series) -> pd.DataFrame:
    future = pd.DataFrame({"ds": pd.to_datetime(target_hours)})
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast[["yhat", "yhat_lower", "yhat_upper"]] = forecast[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)
    return forecast.rename(
        columns={
            "ds": "hour",
            "yhat": "forecast",
            "yhat_lower": "forecast_lower",
            "yhat_upper": "forecast_upper",
        }
    )


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    aligned = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if aligned.empty:
        return {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan")}
    error = aligned["actual"] - aligned["predicted"]
    mae = error.abs().mean()
    rmse = (error.pow(2).mean()) ** 0.5
    wape = error.abs().sum() / max(aligned["actual"].sum(), 1) * 100
    return {"mae": float(mae), "rmse": float(rmse), "wape": float(wape)}


def choose_stations(
    summary: pd.DataFrame,
    top_stations: int,
    min_coverage: float,
    station_ids: list[int] | None,
    all_stations: bool,
) -> list[int]:
    if station_ids:
        return station_ids

    eligible = summary.loc[
        (summary["coverage_rate"] >= min_coverage) & (summary["active_days"] >= 180)
    ].copy()
    if eligible.empty:
        eligible = summary.copy()
    if all_stations:
        return eligible.sort_values("avg_daily_count", ascending=False)["site_id"].astype(int).tolist()
    return (
        eligible.sort_values("avg_daily_count", ascending=False)
        .head(top_stations)["site_id"]
        .astype(int)
        .tolist()
    )


def make_backtest_windows(
    last_hour: pd.Timestamp,
    test_hours: int,
    train_days: int,
    backtest_windows: int,
    backtest_step_hours: int,
) -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    for offset in range(backtest_windows):
        test_end = last_hour - pd.Timedelta(hours=offset * backtest_step_hours)
        test_start = test_end - pd.Timedelta(hours=test_hours - 1)
        train_start = test_start - pd.Timedelta(days=train_days)
        windows.append(
            {
                "backtest_window": offset + 1,
                "train_start": train_start,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
    return windows


def add_metric_row(
    rows: list[dict[str, object]],
    site_id: int,
    model_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    backtest_window: int,
    metric_values: dict[str, float],
) -> None:
    rows.append(
        {
            "site_id": site_id,
            "model": model_name,
            "backtest_window": backtest_window,
            "train_start": train["hour"].min(),
            "train_end": train["hour"].max(),
            "test_start": test["hour"].min(),
            "test_end": test["hour"].max(),
            "test_hours": int(len(test)),
            **metric_values,
        }
    )


def aggregate_backtests(backtests: pd.DataFrame) -> pd.DataFrame:
    if backtests.empty:
        return pd.DataFrame()
    return (
        backtests.groupby(["site_id", "model"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            wape=("wape", "mean"),
            backtest_windows=("backtest_window", "nunique"),
            test_hours=("test_hours", "sum"),
            train_start=("train_start", "min"),
            train_end=("train_end", "max"),
            test_start=("test_start", "min"),
            test_end=("test_end", "max"),
        )
        .sort_values(["site_id", "wape"])
    )


def train_forecasts(
    processed_dir: Path,
    top_stations: int,
    station_ids: list[int] | None,
    horizon_hours: int,
    test_hours: int,
    train_days: int,
    min_coverage: float,
    all_stations: bool,
    backtest_windows: int,
    backtest_step_hours: int,
) -> None:
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").disabled = True
    hourly = pd.read_parquet(processed_dir / "hourly_counts.parquet")
    summary = pd.read_parquet(processed_dir / "station_summary.parquet")
    hourly["hour"] = pd.to_datetime(hourly["hour"])

    selected_sites = choose_stations(summary, top_stations, min_coverage, station_ids, all_stations)
    backtest_rows: list[dict[str, object]] = []
    forecast_rows: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []

    for index, site_id in enumerate(selected_sites, start=1):
        station_hourly = hourly.loc[hourly["site_id"] == site_id, ["site_id", "hour", "count"]].copy()
        station_hourly = station_hourly.sort_values("hour")
        if station_hourly.empty:
            skipped.append({"site_id": site_id, "reason": "no hourly data"})
            continue

        last_hour = station_hourly["hour"].max()
        station_windows = make_backtest_windows(
            last_hour,
            test_hours,
            train_days,
            backtest_windows,
            backtest_step_hours,
        )
        valid_windows = 0
        print(f"[{index:02d}/{len(selected_sites):02d}] site {site_id}: evaluating {len(station_windows)} windows")

        for window in station_windows:
            train = station_hourly.loc[
                (station_hourly["hour"] >= window["train_start"])
                & (station_hourly["hour"] < window["test_start"])
            ].copy()
            test = station_hourly.loc[
                (station_hourly["hour"] >= window["test_start"])
                & (station_hourly["hour"] <= window["test_end"])
            ].copy()

            if len(train) < 24 * 60 or len(test) < 24:
                skipped.append(
                    {
                        "site_id": site_id,
                        "backtest_window": window["backtest_window"],
                        "reason": "not enough train/test data",
                    }
                )
                continue

            valid_windows += 1
            baseline_test = seasonal_naive_predict(train, test["hour"])
            add_metric_row(
                backtest_rows,
                site_id,
                "seasonal_naive",
                train,
                test,
                int(window["backtest_window"]),
                metrics(test["count"], baseline_test),
            )

            feature_training = build_feature_frame(train).dropna(subset=["count"])
            if len(feature_training) < 24 * 30:
                skipped.append(
                    {
                        "site_id": site_id,
                        "backtest_window": window["backtest_window"],
                        "reason": "not enough rows for sklearn models",
                    }
                )
            else:
                y_train = feature_training["count"]
                for model_name, (model, feature_columns) in make_feature_models().items():
                    try:
                        model.fit(feature_training[feature_columns], y_train)
                        model_test = recursive_feature_predict(model, train, test["hour"], feature_columns)
                        add_metric_row(
                            backtest_rows,
                            site_id,
                            model_name,
                            train,
                            test,
                            int(window["backtest_window"]),
                            metrics(test["count"], model_test),
                        )
                    except Exception as exc:
                        skipped.append(
                            {
                                "site_id": site_id,
                                "backtest_window": window["backtest_window"],
                                "reason": f"{model_name} failed: {exc}",
                            }
                        )

            try:
                yearly = train["hour"].max() - train["hour"].min() >= pd.Timedelta(days=365)
                model = fit_prophet(train, yearly=yearly)
                prophet_test = prophet_predict(model, test["hour"])
                add_metric_row(
                    backtest_rows,
                    site_id,
                    "prophet",
                    train,
                    test,
                    int(window["backtest_window"]),
                    metrics(test["count"].reset_index(drop=True), prophet_test["forecast"]),
                )
            except Exception as exc:
                skipped.append(
                    {
                        "site_id": site_id,
                        "backtest_window": window["backtest_window"],
                        "reason": f"prophet failed: {exc}",
                    }
                )

        if valid_windows == 0:
            continue

        future_hours = pd.Series(
            pd.date_range(last_hour + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")
        )
        forecast_train_start = last_hour - pd.Timedelta(days=train_days)
        final_train = station_hourly.loc[station_hourly["hour"] >= forecast_train_start].copy()
        if len(final_train) < 24 * 60:
            final_train = station_hourly.copy()

        forecast_rows.append(
            pd.DataFrame(
                {
                    "site_id": site_id,
                    "hour": future_hours,
                    "model": "seasonal_naive",
                    "forecast": seasonal_naive_predict(station_hourly, future_hours).to_numpy(),
                    "forecast_lower": float("nan"),
                    "forecast_upper": float("nan"),
                }
            )
        )

        feature_training = build_feature_frame(final_train).dropna(subset=["count"])
        if len(feature_training) >= 24 * 30:
            y_train = feature_training["count"]
            for model_name, (model, feature_columns) in make_feature_models().items():
                try:
                    model.fit(feature_training[feature_columns], y_train)
                    forecast_rows.append(
                        pd.DataFrame(
                            {
                                "site_id": site_id,
                                "hour": future_hours,
                                "model": model_name,
                                "forecast": recursive_feature_predict(
                                    model, station_hourly, future_hours, feature_columns
                                ).to_numpy(),
                                "forecast_lower": float("nan"),
                                "forecast_upper": float("nan"),
                            }
                        )
                    )
                except Exception as exc:
                    skipped.append({"site_id": site_id, "reason": f"future {model_name} failed: {exc}"})

        try:
            yearly = final_train["hour"].max() - final_train["hour"].min() >= pd.Timedelta(days=365)
            model = fit_prophet(final_train, yearly=yearly)
            prophet_future = prophet_predict(model, future_hours)
            prophet_future.insert(0, "site_id", site_id)
            prophet_future.insert(2, "model", "prophet")
            forecast_rows.append(prophet_future)
        except Exception as exc:
            skipped.append({"site_id": site_id, "reason": f"future prophet failed: {exc}"})

    backtests = pd.DataFrame(backtest_rows)
    evaluation = aggregate_backtests(backtests)
    forecasts = pd.concat(forecast_rows, ignore_index=True) if forecast_rows else pd.DataFrame()

    if not backtests.empty:
        backtests = backtests.merge(
            summary[["site_id", "station_label", "naam", "gemeente"]],
            on="site_id",
            how="left",
        )
        backtests.to_csv(processed_dir / "model_backtest_windows.csv", index=False)

    if not evaluation.empty:
        evaluation = evaluation.merge(
            summary[["site_id", "station_label", "naam", "gemeente"]],
            on="site_id",
            how="left",
        )
        evaluation.to_csv(processed_dir / "model_evaluation.csv", index=False)

    if not forecasts.empty:
        forecasts = forecasts.merge(
            summary[["site_id", "station_label", "naam", "gemeente"]],
            on="site_id",
            how="left",
        )
        forecasts.to_parquet(processed_dir / "forecast_outputs.parquet", index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_sites": selected_sites,
        "candidate_models": [
            "seasonal_naive",
            "hist_gradient_boosting",
            "prophet",
        ],
        "all_stations": all_stations,
        "top_stations": top_stations,
        "horizon_hours": horizon_hours,
        "test_hours": test_hours,
        "train_days": train_days,
        "backtest_windows": backtest_windows,
        "backtest_step_hours": backtest_step_hours,
        "min_coverage": min_coverage,
        "evaluated_rows": int(len(evaluation)),
        "backtest_rows": int(len(backtests)),
        "forecast_rows": int(len(forecasts)),
        "skipped": skipped,
    }
    (processed_dir / "forecast_run_summary.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote evaluation rows: {len(evaluation):,}")
    print(f"Wrote forecast rows: {len(forecasts):,}")
    if skipped:
        print(f"Skipped stations: {len(skipped)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline, sklearn, and Prophet forecasts.")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--top-stations", type=int, default=12)
    parser.add_argument("--all-stations", action="store_true", help="Train forecasts for all eligible stations.")
    parser.add_argument("--station-ids", default="", help="Optional comma-separated site IDs.")
    parser.add_argument("--horizon-hours", type=int, default=168)
    parser.add_argument("--test-hours", type=int, default=168)
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--backtest-windows", type=int, default=3)
    parser.add_argument("--backtest-step-hours", type=int, default=168)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ids = [int(value.strip()) for value in args.station_ids.split(",") if value.strip()]
    train_forecasts(
        processed_dir=args.processed_dir,
        top_stations=args.top_stations,
        station_ids=ids or None,
        horizon_hours=args.horizon_hours,
        test_hours=args.test_hours,
        train_days=args.train_days,
        min_coverage=args.min_coverage,
        all_stations=args.all_stations,
        backtest_windows=args.backtest_windows,
        backtest_step_hours=args.backtest_step_hours,
    )
