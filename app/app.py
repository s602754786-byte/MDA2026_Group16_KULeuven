from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
HOURLY_PATH = PROCESSED_DIR / "hourly_counts.parquet"
SUMMARY_PATH = PROCESSED_DIR / "station_summary.parquet"
EVALUATION_PATH = PROCESSED_DIR / "model_evaluation.csv"
BACKTEST_PATH = PROCESSED_DIR / "model_backtest_windows.csv"
FORECAST_PATH = PROCESSED_DIR / "forecast_outputs.parquet"
PIPELINE_RUN_PATH = PROCESSED_DIR / "pipeline_run_summary.json"
FORECAST_RUN_PATH = PROCESSED_DIR / "forecast_run_summary.json"
ORCHESTRATION_PATH = PROCESSED_DIR / "pipeline_orchestration.json"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    if not HOURLY_PATH.exists() or not SUMMARY_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), False

    hourly = pd.read_parquet(HOURLY_PATH)
    summary = pd.read_parquet(SUMMARY_PATH)
    hourly["hour"] = pd.to_datetime(hourly["hour"])
    hourly = hourly.merge(
        summary[
            [
                "site_id",
                "station_label",
                "naam",
                "gemeente",
                "long",
                "lat",
                "avg_daily_count",
                "p95_hour_count",
                "growth_pct_recent_vs_previous",
                "coverage_rate",
            ]
        ],
        on="site_id",
        how="left",
    )
    return hourly, summary, True


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def percentile_rank(values: pd.Series, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=values.index)
    return numeric.rank(pct=True, ascending=ascending).fillna(0.5)


def build_priority_table(summary: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    table = summary.copy()
    growth = pd.to_numeric(table["growth_pct_recent_vs_previous"], errors="coerce").fillna(0)
    table["demand_score"] = percentile_rank(table["avg_daily_count"])
    table["peak_pressure_score"] = percentile_rank(table["p95_hour_count"])
    table["growth_score"] = percentile_rank(growth)
    table["coverage_score"] = pd.to_numeric(table["coverage_rate"], errors="coerce").clip(0, 1).fillna(0)

    if not evaluation.empty:
        best_forecast = (
            evaluation.sort_values("wape")
            .groupby("site_id", as_index=False)
            .first()[["site_id", "model", "wape"]]
            .rename(columns={"model": "best_forecast_model", "wape": "best_forecast_wape"})
        )
        table = table.merge(best_forecast, on="site_id", how="left")
        table["forecast_reliability_score"] = percentile_rank(table["best_forecast_wape"], ascending=False)
    else:
        table["best_forecast_model"] = "not trained"
        table["best_forecast_wape"] = pd.NA
        table["forecast_reliability_score"] = table["coverage_score"]

    table["priority_score"] = 100 * (
        0.35 * table["demand_score"]
        + 0.25 * table["peak_pressure_score"]
        + 0.20 * table["growth_score"]
        + 0.10 * table["forecast_reliability_score"]
        + 0.10 * table["coverage_score"]
    )

    peak_median = table["p95_hour_count"].median()

    def recommendation(row: pd.Series) -> str:
        if row["coverage_rate"] < 0.7:
            return "Check sensor/data quality before using for investment decisions"
        if row["priority_score"] >= 75 and row["growth_pct_recent_vs_previous"] >= 20:
            return "Capacity upgrade candidate: high use and recent growth"
        if row["priority_score"] >= 75:
            return "High-demand corridor: review comfort and capacity"
        if row["growth_pct_recent_vs_previous"] >= 40:
            return "Emerging growth: monitor and compare nearby stations"
        if row["p95_hour_count"] >= peak_median:
            return "Peak-hour pressure: inspect rush-hour pattern"
        return "Keep monitoring"

    table["planning_recommendation"] = table.apply(recommendation, axis=1)
    return table.sort_values("priority_score", ascending=False)


def model_comparison_table(evaluation: pd.DataFrame) -> pd.DataFrame:
    if evaluation.empty:
        return pd.DataFrame()

    baseline = evaluation.loc[evaluation["model"] == "seasonal_naive"][
        ["site_id", "station_label", "gemeente", "wape"]
    ].rename(columns={"wape": "baseline_wape"})
    challengers = evaluation.loc[evaluation["model"] != "seasonal_naive"][
        ["site_id", "model", "wape", "mae", "rmse"]
    ].rename(columns={"wape": "model_wape"})
    if baseline.empty or challengers.empty:
        return pd.DataFrame()

    table = challengers.merge(baseline, on="site_id", how="left")
    table["wape_improvement_vs_baseline_pct"] = (
        (table["baseline_wape"] - table["model_wape"])
        / table["baseline_wape"].replace(0, pd.NA)
        * 100
    )
    table["beats_baseline"] = table["model_wape"] < table["baseline_wape"]
    return table.sort_values("wape_improvement_vs_baseline_pct", ascending=False)


HOURLY, SUMMARY, DATA_READY = load_data()
EVALUATION = pd.read_csv(EVALUATION_PATH) if EVALUATION_PATH.exists() else pd.DataFrame()
BACKTESTS = pd.read_csv(BACKTEST_PATH) if BACKTEST_PATH.exists() else pd.DataFrame()
FORECASTS = pd.read_parquet(FORECAST_PATH) if FORECAST_PATH.exists() else pd.DataFrame()
if not FORECASTS.empty:
    FORECASTS["hour"] = pd.to_datetime(FORECASTS["hour"])
FORECAST_READY = not EVALUATION.empty and not FORECASTS.empty
PIPELINE_RUN = read_json(PIPELINE_RUN_PATH)
FORECAST_RUN = read_json(FORECAST_RUN_PATH)
ORCHESTRATION_RUN = read_json(ORCHESTRATION_PATH)
PRIORITY = build_priority_table(SUMMARY, EVALUATION) if DATA_READY else pd.DataFrame()
MODEL_COMPARISON = model_comparison_table(EVALUATION)

if DATA_READY:
    MIN_DATE = HOURLY["hour"].min().date()
    MAX_DATE = HOURLY["hour"].max().date()
    STATION_CHOICES = {
        str(row.site_id): row.station_label
        for row in SUMMARY.sort_values("station_label").itertuples(index=False)
    }
    if FORECAST_READY:
        DEFAULT_STATION = str(int(EVALUATION.sort_values("wape").iloc[0]["site_id"]))
    else:
        DEFAULT_STATION = str(
            SUMMARY.sort_values("avg_daily_count", ascending=False).iloc[0]["site_id"]
        )
else:
    MIN_DATE = pd.Timestamp("2025-01-01").date()
    MAX_DATE = pd.Timestamp("2026-04-30").date()
    STATION_CHOICES = {"": "Run preprocessing first"}
    DEFAULT_STATION = ""


def blank_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    fig.update_layout(template="plotly_white", height=420)
    return fig


def seasonal_naive_forecast(station_hourly: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    history = station_hourly[["hour", "count"]].sort_values("hour").copy()
    if history.empty:
        return pd.DataFrame(columns=["hour", "forecast"])

    last_hour = history["hour"].max()
    future_hours = pd.date_range(last_hour + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")

    lookup = history.set_index("hour")["count"]
    history["hour_of_week"] = history["hour"].dt.dayofweek * 24 + history["hour"].dt.hour
    fallback = history.groupby("hour_of_week")["count"].median()

    forecasts = []
    for hour in future_hours:
        previous_week = hour - pd.Timedelta(days=7)
        hour_of_week = hour.dayofweek * 24 + hour.hour
        value = lookup.get(previous_week, fallback.get(hour_of_week, history["count"].median()))
        forecasts.append({"hour": hour, "forecast": max(float(value), 0.0)})
    return pd.DataFrame(forecasts)


def baseline_validation(station_hourly: pd.DataFrame, horizon_hours: int) -> dict[str, float]:
    history = station_hourly[["hour", "count"]].sort_values("hour").copy()
    if len(history) < horizon_hours + 168:
        return {"MAE": float("nan"), "RMSE": float("nan"), "WAPE": float("nan")}

    test = history.tail(horizon_hours).copy()
    train = history.iloc[:-horizon_hours].copy()
    train_lookup = train.set_index("hour")["count"]
    train["hour_of_week"] = train["hour"].dt.dayofweek * 24 + train["hour"].dt.hour
    fallback = train.groupby("hour_of_week")["count"].median()

    preds = []
    for hour in test["hour"]:
        previous_week = hour - pd.Timedelta(days=7)
        hour_of_week = hour.dayofweek * 24 + hour.hour
        preds.append(train_lookup.get(previous_week, fallback.get(hour_of_week, train["count"].median())))

    error = test["count"].to_numpy() - pd.Series(preds).to_numpy()
    mae = abs(error).mean()
    rmse = (error**2).mean() ** 0.5
    wape = abs(error).sum() / max(test["count"].sum(), 1) * 100
    return {"MAE": mae, "RMSE": rmse, "WAPE": wape}


app_ui = ui.page_navbar(
    ui.nav_panel(
        "Overview",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_date_range("overview_dates", "Date range", start=MIN_DATE, end=MAX_DATE),
                ui.input_slider("top_n", "Stations in rankings", min=5, max=25, value=10),
                width=320,
            ),
            ui.output_ui("status_message"),
            ui.layout_columns(
                ui.value_box("Stations", ui.output_text("station_count"), showcase="sites"),
                ui.value_box("Hourly rows", ui.output_text("row_count"), showcase="rows"),
                ui.value_box("Total cyclists", ui.output_text("total_count"), showcase="count"),
                col_widths=[4, 4, 4],
            ),
            ui.card(
                ui.card_header("Station map"),
                output_widget("station_map"),
            ),
            ui.layout_columns(
                ui.card(ui.card_header("Highest average daily traffic"), ui.output_data_frame("top_traffic_table")),
                ui.card(ui.card_header("Fastest recent growth"), ui.output_data_frame("top_growth_table")),
                col_widths=[6, 6],
            ),
        ),
    ),
    ui.nav_panel(
        "Station Detail",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_selectize("station_id", "Station", choices=STATION_CHOICES, selected=DEFAULT_STATION),
                ui.input_date_range("station_dates", "Date range", start=MIN_DATE, end=MAX_DATE),
                width=360,
            ),
            ui.layout_columns(
                ui.value_box("Avg daily traffic", ui.output_text("station_avg_daily"), showcase="avg"),
                ui.value_box("Peak hour count", ui.output_text("station_peak_hour"), showcase="peak"),
                ui.value_box("Coverage", ui.output_text("station_coverage"), showcase="data"),
                col_widths=[4, 4, 4],
            ),
            ui.card(ui.card_header("Hourly traffic over time"), output_widget("station_timeseries")),
            ui.layout_columns(
                ui.card(ui.card_header("Average pattern by hour"), output_widget("hourly_pattern")),
                ui.card(ui.card_header("Weekday profile"), output_widget("weekday_pattern")),
                col_widths=[6, 6],
            ),
        ),
    ),
    ui.nav_panel(
        "Forecast",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_selectize("forecast_station_id", "Station", choices=STATION_CHOICES, selected=DEFAULT_STATION),
                ui.input_radio_buttons(
                    "forecast_horizon",
                    "Forecast horizon",
                    choices={"24": "Next 24 hours", "168": "Next 7 days hourly"},
                    selected="168",
                ),
                width=360,
            ),
            ui.layout_columns(
                ui.value_box("Baseline WAPE", ui.output_text("baseline_wape"), showcase="base"),
                ui.value_box("Best ML WAPE", ui.output_text("best_ml_wape"), showcase="model"),
                ui.value_box("Best model", ui.output_text("best_model"), showcase="best"),
                col_widths=[4, 4, 4],
            ),
            ui.card(
                ui.card_header("Forecast scope"),
                ui.output_text("forecast_scope_note"),
            ),
            ui.card(
                ui.card_header("Forecast output"),
                output_widget("forecast_plot"),
            ),
            ui.card(
                ui.card_header("Model evaluation across rolling backtests"),
                ui.output_data_frame("forecast_eval_table"),
            ),
            ui.card(
                ui.card_header("Model added value across trained stations"),
                ui.output_data_frame("model_comparison_table"),
            ),
            ui.card(
                ui.card_header("Interpretation"),
                ui.output_text("forecast_note"),
            ),
        ),
    ),
    ui.nav_panel(
        "Planning Insights",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_slider("insight_n", "Rows per table", min=5, max=30, value=12),
                width=300,
            ),
            ui.card(
                ui.card_header("Planning-oriented indicators"),
                ui.p(
                    "These tables turn the monitoring data into signals for infrastructure planning: high demand, fast growth, peak pressure, and data quality risks."
                ),
                ui.p(
                    "Priority score = 35% demand + 25% peak pressure + 20% recent growth + 10% forecast reliability + 10% data coverage."
                ),
                ui.p(
                    "The score is a screening tool, not an automatic investment decision. Low-coverage stations are flagged for sensor or data checks first."
                ),
            ),
            ui.card(
                ui.card_header("Infrastructure priority shortlist"),
                ui.output_data_frame("priority_table"),
            ),
            ui.layout_columns(
                ui.card(ui.card_header("High demand stations"), ui.output_data_frame("demand_table")),
                ui.card(ui.card_header("Peak pressure stations"), ui.output_data_frame("pressure_table")),
                col_widths=[6, 6],
            ),
            ui.layout_columns(
                ui.card(ui.card_header("Fast growth stations"), ui.output_data_frame("growth_table")),
                ui.card(ui.card_header("Lowest data coverage"), ui.output_data_frame("quality_table")),
                col_widths=[6, 6],
            ),
            ui.card(
                ui.card_header("Most predictable forecast stations"),
                ui.output_data_frame("predictability_table"),
            ),
        ),
    ),
    ui.nav_panel(
        "ML Engineering",
        ui.layout_columns(
            ui.value_box("Processed rows", ui.output_text("engineering_rows"), showcase="data"),
            ui.value_box("Model runs", ui.output_text("engineering_models"), showcase="ml"),
            ui.value_box("Forecast horizon", ui.output_text("engineering_horizon"), showcase="time"),
            col_widths=[4, 4, 4],
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Pipeline status"),
                ui.output_data_frame("pipeline_status_table"),
            ),
            ui.card(
                ui.card_header("Generated artifacts"),
                ui.output_data_frame("artifact_table"),
            ),
            col_widths=[6, 6],
        ),
        ui.card(
            ui.card_header("Re-run commands"),
            ui.output_text_verbatim("rerun_commands"),
        ),
        ui.card(
            ui.card_header("Why this matters"),
            ui.p(
                "This page makes the project less like a one-off analysis. It documents how raw CSV files become processed features, how models are retrained, and which artifacts the dashboard consumes."
            ),
        ),
    ),
    title="Bicycle Traffic Monitoring Dashboard",
    window_title="Bicycle Traffic Monitoring Dashboard",
    header=ui.include_css(ROOT / "app" / "www" / "styles.css"),
)


def server(input, output, session):
    @reactive.calc
    def overview_data() -> pd.DataFrame:
        if not DATA_READY:
            return pd.DataFrame()
        start, end = input.overview_dates()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
        return HOURLY.loc[(HOURLY["hour"] >= start_ts) & (HOURLY["hour"] < end_ts)].copy()

    @reactive.calc
    def selected_station_data() -> pd.DataFrame:
        if not DATA_READY or not input.station_id():
            return pd.DataFrame()
        start, end = input.station_dates()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
        site_id = int(input.station_id())
        return HOURLY.loc[
            (HOURLY["site_id"] == site_id) & (HOURLY["hour"] >= start_ts) & (HOURLY["hour"] < end_ts)
        ].copy()

    @reactive.calc
    def forecast_station_data() -> pd.DataFrame:
        if not DATA_READY or not input.forecast_station_id():
            return pd.DataFrame()
        site_id = int(input.forecast_station_id())
        return HOURLY.loc[HOURLY["site_id"] == site_id].copy()

    @reactive.calc
    def selected_evaluation() -> pd.DataFrame:
        if not FORECAST_READY or not input.forecast_station_id():
            return pd.DataFrame()
        site_id = int(input.forecast_station_id())
        return EVALUATION.loc[EVALUATION["site_id"] == site_id].copy()

    @reactive.calc
    def selected_forecasts() -> pd.DataFrame:
        if not FORECAST_READY or not input.forecast_station_id():
            return pd.DataFrame()
        site_id = int(input.forecast_station_id())
        horizon = int(input.forecast_horizon())
        data = FORECASTS.loc[FORECASTS["site_id"] == site_id].copy()
        return data.sort_values("hour").groupby("model", group_keys=False).head(horizon)

    @output
    @render.ui
    def status_message():
        if DATA_READY:
            return ui.div(
                {"class": "status-pill"},
                f"Loaded processed data from {MIN_DATE} to {MAX_DATE}.",
            )
        return ui.div(
            {"class": "missing-data"},
            "Processed data not found. Run: python codes/preprocess.py",
        )

    @output
    @render.text
    def station_count():
        if not DATA_READY:
            return "0"
        return f"{overview_data()['site_id'].nunique():,}"

    @output
    @render.text
    def row_count():
        return f"{len(overview_data()):,}" if DATA_READY else "0"

    @output
    @render.text
    def total_count():
        return f"{int(overview_data()['count'].sum()):,}" if DATA_READY else "0"

    @render_widget
    def station_map():
        data = overview_data()
        if data.empty:
            return blank_figure("No processed data available.")

        station_period = (
            data.groupby(["site_id", "station_label", "naam", "gemeente", "lat", "long"], as_index=False)
            .agg(total_count=("count", "sum"), avg_hourly_count=("count", "mean"))
            .dropna(subset=["lat", "long"])
        )
        station_period["avg_daily_count"] = station_period["total_count"] / max(
            data["hour"].dt.date.nunique(), 1
        )

        fig = px.scatter_mapbox(
            station_period,
            lat="lat",
            lon="long",
            size="avg_daily_count",
            color="avg_daily_count",
            hover_name="station_label",
            hover_data={"avg_daily_count": ":.1f", "total_count": ":,", "lat": False, "long": False},
            color_continuous_scale="Viridis",
            zoom=7,
            height=520,
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"l": 0, "r": 0, "t": 0, "b": 0})
        return fig

    @output
    @render.data_frame
    def top_traffic_table():
        data = overview_data()
        if data.empty:
            return pd.DataFrame()
        days = max(data["hour"].dt.date.nunique(), 1)
        table = (
            data.groupby(["site_id", "station_label"], as_index=False)["count"]
            .sum()
            .assign(avg_daily=lambda d: d["count"] / days)
            .sort_values("avg_daily", ascending=False)
            .head(input.top_n())
        )
        return table[["station_label", "avg_daily", "count"]].round({"avg_daily": 1})

    @output
    @render.data_frame
    def top_growth_table():
        if not DATA_READY:
            return pd.DataFrame()
        table = SUMMARY.sort_values("growth_pct_recent_vs_previous", ascending=False).head(input.top_n())
        return table[["station_label", "recent_avg_daily", "previous_avg_daily", "growth_pct_recent_vs_previous"]].round(1)

    @output
    @render.text
    def station_avg_daily():
        data = selected_station_data()
        if data.empty:
            return "n/a"
        days = max(data["hour"].dt.date.nunique(), 1)
        return f"{data['count'].sum() / days:,.1f}"

    @output
    @render.text
    def station_peak_hour():
        data = selected_station_data()
        return f"{int(data['count'].max()):,}" if not data.empty else "n/a"

    @output
    @render.text
    def station_coverage():
        if not DATA_READY or not input.station_id():
            return "n/a"
        row = SUMMARY.loc[SUMMARY["site_id"] == int(input.station_id())]
        if row.empty:
            return "n/a"
        return f"{row.iloc[0]['coverage_rate'] * 100:.1f}%"

    @render_widget
    def station_timeseries():
        data = selected_station_data()
        if data.empty:
            return blank_figure("No data for this station and date range.")
        daily = data.assign(date=data["hour"].dt.date).groupby("date", as_index=False)["count"].sum()
        fig = px.line(daily, x="date", y="count", markers=False, template="plotly_white")
        fig.update_layout(height=390, xaxis_title="", yaxis_title="Daily cyclists")
        return fig

    @render_widget
    def hourly_pattern():
        data = selected_station_data()
        if data.empty:
            return blank_figure("No data for this station and date range.")
        pattern = data.assign(hour_of_day=data["hour"].dt.hour).groupby("hour_of_day", as_index=False)["count"].mean()
        fig = px.bar(pattern, x="hour_of_day", y="count", template="plotly_white")
        fig.update_layout(height=350, xaxis_title="Hour of day", yaxis_title="Average cyclists")
        return fig

    @render_widget
    def weekday_pattern():
        data = selected_station_data()
        if data.empty:
            return blank_figure("No data for this station and date range.")
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pattern = data.assign(weekday=data["hour"].dt.dayofweek).groupby("weekday", as_index=False)["count"].mean()
        pattern["weekday"] = pattern["weekday"].map(dict(enumerate(names)))
        fig = px.bar(pattern, x="weekday", y="count", template="plotly_white")
        fig.update_layout(height=350, xaxis_title="", yaxis_title="Average hourly cyclists")
        return fig

    @output
    @render.text
    def baseline_wape():
        evaluation = selected_evaluation()
        if not evaluation.empty:
            row = evaluation.loc[evaluation["model"] == "seasonal_naive"]
            if not row.empty:
                return f"{row.iloc[0]['wape']:.1f}%"
        metrics = baseline_validation(forecast_station_data(), int(input.forecast_horizon()))
        return "n/a" if pd.isna(metrics["WAPE"]) else f"{metrics['WAPE']:.1f}%"

    @output
    @render.text
    def best_ml_wape():
        evaluation = selected_evaluation()
        if evaluation.empty:
            return "not trained"
        challengers = evaluation.loc[evaluation["model"] != "seasonal_naive"]
        if challengers.empty:
            return "not trained"
        return f"{challengers['wape'].min():.1f}%"

    @output
    @render.text
    def best_model():
        evaluation = selected_evaluation()
        if evaluation.empty:
            return "baseline"
        row = evaluation.sort_values("wape").iloc[0]
        return str(row["model"]).replace("_", " ")

    @output
    @render.text
    def forecast_scope_note():
        if not DATA_READY:
            return "No processed data loaded yet."
        trained = EVALUATION["site_id"].nunique() if not EVALUATION.empty else 0
        total = SUMMARY["site_id"].nunique() if not SUMMARY.empty else 0
        windows = int(FORECAST_RUN.get("backtest_windows", 1)) if FORECAST_RUN else 1
        step = int(FORECAST_RUN.get("backtest_step_hours", 168)) if FORECAST_RUN else 168
        if FORECAST_READY:
            return (
                f"Monitoring and planning use all {total} stations. Saved forecasts are currently trained for "
                f"{trained} selected stations. Model metrics are averaged over {windows} rolling backtest windows "
                f"spaced {step} hours apart. Use --all-stations to train forecasts for every eligible station."
            )
        return (
            f"Monitoring and planning use all {total} stations, but saved forecast artifacts have not been generated yet."
        )

    @render_widget
    def forecast_plot():
        data = forecast_station_data()
        if data.empty:
            return blank_figure("No data available for forecasting.")
        horizon = int(input.forecast_horizon())
        saved_forecast = selected_forecasts()
        recent = data.sort_values("hour").tail(14 * 24)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["hour"], y=recent["count"], mode="lines", name="Recent history"))
        if saved_forecast.empty:
            forecast = seasonal_naive_forecast(data, horizon)
            fig.add_trace(
                go.Scatter(
                    x=forecast["hour"],
                    y=forecast["forecast"],
                    mode="lines",
                    name="Seasonal naive forecast",
                    line={"dash": "dash"},
                )
            )
        else:
            for model_name, model_data in saved_forecast.groupby("model", sort=False):
                label = str(model_name).replace("_", " ").title()
                fig.add_trace(
                    go.Scatter(
                        x=model_data["hour"],
                        y=model_data["forecast"],
                        mode="lines",
                        name=label,
                        line={"dash": "dash" if model_name == "seasonal_naive" else "solid"},
                    )
                )
                if model_name == "prophet" and model_data["forecast_lower"].notna().any():
                    fig.add_trace(
                        go.Scatter(
                            x=pd.concat([model_data["hour"], model_data["hour"].iloc[::-1]]),
                            y=pd.concat(
                                [model_data["forecast_upper"], model_data["forecast_lower"].iloc[::-1]]
                            ),
                            fill="toself",
                            fillcolor="rgba(31, 138, 112, 0.16)",
                            line={"color": "rgba(255,255,255,0)"},
                            hoverinfo="skip",
                            name="Prophet interval",
                        )
                    )
        fig.update_layout(
            template="plotly_white",
            height=460,
            xaxis_title="",
            yaxis_title="Hourly cyclists",
            legend_orientation="h",
        )
        return fig

    @output
    @render.data_frame
    def forecast_eval_table():
        evaluation = selected_evaluation()
        if evaluation.empty:
            metrics = baseline_validation(forecast_station_data(), int(input.forecast_horizon()))
            return pd.DataFrame(
                [
                    {
                        "model": "seasonal_naive",
                        "mae": metrics["MAE"],
                        "rmse": metrics["RMSE"],
                        "wape": metrics["WAPE"],
                    }
                ]
            ).round(2)
        columns = [
            "model",
            "mae",
            "rmse",
            "wape",
            "backtest_windows",
            "test_hours",
            "test_start",
            "test_end",
        ]
        return evaluation[[column for column in columns if column in evaluation.columns]].round(2)

    @output
    @render.data_frame
    def model_comparison_table():
        if MODEL_COMPARISON.empty:
            return pd.DataFrame()
        table = MODEL_COMPARISON.head(12).copy()
        return table[
            [
                "station_label",
                "gemeente",
                "model",
                "baseline_wape",
                "model_wape",
                "wape_improvement_vs_baseline_pct",
                "beats_baseline",
            ]
        ].round(2)

    @output
    @render.text
    def forecast_note():
        if FORECAST_READY and not selected_evaluation().empty:
            return (
                "This page reads saved model outputs from the forecasting pipeline. "
                "The dashboard compares seasonal naive, histogram gradient boosting, and Prophet across rolling time-based backtests. "
                "The best model is selected per station by average WAPE, then future forecasts are generated after retraining on the latest available history."
            )
        return (
            "This station does not have saved Prophet outputs yet, so the dashboard falls back to an online seasonal naive baseline."
        )

    @output
    @render.data_frame
    def priority_table():
        if PRIORITY.empty:
            return pd.DataFrame()
        table = PRIORITY.head(input.insight_n()).copy()
        table["coverage_pct"] = table["coverage_rate"] * 100
        return table[
            [
                "station_label",
                "gemeente",
                "priority_score",
                "avg_daily_count",
                "p95_hour_count",
                "growth_pct_recent_vs_previous",
                "coverage_pct",
                "best_forecast_model",
                "best_forecast_wape",
                "planning_recommendation",
            ]
        ].round(2)

    @output
    @render.data_frame
    def demand_table():
        if not DATA_READY:
            return pd.DataFrame()
        table = SUMMARY.sort_values("avg_daily_count", ascending=False).head(input.insight_n())
        return table[["station_label", "avg_daily_count", "total_count", "gemeente"]].round(1)

    @output
    @render.data_frame
    def pressure_table():
        if not DATA_READY:
            return pd.DataFrame()
        table = SUMMARY.sort_values("p95_hour_count", ascending=False).head(input.insight_n())
        return table[["station_label", "p95_hour_count", "peak_hour_count", "gemeente"]].round(1)

    @output
    @render.data_frame
    def growth_table():
        if not DATA_READY:
            return pd.DataFrame()
        table = SUMMARY.sort_values("growth_pct_recent_vs_previous", ascending=False).head(input.insight_n())
        return table[["station_label", "recent_avg_daily", "previous_avg_daily", "growth_pct_recent_vs_previous"]].round(1)

    @output
    @render.data_frame
    def quality_table():
        if not DATA_READY:
            return pd.DataFrame()
        table = SUMMARY.sort_values("coverage_rate", ascending=True).head(input.insight_n()).copy()
        table["coverage_pct"] = table["coverage_rate"] * 100
        return table[["station_label", "coverage_pct", "observed_hours", "expected_hours"]].round(1)

    @output
    @render.data_frame
    def predictability_table():
        if not FORECAST_READY:
            return pd.DataFrame()
        table = (
            EVALUATION.sort_values("wape")
            .groupby("site_id", as_index=False)
            .first()
            .sort_values("wape")
            .head(input.insight_n())
        )
        return table[["station_label", "model", "mae", "rmse", "wape", "test_hours"]].round(2)

    @output
    @render.text
    def engineering_rows():
        if PIPELINE_RUN:
            return f"{int(PIPELINE_RUN.get('hourly_rows', 0)):,}"
        return f"{len(HOURLY):,}" if DATA_READY else "0"

    @output
    @render.text
    def engineering_models():
        if EVALUATION.empty:
            return "0"
        windows = int(FORECAST_RUN.get("backtest_windows", 1)) if FORECAST_RUN else 1
        return f"{EVALUATION['model'].nunique()} models / {EVALUATION['site_id'].nunique()} stations / {windows} windows"

    @output
    @render.text
    def engineering_horizon():
        if FORECAST_RUN:
            return f"{int(FORECAST_RUN.get('horizon_hours', 0))} hours"
        return "not trained"

    @output
    @render.data_frame
    def pipeline_status_table():
        rows = [
            {
                "component": "Preprocessing",
                "status": "ready" if DATA_READY else "missing",
                "details": (
                    f"{PIPELINE_RUN.get('monthly_files', 0)} monthly files, "
                    f"{int(PIPELINE_RUN.get('stations_with_counts', 0)):,} stations"
                    if PIPELINE_RUN
                    else "Run codes/preprocess.py"
                ),
                "last_run_utc": PIPELINE_RUN.get("generated_at_utc", "n/a"),
            },
            {
                "component": "Forecast training",
                "status": "ready" if FORECAST_READY else "missing",
                "details": (
                    f"{FORECAST_RUN.get('evaluated_rows', 0)} evaluation rows, "
                    f"{FORECAST_RUN.get('backtest_rows', 0)} backtest rows, "
                    f"{FORECAST_RUN.get('forecast_rows', 0)} forecast rows"
                    if FORECAST_RUN
                    else "Run codes/train_forecasts.py"
                ),
                "last_run_utc": FORECAST_RUN.get("generated_at_utc", "n/a"),
            },
            {
                "component": "Pipeline orchestration",
                "status": "ready" if ORCHESTRATION_RUN else "optional",
                "details": ORCHESTRATION_RUN.get("status", "Use codes/run_pipeline.py for one-command refresh"),
                "last_run_utc": ORCHESTRATION_RUN.get("finished_at_utc", "n/a"),
            },
        ]
        return pd.DataFrame(rows)

    @output
    @render.data_frame
    def artifact_table():
        artifacts = [
            (HOURLY_PATH, "Hourly counts used by all dashboard pages"),
            (SUMMARY_PATH, "Station-level features and planning indicators"),
            (EVALUATION_PATH, "Average model metrics across rolling backtests"),
            (BACKTEST_PATH, "Detailed per-window backtest metrics"),
            (FORECAST_PATH, "Saved 24h/168h forecast outputs"),
            (PIPELINE_RUN_PATH, "Preprocessing run metadata"),
            (FORECAST_RUN_PATH, "Forecast training run metadata"),
            (ORCHESTRATION_PATH, "One-command pipeline run metadata"),
        ]
        rows = []
        for path, role in artifacts:
            exists = path.exists()
            rows.append(
                {
                    "artifact": str(path.relative_to(ROOT)),
                    "role": role,
                    "status": "available" if exists else "missing",
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 2) if exists else 0,
                }
            )
        return pd.DataFrame(rows)

    @output
    @render.text
    def rerun_commands():
        return "\n".join(
            [
                "# Full refresh from raw CSV files",
                "python codes/run_pipeline.py --start-month 2019-08 --end-month 2026-04 --top-stations 12",
                "",
                "# If processed data already exists, only retrain forecasts",
                "python codes/run_pipeline.py --skip-preprocess --top-stations 12 --backtest-windows 3",
                "",
                "# Optional: train forecasts for all eligible stations. This can take much longer.",
                "python codes/run_pipeline.py --skip-preprocess --all-stations --backtest-windows 3",
                "",
                "# Run the dashboard locally",
                "shiny run --host 127.0.0.1 --port 8000 app/app.py",
            ]
        )


app = App(app_ui, server)
