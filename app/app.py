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
OUTLIER_Z_CUTOFF = 2.576


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


def pct_change(current: pd.Series, reference: pd.Series) -> pd.Series:
    current = pd.to_numeric(current, errors="coerce")
    reference = pd.to_numeric(reference, errors="coerce").mask(lambda values: values <= 0)
    return (current - reference) / reference * 100


def recent_daily_frame(hourly: pd.DataFrame, summary: pd.DataFrame, days: int = 28) -> pd.DataFrame:
    daily = (
        hourly[["site_id", "hour", "count"]]
        .assign(date=lambda frame: frame["hour"].dt.floor("D"))
        .groupby(["site_id", "date"], as_index=False, observed=True)["count"]
        .sum()
    )
    if daily.empty:
        return pd.DataFrame(columns=["site_id", "date", "count"])

    max_date = daily["date"].max()
    dates = pd.date_range(max_date - pd.Timedelta(days=days - 1), max_date, freq="D")
    site_ids = summary["site_id"].dropna().astype(int).sort_values()
    full_index = pd.MultiIndex.from_product([site_ids, dates], names=["site_id", "date"])
    return daily.set_index(["site_id", "date"]).reindex(full_index).reset_index()


def period_mean_and_coverage(daily: pd.DataFrame, days: int) -> tuple[pd.Series, pd.Series]:
    if daily.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    max_date = daily["date"].max()
    start = max_date - pd.Timedelta(days=days - 1)
    period = daily.loc[daily["date"] >= start]
    grouped = period.groupby("site_id", observed=True)["count"]
    return grouped.mean(), grouped.apply(lambda values: values.notna().mean())


def build_trend_tables(
    hourly: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if hourly.empty or summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    daily = recent_daily_frame(hourly, summary, days=28)
    table = summary[["site_id", "station_label", "gemeente"]].copy()
    coverage_4w = None
    for label, days in [("1w", 7), ("2w", 14), ("4w", 28)]:
        mean, coverage = period_mean_and_coverage(daily, days)
        table = table.merge(mean.rename(f"ma_{label}"), on="site_id", how="left")
        if days == 28:
            coverage_4w = coverage.rename("recent_coverage_4w")

    if coverage_4w is None:
        table["recent_coverage_4w"] = 0.0
    else:
        table = table.merge(coverage_4w, on="site_id", how="left")
        table["recent_coverage_4w"] = table["recent_coverage_4w"].fillna(0.0)

    table["change_1w_vs_4w_pct"] = pct_change(table["ma_1w"], table["ma_4w"])
    table["recent_coverage_4w_pct"] = table["recent_coverage_4w"] * 100
    table["data_note"] = table["recent_coverage_4w"].map(
        lambda value: "OK" if value >= 0.8 else "Low recent coverage"
    )
    trend_ready = table.loc[table["recent_coverage_4w"] >= 0.8].copy()
    growth = trend_ready.loc[
        (trend_ready["ma_1w"] > trend_ready["ma_2w"])
        & (trend_ready["ma_2w"] > trend_ready["ma_4w"])
    ].copy()
    growth = growth.sort_values("change_1w_vs_4w_pct", ascending=False)

    decline = trend_ready.loc[
        (trend_ready["ma_1w"] < trend_ready["ma_2w"])
        & (trend_ready["ma_2w"] < trend_ready["ma_4w"])
    ].copy()
    decline = decline.sort_values("change_1w_vs_4w_pct")
    return growth, decline


def robust_weekday_baseline(values: pd.Series) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    robust_std = 1.4826 * mad
    return pd.Series(
        {
            "weekday_median": median,
            "weekday_mad": mad,
            "weekday_robust_std": max(float(robust_std), 1.0),
            "weekday_observations": int(values.notna().sum()),
        }
    )


def build_daily_outlier_table(hourly: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty or summary.empty:
        return pd.DataFrame()

    daily = (
        hourly[["site_id", "hour", "count"]]
        .assign(date=lambda frame: frame["hour"].dt.floor("D"))
        .groupby(["site_id", "date"], as_index=False, observed=True)["count"]
        .sum()
    )
    if daily.empty:
        return pd.DataFrame()

    daily["weekday"] = daily["date"].dt.day_name()
    baseline = (
        daily.groupby(["site_id", "weekday"], observed=True)["count"]
        .apply(robust_weekday_baseline)
        .unstack()
        .reset_index()
    )
    scored = daily.merge(baseline, on=["site_id", "weekday"], how="left")
    scored["robust_z"] = (
        (scored["count"] - scored["weekday_median"]) / scored["weekday_robust_std"]
    )
    scored["outlier_direction"] = "normal"
    scored.loc[scored["robust_z"] > OUTLIER_Z_CUTOFF, "outlier_direction"] = "high"
    scored.loc[scored["robust_z"] < -OUTLIER_Z_CUTOFF, "outlier_direction"] = "low"
    scored["is_outlier"] = scored["outlier_direction"] != "normal"

    max_date = scored["date"].max()
    recent_start = max_date - pd.Timedelta(days=27)
    recent = scored.loc[scored["date"] >= recent_start].copy()
    recent["outlier_date"] = recent["date"].where(recent["is_outlier"], pd.NaT)

    summary_rows = (
        recent.groupby("site_id", observed=True)
        .agg(
            recent_days_checked=("date", "nunique"),
            outlier_days_4w=("is_outlier", "sum"),
            high_outlier_days_4w=("outlier_direction", lambda values: int((values == "high").sum())),
            low_outlier_days_4w=("outlier_direction", lambda values: int((values == "low").sum())),
            max_abs_robust_z=("robust_z", lambda values: float(values.abs().max())),
            latest_outlier_date=("outlier_date", "max"),
        )
        .reset_index()
    )
    summary_rows["outlier_rate_4w_pct"] = (
        summary_rows["outlier_days_4w"] / summary_rows["recent_days_checked"].replace(0, pd.NA) * 100
    )
    summary_rows["latest_outlier_date"] = pd.to_datetime(
        summary_rows["latest_outlier_date"], errors="coerce"
    ).dt.date
    summary_rows = summary_rows.merge(
        summary[["site_id", "station_label", "gemeente"]],
        on="site_id",
        how="left",
    )
    return summary_rows.sort_values(
        ["outlier_days_4w", "outlier_rate_4w_pct", "max_abs_robust_z"],
        ascending=[False, False, False],
    )


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
DAILY_OUTLIERS = build_daily_outlier_table(HOURLY, SUMMARY) if DATA_READY else pd.DataFrame()
GROWTH_SIGNALS, DECLINE_SIGNALS = (
    build_trend_tables(HOURLY, SUMMARY)
    if DATA_READY
    else (pd.DataFrame(), pd.DataFrame())
)

if DATA_READY:
    MIN_DATE = HOURLY["hour"].min().date()
    MAX_DATE = HOURLY["hour"].max().date()
    STATION_CHOICES = {
        str(row.site_id): row.station_label
        for row in SUMMARY.sort_values("station_label").itertuples(index=False)
    }
    if FORECAST_READY:
        forecasted_sites = EVALUATION["site_id"].dropna().astype(int).unique()
        default_row = (
            SUMMARY.loc[SUMMARY["site_id"].isin(forecasted_sites)]
            .sort_values("avg_daily_count", ascending=False)
            .iloc[0]
        )
        DEFAULT_STATION = str(int(default_row["site_id"]))
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
            ui.card(ui.card_header("Highest average daily traffic"), ui.output_data_frame("top_traffic_table")),
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
            ui.card(ui.card_header("Daily traffic over time"), output_widget("station_timeseries")),
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
                ui.card_header("Forecast output"),
                output_widget("forecast_plot"),
            ),
            ui.card(
                ui.card_header("Model evaluation across rolling backtests"),
                ui.output_data_frame("forecast_eval_table"),
            ),
            ui.card(
                ui.card_header("Scope and interpretation"),
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
                ui.card_header("Three separate planning questions"),
                ui.div(
                    {"class": "compact-explainer"},
                    ui.p("This page separates planning signals into three focused views, so planners can inspect one question at a time."),
                    ui.tags.ul(
                        ui.tags.li("Fast Growth: ma_1w > ma_2w > ma_4w, highlighting consistently strengthening demand."),
                        ui.tags.li("Fast Decline: ma_1w < ma_2w < ma_4w, highlighting consistently weakening demand."),
                        ui.tags.li("Daily Outliers: flags unusual daily counts against each station's own same-weekday history."),
                    ),
                ),
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Fast Growth",
                    ui.p(
                        "Question: which stations show the strongest recent acceleration? Rule: ma_1w > ma_2w > ma_4w. Ranking: largest positive change_1w_vs_4w_pct."
                    ),
                    ui.output_data_frame("growth_table"),
                ),
                ui.nav_panel(
                    "Fast Decline",
                    ui.p(
                        "Question: which stations show the strongest recent deterioration? Rule: ma_1w < ma_2w < ma_4w. Ranking: most negative change_1w_vs_4w_pct."
                    ),
                    ui.output_data_frame("decline_table"),
                ),
                ui.nav_panel(
                    "Daily Outliers",
                    ui.p(
                        "Question: which stations have the most unusual days in the latest 4 weeks? A day is an outlier when its robust z-score exceeds the two-sided 99% cutoff."
                    ),
                    ui.output_data_frame("daily_outlier_table"),
                ),
            ),
        ),
    ),
    ui.nav_panel(
        "ML Engineering",
        ui.card(
            ui.card_header("End-to-end pipeline"),
            ui.div(
                {"class": "pipeline-flow"},
                ui.div(
                    {"class": "flow-step"},
                    ui.div({"class": "flow-step-number"}, "1"),
                    ui.h5("Ingest raw data"),
                    ui.p("Monthly AWV CSV files and station metadata are read from the local data folder."),
                ),
                ui.div(
                    {"class": "flow-step"},
                    ui.div({"class": "flow-step-number"}, "2"),
                    ui.h5("Build features"),
                    ui.p("Raw counts are aggregated into hourly station data and station-level summary features."),
                ),
                ui.div(
                    {"class": "flow-step"},
                    ui.div({"class": "flow-step-number"}, "3"),
                    ui.h5("Train and validate"),
                    ui.p("Forecast models are trained per station and checked with rolling time-based backtests."),
                ),
                ui.div(
                    {"class": "flow-step"},
                    ui.div({"class": "flow-step-number"}, "4"),
                    ui.h5("Serve dashboard"),
                    ui.p("The dashboard reads saved artifacts, so it stays fast and does not retrain models live."),
                ),
            ),
        ),
        ui.layout_columns(
            ui.value_box("Processed rows", ui.output_text("engineering_rows"), showcase="data"),
            ui.value_box("Forecast stations", ui.output_text("engineering_forecast_stations"), showcase="sites"),
            ui.value_box("Backtest windows", ui.output_text("engineering_backtests"), showcase="validation"),
            ui.value_box("Forecast horizon", ui.output_text("engineering_horizon"), showcase="time"),
            col_widths=[3, 3, 3, 3],
        ),
        ui.card(
            ui.card_header("Engineering takeaway"),
            ui.p(
                "The dashboard is only the presentation layer. Data preparation, model validation, forecast generation, and run metadata are handled by repeatable scripts before the dashboard loads."
            ),
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Run status"),
                ui.output_data_frame("pipeline_status_table"),
            ),
            ui.card(
                ui.card_header("Dashboard artifacts"),
                ui.output_data_frame("artifact_table"),
            ),
            col_widths=[6, 6],
        ),
        ui.card(
            ui.card_header("Operational commands"),
            ui.output_text_verbatim("rerun_commands"),
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
            .agg(
                total_count=("count", "sum"),
                avg_hourly_count=("count", "mean"),
                active_days=("hour", lambda hours: hours.dt.date.nunique()),
            )
            .dropna(subset=["lat", "long"])
        )
        station_period["avg_daily_count"] = station_period["total_count"] / station_period[
            "active_days"
        ].clip(lower=1)

        fig = px.scatter_mapbox(
            station_period,
            lat="lat",
            lon="long",
            size="avg_daily_count",
            color="avg_daily_count",
            hover_name="station_label",
            hover_data={
                "avg_daily_count": ":.1f",
                "active_days": ":,",
                "total_count": ":,",
                "lat": False,
                "long": False,
            },
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
        table = (
            data.groupby(["site_id", "station_label"], as_index=False)
            .agg(
                total_count=("count", "sum"),
                active_days=("hour", lambda hours: hours.dt.date.nunique()),
            )
            .assign(avg_daily=lambda d: d["total_count"] / d["active_days"].clip(lower=1))
            .sort_values("avg_daily", ascending=False)
            .head(input.top_n())
        )
        return table[["station_label", "avg_daily", "active_days", "total_count"]].round(
            {"avg_daily": 1}
        )

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
        fig.update_traces(line_color="#1f8a70")
        fig.update_layout(height=390, xaxis_title="", yaxis_title="Daily cyclists")
        return fig

    @render_widget
    def hourly_pattern():
        data = selected_station_data()
        if data.empty:
            return blank_figure("No data for this station and date range.")
        pattern = data.assign(hour_of_day=data["hour"].dt.hour).groupby("hour_of_day", as_index=False)["count"].mean()
        fig = px.bar(pattern, x="hour_of_day", y="count", template="plotly_white")
        fig.update_traces(marker_color="#f2b35d")
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
        fig.update_traces(marker_color="#3a7ca5")
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
    @render.text
    def forecast_note():
        if not DATA_READY:
            return "No processed data loaded yet."

        trained = EVALUATION["site_id"].nunique() if not EVALUATION.empty else 0
        total = SUMMARY["site_id"].nunique() if not SUMMARY.empty else 0
        windows = int(FORECAST_RUN.get("backtest_windows", 1)) if FORECAST_RUN else 1
        step = int(FORECAST_RUN.get("backtest_step_hours", 168)) if FORECAST_RUN else 168

        if not FORECAST_READY:
            return (
                f"Monitoring and planning use all {total} stations, but saved forecast artifacts have not been generated yet. "
                "This station therefore falls back to an online seasonal naive baseline."
            )

        if FORECAST_RUN.get("all_stations"):
            scope = f"all {trained} eligible stations"
            suffix = "Stations outside this set did not meet the minimum history or coverage rules."
        else:
            scope = f"{trained} selected stations"
            suffix = "Use --all-stations to train forecasts for every eligible station."

        scope_note = (
            f"Monitoring and planning use all {total} stations. Saved forecasts are currently trained for "
            f"{scope}. Model metrics are averaged over {windows} rolling backtest windows "
            f"spaced {step} hours apart. {suffix}"
        )

        if FORECAST_READY and not selected_evaluation().empty:
            return (
                f"{scope_note} This page reads saved model outputs from the forecasting pipeline. "
                "For the selected station, it compares a seasonal naive baseline with histogram gradient boosting across rolling time-based backtests. "
                "The best model for this station is selected by average WAPE, then future forecasts are generated after retraining on the latest available history."
            )
        return (
            f"{scope_note} This station does not have saved model outputs yet, so the dashboard falls back to an online seasonal naive baseline."
        )

    @output
    @render.data_frame
    def growth_table():
        if GROWTH_SIGNALS.empty:
            return pd.DataFrame(
                [{"message": "No reliable fast-growth station detected with the current trend view."}]
            )
        table = GROWTH_SIGNALS.head(input.insight_n()).copy()
        return table[
            [
                "station_label",
                "gemeente",
                "ma_1w",
                "ma_2w",
                "ma_4w",
                "change_1w_vs_4w_pct",
            ]
        ].round(2)

    @output
    @render.data_frame
    def decline_table():
        if DECLINE_SIGNALS.empty:
            return pd.DataFrame(
                [
                    {
                        "message": "No reliable fast-decline station detected with the current trend view."
                    }
                ]
            )
        table = DECLINE_SIGNALS.head(input.insight_n()).copy()
        return table[
            [
                "station_label",
                "gemeente",
                "ma_1w",
                "ma_2w",
                "ma_4w",
                "change_1w_vs_4w_pct",
            ]
        ].round(2)

    @output
    @render.data_frame
    def daily_outlier_table():
        if DAILY_OUTLIERS.empty:
            return pd.DataFrame()
        table = DAILY_OUTLIERS.head(input.insight_n()).copy()
        return table[
            [
                "station_label",
                "gemeente",
                "outlier_days_4w",
                "high_outlier_days_4w",
                "low_outlier_days_4w",
                "outlier_rate_4w_pct",
                "max_abs_robust_z",
            ]
        ].round(2)

    @output
    @render.text
    def engineering_rows():
        if PIPELINE_RUN:
            return f"{int(PIPELINE_RUN.get('hourly_rows', 0)):,}"
        return f"{len(HOURLY):,}" if DATA_READY else "0"

    @output
    @render.text
    def engineering_forecast_stations():
        if EVALUATION.empty:
            return "0"
        total = SUMMARY["site_id"].nunique() if DATA_READY else EVALUATION["site_id"].nunique()
        trained = EVALUATION["site_id"].nunique()
        return f"{trained} / {total}"

    @output
    @render.text
    def engineering_backtests():
        if EVALUATION.empty:
            return "0"
        windows = int(FORECAST_RUN.get("backtest_windows", 1)) if FORECAST_RUN else 1
        return f"{windows} windows"

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
                "step": "1. Preprocessing",
                "status": "ready" if DATA_READY else "missing",
                "what_it_checks": "Raw CSVs -> hourly counts and station summaries",
                "evidence": (
                    f"{PIPELINE_RUN.get('monthly_files', 0)} monthly files, "
                    f"{int(PIPELINE_RUN.get('stations_with_counts', 0)):,} stations, "
                    f"{int(PIPELINE_RUN.get('hourly_rows', 0)):,} hourly rows"
                    if PIPELINE_RUN
                    else "Run codes/preprocess.py"
                ),
                "last_run_utc": PIPELINE_RUN.get("generated_at_utc", "n/a"),
            },
            {
                "step": "2. Forecast training",
                "status": "ready" if FORECAST_READY else "missing",
                "what_it_checks": "Per-station models, rolling backtests, saved forecasts",
                "evidence": (
                    f"{len(FORECAST_RUN.get('selected_sites', []))} stations, "
                    f"{FORECAST_RUN.get('backtest_windows', 0)} windows, "
                    f"{FORECAST_RUN.get('forecast_rows', 0)} forecast rows"
                    if FORECAST_RUN
                    else "Run codes/train_forecasts.py"
                ),
                "last_run_utc": FORECAST_RUN.get("generated_at_utc", "n/a"),
            },
            {
                "step": "3. Orchestration",
                "status": "ready" if ORCHESTRATION_RUN else "optional",
                "what_it_checks": "One command can refresh preprocessing and forecasts",
                "evidence": ORCHESTRATION_RUN.get("status", "Use codes/run_pipeline.py for one-command refresh"),
                "last_run_utc": ORCHESTRATION_RUN.get("finished_at_utc", "n/a"),
            },
            {
                "step": "4. Dashboard serving",
                "status": "ready" if DATA_READY else "missing",
                "what_it_checks": "Shiny reads saved artifacts instead of retraining live",
                "evidence": "App can load processed dashboard inputs" if DATA_READY else "Processed inputs are missing",
                "last_run_utc": "runtime",
            },
        ]
        return pd.DataFrame(rows)

    @output
    @render.data_frame
    def artifact_table():
        artifacts = [
            ("Features", HOURLY_PATH, "Hourly counts used by all dashboard pages"),
            ("Features", SUMMARY_PATH, "Station-level features and planning indicators"),
            ("Model validation", EVALUATION_PATH, "Average model metrics across rolling backtests"),
            ("Model validation", BACKTEST_PATH, "Detailed per-window backtest metrics"),
            ("Forecast serving", FORECAST_PATH, "Saved 24h/168h forecast outputs"),
            ("Run metadata", PIPELINE_RUN_PATH, "Preprocessing run metadata"),
            ("Run metadata", FORECAST_RUN_PATH, "Forecast training run metadata"),
            ("Run metadata", ORCHESTRATION_PATH, "One-command pipeline run metadata"),
        ]
        rows = []
        for stage, path, role in artifacts:
            exists = path.exists()
            rows.append(
                {
                    "stage": stage,
                    "artifact": str(path.relative_to(ROOT)),
                    "status": "available" if exists else "missing",
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 2) if exists else 0,
                    "role": role,
                }
            )
        return pd.DataFrame(rows)

    @output
    @render.text
    def rerun_commands():
        return "\n".join(
            [
                "# 1. Full refresh from raw CSV files",
                "python codes/run_pipeline.py --start-month 2019-08 --end-month 2026-04 --all-stations",
                "",
                "# 2. If processed data already exists, only retrain forecasts",
                "python codes/run_pipeline.py --skip-preprocess --all-stations --backtest-windows 3",
                "",
                "# 3. Run the dashboard locally",
                "shiny run --host 127.0.0.1 --port 8000 app/app.py",
            ]
        )


app = App(app_ui, server)
