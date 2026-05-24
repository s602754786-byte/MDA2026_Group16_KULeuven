# Dashboard Demo Script

Use this as a 5-7 minute dashboard walkthrough.

## 1. Opening Message

Say:

> Our project is a station-level bicycle traffic monitoring, forecasting, and planning-support dashboard for AWV bicycle count data in Flanders. The goal is not only to visualize historical counts, but also to show a repeatable data and ML pipeline behind the dashboard.

Key point:

- Monitoring and planning use all 150 processed stations.
- Saved forecasts are generated for all 147 eligible stations in the latest run.
- The pipeline can be rerun with `--all-stations` when new data is added.

## 2. Overview

Show:

- Date range selector.
- Station map.
- Station count, hourly rows, total cyclists.
- Highest average daily traffic table.

Say:

> This page answers where bicycle traffic is concentrated. The map gives a spatial overview, while the ranking table helps identify high-demand stations.

## 3. Station Detail

Choose a busy station, for example:

- `leuven totem | Leuven (#107)`
- `Nieuwpoort teller 1 | Nieuwpoort (#64)`
- `Hasselt-Kempische brug | Hasselt (#138)`

Show:

- Daily time series.
- Average pattern by hour.
- Weekday profile.
- Coverage value.

Say:

> This page lets planners inspect one counting station in detail. It shows whether the station has commuter peaks, weekday-weekend differences, and reliable enough data coverage for analysis.

## 4. Forecast

Show:

- 24-hour and 168-hour horizon options.
- Recent history plus forecast curves.
- Model evaluation table.
- Scope and interpretation note.

Say:

> We compare two models: a seasonal naive baseline and histogram gradient boosting. The metrics are not based on a single final week only; they are averaged over three rolling backtest windows. The best model is selected per station based on WAPE.

Mention:

- Seasonal naive is the baseline.
- Histogram gradient boosting is the feature-based sklearn model.
- HGB currently wins for most of the forecasted stations, but the baseline remains competitive.

## 5. Planning Insights

Show:

- Fast Growth tab: stations where short-term average demand is rising.
- Fast Decline tab: stations where short-term average demand is falling.
- Daily Outliers tab: stations with many statistically unusual days in the latest 4 weeks.

Say:

> This page separates three planning questions: which stations show strengthening demand, which stations show weakening demand, and which stations recently had many unusual daily counts.

Explain the moving-average rule:

- Growth and decline are based on 1-week, 2-week, and 4-week average daily counts.
- Fast Growth uses `ma_1w > ma_2w > ma_4w`.
- Fast Decline uses `ma_1w < ma_2w < ma_4w`.
- Daily Outliers compares each day only with the same station and same weekday.
- The baseline is robust: weekday median plus MAD-based scale.
- A day is marked as outlier if `abs(robust z-score) > 2.576`, using a two-sided 99% cutoff.
- The table ranks stations by the number of outlier days in the latest 4 weeks.

## 6. ML Engineering

Show:

- Processed rows.
- Forecast stations.
- Backtest windows and forecast horizon.
- End-to-end pipeline.
- Run status and dashboard artifacts.
- Operational commands.

Say:

> This page responds directly to the ML engineering requirement. It shows that the dashboard is backed by a reproducible pipeline: raw CSV files are processed into Parquet, models are evaluated with rolling backtests, forecast outputs are saved, and the dashboard reads those artifacts.

## 7. Closing Message

Say:

> The main contribution is a small modern data analytics system: it combines data ingestion, preprocessing, feature engineering, model evaluation, forecasting, dashboarding, and planning-oriented interpretation. Future work could add weather, holidays, and scheduled monthly retraining.
