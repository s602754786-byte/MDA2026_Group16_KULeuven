# Dashboard Demo Script

Use this as a 5-7 minute dashboard walkthrough.

## 1. Opening Message

Say:

> Our project is a station-level bicycle traffic monitoring, forecasting, and planning-support dashboard for AWV bicycle count data in Flanders. The goal is not only to visualize historical counts, but also to show a repeatable data and ML pipeline behind the dashboard.

Key point:

- Monitoring and planning use all 150 processed stations.
- Saved advanced forecasts are generated for the busiest 12 eligible stations by default to keep the demo fast.
- The pipeline can be rerun for all eligible stations with `--all-stations`.

## 2. Overview

Show:

- Date range selector.
- Station map.
- Station count, hourly rows, total cyclists.
- Highest average daily traffic table.
- Fastest recent growth table.

Say:

> This page answers where bicycle traffic is concentrated and which locations are changing quickly. The map gives a spatial overview, while the ranking tables help identify high-demand and fast-growth stations.

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

- Forecast scope card.
- 24-hour and 168-hour horizon options.
- Recent history plus forecast curves.
- Model evaluation table.
- Model added value table.

Say:

> We compare three models: seasonal naive, histogram gradient boosting, and Prophet. The metrics are not based on a single final week only; they are averaged over three rolling backtest windows. The best model is selected per station based on WAPE.

Mention:

- Seasonal naive is the baseline.
- Histogram gradient boosting is the feature-based sklearn model.
- Prophet is the time-series benchmark.
- HGB currently wins for most of the forecasted stations, but the baseline remains competitive.

## 5. Planning Insights

Show:

- Infrastructure priority shortlist.
- Priority score formula.
- High demand stations.
- Peak pressure stations.
- Fast growth stations.
- Lowest data coverage.
- Most predictable stations.

Say:

> This page translates monitoring and forecasting outputs into planning signals. The priority score combines demand, peak pressure, recent growth, forecast reliability, and data coverage. It is not an automatic investment decision, but a screening tool that tells planners where to look first.

Explain recommendation examples:

- `Capacity upgrade candidate`: high use and recent growth.
- `High-demand corridor`: high overall demand.
- `Peak-hour pressure`: high rush-hour pressure.
- `Check sensor/data quality`: coverage is too low for reliable decisions.
- `Keep monitoring`: no urgent signal yet.

## 6. ML Engineering

Show:

- Processed rows.
- Models, stations, and backtest windows.
- Pipeline status.
- Generated artifacts.
- Re-run commands.

Say:

> This page responds directly to the ML engineering requirement. It shows that the dashboard is backed by a reproducible pipeline: raw CSV files are processed into Parquet, models are evaluated with rolling backtests, forecast outputs are saved, and the dashboard reads those artifacts.

## 7. Closing Message

Say:

> The main contribution is a small modern data analytics system: it combines data ingestion, preprocessing, feature engineering, model evaluation, forecasting, dashboarding, and planning-oriented interpretation. Future work could add weather, holidays, and scheduled monthly retraining.
