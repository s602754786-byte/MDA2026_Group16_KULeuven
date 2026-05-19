# Bicycle Traffic Monitoring and Forecasting

Modern Data Analytics group project for station-level bicycle traffic monitoring, short-term forecasting, and planning support using the AWV `fietstellingen` dataset.

## Project Summary

This project builds a small modern data analytics system rather than a one-off analysis. It processes raw monthly bicycle count CSV files, creates hourly station-level Parquet datasets, trains and evaluates forecasting models with rolling backtests, and serves the results through a Shiny for Python dashboard.

The dashboard supports:

- monitoring all processed bicycle counting stations;
- inspecting station-level historical patterns;
- forecasting future hourly counts for selected high-demand stations;
- comparing forecasting models against a seasonal naive baseline;
- identifying planning-relevant stations using demand, peak pressure, growth, forecast reliability, and data coverage;
- documenting pipeline status and generated artifacts.

## Repository Structure

```text
.
|-- app/
|   |-- app.py                 # Shiny dashboard
|   `-- www/styles.css         # Dashboard styling
|-- codes/
|   |-- preprocess.py          # Raw CSV -> processed hourly/station datasets
|   |-- train_forecasts.py     # Rolling backtests and future forecasts
|   `-- run_pipeline.py        # One-command pipeline orchestration
|-- proposal/                  # Project proposal files
|-- report/
|   |-- MDA_Group16_Report.tex
|   `-- MDA_Group16_Report.pdf
|-- DASHBOARD_README.md        # Detailed local run instructions
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

The raw data folder `fietstellingen_csv/` and generated files in `data/processed/` are intentionally ignored by Git because they are large and reproducible.

## Data

Input data comes from the public AWV bicycle count dataset:

- Dataset catalogue: <https://www.vlaanderen.be/datavindplaats/catalogus/fietstellingen-awv>
- Direct CSV files: <https://opendata.apps.mow.vlaanderen.be/fietstellingen/index.html>

The pipeline expects the downloaded raw files in:

```text
fietstellingen_csv/
```

The current local run processes 81 monthly files from August 2019 to April 2026, producing 5,311,003 hourly station records for 150 stations.

## Setup

Create or activate a Python environment, for example through Anaconda, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Pipeline

Full refresh from raw CSV files:

```bash
python codes/run_pipeline.py --start-month 2019-08 --end-month 2026-04 --top-stations 12 --backtest-windows 3
```

If processed Parquet files already exist and only forecasts need to be refreshed:

```bash
python codes/run_pipeline.py --skip-preprocess --top-stations 12 --backtest-windows 3
```

To train forecasts for every eligible station instead of only the busiest 12:

```bash
python codes/run_pipeline.py --skip-preprocess --all-stations --backtest-windows 3
```

The default forecast run trains the busiest 12 eligible stations to keep the demo fast. Monitoring and planning views still use all 150 processed stations.

## Run the Dashboard

```bash
shiny run --host 127.0.0.1 --port 8000 app/app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Dashboard Demo Flow

Use this sequence for a clear presentation:

1. `Overview`: show the map, total rows, total cyclists, and top traffic stations.
2. `Station Detail`: select a busy station and explain daily, hourly, and weekday patterns.
3. `Forecast`: show 24-hour and 168-hour forecasts, rolling backtest metrics, and best model selection.
4. `Planning Insights`: explain the priority score and planning recommendations.
5. `ML Engineering`: show that the dashboard is backed by a reproducible pipeline and saved artifacts.

See `DEMO_SCRIPT.md` for a more detailed speaking script.

## Forecasting

The current forecasting pipeline compares:

- `seasonal_naive`;
- `hist_gradient_boosting`;
- `prophet`.

Model evaluation uses 3 rolling backtest windows with a 168-hour test horizon. The best model is selected per station based on average WAPE.

## Docker

Docker verification note: this repository includes a Dockerfile, but Docker CLI was not available on the current machine during the latest project check (`docker: command not found`). The commands below should be run after Docker Desktop or another Docker runtime is installed.

Build the dashboard image:

```bash
docker build -t mda-bike-dashboard .
```

Run the dashboard with local processed data mounted into the container:

```bash
docker run --rm -p 8000:8000 -v "$PWD/data/processed:/app/data/processed" mda-bike-dashboard
```

## Report

The project report is available at:

```text
report/MDA_Group16_Report.pdf
```

It describes the data pipeline, model evaluation, dashboard design, planning logic, limitations, and how the implementation responds to the proposal feedback.
