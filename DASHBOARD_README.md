# Bicycle Traffic Monitoring Dashboard

This project is a small Modern Data Analytics system for AWV bicycle count data. It turns raw monthly CSV files into processed hourly station data, trains forecasting models, stores evaluation artifacts, and serves the results in a Shiny dashboard.

## 1. Open the project

Open this folder in VS Code:

```bash
/Users/suzeren/Documents/MDA2026_Group16_KULeuven
```

Select the Python interpreter from your Anaconda environment.

## 2. Install packages if needed

```bash
pip install -r requirements.txt
```

## 3. Run the data pipeline

Recommended one-command refresh:

```bash
python codes/run_pipeline.py --start-month 2019-08 --end-month 2026-04 --top-stations 12 --backtest-windows 3
```

If the processed parquet files already exist and you only want to retrain forecasts:

```bash
python codes/run_pipeline.py --skip-preprocess --top-stations 12 --backtest-windows 3
```

The default forecast command trains the busiest 12 stations to keep the demo fast. Monitoring and planning pages still use all processed stations. To train forecasts for every eligible station, run:

```bash
python codes/run_pipeline.py --skip-preprocess --all-stations --backtest-windows 3
```

Model metrics are averaged over rolling backtest windows. This is more robust than evaluating only the final week once.

The pipeline writes a run manifest to:

```text
data/processed/pipeline_orchestration.json
```

## 4. Run steps manually if needed

Build the processed hourly data from all downloaded monthly CSV files:

```bash
python codes/preprocess.py --start-month 2019-08 --end-month 2026-04
```

## 5. Train forecast outputs

Train the seasonal naive baseline, histogram gradient boosting, and Prophet for the busiest stations:

```bash
python codes/train_forecasts.py --top-stations 12 --horizon-hours 168 --test-hours 168 --train-days 730 --backtest-windows 3
```

To train forecasts for all eligible stations:

```bash
python codes/train_forecasts.py --all-stations --horizon-hours 168 --test-hours 168 --train-days 730 --backtest-windows 3
```

## 6. Run the dashboard locally

```bash
shiny run --host 127.0.0.1 --port 8000 app/app.py
```

Then open:

```text
http://127.0.0.1:8000
```

## 7. Optional Docker run

Docker verification note: Docker CLI was not available on the current machine during the latest project check (`docker: command not found`). Run the following commands after installing Docker Desktop or another Docker runtime.

Build the image:

```bash
docker build -t mda-bike-dashboard .
```

Run the dashboard with local processed data mounted into the container:

```bash
docker run --rm -p 8000:8000 -v "$PWD/data/processed:/app/data/processed" mda-bike-dashboard
```

## Notes

The raw data folder `fietstellingen_csv/` and generated files in `data/processed/` are ignored by Git. This keeps GitHub clean while still allowing the dashboard to run locally from generated artifacts.

The dashboard is in English. Station names and municipality names come from the original data source, so some location labels remain in Dutch.
