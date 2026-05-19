from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "fietstellingen_csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed"

COUNT_COLUMNS = ["site_id", "richting", "type", "van", "tot", "aantal"]
SITE_COLUMNS = [
    "site_id",
    "site_nr",
    "long",
    "lat",
    "naam",
    "domein",
    "wegnr",
    "district",
    "gemeente",
    "interval",
    "datum_van",
]


def month_from_path(path: Path) -> str | None:
    match = re.search(r"data-(\d{4}-\d{2})\.csv$", path.name)
    return match.group(1) if match else None


def list_count_files(raw_dir: Path, start_month: str, end_month: str | None) -> list[Path]:
    files = []
    for path in sorted(raw_dir.glob("data-*.csv")):
        month = month_from_path(path)
        if month is None:
            continue
        if month < start_month:
            continue
        if end_month is not None and month > end_month:
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No monthly count files found in {raw_dir} for {start_month} to {end_month or 'latest'}."
        )
    return files


def read_sites(raw_dir: Path) -> pd.DataFrame:
    sites_path = raw_dir / "sites.csv"
    sites = pd.read_csv(sites_path, names=SITE_COLUMNS)
    sites["datum_van"] = pd.to_datetime(sites["datum_van"], errors="coerce")
    return sites


def aggregate_file(path: Path, chunksize: int) -> pd.DataFrame:
    monthly_parts: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        names=COUNT_COLUMNS,
        dtype={
            "site_id": "int32",
            "richting": "category",
            "type": "category",
            "aantal": "float32",
        },
        chunksize=chunksize,
    )

    for chunk in reader:
        cyclists = chunk.loc[chunk["type"] == "FIETSERS", ["site_id", "van", "aantal"]].copy()
        if cyclists.empty:
            continue
        cyclists["aantal"] = cyclists["aantal"].fillna(0)
        cyclists["hour"] = pd.to_datetime(cyclists["van"], errors="coerce").dt.floor("h")
        cyclists = cyclists.dropna(subset=["hour"])
        hourly = (
            cyclists.groupby(["site_id", "hour"], as_index=False, observed=True)["aantal"]
            .sum()
            .rename(columns={"aantal": "count"})
        )
        monthly_parts.append(hourly)

    if not monthly_parts:
        return pd.DataFrame(columns=["site_id", "hour", "count"])

    monthly = pd.concat(monthly_parts, ignore_index=True)
    return monthly.groupby(["site_id", "hour"], as_index=False, observed=True)["count"].sum()


def build_station_summary(hourly: pd.DataFrame, sites: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly = hourly.copy()
    hourly["date"] = hourly["hour"].dt.date
    hourly["weekday"] = hourly["hour"].dt.dayofweek
    hourly["hour_of_day"] = hourly["hour"].dt.hour

    daily = hourly.groupby(["site_id", "date"], as_index=False, observed=True)["count"].sum()
    daily["date"] = pd.to_datetime(daily["date"])

    max_date = daily["date"].max()
    recent_start = max_date - pd.Timedelta(days=89)
    previous_start = recent_start - pd.Timedelta(days=90)

    recent = daily.loc[daily["date"] >= recent_start]
    previous = daily.loc[(daily["date"] >= previous_start) & (daily["date"] < recent_start)]

    recent_avg = recent.groupby("site_id", observed=True)["count"].mean().rename("recent_avg_daily")
    previous_avg = previous.groupby("site_id", observed=True)["count"].mean().rename("previous_avg_daily")

    coverage = hourly.groupby("site_id", observed=True).agg(
        first_hour=("hour", "min"),
        last_hour=("hour", "max"),
        observed_hours=("hour", "nunique"),
        total_count=("count", "sum"),
        active_hours=("count", lambda s: int((s > 0).sum())),
        peak_hour_count=("count", "max"),
        p95_hour_count=("count", lambda s: float(s.quantile(0.95))),
    )
    coverage["expected_hours"] = (
        (coverage["last_hour"] - coverage["first_hour"]).dt.total_seconds() / 3600 + 1
    ).clip(lower=1)
    coverage["coverage_rate"] = coverage["observed_hours"] / coverage["expected_hours"]

    active_days = daily.groupby("site_id", observed=True)["date"].nunique().rename("active_days")
    avg_daily = daily.groupby("site_id", observed=True)["count"].mean().rename("avg_daily_count")

    summary = (
        coverage.join(active_days)
        .join(avg_daily)
        .join(recent_avg)
        .join(previous_avg)
        .reset_index()
    )
    summary["growth_pct_recent_vs_previous"] = (
        (summary["recent_avg_daily"] - summary["previous_avg_daily"])
        / summary["previous_avg_daily"].replace(0, pd.NA)
        * 100
    )

    sites_keep = sites[
        ["site_id", "site_nr", "long", "lat", "naam", "gemeente", "district", "datum_van"]
    ].copy()
    summary = summary.merge(sites_keep, on="site_id", how="left")
    summary["station_label"] = (
        summary["naam"].fillna("Unknown station")
        + " | "
        + summary["gemeente"].fillna("Unknown municipality")
        + " (#"
        + summary["site_id"].astype(str)
        + ")"
    )

    quality = summary[
        [
            "site_id",
            "station_label",
            "first_hour",
            "last_hour",
            "observed_hours",
            "expected_hours",
            "coverage_rate",
            "active_hours",
        ]
    ].copy()
    quality["coverage_rate"] = quality["coverage_rate"].round(3)

    return summary, quality


def preprocess(raw_dir: Path, out_dir: Path, start_month: str, end_month: str | None, chunksize: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_count_files(raw_dir, start_month, end_month)
    print(f"Processing {len(files)} monthly files from {month_from_path(files[0])} to {month_from_path(files[-1])}")

    hourly_parts: list[pd.DataFrame] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index:02d}/{len(files):02d}] {path.name}")
        hourly_parts.append(aggregate_file(path, chunksize))

    hourly = pd.concat(hourly_parts, ignore_index=True)
    hourly = hourly.groupby(["site_id", "hour"], as_index=False, observed=True)["count"].sum()
    hourly["count"] = hourly["count"].astype("int32")

    sites = read_sites(raw_dir)
    summary, quality = build_station_summary(hourly, sites)

    hourly_path = out_dir / "hourly_counts.parquet"
    summary_path = out_dir / "station_summary.parquet"
    quality_path = out_dir / "data_quality_summary.csv"
    run_summary_path = out_dir / "pipeline_run_summary.json"

    hourly.to_parquet(hourly_path, index=False)
    summary.to_parquet(summary_path, index=False)
    quality.to_csv(quality_path, index=False)
    run_summary_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "raw_dir": str(raw_dir),
                "start_month": month_from_path(files[0]),
                "end_month": month_from_path(files[-1]),
                "monthly_files": len(files),
                "hourly_rows": int(len(hourly)),
                "stations_with_counts": int(summary["site_id"].nunique()),
                "outputs": {
                    "hourly_counts": str(hourly_path),
                    "station_summary": str(summary_path),
                    "data_quality_summary": str(quality_path),
                },
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote {len(hourly):,} hourly rows to {hourly_path}")
    print(f"Wrote {len(summary):,} station rows to {summary_path}")
    print(f"Wrote data quality summary to {quality_path}")
    print(f"Wrote pipeline run summary to {run_summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess AWV bicycle counts for the Shiny dashboard.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-month", default="2019-08", help="First monthly file to process, YYYY-MM.")
    parser.add_argument("--end-month", default=None, help="Last monthly file to process, YYYY-MM.")
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess(args.raw_dir, args.out_dir, args.start_month, args.end_month, args.chunksize)
