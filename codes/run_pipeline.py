from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "fietstellingen_csv"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def run_step(name: str, command: list[str]) -> dict[str, object]:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    started = time.time()
    result = subprocess.run(command, cwd=ROOT, check=False)
    duration = round(time.time() - started, 2)
    step = {
        "name": name,
        "command": command,
        "exit_code": result.returncode,
        "duration_seconds": duration,
    }
    if result.returncode != 0:
        step["status"] = "failed"
    else:
        step["status"] = "completed"
    return step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MDA bicycle dashboard data pipeline.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--start-month", default="2019-08")
    parser.add_argument("--end-month", default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--top-stations", type=int, default=12)
    parser.add_argument("--all-stations", action="store_true")
    parser.add_argument("--station-ids", default="", help="Optional comma-separated site IDs for forecasting.")
    parser.add_argument("--horizon-hours", type=int, default=168)
    parser.add_argument("--test-hours", type=int, default=168)
    parser.add_argument("--train-days", type=int, default=730)
    parser.add_argument("--backtest-windows", type=int, default=3)
    parser.add_argument("--backtest-step-hours", type=int, default=168)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-forecast", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.processed_dir / "pipeline_orchestration.json"
    manifest: dict[str, object] = {
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "status": "running",
        "steps": [],
    }
    write_manifest(manifest_path, manifest)

    steps: list[tuple[str, list[str]]] = []
    if not args.skip_preprocess:
        preprocess_command = [
            sys.executable,
            "codes/preprocess.py",
            "--raw-dir",
            str(args.raw_dir),
            "--out-dir",
            str(args.processed_dir),
            "--start-month",
            args.start_month,
            "--chunksize",
            str(args.chunksize),
        ]
        if args.end_month:
            preprocess_command.extend(["--end-month", args.end_month])
        steps.append(("Preprocess raw CSV files", preprocess_command))

    if not args.skip_forecast:
        forecast_command = [
            sys.executable,
            "codes/train_forecasts.py",
            "--processed-dir",
            str(args.processed_dir),
            "--top-stations",
            str(args.top_stations),
            "--horizon-hours",
            str(args.horizon_hours),
            "--test-hours",
            str(args.test_hours),
            "--train-days",
            str(args.train_days),
            "--backtest-windows",
            str(args.backtest_windows),
            "--backtest-step-hours",
            str(args.backtest_step_hours),
            "--min-coverage",
            str(args.min_coverage),
        ]
        if args.station_ids:
            forecast_command.extend(["--station-ids", args.station_ids])
        if args.all_stations:
            forecast_command.append("--all-stations")
        steps.append(("Train and evaluate forecasts", forecast_command))

    exit_code = 0
    completed_steps: list[dict[str, object]] = []
    for name, command in steps:
        step = run_step(name, command)
        completed_steps.append(step)
        manifest["steps"] = completed_steps
        if step["status"] == "failed":
            manifest["status"] = "failed"
            exit_code = int(step["exit_code"])
            break
        write_manifest(manifest_path, manifest)

    if not steps:
        manifest["status"] = "skipped"
    elif exit_code == 0:
        manifest["status"] = "completed"
    manifest["finished_at_utc"] = utc_now()
    write_manifest(manifest_path, manifest)

    print(f"\nPipeline {manifest['status']}. Manifest: {manifest_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
