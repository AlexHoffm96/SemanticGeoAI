#!/usr/bin/env python3

from pathlib import Path
from scripts.utils.read_csv import read_csv
from scripts.yolo.yolo_modeltrainer import train_yolo
from scripts.utils.config_loader import (
    YOLO_RUNS_PATH, YOLO_RUNS_CONFIG
)


def run_trainings():
    # read CSV
    csv_path = Path(YOLO_RUNS_CONFIG)
    targets = read_csv(csv_path)

    print(f"[INFO] Loading {len(targets)} training targets from {csv_path}")


    for count_row, config in enumerate(targets, 1):
        run_name = config.get("run_name", f"run_{count_row}")
        run_dir = Path(YOLO_RUNS_PATH)
        run_subdir = run_dir / run_name

        print(f"\n=== Start Run {count_row}/{len(targets)} ===")
        print(f" run_dir: {run_subdir}")
        # integrating resume structure: don't go into runs that are already finished,
        # but go into runs if they are meant to be resumed
        resume = str(config.get("resume", "false")).strip().lower() in {"1", "true", "yes"}
        run_overwrite = str(config.get("run_overwrite", "false")).strip().lower() in {"1", "true", "yes"}

        if run_subdir.exists() and not resume and not run_overwrite:
            print(f"[SKIP] {run_name} already exists, skipping.")
            continue

        print(f" params: {config}")

        # start training
        results = train_yolo(
            data_yaml="scripts/yolo/yolo.yaml",
            run_dir=run_dir,
            **config
        )

        # print result metrics
        if hasattr(results, "metrics"):
            print(f"[DONE] {run_name} metrics: {results.metrics}")
        elif hasattr(results, "results_dict"):
            print(f"[DONE] {run_name} results: {results.results_dict}")

    print("\n[DONE] All scheduled training runs completed")


if __name__ == "__main__":
    run_trainings()