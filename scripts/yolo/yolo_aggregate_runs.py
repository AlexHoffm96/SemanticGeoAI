import csv
import time
from scripts.utils.write_csv import write_csv
from pathlib import Path

from scripts.utils.read_csv import read_csv
from scripts.utils.config_loader import (
    # paths, targets, and metrics for evaluation
    YOLO_METRICS, YOLO_RUNS_PATH, YOLO_RUNS_CONFIG, defaults_from_ini,
)


ARGS_KEEP = [
    "model", "optimizer", "learning_rate", "lrf", "momentum", "weight_decay",
    "epochs", "batch", "imgsz", "seed", "device", "workers", "patience", "cache", "resume"
]

def read_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def last_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    return rows[-1] if rows else None

# safe_float avoids type errors in the rec_line calculation
def safe_float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

# Collect all hyperparameters:


def merge_hparams(config_row: dict[str, object]) -> dict[str, object]:
    # Load defaults from config.ini
    base = defaults_from_ini().copy()
    # Apply overrides from the CSV row
    base.update({key: value for key, value in config_row.items() if value is not None})
    if config_row.get("lr_0") is not None:
        base["learning_rate"] = config_row["lr_0"]

    # Return the merged result
    out = {k: base.get(k) for k in ARGS_KEEP}
    out["run_name"] = base.get("run_name")
    return out

# Collect data for a single run from its last results row
def collect_one_run(run_dir: Path, hparams: dict[str, object]) -> dict[str, object] | None:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print(f"No results.csv found: {results_csv}")
        return None
    rows = read_results_csv(results_csv)
    last = last_row(rows)
    if last is None:
        print(f"[SKIP] results file is empty: {results_csv}")
        return None

    # Record timestamp, directory, and name for this run
    rec_line = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime((run_dir / "results.csv").stat().st_mtime)),
        "run_dir": str(run_dir),
        "run_name": hparams.get("run_name") or run_dir.name,
    }

    # Map final metrics from the last epoch row
    for key, col in YOLO_METRICS.items():
        rec_line[key] = safe_float(last.get(col))

    # F1 score: harmonic mean of precision and recall
    # see https://docs.ultralytics.com/guides/yolo-performance-metrics/#object-detection-metrics
    precision, recall = safe_float(rec_line.get("precision")), safe_float(rec_line.get("recall"))
    rec_line["f1"] = (2 * precision * recall / (precision + recall)) if\
        (precision and recall and (precision + recall) > 0) else None

    # Sum total training time
    if rows and "time" in rows[0]:
        total = 0.0
        for row in rows:
            v = safe_float(row.get("time"))
            if v is not None:
                total += v
        rec_line["time_total_s"] = total
    else:
        rec_line["time_total_s"] = None

    # Best and last checkpoints
    weights = run_dir / "weights"
    rec_line["best_pt"] = str((weights / "best.pt").resolve()) if (weights / "best.pt").exists() else ""
    rec_line["last_pt"] = str((weights / "last.pt").resolve()) if (weights / "last.pt").exists() else ""

    # Append hyperparameters
    rec_line.update({param: hparams.get(param) for param in ARGS_KEEP})
    return rec_line


# Aggregate all runs listed in YOLO_RUNS_CONFIG from YOLO_RUNS_PATH
# into a metrics.csv
def aggregate_runs() -> Path:
    # load planned runs
    targets: list[dict] = read_csv(YOLO_RUNS_CONFIG)
    if not targets:
        raise RuntimeError(f"No Training targets found in {YOLO_RUNS_CONFIG}.")

    records: list[dict] = []
    for config in targets:
        # merge hyperparameters from defaults and CSV overrides
        hparams = merge_hparams(config)
        run_name = (hparams.get("run_name") or config.get("run_name") or "").strip()
        if not run_name:
            print("[WARNING] run name missing, skipping line.")
            continue

        # collect run folder and results
        run_dir = Path(YOLO_RUNS_PATH) / run_name
        rec_line = collect_one_run(run_dir, hparams)
        if rec_line:
            records.append(rec_line)
        else:
            print(f"[SKIP] no results found for {run_name} run.")

    if not records:
        raise RuntimeError(f"No results found for {YOLO_RUNS_CONFIG}.")

    # determine output path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(YOLO_RUNS_PATH)
    out_csv = out_dir / f"metrics_{timestamp}.csv"

    # fixed column order for the output
    preferred_cols = [
        "timestamp", "run_name", "run_dir",
        # hyperparameters
        "model", "optimizer", "learning_rate", "lrf", "momentum", "weight_decay",
        "epochs", "batch", "imgsz", "seed", "device", "workers", "patience", "cache", "resume",
        # metrics
        "precision", "recall", "f1", "map50", "map50_95",
        "train_box_loss", "train_cls_loss", "train_dfl_loss",
        "val_box_loss", "val_cls_loss", "val_dfl_loss",
        # misc
        "time_total_s", "best_pt", "last_pt",
    ]

    # write CSV
    write_csv(records, out_csv, fieldnames=preferred_cols)
    print(f"[OK] aggregated {out_csv}")
    print(f"[INFO] collected {len(records)} runs.")
    return out_csv


if __name__ == "__main__":
    aggregate_runs()