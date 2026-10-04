# Learning curve plotter for comparing multiple runs
# YOLO does not support this out of the box, extra plotter needed

from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from scripts.utils.config_loader import YOLO_RUNS_PATH

OUT_DIR = Path(YOLO_RUNS_PATH) / "figures_curves"
METRICS = ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]
# True: best run per optimizer, False: all runs in one plot (Anhang A)
GROUPED = False


# every folder with a results.csv is a run, the optimizer comes from YOLO's args.yaml
def load_runs(runs_path: Path) -> list[dict]:
    runs = []
    for run_dir in sorted(runs_path.iterdir()):
        if not (run_dir / "results.csv").exists():
            continue
        with (run_dir / "args.yaml").open("r") as f:
            optimizer = yaml.safe_load(f)["optimizer"]
        runs.append({"name": run_dir.name,
                     "optimizer": optimizer,
                     "results": pd.read_csv(run_dir / "results.csv")})
    return runs


# best run of each optimizer. last epoch as comparison, since they are far from overfitting with small epochs
def best_per_optimizer(runs: list[dict], metric: str) -> list[dict]:
    best = {}
    for run in runs:
        last_value = run["results"][metric].iloc[-1]
        optimizer = run["optimizer"]
        if optimizer not in best or last_value > best[optimizer]["results"][metric].iloc[-1]:
            best[optimizer] = run
    return [best[optimizer] for optimizer in sorted(best)]


# save one line per run
def plot(runs: list[dict], metric: str, title: str, outfile: Path, show_optimizer: bool) -> None:
    fig, ax = plt.subplots()
    for run in runs:
        label = f"{run['optimizer']} | {run['name']}" if show_optimizer else run["name"]
        ax.plot(run["results"]["epoch"], run["results"][metric], label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)
    print(f"[DONE] {outfile}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs(Path(YOLO_RUNS_PATH))
    for metric in METRICS:
        name = metric.replace("/", "_")   # no "/" in file names
        if GROUPED:
            plot(best_per_optimizer(runs, metric), metric, f"{metric}: best per optimizer",
                 OUT_DIR / f"compare_best_optimizer_{name}.png", show_optimizer=True)
        else:
            plot(runs, metric, f"{metric}: all runs",
                 OUT_DIR / f"overlay_all_{name}.png", show_optimizer=False)
