import csv
from pathlib import Path

# Read from CSV, return a list of dicts
# Should work for most .csvs that are built in the form of:
# Header as keys (columnnames), rowvalues as values
# So far only used for training runs: Each row represents one training run with its hyperparameters
def read_csv(csv_path: str | Path) -> list[dict]:
    csv_path = Path(csv_path)
    runs = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                #entries can only be numbers for config.csv
                if v is None or v.strip() == "": #random blank spaces in .csv
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v.strip()
            runs.append(parsed)
    return runs

# runs-structure:
#{
#   {"epochs" : 200, "lr0": 0.01; ....}
#   {"epochs" : 200, "lr0": 0.001; ....}
#}