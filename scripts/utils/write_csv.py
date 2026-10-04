import csv
from pathlib import Path

# Write a list of dicts as a CSV file
# keys as header, values into rows
def write_csv(
    dict_list: list[dict],
    out_file: Path,
    fieldnames: list[str] | None = None #fixed/defined column order
) -> None:
    if not dict_list:
        print("[INFO] No list of dictionaries to write")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(dict_list[0].keys())

    with out_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_list)

    print(f"[DONE] Dictionary list written to {out_file}")