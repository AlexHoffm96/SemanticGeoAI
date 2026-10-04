import csv
import requests
import time
from pathlib import Path
from scripts.utils.config_loader import IMG_DIR, META_CSV, TIMEOUT, MAX_TOTAL, SLEEP_SECONDS

### SEPTEMBER 2026 UPDATE: there is now a mapillary python sdk: https://leafmap.org/maplibre/mapillary/
### Simplifying everything; entire /mapillary folder is now obsolete

# Loading metadata from CSV
def load_metadata(meta_csv: Path = META_CSV):
    if not meta_csv.exists():
        raise FileNotFoundError(f"CSV missing: {meta_csv}")
    with meta_csv.open("r", encoding="utf-8") as f_metadata:
        return list(csv.DictReader(f_metadata))

# Image download function
# Separated from the metadata fetch, since some long download times tended to me not robust enough otherwise
def download_images(max_images=MAX_TOTAL, meta_csv=META_CSV, img_dir=IMG_DIR):
    rows = load_metadata(meta_csv)
    saved = 0

    for row in rows:
        if saved >= max_images:
            break

        # metadata sometimes lacks id or url
        imgid = row.get("id")
        url = row.get("thumb_1024_url")
        if not imgid or not url:
            continue

        path = img_dir / f"{imgid}.jpg"
        if path.exists() and path.stat().st_size > 0:
            continue

        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            path.write_bytes(r.content)

            saved += 1
            print(f"[INFO] Downloaded {imgid} -> {path.name}")

            time.sleep(SLEEP_SECONDS)

        except requests.HTTPError as e:
            print(f"[WARN] HTTP {e.response.status_code} for {imgid}, skip for now")
            time.sleep(1.0)
        except Exception as e:
            print(f"[WARN] Download failed for {imgid}: {e}")

    print(f"[DONE] Downloaded new images: {saved}")

if __name__ == "__main__":
    download_images()