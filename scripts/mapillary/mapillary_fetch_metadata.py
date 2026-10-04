import csv
import math
import random
import requests
import time
from pathlib import Path
from scripts.utils.config_loader import (TOKEN, HEADERS, URL, FIELDS, META_CSV,
                                         FIELDNAMES, BBOX, TILING_ROWS, TILING_COLS,
                                         CELL_IMAGE_LIMIT, MAX_TOTAL, SLEEP_SECONDS, TIMEOUT,
                                         REQUEST_BOX_SIZE, MAX_REQUESTS_PER_CELL)

# Create CSV file, write the header row
def ensure_header(meta_csv: Path) -> None:
    if not meta_csv.exists():
        with meta_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

# Load downloaded image IDs to avoid duplicate downloads
# using sets instead of list here for big databases is important!
# already ~2 mins for current amount of ~29k images (for list instead of set)
def load_seen(meta_csv: Path) -> set[str]:
    start_time = time.time()
    seen = set()
    if meta_csv.exists():
        with meta_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    seen.add(row["id"])
    print(f"[INFO] {len(seen)} images loaded in {time.time() - start_time} seconds")
    return seen

# Split bounding box of world into cells for API coverage.
# Divides the bounding box into rows*cols cells
# cells consist of bbox tuples
def split_bbox(bbox: tuple[float, float, float, float], rows=3, cols=3) -> list:
    min_lon, min_lat, max_lon, max_lat = bbox
    dlon_total = max_lon - min_lon
    dlat_total = max_lat - min_lat
    # size of single cell
    dlon = dlon_total / cols
    dlat = dlat_total / rows
    cells = []
    for r in range(rows):   # build individual cells
        for c in range(cols):
            lon0 = min_lon + c * dlon
            lat0 = min_lat + r * dlat
            lon1 = lon0 + dlon
            lat1 = lat0 + dlat
            cells.append((lon0, lat0, lon1, lat1))
    return cells

# Mapillary API only allows 0.01° * 0.01° square requests since somewhere around may 2026
# windows inside a cell since API rejects bboxes above 0.010 square degrees
# solution: request centre point of multiple areas in case one is empty
def small_request_boxes(cell, size=REQUEST_BOX_SIZE, per_side=4) -> list:
    lon0, lat0, lon1, lat1 = cell
    boxes = []
    # build n x n, usually 4x4, cells from single cell, then take the centre point as cx and cy
    for row in range(per_side):
        for col in range(per_side):
            cx = lon0 + (lon1 - lon0) * (2 * col + 1) / (2 * per_side)
            cy = lat0 + (lat1 - lat0) * (2 * row + 1) / (2 * per_side)
            boxes.append((cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2))
    return boxes


# moving request boxes inside the cell, until it finds image coverage
def attempt_request_box(cell, max_requests=MAX_REQUESTS_PER_CELL, size=REQUEST_BOX_SIZE) -> list:
    # calculate cell size, building up n*n grid so there are always enough squares for the input
    # e.g. input 12, int(sqrt(12)-1)+1 = 16 --> enough tiles for 12 requests
    per_side = int(math.sqrt(max_requests - 1)) + 1 if max_requests > 1 else 1
    boxes = small_request_boxes(cell, size=size, per_side=per_side)
    # avoiding distributions towards a specific direction
    random.Random(str(cell)).shuffle(boxes)
    for attempt, box in enumerate(boxes[:max_requests], start=1):
        images = request_images(box)
        if images:
            print(f"[INFO] {len(images)} images after {attempt} of {max_requests} probes")
            return images
    return []


# some areas are returning null. Assumption: image limit is exceeded, API can't respond in time
# solution: size of the request area is halfed until it does not return null
# build after documentation: https://www.mapillary.com/developer/api-documentation/faq?locale=de_DE
def request_images(cell, limit=CELL_IMAGE_LIMIT, attempts=4) -> list:
    for attempt in range(attempts):
        # constructing dict from cells, fields, bbox-tuples, image_limit
        try:
            params = {"fields": FIELDS, "bbox": ",".join(map(str, cell)), "limit": limit}
            r = requests.get(URL, headers=HEADERS, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                try:
                    return r.json().get("data", []) or []
                except ValueError as e:
                    print(f"[ERROR] no valid json for {cell}: {e}")
                    return []
        except requests.RequestException as e:
            print(f"[WARNING] No answer for {cell}: {e}")

        # build smaller window before retrying
        lon0, lat0, lon1, lat1 = cell
        cx = (lon0 + lon1) / 2
        cy = (lat0 + lat1) / 2
        dx = (lon1 - lon0) / 4
        dy = (lat1 - lat0) / 4
        cell = (cx - dx, cy - dy, cx + dx, cy + dy)
    return []


# Extract coordinates
# Read coordinates safely in case some images have no coordinate data
# API-Response in the form of:
# {    "id": "0123456789",
#     "computed_geometry": {"type": "Point", "coordinates": [10.6001, 51.7502]},   || most precise mapillary data, but sometimes missing.
#     "geometry":          {"type": "Point", "coordinates": [10.6, 51.75]},
#     "captured_at":
#     .
#     .
#     .
# }
# more information of response structure under: https://www.mapillary.com/developer/api-documentation?locale=de_DE#image, Entities-Image chapter
def extract_lon_lat(image: dict) -> tuple:
    geom = image.get("computed_geometry") or image.get("geometry") or {}
    coords = geom.get("coordinates", [None, None])  #  KeyError without [None, None]
    lon = coords[0] if len(coords) > 0 else None
    lat = coords[1] if len(coords) > 1 else None
    return lon, lat


# Main API fetch function
def fetch_images() -> None:
    # API-token set in e.g. ~/.profile, needs to be requested from mapillary website
    if not TOKEN:
        raise RuntimeError("MAPILLARY_TOKEN missing")
    ensure_header(META_CSV)
    seen = load_seen(META_CSV)
    saved = 0

    cells = split_bbox(BBOX, TILING_ROWS, TILING_COLS)

    # Write image metadata row by row with appending
    # fieldnames are CSV column names.
    with open(META_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)

        # loop over cells
        for index, cell in enumerate(cells, start=1):
            # moved max images up, so the loop always instantly stops
            if saved >= MAX_TOTAL:
                break
            print(f"[INFO] Requesting Tile {index}/{len(cells)}: {cell}")

            # Tiles must be valid: lon0 < lon1 and lat0 < lat1
            if not (cell[2] > cell[0] and cell[3] > cell[1]):
                print(f"[WARNING] Skipping Tile {index}/{len(cells)}: {cell}")
                continue
            images = attempt_request_box(cell)

            if not images:
                time.sleep(SLEEP_SECONDS)
                continue

            for image in images:
                # case should never activate, but want to be safe
                if saved >= MAX_TOTAL:
                    break
                iid = image.get("id")
                if not iid or iid in seen:
                    continue

                lon, lat = extract_lon_lat(image) #could write None, None right now, maybe change it later
                cap = image.get("captured_at")
                ang = image.get("compass_angle")
                turl = image.get("thumb_1024_url")

                # writing dict to excel, so I dont mess up the order
                writer.writerow({
                    "id": iid,
                    "lon": lon,
                    "lat": lat,
                    "captured_at": cap,
                    "compass_angle": ang,
                    "thumb_1024_url": turl  # 1024px thumbnail; recommended resolution per Mapillary API docs (obsolete, recommendation no longer in docs as of 09/2026 :/)
                })

                seen.add(iid)
                saved += 1

            csv_file.flush()  # write through, so the csv can be watched while running
            time.sleep(SLEEP_SECONDS)  # prevent API from being overwhelmed :)

    print(f"Downloaded imagedata: {saved}")
    print(f"CSV Written: {META_CSV}")



if __name__ == "__main__":
    fetch_images()