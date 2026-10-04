# Test images around the reference images, Chapter 3.6

import csv
import math
import random
import time
import numpy as np

from scripts.localization.localization_db import db_builder
from scripts.mapillary.mapillary_fetch_metadata import request_images, extract_lon_lat
from scripts.mapillary.mapillary_load_image import download_images
from scripts.utils.localization_math import MEAN_EARTH_RADIUS, nearest_reference_km
from scripts.utils.config_loader import (REQUEST_BOX_SIZE, SLEEP_SECONDS, FIELDNAMES,
                                         TEST_IMG_DIR, TEST_META_CSV, RINGS_KM, PER_CENTRE, MIN_DISTANCE_KM)


### SEPTEMBER 2026 UPDATE: there is now a mapillary python sdk: https://leafmap.org/maplibre/mapillary/
### Simplifying everything; entire /mapillary folder is now obsolete

KM_PER_DEGREE = 2 * math.pi * MEAN_EARTH_RADIUS / 360

# got from the cells centrepoint to a specific direction alongside a cirle with 30/100 KM
def offset_box(lon, lat, km, degree_directions, size=REQUEST_BOX_SIZE) -> tuple[float, float, float, float]:
    cy = lat + km * math.cos(math.radians(degree_directions)) / KM_PER_DEGREE # results in degree offset towards lat
    cx = lon + km * math.sin(math.radians(degree_directions)) / (KM_PER_DEGREE * math.cos(math.radians(lat))) #adjusting for smaller longitudes towards the north
    return cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2


def fetch_testset(rings_km=RINGS_KM) -> None:
    references = db_builder().coordinates
    # 0.1 degree cells as centres, unique removes duplicates, so only single reference centre points remain
    centres = np.unique(np.round(references, 1), axis=0)
    TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()

    with TEST_META_CSV.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=FIELDNAMES + ["ring_km", "distance_km"]) # additional fieldnames
        writer.writeheader()
        # first 30km, then 100km ring
        for ring in rings_km:
            # counting through all centre points from reference data
            for index, (lon, lat) in enumerate(centres, start=1):
                degrees_directions = list(range(0, 360, 45))
                random.Random(f"{lon},{lat},{ring}").shuffle(degrees_directions)
                found = 0
                for degrees_direction in degrees_directions:
                    if found == PER_CENTRE:
                        break
                    images = request_images(offset_box(lon, lat, ring, degrees_direction), limit=1)
                    time.sleep(SLEEP_SECONDS)
                    if not images or images[0].get("id") in seen:
                        continue
                    image = images[0]
                    ilon, ilat = extract_lon_lat(image)
                    if ilon is None or ilat is None:
                        continue
                    distance = nearest_reference_km(ilon, ilat, references)
                    if distance < MIN_DISTANCE_KM:
                        continue
                    writer.writerow({"id": image["id"], "lon": ilon, "lat": ilat,
                                     "captured_at": image.get("captured_at"),
                                     "compass_angle": image.get("compass_angle"),
                                     "thumb_1024_url": image.get("thumb_1024_url"),
                                     "ring_km": ring, "distance_km": round(distance, 1)})
                    seen.add(image["id"])
                    found += 1
                metadata_file.flush()
                print(f"[INFO] ring {ring} km: centre {index} of {len(centres)}, {found} images")

    download_images(meta_csv=TEST_META_CSV, img_dir=TEST_IMG_DIR)


if __name__ == "__main__":
    fetch_testset()
