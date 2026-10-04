# database for retrieval setup, Chapter 3.4. and 3.7.
# so far, everything is set up within .csv and numpy structures, deployment of an actual database (sqlite or similar) to be implemented later

import csv
from pathlib import Path
from typing import NamedTuple
import numpy as np

from scripts.aggregation.aggregation_filter import filter_vectors, apply_confidence_threshold
from scripts.aggregation.aggregation_normalization import load_parameters, apply_normalization
from scripts.utils.config_loader import AGG_RAW_VECTORS, AGG_VECTOR_IDS, META_CSV, CONFIDENCE_THRESHOLD

class RefDB (NamedTuple):
    vectors: np.ndarray
    im_ids: np.ndarray
    coordinates: np.ndarray

# https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
# axis = -1: if matrix, then only last axis is considered
# enables vector and matrix input
# ord = 2 for L2-Normalization
def l2_norm(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors, axis= -1, ord = 2, keepdims = True)

# apply normalization to vectors
def normalize_vectors(raw_vectors:np.ndarray, weights: np.ndarray) -> np.ndarray:
    threshold_vector = apply_confidence_threshold(raw_vectors)
    weight_vector = apply_normalization(threshold_vector, weights)
    normalized_vectors = (weight_vector/l2_norm(weight_vector)).astype(np.float32)
    return normalized_vectors

# only if lon and lat are present, the image id gets a coordinate
def load_geotags_from_csv(meta_csv: Path = META_CSV) -> dict[str, tuple[float, float]]:
    geotags = {}
    with Path(meta_csv).open("r", encoding="utf-8") as csv_file_metadata:
        for row in csv.DictReader(csv_file_metadata):
            if row["lon"] and row["lat"]:
                geotags[row["id"]] = float(row["lon"]), float(row["lat"])
    print(f"[INFO] {len(geotags)} loaded geotags from {meta_csv}")
    return geotags


# only the imagevectors, where lon and lat are present, are kept (compare to load_geotags_from_csv)
def remove_vectors_without_geotag(vectors: np.ndarray, image_ids: np.ndarray,
                      geotags: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_tagged_ids = []
    row = 0
    for image_id in image_ids:
        if image_id in geotags:
            keep_tagged_ids.append(row)
        row += 1

    kept_ids = image_ids[keep_tagged_ids]
    geo_coordinates = np.array([geotags[image_id] for image_id in kept_ids])
    print(f"[INFO] Removed vectors without geotags: {len(kept_ids)} / {len(image_ids)} kept")
    return vectors[keep_tagged_ids], kept_ids, geo_coordinates


def db_builder() -> RefDB:
    raw_vectors = np.load(AGG_RAW_VECTORS)
    # remove "id" as first entry, write in with splitlines, remove empty spaces
    # target format: im_ids = ["3292343", "3203494", ....]
    # im_ids = [line.strip() for line in AGG_VECTOR_IDS.read_text(encoding="utf-8").splitlines()[1:]]

    with AGG_VECTOR_IDS.open("r", encoding="utf-8") as f:
         im_ids = np.array([row["id"] for row in csv.DictReader(f)], dtype = str)

    vectors, im_ids = filter_vectors(apply_confidence_threshold(raw_vectors), im_ids)
    vectors, im_ids, coordinates = remove_vectors_without_geotag(vectors, im_ids, load_geotags_from_csv())

    vectors = normalize_vectors(vectors, load_parameters())
    return RefDB(vectors, im_ids, coordinates)


if __name__ == "__main__":
    print(f"BUILDING REFERENCE DATABASE")
    print(f" Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    database = db_builder()
