# retrieval, Chapter 3.5.
# cosine similarity between query image vector and db vectors to find nearest neighbour

from pathlib import Path

import numpy as np

from scripts.aggregation.detect_features import load_models, detect
from scripts.aggregation.aggregation_pyramid import build_vector
from scripts.aggregation.aggregation_filter import is_usable, apply_confidence_threshold
from scripts.aggregation.aggregation_normalization import load_parameters
from scripts.localization.localization_db import db_builder, normalize_vectors, RefDB


#models contains pretrained at 0 and finetuned at 1
def vector_query_im(image_path: Path, models: tuple, factor: np.ndarray) -> np.ndarray | None:
    raw_vector = build_vector(detect(Path(image_path), models[0], models[1]))
    if not is_usable(apply_confidence_threshold(raw_vector)):
        return None
    return normalize_vectors(raw_vector, factor)

def similarities(database: RefDB, query: np.ndarray) -> np.ndarray:
    return database.vectors @ query # @ for matrix multiplication

# tested implementation of k-NN inspired matching, where the best matches are all taken, chapter 5.5 page 71
# bad matching at 200km range and all in all rather confusing results.
# test in main does not regularly put an image into its own cluster. so only k = 1 considered for now
def closest_matches(database:RefDB, query: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    similarity_array = similarities(database, query)
    # [::-1] to change direction, take first k
    ranking_similarity_array = np.argsort(similarity_array)[::-1][:k]
    return ranking_similarity_array, similarity_array[ranking_similarity_array]


def nearest_neighbour(image_path: Path, database: RefDB, models: tuple, factor: np.ndarray) -> tuple[float, float] | None:
    query = vector_query_im(image_path, models, factor)
    if query is None:
        print(f"[INFO] no match found for {image_path}")
        return None

    rows, scores = closest_matches(database, query, k = 1)
    lon, lat = database.coordinates[rows[0]]
    print(f"[INFO] Closest Match for {Path(image_path).name}: {database.im_ids[rows[0]]}")
    print(f"Match at {lat: .6f}, {lon: .6f}, similarity: {scores[0]:.4f}")
    return float(lon), float(lat)


if __name__ == "__main__":
    print("RETRIEVAL")
    database = db_builder()
    # models = load_models()
    # factor = load_parameters()

    # test for n (here: 29) most similar imaged to one (the very first) image in db, is it in the same cluster?
    rows, scores = closest_matches(database, database.vectors[0], k= 30)
    for row, score in zip(rows, scores):
        print(f"[INFO] rank {score:.4f} - {database.im_ids[row]} - {database.coordinates[row]}")