# building the aggregated vector, Chapter 3.3 and Chapter 3.7

from pathlib import Path

import numpy as np

from scripts.aggregation.detect_features import load_models, detect
from scripts.aggregation.aggregation_pyramid import build_vector, index_cell_and_dimension
from scripts.aggregation.aggregation_filter import apply_confidence_threshold, filter_vectors, coverage
from scripts.aggregation.aggregation_normalization import calculate_parameters, save_parameters
from scripts.utils.config_loader import (
    IMG_DIR, AGG_VECTORS_DIR, AGG_RAW_VECTORS, AGG_RAW_VECTORS_CSV, AGG_VECTOR_IDS,
    CONFIDENCE_THRESHOLD, DETECTION_FLOOR, N_CELLS, FEATURE_CLASS_NAMES,
)
from scripts.utils.write_csv import write_csv


# detect and aggregate every image
def build_raw_vectors(image_dir: Path = IMG_DIR) -> tuple[np.ndarray, list[str]]:
    images = sorted(image_dir.glob("*.jpg"))
    model_pretrained, model_finetuned = load_models()
    vectors = []
    image_ids = [] # using filenames without suffix as ids
    for position, image_path in enumerate(images, start=1):
        try:
            vectors.append(build_vector(detect(image_path, model_pretrained, model_finetuned)))
            image_ids.append(image_path.stem)
        except Exception as e:
            print(f"[WARN] skipped {image_path.name}: {e}")
        if position % 100 == 0:
            # it takes a while, so there's a small [INFO] print after 100 images for control purposes
            print(f"[INFO] {position} of {len(images)} images processed")
    return np.array(vectors, dtype=np.float32), image_ids


def store_raw_vectors(raw_vectors: np.ndarray, image_ids: list[str]) -> None:
    # .npy and id list for the db, rows of both files belong together
    AGG_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(AGG_RAW_VECTORS, raw_vectors)
    print(f"[DONE] {raw_vectors.shape} raw vectors written to {AGG_RAW_VECTORS}")
    write_csv([{"id": image_id} for image_id in image_ids], AGG_VECTOR_IDS, fieldnames=["id"])

    '''
    # readable .csv for control purposes, not read by the pipeline
    labels = [index_cell_and_dimension(dimension) for dimension in range(raw_vectors.shape[1])]
    rows = []
    for image_id, vector in zip(image_ids, raw_vectors):
        for dimension in np.flatnonzero(vector):    # positions of all entries > 0
            dimension = int(dimension)
            rows.append({"id": image_id,
                         "dimension": dimension,
                         "k": dimension // N_CELLS,
                         "class": FEATURE_CLASS_NAMES[dimension // N_CELLS],
                         "z": dimension % N_CELLS,
                         "label": labels[dimension],
                         "value": f"{vector[dimension]:.4f}"})
    write_csv(rows, AGG_RAW_VECTORS_CSV)
    '''


def run():
    print("AGGREGATION CONTROLLER START")

    print(f"\n [INFO] Detect features and build initial raw vectors (floor: {DETECTION_FLOOR})")
    raw_vectors, image_ids = build_raw_vectors()
    store_raw_vectors(raw_vectors, image_ids)

    print(f"\n [INFO] Apply confidence threshold {CONFIDENCE_THRESHOLD} and image filter")
    thresholded = apply_confidence_threshold(raw_vectors)
    print(f"[INFO] coverage {coverage(thresholded):.3f}")
    thresholded, image_ids = filter_vectors(thresholded, image_ids)

    print("\n [INFO] Estimate normalization parameters")
    weights, counts = calculate_parameters(thresholded)
    save_parameters(weights, counts)

    print("\nAGGREGATION CONTROLLER FINISHED")


if __name__ == "__main__":
    run()
