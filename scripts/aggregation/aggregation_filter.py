# Filtering, Chapter 3.3.1 and Chapter 3.5.

# Detection: entries below the confidence threshold are set to zero.

# Image: images without a usable detection are discarded
# Runs before the normalization


import numpy as np
from scripts.utils.config_loader import CONFIDENCE_THRESHOLD

# values below confidence threshhold = 0, chapter 3.3.3 page 45
# https://numpy.org/devdocs/reference/generated/numpy.where.html
def apply_confidence_threshold(raw_vectors: np.ndarray,
                               threshold: float = CONFIDENCE_THRESHOLD) -> np.ndarray:
    return np.where(raw_vectors >= threshold, raw_vectors, 0.0)

# Single image, false if every detection is below the confidence threshold
def is_usable(raw_vector: np.ndarray) -> bool:
    return bool(np.any(raw_vector > 0.0))

# Reference/database set
def filter_vectors(raw_vectors: np.ndarray, image_ids: np.ndarray | list[str]) -> tuple[np.ndarray, np.ndarray]:
    keep = np.any(raw_vectors > 0.0, axis=1)
    kept_ids = np.asarray(image_ids)[keep]
    dropped = len(image_ids) - len(kept_ids)
    print(f"[INFO] image filter: {len(kept_ids)} of {len(image_ids)} images kept, {dropped} dropped")
    return raw_vectors[keep], kept_ids

# images that pass the filter
def coverage(raw_vectors: np.ndarray) -> float:
    if len(raw_vectors) == 0:
        return 0.0
    return float(np.any(raw_vectors > 0.0, axis=1).mean())
