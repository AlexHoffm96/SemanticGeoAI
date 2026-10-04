# Spatial pyramid aggregation, chapter 3.3.2.

# N_CLASSES x N_CELLS
# content of cell is the maximum confidence per class in cell

import numpy as np

from scripts.utils.config_loader import (
    PYRAMID_LEVELS, N_CELLS, N_CLASSES, FEATURE_CLASS_NAMES,
)


# Cell indexing: in which cell is a point in the image?
# Chapter 3.3.2, page 44
def cell_indices(cx: float, cy: float) -> list[int]:
    #Return cell indices for their respective pyramid levels
    indices = []
    offset = 0
    for level in PYRAMID_LEVELS:
        column  = min(int(cx * level), level - 1)
        row     = min(int(cy * level), level - 1)
        indices.append(offset + row * level + column)
        offset  += level * level
    return indices

# Aggregation operator, aggregate detections into (N_CLASSES, N_CELLS) table, only keep max
def build_max_table(detections: list[tuple[int, float, float, float]]) -> np.ndarray:
    table = np.zeros((N_CLASSES, N_CELLS), dtype=np.float32)
    for class_index, cell_confidence, cx, cy in detections:
        for cell in cell_indices(cx, cy):
            # maximum operator: strongest cell_confidence wins
            if cell_confidence > table[class_index, cell]:
                table[class_index, cell] = cell_confidence
    return table

# Chapter 3.3.3., page 45, table -> feature vector (concatenate = verketten), all entries into a single row for 693 x 1 vector
def concatenate(table: np.ndarray) -> np.ndarray:
    return table.reshape(-1)

# readable vector index, chapter 3.3.3, page 45
# deducting class k and cell z from vector index i (inverse of i = k * N_CELLS + z),
# cell z also describes the pyramid level, 0 -> 1, 1-4 = 2, 5 - 20 = 3
def index_cell_and_dimension(dimension: int) -> str:
    k = dimension // N_CELLS
    z = dimension % N_CELLS
    offset = 0
    for level in PYRAMID_LEVELS:
        if z < offset + level * level:
            local = z - offset
            return f"k={k} ({FEATURE_CLASS_NAMES[k]}), z={z} ({level}x{level}[row {local // level}, col {local % level}])"
        offset += level * level
    raise IndexError(f"cell index {z} outside {N_CELLS} cells")

def build_vector(detections: list[tuple[int, float, float, float]]) -> np.ndarray:
    return concatenate(build_max_table(detections))


if __name__ == "__main__":
    print(f"[INFO] pyramid levels {PYRAMID_LEVELS} -> {N_CELLS} cells")
    print(f"[INFO] {N_CLASSES} classes -> {N_CLASSES * N_CELLS} dimensions")
    print(f"[INFO] example dimension 67: {index_cell_and_dimension(67)}")

