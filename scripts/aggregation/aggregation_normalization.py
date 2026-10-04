# dimension normalisation, Kapitel 3.3.3., page 46

# Two functions:
# calculate_parameters once, over the whole reference set, reference path only
# apply_normalization per image on both paths (db and query), using the estimated parameters


import csv
from pathlib import Path

import numpy as np

from scripts.utils.config_loader import AGG_NORM_PARAMS, N_CELLS, FEATURE_CLASS_NAMES
from scripts.utils.write_csv import write_csv
from scripts.aggregation.aggregation_pyramid import index_cell_and_dimension

PARAMETER_FIELDS = ["dimension", "k", "class", "z", "label", "n", "w"]


def calculate_parameters(raw_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # Estimate the weighting factor w and the n counts.

    # printed (simplified) rawvektor as reference:
    # shape: (n_images, 693)
    # [0.  0. ...  0.   0.
    #  0.  0. ...  0.81 0.
    #         ...
    #  0.  0. ...  0.   0.
    #  0.  0. ...  0.   0.]

    # check for detection in vector
    occupied = raw_vectors > 0.0
    # check number of rows in rawvectors for total image count
    n_images = raw_vectors.shape[0]
    # numpy array, count all column entries (axis = 0) to see how many images a dimension is occupied (C.1, page 118)
    image_count_in_dims = occupied.sum(axis=0)

    # formula on page 46
    weights = np.log((1.0 + n_images) / (1.0 + image_count_in_dims)) + 1.0

    print(f"[INFO] parameters calculated from {n_images} reference images")
    print(f"[INFO] {int((image_count_in_dims == 0).sum())} / {len(image_count_in_dims)} dimensions unoccupied in dataset")
    return weights.astype(np.float32), image_count_in_dims.astype(np.int32)


# Application, both paths
# takes one raw vector or a matrix of them
def apply_normalization(raw_vectors: np.ndarray, factor: np.ndarray) -> np.ndarray:
    return raw_vectors * factor


def save_parameters(weights: np.ndarray, counts: np.ndarray, path=AGG_NORM_PARAMS) -> None:
    parameters = [
        {
            "dimension":    dimension,                                  # i = k * N_CELLS + z
            "k":            dimension // N_CELLS,                       # class index
            "class":        FEATURE_CLASS_NAMES[dimension // N_CELLS],
            "z":            dimension % N_CELLS,                        # cell index
            "label":        index_cell_and_dimension(dimension),        # readable label
            "n":            int(counts[dimension]),
            "w":            f"{weights[dimension]:.9g}",
        }
        for dimension in range(len(weights))
    ]
    write_csv(parameters, Path(path), fieldnames=PARAMETER_FIELDS)


def load_parameters(path=AGG_NORM_PARAMS) -> np.ndarray:
    weights = []
    with Path(path).open("r", encoding="utf-8") as csvfile:
        for row in csv.DictReader(csvfile):
            weights.append(float(row["w"]))
    print(f"[INFO] {len(weights)} normalization parameters loaded from {path}")
    return np.array(weights, dtype=np.float32)