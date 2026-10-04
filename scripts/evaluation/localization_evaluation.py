# Localization evaluation, Chapter 3.6
# plus the fuctions needed for the analysis of the reference set, Chapter 4 and 5

import csv
import numpy as np
# numpy does not contain a direct implementation of spearman rank correlation correlation
# scipy library is used: https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.spearmanr.html
from scipy import stats

from scripts.aggregation.detect_features import load_models
from scripts.aggregation.aggregation_normalization import load_parameters
from scripts.localization.localization_db import db_builder
from scripts.localization.localization_retrieval import vector_query_im, closest_matches
from scripts.utils.config_loader import (TEST_META_CSV, TEST_IMG_DIR, AGG_RAW_VECTORS, CONFIDENCE_THRESHOLD,
                                         N_CLASSES, N_CELLS, FEATURE_CLASS_NAMES, EVAL_DIR,
                                         THRESHOLDS_KM, DIMENSION_GROUPS, CLUSTER_RANGE_KM)
from scripts.utils.write_csv import write_csv
from scripts.utils.localization_math import haversine_km

# Chapter 5.1
# comparing all input images with all db-vectors
# result: one row per test image with all metrics (localization_per_image.csv),
# rank correlations/class as class_rhos (localization_classes.csv),
# compared mean and meadian distanz per image (localization_mean_distance.csv)
def evaluate_images(database, models, weights)\
        -> tuple[list[dict], dict[int, list[float]], dict[int, int], np.ndarray]:
    with TEST_META_CSV.open(encoding="utf-8") as f:
        test_images = list(csv.DictReader(f))
    rows, skipped, image_distances = [], {}, []
    class_rhos = {k: [] for k in range(N_CLASSES)}

    for index, image in enumerate(test_images, start=1):
        # setting up query image vectors for 30 and 100 km rings
        ring = int(image["ring_km"])
        path = TEST_IMG_DIR / f"{image['id']}.jpg"
        query = vector_query_im(path, models, weights) if path.exists() else None
        if query is None:   # if no file or no detection above the threshold, no estimate
            skipped[ring] = skipped.get(ring, 0) + 1
            continue

        match, similarity = closest_matches(database, query, k=1)
        match, similarity = int(match[0]), float(similarity[0])
        # print(f"[TEST] matching images: {match}, similarity: {similarity} \n") # match as row number in db

        # distances of query image coordinates to reference image with haversine equation, page 50
        distances = haversine_km(float(image["lon"]), float(image["lat"]),
                                 database.coordinates[:, 0], database.coordinates[:, 1])
        median_distance = float(np.median(distances))
        image_distances.append((median_distance, float(distances.mean())))
        # print (f"[TEST] distances: {distances} \n median distances {median_distance} \n image distances {image_distances}")

        row = {"id":            image["id"],
               "ring_km":       ring,
               "error_km":      round(float(distances[match]), 1),
               "random_km":     round(median_distance, 1),
               "best_km":       round(float(distances.min()), 1),
               "similarity":    round(similarity, 4),
               "occupied_dims": int((query > 0).sum())}

        # chapter 4.2.2, page 60
        # what percentage of reference pictures is closer than the thresholds?
        # -> this represents the random localization
        for threshold in THRESHOLDS_KM:
            row[f"random_share_{threshold}"] = round(float((distances < threshold).mean()), 5)

        # spearman rank correlation between similarity and distance over all reference images
        # https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.stats.spearmanr.html, no numpy functionality for spearman
        row["similarity_distance_rho"] = round(float(stats.spearmanr(database.vectors @ query, distances).statistic), 3)
        rows.append(row)

        # main test for hypothesis:
        # spearman correlation between class-similarities and distances
        # all 21 pyramid cells summed up into a single class
        class_contribution = (database.vectors * query).reshape(-1, N_CLASSES, N_CELLS).sum(axis=2)
        #print(f"[TEST] class contribution shape: {class_contribution.shape}")

        # 21_pyr_cell_sum = query.reshape(N_CLASSES, N_CELLS).sum(axis=1)
        for k in np.flatnonzero(query.reshape(N_CLASSES, N_CELLS).sum(axis=1)):
            # are there at least 2 contributions to measure?
            if len(np.unique(class_contribution[:, k])) > 1:
                class_rhos[k].append(float(stats.spearmanr(class_contribution[:, k], distances).statistic))

        if index % 100 == 0:
            print(f"[INFO] {index} of {len(test_images)} test images")

    return rows, class_rhos, skipped, np.array(image_distances)


# allocating images towards either 30km or 100 km distance testset, chapter 4.2.2
def testset_allocation(rows, skipped) -> list[dict]:
    summary = []
    for ring in sorted({row["ring_km"] for row in rows}):
        part    = [row for row in rows if row["ring_km"] == ring]
        error   = np.array([row["error_km"] for row in part])
        random  = np.array([row["random_km"] for row in part])
        best    = np.array([row["best_km"] for row in part])
        rho     = np.array([row["similarity_distance_rho"] for row in part])
        summary.append({"ring_km": ring,
                        "images": len(part),
                        "without_estimate": skipped.get(ring, 0),
                        "median_error_km": round(float(np.median(error)), 1),
                        "median_random_km": round(float(np.median(random)), 1),
                        "median_best_km": round(float(np.median(best)), 1),
                        "closer_than_random_pct": round(100 * float(np.mean(error < random)), 1),
                        "median_error_reduction_pct": round(100 * (1 - float(np.median(error) / np.median(random))), 1),
                        "median_similarity_distance_rho": round(float(np.median(rho)), 3),
                        "similarity_distance_rho_negative_pct": round(100 * float(np.mean(rho < 0)), 1)})
    return summary


# how big is the percentage of query images that is within a threshold, compared against random assignment
# used as summary for 30km /100 km rings
def threshold_summary(rows)-> list[dict]:
    summary = []
    for ring in sorted({row["ring_km"] for row in rows}):
        part = [row for row in rows if row["ring_km"] == ring]
        error = np.array([row["error_km"] for row in part])
        for threshold in THRESHOLDS_KM:
            summary.append({"ring_km": ring,
                            "threshold_km": threshold,
                            "method_pct": round(100 * float(np.mean(error < threshold)), 1),
                            "random_pct": round(100 * float(np.mean([row[f"random_share_{threshold}"] for row in part])), 1)})
    return summary


# distance and threshold comparisons, per testset and per occupied dimensions (4.2.2 page 60 and 61)
def dimension_summary(rows) -> list[dict]:
    summary = []
    for ring in sorted({row["ring_km"] for row in rows}):
        for low, high in DIMENSION_GROUPS:
            part = [row for row in rows if row["ring_km"] == ring and low <= row["occupied_dims"] <= high]
            if not part:
                continue
            error = np.array([row["error_km"] for row in part])
            random = np.array([row["random_km"] for row in part])
            rho = np.array([row["similarity_distance_rho"] for row in part])
            group = {"ring_km": ring,
                     "dims_from": low,
                     "dims_to": high,
                     "images": len(part),
                     "median_error_km": round(float(np.median(error)), 1),
                     "median_random_km": round(float(np.median(random)), 1),
                     "closer_than_random_pct": round(100 * float(np.mean(error < random)), 1),
                     "median_similarity_distance_rho": round(float(np.median(rho)), 3),
                     "similarity_distance_rho_negative_pct": round(100 * float(np.mean(rho < 0)), 1)}
            for threshold in THRESHOLDS_KM:
                group[f"method_pct_{threshold}"] = round(100 * float(np.mean(error < threshold)), 1)
                group[f"random_pct_{threshold}"] = round(100 * float(np.mean([row[f"random_share_{threshold}"] for row in part])), 1)
            summary.append(group)
    return summary


# summary of class correlations, taking the median over all test images
def class_correlation_summary(class_rhos) -> list[dict]:
    summary = []
    for k, name in enumerate(FEATURE_CLASS_NAMES):
        rho = np.array(class_rhos[k])
        summary.append({"class": name,
                        "queries": len(rho),
                        "median_rho": round(float(np.median(rho)), 3) if len(rho) else "",
                        "negative_pct": round(100 * float(np.mean(rho < 0)), 1) if len(rho) else ""})
    return summary


# comparing different methods of the randomised localization value, Chapter 5.1
# different combinations of median and mean (per image, and over all images)
def mean_distance_summary(rows, image_distances):
    summary = []
    for ring in sorted({row["ring_km"] for row in rows}):
        part = [index for index, row in enumerate(rows) if row["ring_km"] == ring]

        medians     = image_distances[part, 0]  # median distance per image
        means       = image_distances[part, 1]  # mean distance per image
        variants    = [("median", "median", float(np.median(medians))),
                        ("median", "mean", float(np.mean(medians))),
                        ("mean", "median", float(np.median(means))),
                        ("mean", "mean", float(np.mean(means)))]

        used_variant = variants[0][2]   # chapter 3.6
        for image_name, set_name, distance in variants:
            summary.append({"ring_km": ring,
                            "images": len(part),
                            "per_image": image_name, "per_test_set": set_name,
                            "distance_km": round(distance, 1),
                            "deviation_pct": round(100 * (distance / used_variant - 1), 1)})
    return summary


# Chapter 4.2.1
# confidence values of all classes, summarized
def class_confidence_summary():
    vectors = np.load(AGG_RAW_VECTORS).reshape(-1, N_CLASSES, N_CELLS)
    summary = []
    for k, name in enumerate(FEATURE_CLASS_NAMES):
        entries = vectors[:, k, :][vectors[:, k, :] >= CONFIDENCE_THRESHOLD]
        summary.append({"class": name,
                        "entries": entries.size,
                        "mean_confidence": round(float(entries.mean()), 3) if entries.size else ""})
    return summary


# how close are images to other clusters?
def cluster_distance(coordinates):
    lons, lats = coordinates[:, 0], coordinates[:, 1]
    cluster = list(range(len(coordinates)))   #  an image is its own cluster at the start

    def root(image):
        while cluster[image] != image:
            image = cluster[image]
        return image

    nearest = []
    for i in range(len(coordinates)):
        distances = haversine_km(lons[i], lats[i], lons, lats)
        for j in np.flatnonzero(distances < CLUSTER_RANGE_KM):   # all images within range are merged
            cluster[root(j)] = root(i)
        nearest.append(distances[distances >= CLUSTER_RANGE_KM].min())
    clusters = len({root(i) for i in range(len(coordinates))})

    print(f"[INFO] {len(coordinates)} images in {clusters} clusters")
    print(f"[INFO] {len(coordinates)} images, mean distance to nearest different cluster is {np.mean(nearest):.2f} km")


# spearman correlation to csv, Chapter 4.2
def spearman_csv_writer(x, y, label, out_csv):
    rho, p = stats.spearmanr(x, y)
    print(f"[INFO] spearman {label}: rho {rho:.3f}, p {p:.3g}, {len(x)} test images")
    write_csv([{"images": len(x), "rho": round(float(rho), 3), "p": float(f"{p:.3g}")}], out_csv)


def evaluate(out_dir=EVAL_DIR):
    database = db_builder()
    rows, class_rhos, skipped, image_distances = evaluate_images(database, load_models(), load_parameters())
    errors = [row["error_km"] for row in rows]

    write_csv(rows, out_dir / "localization_per_image.csv")
    write_csv(testset_allocation(rows, skipped), out_dir / "localization_distance.csv")
    write_csv(threshold_summary(rows), out_dir / "localization_thresholds.csv")
    write_csv(dimension_summary(rows), out_dir / "localization_per_dim.csv")
    write_csv(class_correlation_summary(class_rhos), out_dir / "localization_classes.csv")
    write_csv(mean_distance_summary(rows, image_distances), out_dir / "localization_mean_distance.csv")
    write_csv(class_confidence_summary(), out_dir / "localization_class_confidence.csv")

    spearman_csv_writer([row["similarity"] for row in rows], errors,
             "similarity to error", out_dir / "localization_spearman.csv")
    spearman_csv_writer([row["occupied_dims"] for row in rows], errors,
             "occupied dimensions to distance error", out_dir / "localization_dim_correlation.csv")

    cluster_distance(database.coordinates)


if __name__ == "__main__":
    evaluate()
