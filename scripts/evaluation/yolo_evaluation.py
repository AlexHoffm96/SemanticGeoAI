# Per-class metrics of YOLO model
# evaluation based on vistas validation set

from pathlib import Path
import numpy as np
from ultralytics import YOLO

from scripts.utils.config_loader import (AGG_MODEL_FINETUNED, YOLO_YAML_PATH,
                                         YOLO_IMGSZ, YOLO_BATCH, YOLO_DEVICE, EVAL_DIR)
from scripts.utils.write_csv import write_csv

OUT_CSV_PER_CLASS = EVAL_DIR / "per_class.csv"
OUT_CSV_CONFUSION = EVAL_DIR / "confusion.csv"



def evaluate(finetuned_model_weights: Path = AGG_MODEL_FINETUNED,
             out_csv_per_class: Path = OUT_CSV_PER_CLASS,
             out_csv_confusion: Path = OUT_CSV_CONFUSION) -> None:
    print(f"[INFO] Evaluating {finetuned_model_weights}")

    # loading finetuned model and let it run over entire validation set
    # project= points towards data/processed/evaluation/val, results are saved there
    results = YOLO(str(finetuned_model_weights)).val(data=str(YOLO_YAML_PATH), split="val", imgsz=YOLO_IMGSZ,
                                                     batch=YOLO_BATCH, device=YOLO_DEVICE,
                                                     project=str(out_csv_per_class.parent), name="val", exist_ok=True, plots=True)
    rows = []
    # ap_class_index contains all class-IDs of val set
    # iterated as tuple to get position since some class_ids can be empty
    for position, class_id in enumerate(results.box.ap_class_index):
        class_id = int(class_id)
        precision, recall, map50, map50_95 = results.box.class_result(position)
        rows.append({"class": results.names[class_id],
                     "images": int(results.nt_per_image[class_id]),
                     "instances": int(results.nt_per_class[class_id]),
                     "precision": round(float(precision), 4),
                     "recall": round(float(recall), 4),
                     "map50": round(float(map50), 4),
                     "map50_95": round(float(map50_95), 4)})


    #pearson correlation between number of instances and detection quality map50-95, see chapter 4.2.3,  p. 63
    #https://numpy.org/devdocs/reference/generated/numpy.corrcoef.html
    #https://numpy.org/devdocs/reference/generated/numpy.log.html
    array_instances = np.log([row["instances"] for row in rows])
    array_map50_95 = np.array([row["map50_95"] for row in rows])
    correlation = np.corrcoef(array_instances, array_map50_95)[0, 1]

    print(f"[INFO] correlation log(instances) to map50_95: {correlation:.3f}")
    write_csv(rows, out_csv_per_class)


    # confusion matrix
    labels = list(results.names.values()) + ["background"]
    matrix = results.confusion_matrix.matrix
    matrix = matrix / np.maximum(matrix.sum(axis=0, keepdims=True), 1)  # share of the true class
    confusion = []
    for i, label in enumerate(labels):
        row = {"predicted": label}
        row.update({labels[j]: round(float(matrix[i, j]), 3) for j in range(len(labels))})
        confusion.append(row)
    write_csv(confusion, out_csv_confusion)


if __name__ == "__main__":
    evaluate()
