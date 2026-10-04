# Transformation, Chapter 3.3.1.
# both models are used, the results are combined
# Each image results in a List of tuples
# [(class_index, confidence, cx, cy), ...]
# Other information is not stored, changes towards more semantic information need to be implemented here first

from pathlib import Path

from ultralytics import YOLO

from scripts.utils.config_loader import (
    AGG_MODEL_PRETRAINED, AGG_MODEL_FINETUNED, PRETRAINED_ID_TO_INDEX, DETECTION_FLOOR,
)

def load_models() -> tuple[YOLO, YOLO]:
    return YOLO(str(AGG_MODEL_PRETRAINED)), YOLO(str(AGG_MODEL_FINETUNED))


# https://docs.ultralytics.com/tasks/detect
def _boxes(model: YOLO, image_path: Path) -> list[tuple[int, float, float, float]]:
    # all instance-boxes in image entry of list output
    results = model(str(image_path), conf=DETECTION_FLOOR, verbose=False)
    boxes = []
    for result in results:
        xywhn   = result.boxes.xywhn
        classes = result.boxes.cls.int()
        confidence   = result.boxes.conf
        # .item() takes values from graphics driver (values are pytorch tensors), error without
        for i in range(len(classes)):
            boxes.append((classes[i].item(), confidence[i].item(), xywhn[i, 0].item(), xywhn[i, 1].item()))
    return boxes


def detect(image_path: Path, model_pretrained: YOLO,
           model_finetuned: YOLO) -> list[tuple[int, float, float, float]]:
    # Modell 2: finetuned classes, output list of boxes of class 0 - 22
    detections = _boxes(model_finetuned, image_path)
    # Modell 1: iterate through coco-classes, only defined dimensions have a value and coco-ID
    # add on top of detections for aggregated output of both models
    for class_id, confidence, cx, cy in _boxes(model_pretrained, image_path):
        class_index = PRETRAINED_ID_TO_INDEX.get(class_id)
        if class_index is not None:
            detections.append((class_index, confidence, cx, cy))
    return detections
