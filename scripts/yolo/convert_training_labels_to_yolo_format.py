from pathlib import Path
import yaml, json, shutil

# Configuration from yolo.yaml
with open("scripts/yolo/yolo.yaml") as f:
    config = yaml.safe_load(f)

RAW = Path(config["raw"])
TRAIN_RAW = RAW / config["train"]
VALIDATION_RAW = RAW / config["val"]
PROCESSED = Path(config["path"])

VISTA_CATEGORIES = list(config["panoptic_categories"].values())
YOLO_CATEGORIES = {name: i for i, name in enumerate(VISTA_CATEGORIES)}

# Normalized conversion from Vistas [x, y, w, h] (pixel values) to YOLO [xc, yc, ww, hh]
# xc and yc are center points in YOLO; Vistas uses top-left corner (panoptic segmentation convention)
# YOLO expects normalized width and height values relative to image dimensions
def vistas_to_yolo_bbox(bbox, w_image, h_image):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    # centre point
    xc = (x + w / 2) / w_image
    yc = (y + h / 2) / h_image

    # Normalized width and height: bbox dimension divided by image dimension
    ww = w / w_image
    hh = h / h_image
    if ww <= 0 or hh <= 0:
        return None
    return [xc, yc, ww, hh]

# Convert the full split dataset to YOLO format
def convert_split_to_yolo(split: str):

    # Set paths
    raw_root = RAW / split
    processed_root = PROCESSED / split
    image_out = processed_root / "images"
    label_out = processed_root / "labels"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    # Load panoptic JSON
    # Panoptic JSON maps segments to categories
    panoptic_json = raw_root / "v2.0" / "panoptic" / "panoptic_2020.json"
    with open(panoptic_json, "r", encoding="utf-8") as f_pan:
        panoptic = json.load(f_pan)

    # Mapping: category_id -> name
    categories = {category["id"]: category["name"] for category in panoptic["categories"]}

    # Iterate over images
    for image_info in panoptic["images"]:
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        w_image, h_image = image_info["width"], image_info["height"]

        # Copy image to output path
        source_image = raw_root / "images" / file_name
        destination_image = image_out / file_name
        if not destination_image.exists():
            shutil.copy(source_image, destination_image)

        label_path = label_out / (Path(file_name).stem + ".txt")

        # Find annotation for this image
        annotation = None
        for ann in panoptic["annotations"]:
            if ann["image_id"] == image_id:
                annotation = ann
                break
        if annotation is None:
            continue

        # Open label file and iterate over all segments for this image,
        # look up the Vistas category and check if it maps to a target YOLO class,
        # write the resulting YOLO class ID and converted bounding box to the label file
        with open(label_path, "w", encoding="utf-8") as f_label:
            for segment in annotation["segments_info"]:
                category_id = segment["category_id"]
                if category_id not in categories:
                    continue
                category_name = categories[category_id]

                if category_name not in YOLO_CATEGORIES:
                    continue

                yolo_class = YOLO_CATEGORIES[category_name]
                bbox = segment["bbox"]
                yolo_bbox = vistas_to_yolo_bbox(bbox, w_image, h_image)
                if yolo_bbox is None:
                    continue

                f_label.write(f"{yolo_class} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")


    print(f"[DONE] Split '{split}' -> {image_out}/{label_out}")


if __name__ == "__main__":
    print("Converting training data to YOLO format")
    convert_split_to_yolo("training")
    print("[DONE] Conversion of training data to YOLO format completed")

    print("Converting validation data to YOLO format")
    convert_split_to_yolo("validation")
    print("[DONE] Conversion of validation data to YOLO format completed")