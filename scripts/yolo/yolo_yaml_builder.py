from pathlib import Path
import yaml

# all config loaded centrally from config_loader
from scripts.utils.config_loader import (
    YOLO_RAW, YOLO_TRAIN, YOLO_VAL, YOLO_PROCESSED,
    VISTA_CATEGORIES, YOLO_NAMES
)

YOLO_YAML = Path("scripts/yolo/yolo.yaml")

def build_yolo_yaml():
    # write keys as ints (0,1,2,...) in YAML
    panoptic = {int(k): v for k, v in VISTA_CATEGORIES.items()}
    names    = {int(k): v for k, v in YOLO_NAMES.items()}

    yolo_yaml = {
        "raw": str(YOLO_RAW),
        "train": str(YOLO_TRAIN),
        "val": str(YOLO_VAL),
        "path": str(YOLO_PROCESSED),
        "panoptic_categories": panoptic,
        "names": names,
    }

    YOLO_YAML.parent.mkdir(parents=True, exist_ok=True)
    with YOLO_YAML.open("w", encoding="utf-8") as f:
        yaml.dump(yolo_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"[DONE] Wrote {YOLO_YAML.resolve()}")

if __name__ == "__main__":
    build_yolo_yaml()