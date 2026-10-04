import os
import configparser
from pathlib import Path

import sys
if sys.version_info < (3, 10):
    raise RuntimeError("Installation of Python >= 3.10 required")


# Config Loader
def load_config(path="setup/config.ini"):
    config = configparser.ConfigParser()
    config.read(path)
    return config
config = load_config("setup/config.ini")

# YOLO Training Config
YOLO_EPOCHS = config.getint("TRAIN_YOLO", "epochs", fallback=30)
YOLO_BATCH = config.getint("TRAIN_YOLO", "batch", fallback=16)
YOLO_IMGSZ = config.getint("TRAIN_YOLO", "imgsz", fallback=640)
YOLO_MODEL = config.get("TRAIN_YOLO", "model", fallback="yolo11n.pt")
YOLO_SEED = config.getint("TRAIN_YOLO", "seed", fallback=42)
YOLO_DEVICE   = config.get("TRAIN_YOLO", "device", fallback="0")
YOLO_WORKERS  = config.getint("TRAIN_YOLO", "workers", fallback=4)
YOLO_PATIENCE = config.getint("TRAIN_YOLO", "patience", fallback=20)
YOLO_CACHE    = config.get("TRAIN_YOLO", "cache", fallback="false")
YOLO_RESUME   = config.getboolean("TRAIN_YOLO", "resume", fallback=False)
YOLO_RUNNAME = config.get("TRAIN_YOLO", "run_name", fallback="yolo_run")
YOLO_RUNOVERWRITE = config.getboolean("TRAIN_YOLO", "run_overwrite", fallback=False)
YOLO_OPTIMIZER = config.get("TRAIN_YOLO", "optimizer", fallback="SGD")
YOLO_LR = config.getfloat("TRAIN_YOLO", "lr_0", fallback=0.01)
YOLO_LRMULTIPLIER = config.getfloat("TRAIN_YOLO", "lr_f", fallback=0.01)
YOLO_MOMENTUM = config.getfloat("TRAIN_YOLO", "momentum", fallback=0.90)
YOLO_WEIGHTDECAY = config.getfloat("TRAIN_YOLO", "weight_decay", fallback=0.0)

# YOLO Result Metrics
YOLO_METRICS = dict(config.items("METRICS_YOLO"))
YOLO_PRECISION = config.get("METRICS_YOLO", "precision", fallback="metrics/precision(B)")
YOLO_RECALL = config.get("METRICS_YOLO", "recall", fallback="metrics/recall(B)")
YOLO_MAP50 = config.get("METRICS_YOLO", "map50", fallback="metrics/map50(B)")
YOLO_MAP50_95 = config.get("METRICS_YOLO", "map50_95", fallback="metrics/map50_95(B)")
YOLO_TRAIN_BOX_LOSS = config.get("METRICS_YOLO", "train_box_loss", fallback="train/train_box_loss")
YOLO_TRAIN_CLS_LOSS = config.get("METRICS_YOLO", "train_cls_loss", fallback="train/train_cls_loss")
YOLO_TRAIN_DFL_LOSS = config.get("METRICS_YOLO", "train_dfl_loss", fallback="train/train_dfl_loss")
YOLO_VAL_BOX_LOSS = config.get("METRICS_YOLO", "val_box_loss", fallback="val/val_box_loss")
YOLO_VAL_CLS_LOSS = config.get("METRICS_YOLO", "val_cls_loss", fallback="val/val_cls_loss")
YOLO_VAL_DFL_LOSS = config.get("METRICS_YOLO", "val_dfl_loss", fallback="test/val_dfl_loss")

# YOLO Conversion Paths
YOLO_RAW = Path(config.get("PATHS_YOLO", "raw"))
YOLO_TRAIN = Path(config.get("PATHS_YOLO", "train"))
YOLO_VAL = Path(config.get("PATHS_YOLO", "val"))
YOLO_PROCESSED = Path(config.get("PATHS_YOLO", "path"))

# YOLO Training Runs Config
YOLO_RUNS_PATH = Path(config.get("PATHS_YOLO", "runs_path"))
YOLO_RUNS_CONFIG = Path(config.get("PATHS_YOLO", "runs_config"))
YOLO_YAML_PATH = Path(config.get("PATHS_YOLO", "yaml_path"))
YOLO_MODELS_DIR = Path(config.get("PATHS_YOLO", "models_dir"))

# YOLO Categories
VISTA_CATEGORIES = dict(config["CATEGORIES_PANOPTIC"])
YOLO_NAMES = dict(config["CATEGORIES_YOLO_NAMES"])

# API Config
TOKEN = os.getenv(config.get("API_MAPILLARY","token_env", fallback= "MAPILLARY_TOKEN"))
HEADERS = {"Authorization": f"OAuth {TOKEN}"}
URL = config.get("API_MAPILLARY","url")
FIELDS = config.get("API_MAPILLARY","fields")

# Output directories and CSV
OUT = Path(config.get("PATHS_MAPILLARY","out_dir"))
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = OUT / config.get("PATHS_MAPILLARY", "images_subdir")
IMG_DIR.mkdir(parents=True, exist_ok=True)
META_CSV = OUT / config.get("PATHS_MAPILLARY","meta_csv")
FIELDNAMES = [f.strip() for f in config["PATHS_MAPILLARY"]["fieldnames"].split(",")]

# Bounding Box
BBOX = (
    config.getfloat("BBOX_MAPILLARY", "min_lon"),
    config.getfloat("BBOX_MAPILLARY", "min_lat"),
    config.getfloat("BBOX_MAPILLARY", "max_lon"),
    config.getfloat("BBOX_MAPILLARY", "max_lat"),
)

# Tiling Parameters, dividing bbox into cells
# (tiles in old versions, cells more according to literature in my thesis, they mean practically the same)
TILING_ROWS = config.getint("TILING_MAPILLARY", "rows", fallback=3)
TILING_COLS = config.getint("TILING_MAPILLARY", "cols", fallback=3)
REQUEST_BOX_SIZE = config.getfloat("TILING_MAPILLARY", "request_box_size", fallback=0.05)
MAX_REQUESTS_PER_CELL = config.getint("TILING_MAPILLARY", "max_probes_per_cell", fallback=12)
CELL_IMAGE_LIMIT = config.getint("TILING_MAPILLARY", "cell_image_limit", fallback=100)
MAX_TOTAL = config.getint("TILING_MAPILLARY", "max_total", fallback=1000)
SLEEP_SECONDS = config.getfloat("TILING_MAPILLARY", "sleep_seconds", fallback=0.2)
TIMEOUT = config.getint("TILING_MAPILLARY", "timeout", fallback=20)

#Testset Parameters
RINGS_KM = tuple(int(km) for km in config.get("TESTSET_MAPILLARY", "rings_km", fallback = "30, 100").split(","))
PER_CENTRE = config.getint("TESTSET_MAPILLARY", "per_centre", fallback=2)
MIN_DISTANCE_KM = config.getint("TESTSET_MAPILLARY", "min_distance_km", fallback=10)


def defaults_from_ini() -> dict[str, object]:
    return dict(model=YOLO_MODEL, epochs=YOLO_EPOCHS, batch=YOLO_BATCH, imgsz=YOLO_IMGSZ, seed=YOLO_SEED,
                device=YOLO_DEVICE, workers=YOLO_WORKERS, patience=YOLO_PATIENCE, cache=YOLO_CACHE, resume=YOLO_RESUME,
                run_name=YOLO_RUNNAME, run_overwrite=YOLO_RUNOVERWRITE, optimizer=YOLO_OPTIMIZER, learning_rate=YOLO_LR,
                lrf=YOLO_LRMULTIPLIER, momentum=YOLO_MOMENTUM, weight_decay=YOLO_WEIGHTDECAY)

# Feature Aggregation
PYRAMID_LEVELS = tuple(
    int(level) for level in config.get("AGGREGATION", "pyramid_levels", fallback="1,2,4").split(",")
)
N_CELLS = sum(level * level for level in PYRAMID_LEVELS)
CONFIDENCE_THRESHOLD = config.getfloat("AGGREGATION", "confidence_threshold", fallback=0.25)
DETECTION_FLOOR = config.getfloat("AGGREGATION", "detection_floor", fallback=0.1)
assert DETECTION_FLOOR <= CONFIDENCE_THRESHOLD, "confidence_threshold must not be below detection_floor"

# Feature Classes
# Sorting Classes of both models so they are in correct sequence
N_CLASSES_FINETUNED = len(YOLO_NAMES)
YOLO_NAMES_PRETRAINED = dict(config["CATEGORIES_YOLO_PRETRAINED"])
N_CLASSES = N_CLASSES_FINETUNED + len(YOLO_NAMES_PRETRAINED)
N_DIMENSIONS = N_CLASSES * N_CELLS

PRETRAINED_ID_TO_INDEX = {
    int(coco_id): (N_CLASSES_FINETUNED + position) # breaks if only coco_id instead of int(coco_id)
    for position, coco_id in enumerate(YOLO_NAMES_PRETRAINED)
}

# counting up list: vector index -> class name, both models
# Modell 2 (fine-tuned): vector indices 0 - 22, Modell 1 (pretrained): 23 - 32
FEATURE_CLASS_NAMES = ([YOLO_NAMES[str(i)] for i in range(len(YOLO_NAMES))] +
[YOLO_NAMES_PRETRAINED[coco_id] for coco_id in YOLO_NAMES_PRETRAINED])

# Aggregation Paths
AGG_MODEL_PRETRAINED = Path(config.get("PATHS_AGGREGATION", "model_pretrained"))
AGG_MODEL_FINETUNED = Path(config.get("PATHS_AGGREGATION", "model_finetuned"))
AGG_VECTORS_DIR = Path(config.get("PATHS_AGGREGATION", "vectors_dir"))
# one base name for both files: .npy for the db, .csv for control purposes
AGG_RAW_VECTORS = (AGG_VECTORS_DIR / config.get("PATHS_AGGREGATION", "raw_vectors")).with_suffix(".npy")
AGG_RAW_VECTORS_CSV = AGG_RAW_VECTORS.with_suffix(".csv")
AGG_VECTOR_IDS = AGG_VECTORS_DIR / config.get("PATHS_AGGREGATION", "vector_ids")
AGG_NORM_PARAMS = AGG_VECTORS_DIR / config.get("PATHS_AGGREGATION", "norm_params")

# Evaluation Paths
TEST_OUT = Path(config.get("PATHS_EVALUATION", "test_dir"))
TEST_IMG_DIR = TEST_OUT / config.get("PATHS_EVALUATION", "test_images_subdir")
TEST_META_CSV = TEST_OUT / config.get("PATHS_EVALUATION", "test_meta_csv")
EVAL_DIR = Path(config.get("PATHS_EVALUATION", "evaluation_dir"))

# Evaluation Parameters
THRESHOLDS_KM = tuple(int(km) for km in config.get("EVALUATION", "thresholds_km").split(","))
DIMENSION_GROUPS = tuple(tuple(int(dim_boundary) for dim_boundary in group.split("-"))
                         for group in config.get("EVALUATION", "dimension_groups").split(","))
CLUSTER_RANGE_KM = config.getint("EVALUATION", "cluster_range_km")
