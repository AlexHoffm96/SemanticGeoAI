# Train a YOLO model with configurable hyperparameters and run directory
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

from scripts.utils.yolomodel_loader import resolve_model_file
from scripts.utils.config_loader import (
    YOLO_MODEL, YOLO_EPOCHS, YOLO_BATCH, YOLO_IMGSZ, YOLO_DEVICE, YOLO_WORKERS,
    YOLO_PATIENCE, YOLO_CACHE, YOLO_RESUME, YOLO_SEED, YOLO_RUNNAME, YOLO_RUNOVERWRITE,
    YOLO_OPTIMIZER, YOLO_LR, YOLO_LRMULTIPLIER, YOLO_WEIGHTDECAY, YOLO_MOMENTUM,
    YOLO_YAML_PATH, YOLO_RUNS_PATH, YOLO_MODELS_DIR
)

# Directory for pretrained model weights
models_dir = YOLO_MODELS_DIR
models_dir.mkdir(parents=True, exist_ok=True)

def _to_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "on"}

def log_device():
    if torch.cuda.is_available():
        print("PyTorch backend: CUDA/ROCm")
        print(f"torch.version.hip: {getattr(torch.version, 'hip', None)}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch backend: CPU only")

# Start YOLO training based on config.ini and yolo.yaml
def train_yolo(
        data_yaml: str | Path = "scripts/yolo/yolo.yaml",
        run_dir: str | Path = "runs/detect",
        **overrides,  # allows key-value overrides like epochs=10, imgsz=1024
):
    # Load training config
    data_yaml = Path(data_yaml)
    with data_yaml.open("r", encoding="utf-8") as f:
        yaml_config = yaml.safe_load(f)

    # Resolve and load the YOLO model
    resume      = _to_bool(overrides.get("resume", YOLO_RESUME))

    model_file = overrides.get("model", YOLO_MODEL)
    if resume:
        run_name_early = overrides.get("run_name", YOLO_RUNNAME)
        model_file = str(Path(run_dir) / run_name_early / "weights" / "last.pt")
        print(f"[RESUME] Using checkpoint: {model_file}")
    elif Path(model_file).suffix == ".pt":
        model_file = resolve_model_file(model_file, models_dir)
    print(f"Resolved model path: {model_file}  | exists={Path(model_file).exists()}")

    # Defaults from config.ini, optionally overridden via **overrides
    epochs      = int(overrides.get("epochs", YOLO_EPOCHS))
    batch       = int(overrides.get("batch",  YOLO_BATCH))
    imgsz       = int(overrides.get("imgsz",  YOLO_IMGSZ))
    seed        = int(overrides.get("seed", YOLO_SEED))
    device      = overrides.get("device", YOLO_DEVICE)
    workers     = int(overrides.get("workers", YOLO_WORKERS))
    patience    = int(overrides.get("patience", YOLO_PATIENCE))
    _cache_raw  = str(overrides.get("cache", YOLO_CACHE)).strip().lower()
    cache       = "disk" if _cache_raw == "disk" else _to_bool(_cache_raw)
    run_name    = overrides.get("run_name", YOLO_RUNNAME)
    run_overwrite = _to_bool(overrides.get("run_overwrite", YOLO_RUNOVERWRITE))
    optimizer   = overrides.get("optimizer", YOLO_OPTIMIZER)
    lr          = float(overrides.get("lr_0", YOLO_LR))
    lr_multiplier = float(overrides.get("lr_f", YOLO_LRMULTIPLIER))
    momentum    = float(overrides.get("momentum", YOLO_MOMENTUM))
    weight_decay = float(overrides.get("weight_decay", YOLO_WEIGHTDECAY))

    print("[YOLO TRAIN]")
    print(f" model: {model_file}")
    print(f" epochs: {epochs}  batch: {batch}  imgsz: {imgsz}")
    print(f" learning_rate: {lr}")
    print(f" device: {device}  workers: {workers}  patience: {patience}")
    print(f" data: {data_yaml}  (train={yaml_config['train']}  val={yaml_config['val']})")
    print(f" cache: {cache}  resume: {resume}")
    print(f" run: {Path(run_dir)/run_name}  exist_ok={run_overwrite}")


    model = YOLO(model_file)
    results = model.train(
        # Base
        data        = str(data_yaml),
        epochs      = epochs,    # 100
        batch       = batch,     # 16
        imgsz       = imgsz,     # 640
        device      = None if device in ("", "None", "none") else device,  # "0"
        workers     = workers,   # 6
        seed        = seed,      # 42
        project     = run_dir,
        name        = run_name,
        exist_ok    = run_overwrite,
        patience    = patience,  # 20
        cache       = cache,     # True
        resume      = resume,    # False
        verbose     = True,
        plots       = True,
        save        = True,
        val         = True,
        deterministic= False,

        # Optimizer & LR
        optimizer   = optimizer,
        lr0         = lr,
        lrf         = lr_multiplier,
        momentum    = momentum,
        weight_decay= weight_decay,

        # Other
        cos_lr      = False,
        warmup_epochs= 3.0,
        warmup_momentum= 0.80,
        warmup_bias_lr= 0.10,
        amp         = False,
        nbs         = 64,
        save_period = -1,
        auto_augment= "randaugment",
        hsv_h       = 0.015,
        hsv_s       = 0.70,
        hsv_v       = 0.40,
        degrees     = 0.0,
        translate   = 0.10,
        scale       = 0.50,
        shear       = 0.0,
        perspective = 0.0,
        fliplr      = 0.50,
        flipud      = 0.00,
        mosaic      = 1.0,
        close_mosaic= 10,
        mixup       = 0.0,
        copy_paste  = 0.0,
        copy_paste_mode= "flip",
        erasing     = 0.4,
        cutmix      = 0.0,
        dropout     = 0.0,
        bgr         = 0.0,
        box         = 7.5,
        cls         = 0.5,
        dfl         = 1.5,
        kobj        = 1.0,
        pose        = 12.0,
        rect        = False,
        multi_scale = False,
        single_cls  = False,
        classes     = None,
        fraction    = 1.0,
        pretrained  = True,
        freeze      = None,
        conf        = None,
        iou         = 0.70,
        max_det     = 300,
        nms         = False,
        agnostic_nms= False,
        format      = "torchscript",
        simplify    = True,
        show        = False,
        show_boxes  = True,
        show_conf   = True,
        show_labels = True,
        save_conf   = False,
        save_txt    = False,
        save_json   = False,
    )

    # Save results and return save_dir
    save_dir = model.trainer.save_dir
    print(f"[DONE] Yolo-Trainingresults in: {save_dir}")

    return results


if __name__ == "__main__":
    print("Hardware log:")
    log_device()
    results = train_yolo(YOLO_YAML_PATH, YOLO_RUNS_PATH)
    print(results)
    print(results.metrics)
    print(results.results_dict)