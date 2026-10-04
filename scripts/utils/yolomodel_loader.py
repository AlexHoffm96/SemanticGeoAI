# Downloading YOLO Model if not available locally (at the directory the model_dir points towards)
from pathlib import Path
from ultralytics import SETTINGS
SETTINGS.update({'weights_dir': 'data/models'})  # arguments taken from print(settings)
from ultralytics.utils.downloads import safe_download # option of retrying, unzipping and deleting downloaded file: https://docs.ultralytics.com/reference/utils/downloads#ultralytics.utils.downloads.safe_download


# model download
def resolve_model_file(name: str, model_dir: Path) -> str:
    directory = Path(name)
    if directory.exists():
        return str(directory.resolve())
    path = model_dir / name
    if not path.exists():
        print(f"[INFO] Downloading model: {name} to: {path}")
        model_dir.mkdir(parents=True, exist_ok=True)
        safe_download(
            url=f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}", #v8.3 as release tag contains yolo11
            dir=model_dir,
            file=name,
        )
    return str(path.resolve())
