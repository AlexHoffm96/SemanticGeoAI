#!/usr/bin/env bash
# Local setup: Python dependencies, containing PyTorch with ROCm for AMD GPUs
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
