#!/usr/bin/env bash
source /media/jrainey/T7/venvs/tf-t7/bin/activate

VENV_SITE=$(python3 - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

export LD_LIBRARY_PATH="$VENV_SITE/nvidia/cudnn/lib:$VENV_SITE/nvidia/cublas/lib:$VENV_SITE/nvidia/cuda_runtime/lib:$VENV_SITE/nvidia/cufft/lib:$VENV_SITE/nvidia/curand/lib:$VENV_SITE/nvidia/cusolver/lib:$VENV_SITE/nvidia/cusparse/lib:$VENV_SITE/nvidia/nccl/lib:$VENV_SITE/nvidia/nvjitlink/lib:$VENV_SITE/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"