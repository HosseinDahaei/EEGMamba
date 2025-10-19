#!/usr/bin/env bash
set -euo pipefail

# ==============================
# EEGMamba Installer (Linux)
# Python 3.10 + CUDA 12.1 stack
# ==============================

# ---- Editable defaults ----
ENV_NAME="${ENV_NAME:-eegmamba}"
REPO_URL="${REPO_URL:-https://github.com/HosseinDahaei/EEGMamba-updated.git}"
REPO_DIR="${REPO_DIR:-EEGMamba-updated}"
PYTHON_VER="${PYTHON_VER:-3.10}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
# Pinned versions (Py3.10-friendly)
TORCH_VER="${TORCH_VER:-2.5.1+cu121}"
TV_VER="${TV_VER:-0.20.1+cu121}"
TA_VER="${TA_VER:-2.5.1+cu121}"
TRITON_VER="${TRITON_VER:-3.1.0}"
MAMBA_SSM_VER="${MAMBA_SSM_VER:-2.2.6.post3}"
CAUSAL_CONV_TAG="${CAUSAL_CONV_TAG:-v1.5.2}"

# Scientific stack pins compatible with Python 3.10
NUMPY_VER="${NUMPY_VER:-2.2.2}"
SCIPY_VER="${SCIPY_VER:-1.14.1}"
SKLEARN_VER="${SKLEARN_VER:-1.6.1}"
PANDAS_VER="${PANDAS_VER:-2.3.2}"
EINOPS_VER="${EINOPS_VER:-0.8.1}"
RET_VER="${RET_VER:-0.6.3}"   # rotary-embedding-torch
TQDM_VER="${TQDM_VER:-4.66.5}"
MATP_VER="${MATP_VER:-3.9.2}"
SEABORN_VER="${SEABORN_VER:-0.13.2}"
PYYAML_VER="${PYYAML_VER:-6.0.2}"

# ---- Helpers ----
have() { command -v "$1" >/dev/null 2>&1; }

echo "==> EEGMamba installer started"

# 0) Basic checks
if ! have nvidia-smi; then
  echo "!! NVIDIA driver not detected. Install a CUDA 12.1-capable driver first (nvidia-smi should work)."
  exit 1
fi

# 1) Conda (Miniconda/Anaconda) check
if ! have conda; then
  echo "==> Conda not found. Installing Miniconda (silent) ..."
  TMP_MINI="$(mktemp -d)"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$TMP_MINI/mini.sh"
  bash "$TMP_MINI/mini.sh" -b -p "$HOME/miniconda3"
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
  rm -rf "$TMP_MINI"
else
  eval "$(conda shell.bash hook)"
fi

# 2) Create/activate env
if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  echo "==> Conda env '${ENV_NAME}' exists; reusing."
else
  echo "==> Creating conda env '${ENV_NAME}' (Python ${PYTHON_VER})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VER}"
fi
conda activate "${ENV_NAME}"

# 3) Ensure pip/setuptools/wheel up-to-date
python -m pip install -U pip setuptools wheel

# 4) Install PyTorch CUDA 12.1 wheels
echo "==> Installing PyTorch (${TORCH_VER}) + CUDA 12.1 wheels"
python -m pip install \
  "torch==${TORCH_VER}" "torchvision==${TV_VER}" "torchaudio==${TA_VER}" \
  --index-url "${TORCH_INDEX}"

python - <<'PY'
import torch
print("[check] torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available in torch. Check driver and chosen wheel.")
PY

# 5) Core deps
echo "==> Installing core dependencies (Py3.10 friendly pins)"
python -m pip install \
  "mamba-ssm==${MAMBA_SSM_VER}" \
  "einops==${EINOPS_VER}" \
  "rotary-embedding-torch==${RET_VER}" \
  "tqdm==${TQDM_VER}" \
  "numpy==${NUMPY_VER}" \
  "scipy==${SCIPY_VER}" \
  "scikit-learn==${SKLEARN_VER}" \
  "pandas==${PANDAS_VER}" \
  "matplotlib==${MATP_VER}" \
  "seaborn==${SEABORN_VER}" \
  "pyyaml==${PYYAML_VER}"

# 6) Triton aligned with torch 2.5.x
echo "==> Installing triton ${TRITON_VER}"
python -m pip install "triton==${TRITON_VER}"

# 7) Build causal-conv1d against current Torch (no isolated build)
echo "==> Building causal-conv1d (${CAUSAL_CONV_TAG}) from Dao-AILab repo"
# Give JIT/compiles a clean, roomy place
mkdir -p "$PWD/.torch_extensions" "$PWD/.tmp"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
export TMPDIR="$PWD/.tmp"

# Try to set a sane arch automatically (optional)
PY_ARCH=$(python - <<'PY'
import torch
try:
    cc = torch.cuda.get_device_capability()
    print(f"{cc[0]}.{cc[1]}")
except Exception:
    print("")
PY
)
if [ -n "$PY_ARCH" ]; then
  export TORCH_CUDA_ARCH_LIST="$PY_ARCH"
fi

# Clear stale caches before build
rm -rf "$TORCH_EXTENSIONS_DIR" ~/.cache/torch_extensions ~/.triton ~/.cache/pytriton 2>/dev/null || true

set +e
python -m pip install --no-build-isolation --no-cache-dir \
  "git+https://github.com/Dao-AILab/causal-conv1d@${CAUSAL_CONV_TAG}"
CC_STATUS=$?
set -e
if [ $CC_STATUS -ne 0 ]; then
  echo "!! causal-conv1d build failed (likely toolchain/space). You can run without fused kernels:"
  echo "   export MAMBA_SSM_DISABLE_TRITON=1"
  echo "   export MAMBA_SSM_FORCE_REF=1"
fi

# 8) Clone repo (if missing)
if [ ! -d "${REPO_DIR}" ]; then
  echo "==> Cloning repo ${REPO_URL}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

# 9) Optional: patch quick_example.py to silence torch.load warning
PATCH_FILE="${REPO_DIR}/quick_example.py"
if grep -q 'torch.load("pretrained_weights/pretrained_weights.pth", map_location=device)' "$PATCH_FILE" 2>/dev/null; then
  echo "==> Patching quick_example.py to use weights_only=True"
  sed -i 's|torch.load("pretrained_weights/pretrained_weights.pth", map_location=device)|torch.load("pretrained_weights/pretrained_weights.pth", map_location=device, weights_only=True)|' "$PATCH_FILE"
fi

# 10) Final verify summary
python - <<'PY'
import torch, sys
print("[summary] torch:", torch.__version__, "cuda:", torch.cuda.is_available())
try:
    import triton
    print("[summary] triton:", triton.__version__)
except Exception as e:
    print("[summary] triton import failed:", e)
try:
    import mamba_ssm
    print("[summary] mamba-ssm OK")
except Exception as e:
    print("[summary] mamba-ssm import failed:", e)
try:
    import causal_conv1d
    print("[summary] causal-conv1d OK")
except Exception as e:
    print("[summary] causal-conv1d not available (using reference path is fine):", e)
PY

cat <<'MSG'

===========================================================
✅ Install finished.

Next steps:
  conda activate eegmamba
  cd EEGMamba-updated
  # (optional) if causal-conv1d build failed, run:
  # export MAMBA_SSM_DISABLE_TRITON=1
  # export MAMBA_SSM_FORCE_REF=1
  python quick_example.py

Expected: "Logits shape: torch.Size([8, 4])"
===========================================================
MSG
