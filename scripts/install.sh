#!/usr/bin/env bash
set -euo pipefail

# One-command setup & diagnostic check for 21cmCU.
# Usage:
#   ./scripts/install.sh [--cpu] [--cuda-archs 89|auto] [--venv .venv]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FORCE_CPU=0
CUDA_ARCHS="${CUDA_ARCHS:-auto}"
VENV_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      FORCE_CPU=1
      shift
      ;;
    --cuda-archs)
      CUDA_ARCHS="${2:?--cuda-archs requires an argument}"
      shift 2
      ;;
    --venv)
      VENV_DIR="${2:?--venv requires a path}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Install 21cmCU in editable mode after checking toolchain and CUDA dependencies.

Options:
  --cpu                 Build CPU-only (USE_CUDA=0) even if nvcc is installed
  --cuda-archs <archs>  Target SM list or 'auto' (default: auto)
  --venv <path>         Create/activate a Python 3.12 venv before installing
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -n "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON_BIN="python"
fi

echo "==> Checking Python >= 3.12 and headers..."
"$PYTHON_BIN" -c '
import sys, sysconfig
from pathlib import Path
if sys.version_info < (3, 12):
    raise SystemExit(f"ERROR: Python >= 3.12 required, found {sys.version.split()[0]}")
inc = Path(sysconfig.get_path("include")) / "Python.h"
if not inc.exists():
    raise SystemExit(
        f"ERROR: Missing Python.h ({inc}). Install python3.12-dev / conda python headers."
    )
print(f"Python {sys.version.split()[0]} OK (headers: {inc})")
'

if ! command -v gcc >/dev/null 2>&1; then
  echo "ERROR: gcc not found. Install build-essential / gcc." >&2
  exit 1
fi

echo "==> Checking GSL & FFTW3 float libraries..."
"$PYTHON_BIN" -c '
import ctypes.util, shutil
missing = []
for libname, pkg in [("gsl", "libgsl-dev"), ("fftw3f", "libfftw3-dev")]:
    if ctypes.util.find_library(libname) is None and not shutil.which("pkg-config"):
        missing.append(f"{libname} ({pkg})")
if missing:
    print("WARNING: Could not auto-locate:", ", ".join(missing))
    print("If compilation fails, install libgsl-dev & libfftw3-dev (Ubuntu) or gsl & fftw (conda).")
'

USE_CUDA_VAL=0
if [[ "$FORCE_CPU" -eq 0 ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    USE_CUDA_VAL=1
  else
    latest_nvcc="$(ls -1d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "$latest_nvcc" ]]; then
      export PATH="$(dirname "$latest_nvcc"):$PATH"
      USE_CUDA_VAL=1
    else
      echo "INFO: nvcc not found; falling back to CPU build (USE_CUDA=0). Pass CUDA in PATH to enable GPU."
    fi
  fi
fi

echo "==> Building & installing 21cmCU (USE_CUDA=$USE_CUDA_VAL, CUDA_ARCHS=$CUDA_ARCHS)..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools cffi
USE_CUDA="$USE_CUDA_VAL" CUDA_ARCHS="$CUDA_ARCHS" "$PYTHON_BIN" -m pip install -e . --no-build-isolation

echo "==> Running 21cmcu doctor..."
"$PYTHON_BIN" -m py21cmcu.cli doctor
