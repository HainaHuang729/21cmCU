#!/usr/bin/env bash
set -euo pipefail

# Build reproducible CPU or CUDA wheels (and optional sdist) + SHA256SUMS.txt.
# Usage:
#   ./scripts/build_wheels.sh [--cpu | --cuda] [--cuda-archs "75,80,86,89,90"] [--no-sdist]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="cpu"
CUDA_ARCHS="${CUDA_ARCHS:-75,80,86,89,90}"
BUILD_SDIST=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      MODE="cpu"
      shift
      ;;
    --cuda)
      MODE="cuda"
      shift
      ;;
    --cuda-archs)
      CUDA_ARCHS="${2:?--cuda-archs requires an argument}"
      shift 2
      ;;
    --no-sdist)
      BUILD_SDIST=0
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Build 21cmCU Python distribution artifacts in dist/ and verify metadata via twine.

Options:
  --cpu                 Build CPU-only wheel (USE_CUDA=0, default)
  --cuda                Build CUDA-enabled wheel (USE_CUDA=1)
  --cuda-archs <archs>  Comma/semicolon CUDA SM list when --cuda is set (default: 75,80,86,89,90)
  --no-sdist            Build wheel only (skip source tarball)
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
rm -rf dist

if [[ "$MODE" == "cuda" ]]; then
  USE_CUDA_VAL=1
else
  USE_CUDA_VAL=0
fi

BUILD_ARGS=("--wheel")
if [[ "$BUILD_SDIST" -eq 1 ]]; then
  BUILD_ARGS=("--sdist" "--wheel")
fi

echo "==> Building 21cmCU distributions (USE_CUDA=$USE_CUDA_VAL, CUDA_ARCHS=$CUDA_ARCHS)..."
USE_CUDA="$USE_CUDA_VAL" CUDA_ARCHS="$CUDA_ARCHS" "$PYTHON_BIN" -m build "${BUILD_ARGS[@]}"

if "$PYTHON_BIN" -m twine --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m twine check dist/*
else
  echo "WARNING: twine not installed; skipping twine check."
fi

(
  cd dist
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- * > SHA256SUMS.txt
  else
    shasum -a 256 -- * > SHA256SUMS.txt
  fi
)

echo "==> Built artifacts in dist/:"
ls -lh dist/
