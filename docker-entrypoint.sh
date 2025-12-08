#!/usr/bin/env sh

set -e

IMG_DIR="${DATA_ROOT}/${SCENE_NAME}/images"

echo "Checking image directory: ${IMG_DIR}"

if ! ls "$IMG_DIR" | grep -qEi '\.(jpe?g|png)$'; then
  echo "❌ No image files found in $IMG_DIR"
  exit 1
fi

echo "✅ Required files found. Starting main service ($@)..."

exec "$@"