#!/bin/bash

set -e

# --- Config ---
INPUT_DIR="/data/input"
OUTPUT_DIR="${INPUT_DIR}/custom/${SCENE_NAME}/images"
INPUT_FILE=$(find "$INPUT_DIR" -maxdepth 1 -type f -iregex ".*\.\(bag\|mp4\|mov\|mkv\|avi\)$" -print -quit 2>/dev/null)

# --- Validation ---
[ -z "$INPUT_FILE" ] && echo "Error: No valid input video found (checked .bag, .mp4, .mov, .mkv, .avi)." >&2 && exit 1

if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo "Images exist. Skipping."
    exit 0
fi

# --- Processing ---
echo "Processing: $INPUT_FILE"
mkdir -p "$OUTPUT_DIR"

if [[ "$INPUT_FILE" == *.bag ]]; then
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf '$TEMP_DIR'" EXIT

    rs-convert -i "$INPUT_FILE" -p "$TEMP_DIR/frame" > /dev/null
    
    SOURCE="$TEMP_DIR/*.png"
    PRE_FLAGS="-framerate ${CAPTURE_FRAMERATE} -pattern_type glob"
else
    SOURCE="$INPUT_FILE"
    PRE_FLAGS=""
fi

ffmpeg -y $PRE_FLAGS -i "$SOURCE" \
    -vf "fps=${FPS_EXTRACT},scale=${IMG_WIDTH}:${IMG_HEIGHT}:force_original_aspect_ratio=increase,crop=${IMG_WIDTH}:${IMG_HEIGHT},format=rgb24" \
    -c:v png -compression_level 6 -pix_fmt rgb24 \
    "$OUTPUT_DIR/frame_%04d.png"

IMAGE_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.png" | wc -l)

echo "Done: $OUTPUT_DIR - Extracted **$IMAGE_COUNT** images."