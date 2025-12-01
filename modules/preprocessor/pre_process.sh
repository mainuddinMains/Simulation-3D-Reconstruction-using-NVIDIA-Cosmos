#!/bin/bash

set -e

# --- Config ---
INPUT_DIR="/data/input"
OUTPUT_DIR="${INPUT_DIR}/custom/${SCENE_NAME}/images"
INPUT_FILE=$(find "$INPUT_DIR" -maxdepth 1 \( -name "*.bag" -o -name "*.mp4" \) -print -quit 2>/dev/null)

# --- Validation ---
[ -z "$INPUT_FILE" ] && echo "Error: No input file found." >&2 && exit 1

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
    -vf "fps=${FPS_EXTRACT},scale=512:512:force_original_aspect_ratio=increase,crop=512:512,format=rgb24" \
    -c:v png -compression_level 6 -pix_fmt rgb24 \
    "$OUTPUT_DIR/frame_%04d.png"

echo "Done: $OUTPUT_DIR"