#!/bin/bash

set -e

# --- Configuration ---
INPUT_DIR="/data/input"
INPUT_FILE=$(find "$INPUT_DIR" -maxdepth 1 \( -name "*.bag" -o -name "*.mp4" \) -print -quit)
OUTPUT_DIR="${INPUT_DIR}/custom/${SCENE_NAME}/images"

# --- Validation ---
if [ -z "$INPUT_FILE" ]; then
    echo "Error: No .bag or .mp4 file found in $INPUT_DIR" >&2
    exit 1
fi

if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR")" ]; then
    echo "--- [Preprocessor] Images already exist. Skipping. ---"
    exit 0
fi

# --- Processing ---
echo "--- [Preprocessor] Processing: $INPUT_FILE ---"
mkdir -p "$OUTPUT_DIR"

if [[ "$INPUT_FILE" == *.bag ]]; then
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf '${TEMP_DIR}'" EXIT SIGINT SIGTERM

    echo "--- [Preprocessor] Extracting .bag frames ---"
    rs-convert -i "$INPUT_FILE" -p "$TEMP_DIR/frame" > /dev/null

    ffmpeg -y -framerate "${CAPTURE_FRAMERATE}" \
        -pattern_type glob -i "$TEMP_DIR/*.png" \
        -vf "fps=${FPS_EXTRACT}" \
        "$OUTPUT_DIR/frame_%04d.png"

elif [[ "$INPUT_FILE" == *.mp4 ]]; then
    echo "--- [Preprocessor] Converting .mp4 file ---"
    ffmpeg -y -i "$INPUT_FILE" \
        -vf "fps=${FPS_EXTRACT}" \
        "$OUTPUT_DIR/frame_%04d.png"
fi

echo "--- [Preprocessor] Done. ---"