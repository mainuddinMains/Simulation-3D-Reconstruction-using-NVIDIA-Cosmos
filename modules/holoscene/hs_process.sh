#!/bin/bash

set -e


############################################
# Configuration
############################################
export SCENE_NAME="${SCENE_NAME}"
export DATA_ROOT="${DATA_ROOT}"

export IMG_WIDTH="${IMG_WIDTH}"
export IMG_HEIGHT="${IMG_HEIGHT}"

DATA_DIR="${DATA_ROOT}/${SCENE_NAME}"
CACHE_ROOT="/root/.cache"

WONDER3D_DIR="${CACHE_ROOT}/ckpts"
LAMA_DIR="${CACHE_ROOT}/lama"
LAMA_CHECK_FILE="${LAMA_DIR}/big-lama/models/best.ckpt"
OMNIDATA_FILE="${CACHE_ROOT}/omnidata/omnidata_dpt_normal_v2.ckpt"

mkdir -p /tmp/confs
envsubst < "confs/base.conf" > "/tmp/confs/base.conf"
envsubst < "confs/post.conf" > "/tmp/confs/post.conf"
envsubst < "confs/tex.conf"  > "/tmp/confs/tex.conf"

BASE_CONF="/tmp/confs/base.conf"
POST_CONF="/tmp/confs/post.conf"
TEX_CONF="/tmp/confs/tex.conf"


############################################
# Check/Download Models
############################################
# --- Wonder3D+ ---
if [ ! -d "${WONDER3D_DIR}" ] || [ -z "$(ls -A "${WONDER3D_DIR}" 2>/dev/null || true)" ]; then
    echo "--- Downloading Wonder3D+ models... ---"
    mkdir -p "${WONDER3D_DIR}"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='flamehaze1115/Wonder3D_plus', local_dir='${WONDER3D_DIR}')"
else
    echo "--- Wonder3D+ models found. ---"
fi

# --- LaMa ---
if [ ! -f "${LAMA_CHECK_FILE}" ]; then
    echo "--- Downloading LaMa model... ---"
    mkdir -p "${LAMA_DIR}"
    pushd "${LAMA_DIR}" > /dev/null
        curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
        unzip -o big-lama.zip
        rm big-lama.zip
    popd > /dev/null
else
    echo "--- LaMa model found. ---"
fi

# --- Omnidata ---
if [ ! -f "${OMNIDATA_FILE}" ]; then
    echo "--- Downloading Omnidata model... ---"
    mkdir -p "$(dirname "$OMNIDATA_FILE")"
    wget -O "${OMNIDATA_FILE}" "https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt"
else
    echo "--- Omnidata model found. ---"
fi


############################################
# Symlink Setup
############################################
ln -sfn "$WONDER3D_DIR" /app/holoscene/ckpts
ln -sfn "$LAMA_DIR/big-lama" /app/holoscene/lama/big-lama
ln -sfn "$OMNIDATA_FILE" /app/holoscene/omnidata_dpt_normal_v2.ckpt


############################################
# Stage 0: Priors (Marigold)
############################################
echo "--- Stage 0: Generating Priors ---"
python3 marigold/run.py \
    --checkpoint="GonzaloMG/marigold-e2e-ft-normals" \
    --modality normals \
    --input_rgb_dir="${DATA_DIR}/images" \
    --output_dir="${DATA_DIR}/"

python3 marigold/run.py \
    --checkpoint="GonzaloMG/marigold-e2e-ft-depth" \
    --modality depth \
    --input_rgb_dir="${DATA_DIR}/images" \
    --output_dir="${DATA_DIR}/"


############################################
# Training Stages
############################################
echo "--- Stage 1: Initial Reconstruction ---"
python3 training/exp_runner.py --conf "${BASE_CONF}"

echo "--- Stage 2: Post-processing ---"
python3 training/exp_runner_post.py --conf "${POST_CONF}" \
    --is_continue --timestamp latest --checkpoint latest

echo "--- Stage 3: Texture Refinement ---"
python3 training/exp_runner_texture.py --conf "${TEX_CONF}" \
    --is_continue --timestamp latest --checkpoint latest

echo "--- Stage 4: Gaussian on Mesh ---"
python3 training/exp_runner_gaussian_on_mesh.py --conf "${TEX_CONF}" \
    --is_continue --timestamp latest --checkpoint latest


############################################
# Export
############################################
echo "--- Training stages complete, exporting results... ---"

python3 export/export_glb.py    --conf "${TEX_CONF}" --timestamp latest
python3 export/export_usd.py    --conf "${TEX_CONF}" --timestamp latest
python3 export/export_gs_usd.py --conf "${TEX_CONF}" --timestamp latest

echo "--- Export complete. ---"