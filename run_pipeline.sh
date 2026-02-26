#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.yml"

ALL_STAGES=("preprocessor" "da3" "sam3" "holoscene")
RUN_STAGES=("${ALL_STAGES[@]}")

FROM_STAGE=""
ONLY_STAGE=""
DRY_RUN=0
NO_BUILD=0

usage() {
    cat <<'EOF'
Usage:
  ./run_pipeline.sh [--from <stage>] [--only <stage>] [--dry-run] [--no-build]

Options:
  --from <stage>   Start from stage: preprocessor | da3 | sam3 | holoscene
  --only <stage>   Run only a single stage.
  --dry-run        Print commands without executing them.
  --no-build       Skip image rebuild (`docker compose up` instead of `up --build`).
  -h, --help       Show this help.
EOF
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

info() {
    echo "[INFO] $*"
}

warn() {
    echo "[WARN] $*" >&2
}

stage_is_valid() {
    local candidate="$1"
    local s
    for s in "${ALL_STAGES[@]}"; do
        if [[ "${s}" == "${candidate}" ]]; then
            return 0
        fi
    done
    return 1
}

stage_in_run_list() {
    local target="$1"
    local s
    for s in "${RUN_STAGES[@]}"; do
        if [[ "${s}" == "${target}" ]]; then
            return 0
        fi
    done
    return 1
}

build_stage_list() {
    local start="$1"
    local found=0
    RUN_STAGES=()
    local s
    for s in "${ALL_STAGES[@]}"; do
        if [[ "${s}" == "${start}" ]]; then
            found=1
        fi
        if [[ ${found} -eq 1 ]]; then
            RUN_STAGES+=("${s}")
        fi
    done
}

map_data_path_to_host() {
    local container_path="$1"
    if [[ "${container_path}" == "/data/"* ]]; then
        echo "${ROOT_DIR}/data/${container_path#/data/}"
        return 0
    fi
    return 1
}

require_file() {
    local f="$1"
    [[ -f "${f}" ]] || die "Required file missing: ${f}"
}

require_non_empty_var() {
    local k="$1"
    [[ -n "${!k:-}" ]] || die "Missing required value in .env: ${k}"
}

contains_supported_input() {
    local input_dir="$1"
    find "${input_dir}" -maxdepth 1 -type f \
        \( -iname '*.bag' -o -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.avi' \) \
        -print -quit | grep -q .
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --from)
                [[ $# -ge 2 ]] || die "--from requires a stage."
                FROM_STAGE="$2"
                shift 2
                ;;
            --only)
                [[ $# -ge 2 ]] || die "--only requires a stage."
                ONLY_STAGE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --no-build)
                NO_BUILD=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac
    done
}

configure_run_stages() {
    if [[ -n "${FROM_STAGE}" && -n "${ONLY_STAGE}" ]]; then
        die "Use either --from or --only, not both."
    fi

    if [[ -n "${ONLY_STAGE}" ]]; then
        stage_is_valid "${ONLY_STAGE}" || die "Invalid --only stage: ${ONLY_STAGE}"
        RUN_STAGES=("${ONLY_STAGE}")
        return
    fi

    if [[ -n "${FROM_STAGE}" ]]; then
        stage_is_valid "${FROM_STAGE}" || die "Invalid --from stage: ${FROM_STAGE}"
        build_stage_list "${FROM_STAGE}"
    fi
}

preflight() {
    info "Running preflight checks..."

    require_file "${COMPOSE_FILE}"
    require_file "${ENV_FILE}"
    require_file "${ROOT_DIR}/modules/sam3/prompts.txt"
    require_file "${ROOT_DIR}/modules/holoscene/confs/base.conf"
    require_file "${ROOT_DIR}/modules/holoscene/confs/post.conf"
    require_file "${ROOT_DIR}/modules/holoscene/confs/tex.conf"

    command -v docker >/dev/null 2>&1 || die "Docker is not installed or not in PATH."
    docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is unavailable."
    docker info >/dev/null 2>&1 || die "Docker daemon is not reachable."

    set +u
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
    set -u

    require_non_empty_var "SCENE_NAME"
    require_non_empty_var "DATA_ROOT"
    require_non_empty_var "OUTPUT_ROOT"
    require_non_empty_var "IMG_HEIGHT"
    require_non_empty_var "IMG_WIDTH"
    require_non_empty_var "TORCH_CUDA_ARCH_LIST"
    require_non_empty_var "CAPTURE_FRAMERATE"
    require_non_empty_var "FPS_EXTRACT"
    require_non_empty_var "SAM3_MIN_SCORE"
    require_non_empty_var "SAM3_MIN_FRAME_DURATION"

    if stage_in_run_list "sam3"; then
        require_non_empty_var "HF_TOKEN"
    fi

    local input_dir="${ROOT_DIR}/data/input"
    [[ -d "${input_dir}" ]] || die "Missing input directory: ${input_dir}"

    if stage_in_run_list "preprocessor"; then
        contains_supported_input "${input_dir}" || die \
            "No input media found in ${input_dir}. Add one of: .bag, .mp4, .mov, .mkv, .avi."
    fi

    local data_root_host=""
    if data_root_host="$(map_data_path_to_host "${DATA_ROOT}")"; then
        local scene_dir="${data_root_host}/${SCENE_NAME}"
        local image_dir="${scene_dir}/images"
        local transforms_file="${scene_dir}/transforms.json"
        local mask_dir="${scene_dir}/instance_mask"

        if ! stage_in_run_list "preprocessor" && (stage_in_run_list "da3" || stage_in_run_list "sam3" || stage_in_run_list "holoscene"); then
            [[ -d "${image_dir}" ]] || die "Missing extracted images directory: ${image_dir}"
            find "${image_dir}" -maxdepth 1 -type f -name '*.png' -print -quit | grep -q . || die \
                "No extracted PNG frames found in ${image_dir}"
        fi

        if ! stage_in_run_list "da3" && (stage_in_run_list "sam3" || stage_in_run_list "holoscene"); then
            [[ -f "${transforms_file}" ]] || die "Missing DA3 output: ${transforms_file}"
        fi

        if ! stage_in_run_list "sam3" && stage_in_run_list "holoscene"; then
            [[ -d "${mask_dir}" ]] || die "Missing SAM3 output directory: ${mask_dir}"
            find "${mask_dir}" -maxdepth 1 -type f -name '*.png' -print -quit | grep -q . || die \
                "No mask PNGs found in ${mask_dir}"
        fi
    else
        warn "DATA_ROOT=${DATA_ROOT} does not map to ./data; skipping host-side artifact checks."
    fi

    info "Preflight passed."
}

run_pipeline() {
    local s
    local cmd
    for s in "${RUN_STAGES[@]}"; do
        info "Starting stage: ${s}"
        if [[ ${NO_BUILD} -eq 1 ]]; then
            cmd=(docker compose up "${s}")
        else
            cmd=(docker compose up --build "${s}")
        fi

        if [[ ${DRY_RUN} -eq 1 ]]; then
            echo "[DRY-RUN] ${cmd[*]}"
        else
            (cd "${ROOT_DIR}" && "${cmd[@]}")
        fi
        info "Completed stage: ${s}"
    done
}

main() {
    parse_args "$@"
    configure_run_stages

    info "Selected stages: ${RUN_STAGES[*]}"
    preflight
    run_pipeline
    info "Pipeline finished successfully."
}

main "$@"
