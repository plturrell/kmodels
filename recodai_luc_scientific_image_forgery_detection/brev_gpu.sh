#!/usr/bin/env bash
#
# Delegate to the repo-level Brev GPU helper while scoping paths to this project.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"

export BREV_LOCAL_DIR="${PROJECT_DIR}"
export BREV_REMOTE_DIR="~/projects/competitions/recodai_luc_scientific_image_forgery_detection"

exec "${REPO_ROOT}/scripts/brev_gpu.sh" "$@"
