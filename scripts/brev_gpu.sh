#!/usr/bin/env bash
#
# Helper for managing Brev GPU instances for this repository.
# Usage: ./scripts/brev_gpu.sh <command> [args]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="$(basename "${REPO_ROOT}")"

DEFAULT_INSTANCE="${BREV_INSTANCE_NAME:-competitions-gpu}"
DEFAULT_GPU="${BREV_GPU_TYPE:-n1-highmem-4:nvidia-tesla-t4:1}"
DEFAULT_CPU="${BREV_CPU_TYPE:-}"
DEFAULT_LOCAL_PATH="${BREV_LOCAL_DIR:-${REPO_ROOT}}"
DEFAULT_REMOTE_PATH="${BREV_REMOTE_DIR:-~/projects/${PROJECT_NAME}}"

print_usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  create [instance]          Create a GPU instance (default: ${DEFAULT_INSTANCE})
  start [instance]           Start an existing instance
  stop [instance]            Stop the instance
  delete [instance]          Delete the instance (irreversible)
  status [instance]          Show status for the instance
  shell [instance]           Open an interactive shell on the instance
  sync-up [instance] [dir]   Copy local repo (or dir) to the instance
  sync-down [instance] [dir] Copy remote repo back to local

Environment overrides:
  BREV_INSTANCE_NAME   Default instance name (current: ${DEFAULT_INSTANCE})
  BREV_GPU_TYPE        GPU type for creation (current: ${DEFAULT_GPU})
  BREV_CPU_TYPE        CPU shape override for creation
  BREV_LOCAL_DIR       Local path to sync (current: ${DEFAULT_LOCAL_PATH})
  BREV_REMOTE_DIR      Remote path to sync (current: ${DEFAULT_REMOTE_PATH})

Examples:
  ./scripts/brev_gpu.sh create
  ./scripts/brev_gpu.sh sync-up
  ./scripts/brev_gpu.sh shell
  ./scripts/brev_gpu.sh sync-down competitions-gpu outputs
EOF
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

ensure_brev() {
  if ! command_exists brev; then
    echo "Error: Brev CLI not found. Install via 'brew install brevdev/homebrew-brev/brev'." >&2
    exit 1
  fi
}

resolve_instance() {
  local override="${1:-}"
  if [[ -n "${override}" ]]; then
    echo "${override}"
  else
    echo "${DEFAULT_INSTANCE}"
  fi
}

resolve_local_path() {
  local override="${1:-}"
  if [[ -n "${override}" ]]; then
    local candidate
    if [[ "${override}" == /* ]]; then
      candidate="${override}"
    else
      candidate="${REPO_ROOT}/${override}"
    fi

    if [[ -d "${candidate}" ]]; then
      echo "$(cd "${candidate}" && pwd)"
    else
      echo "${candidate}"
    fi
  else
    echo "${DEFAULT_LOCAL_PATH}"
  fi
}

resolve_remote_path() {
  local override="${1:-}"
  if [[ -n "${override}" ]]; then
    echo "${override}"
  else
    echo "${DEFAULT_REMOTE_PATH}"
  fi
}

create_instance() {
  ensure_brev
  local instance
  instance="$(resolve_instance "$1")"
  local gpu_shape="${BREV_GPU_TYPE:-${DEFAULT_GPU}}"
  local cpu_shape="${BREV_CPU_TYPE:-${DEFAULT_CPU}}"

  echo "Creating Brev instance '${instance}' with GPU '${gpu_shape}'${cpu_shape:+ and CPU '${cpu_shape}'}..."

  local args=(create "${instance}" --gpu "${gpu_shape}")
  if [[ -n "${cpu_shape}" ]]; then
    args+=(--cpu "${cpu_shape}")
  fi

  brev "${args[@]}"
}

start_instance() {
  ensure_brev
  local instance
  instance="$(resolve_instance "$1")"
  echo "Starting Brev instance '${instance}'..."
  brev start "${instance}"
}

stop_instance() {
  ensure_brev
  local instance
  instance="$(resolve_instance "$1")"
  echo "Stopping Brev instance '${instance}'..."
  brev stop "${instance}"
}

delete_instance() {
  ensure_brev
  local instance
  instance="$(resolve_instance "$1")"
  read -r -p "Delete Brev instance '${instance}'? This cannot be undone. [y/N] " confirm
  if [[ "${confirm}" == "y" || "${confirm}" == "Y" ]]; then
    brev delete "${instance}"
  else
    echo "Aborted."
  fi
}

status_instance() {
  ensure_brev
  local instance
  instance="$(resolve_instance "$1")"
  brev status "${instance}"
}

shell_instance() {
  ensure_brev
  local instance
  instance="$(resolve_instance "$1")"
  echo "Opening shell on '${instance}'. Exit the shell when you're done."
  brev shell "${instance}"
}

sync_up() {
  ensure_brev
  local instance local_path remote_path
  instance="$(resolve_instance "$1")"
  local_path="$(resolve_local_path "${2:-}")"
  remote_path="$(resolve_remote_path "${3:-}")"

  if [[ ! -d "${local_path}" ]]; then
    echo "Error: local path '${local_path}' does not exist or is not a directory." >&2
    exit 1
  fi

  echo "Syncing local '${local_path}' -> '${instance}:${remote_path}'..."
  brev copy "${local_path}" "${instance}:${remote_path}"
}

sync_down() {
  ensure_brev
  local instance remote_path local_path
  instance="$(resolve_instance "$1")"
  remote_path="$(resolve_remote_path "${2:-}")"
  local_path="$(resolve_local_path "${3:-}")"

  mkdir -p "${local_path}"

  echo "Syncing remote '${instance}:${remote_path}' -> '${local_path}'..."
  brev copy "${instance}:${remote_path}" "${local_path}"
}

main() {
  if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
  fi

  local cmd="$1"
  shift

  case "${cmd}" in
    create)
      create_instance "${1:-}"
      ;;
    start)
      start_instance "${1:-}"
      ;;
    stop)
      stop_instance "${1:-}"
      ;;
    delete)
      delete_instance "${1:-}"
      ;;
    status)
      status_instance "${1:-}"
      ;;
    shell)
      shell_instance "${1:-}"
      ;;
    sync-up)
      sync_up "${1:-}" "${2:-}" "${3:-}"
      ;;
    sync-down)
      sync_down "${1:-}" "${2:-}" "${3:-}"
      ;;
    help|-h|--help)
      print_usage
      ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      print_usage
      exit 1
      ;;
  esac
}

main "$@"
