#!/usr/bin/env bash
set -euo pipefail

HERE="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NOXIM_DIR="${NOXIM_DIR:-${HERE}/noxim-src}"
NOXIM_REPO="${NOXIM_REPO:-https://github.com/davidepatti/noxim.git}"
NOXIM_COMMIT="${NOXIM_COMMIT:-1b7d99923040ef5bda15ec519ef0071edf2dfad8}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Clones ${NOXIM_REPO} at ${NOXIM_COMMIT} into ${NOXIM_DIR},
applies patches/*.patch, downloads SystemC and yaml-cpp, and builds bin/noxim.
Re-running it on an existing checkout only rebuilds.

Environment overrides: NOXIM_DIR, NOXIM_REPO, NOXIM_COMMIT
EOF
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
fi

if [[ ! -d "${NOXIM_DIR}/.git" ]]; then
  echo "[setup] Cloning Noxim into ${NOXIM_DIR}"
  git clone "${NOXIM_REPO}" "${NOXIM_DIR}"
  git -C "${NOXIM_DIR}" checkout --detach "${NOXIM_COMMIT}"

  for patch in "${HERE}"/patches/*.patch; do
    echo "[setup] Applying $(basename "${patch}")"
    git -C "${NOXIM_DIR}" apply "${patch}"
  done
else
  echo "[setup] Reusing existing checkout ${NOXIM_DIR}"
fi

echo "[setup] Building Noxim (first build also fetches SystemC and yaml-cpp)"
(cd "${NOXIM_DIR}" && ./build.sh -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)")

echo "[setup] Done: ${NOXIM_DIR}/bin/noxim"
