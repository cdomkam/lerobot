#!/usr/bin/env bash
set -euo pipefail

LEROBOT_MAIN_DIR="${LEROBOT_MAIN_DIR:-vendor/lerobot-main}"

if [[ ! -d "${LEROBOT_MAIN_DIR}/.git" ]]; then
  echo "Missing git checkout: ${LEROBOT_MAIN_DIR}" >&2
  exit 1
fi

git -C "${LEROBOT_MAIN_DIR}" \
  -c filter.lfs.smudge= \
  -c filter.lfs.process= \
  -c filter.lfs.required=false \
  pull --ff-only

git -C "${LEROBOT_MAIN_DIR}" rev-parse --short HEAD
