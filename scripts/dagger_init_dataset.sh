#!/usr/bin/env bash
set -euo pipefail

BASE_DATASET_REPO_ID="${BASE_DATASET_REPO_ID:-ofcourseistillloveyou/so-101-feed-me}"
DAGGER_LOCAL_DATASET_ID="${DAGGER_LOCAL_DATASET_ID:-cdomkam/so101_feed_me_dagger_r1}"
LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
TARGET_DIR="${TARGET_DIR:-${LEROBOT_HOME}/${DAGGER_LOCAL_DATASET_ID}}"
FORCE="${FORCE:-false}"

if [[ -e "${TARGET_DIR}" && "${FORCE}" != "true" ]]; then
  echo "Target dataset already exists: ${TARGET_DIR}" >&2
  echo "Set FORCE=true to replace it, or choose DAGGER_LOCAL_DATASET_ID=..." >&2
  exit 1
fi

uv run --with huggingface_hub python - <<PY
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

base_repo = "${BASE_DATASET_REPO_ID}"
target_dir = Path("${TARGET_DIR}")
force = "${FORCE}" == "true"

source_dir = Path(snapshot_download(repo_id=base_repo, repo_type="dataset"))
if target_dir.exists() and force:
    shutil.rmtree(target_dir)

target_dir.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source_dir, target_dir, symlinks=False)

print(f"Initialized DAgger dataset from {base_repo}")
print(f"Local dataset id: ${DAGGER_LOCAL_DATASET_ID}")
print(f"Path: {target_dir}")
PY
