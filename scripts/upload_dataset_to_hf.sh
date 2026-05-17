#!/usr/bin/env bash
set -euo pipefail

SOURCE_DATASET_ID="${SOURCE_DATASET_ID:-cdomkam/so101_recording_strawberry_20260516_161110}"
TARGET_REPO_ID="${TARGET_REPO_ID:-ofcourseistillloveyou/so101_recording_strawberry_20260516_161110}"
LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
SOURCE_DIR="${SOURCE_DIR:-${LEROBOT_HOME}/${SOURCE_DATASET_ID}}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-Upload SO-101 recording ${SOURCE_DATASET_ID}}"

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "Dataset not found: ${SOURCE_DIR}" >&2
  exit 1
fi

echo "Source dataset: ${SOURCE_DATASET_ID}"
echo "Source folder:  ${SOURCE_DIR}"
echo "Target repo:    https://huggingface.co/datasets/${TARGET_REPO_ID}"

uv run --with huggingface_hub python - <<PY
from huggingface_hub import HfApi

source_dir = "${SOURCE_DIR}"
target_repo_id = "${TARGET_REPO_ID}"
commit_message = "${COMMIT_MESSAGE}"

api = HfApi()
try:
    api.whoami()
except Exception as exc:
    raise SystemExit(
        "Hugging Face auth failed. Run: uvx --from huggingface_hub hf auth login"
    ) from exc

api.create_repo(repo_id=target_repo_id, repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path=source_dir,
    repo_id=target_repo_id,
    repo_type="dataset",
    path_in_repo=".",
    commit_message=commit_message,
)

print(f"Uploaded to https://huggingface.co/datasets/{target_repo_id}")
PY
