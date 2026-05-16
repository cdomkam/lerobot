#!/usr/bin/env bash
set -euo pipefail

POLICY_REPO_ID="${POLICY_REPO_ID:-ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1}"
LOCAL_DIR="${LOCAL_DIR:-models/${POLICY_REPO_ID//\//__}}"

mkdir -p "${LOCAL_DIR}"

echo "Downloading policy repo: ${POLICY_REPO_ID}"
echo "Local cache:             ${LOCAL_DIR}"

exec uvx --from huggingface_hub hf download "${POLICY_REPO_ID}" \
  --repo-type model \
  --local-dir "${LOCAL_DIR}"
