#!/usr/bin/env bash
set -euo pipefail

DAGGER_LOCAL_DATASET_ID="${DAGGER_LOCAL_DATASET_ID:-cdomkam/so101_feed_me_dagger_r1}"
DAGGER_HF_DATASET_REPO_ID="${DAGGER_HF_DATASET_REPO_ID:-ofcourseistillloveyou/so-101-feed-me-dagger-r1}"

exec env \
  SOURCE_DATASET_ID="${DAGGER_LOCAL_DATASET_ID}" \
  TARGET_REPO_ID="${DAGGER_HF_DATASET_REPO_ID}" \
  ./scripts/upload_dataset_to_hf.sh
