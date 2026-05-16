#!/usr/bin/env bash
set -euo pipefail

DAGGER_HF_DATASET_REPO_ID="${DAGGER_HF_DATASET_REPO_ID:-ofcourseistillloveyou/so-101-feed-me-dagger-r1}"
PREVIOUS_POLICY_REPO_ID="${PREVIOUS_POLICY_REPO_ID:-ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1}"
NEXT_POLICY_REPO_ID="${NEXT_POLICY_REPO_ID:-ofcourseistillloveyou/act-so101-feed-me-dagger-r1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-so101-feed-me-dagger-r1-${RUN_ID}}"
STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-16}"

exec env \
  DATASET_REPO_ID="${DAGGER_HF_DATASET_REPO_ID}" \
  PRETRAINED_POLICY_REPO_ID="${PREVIOUS_POLICY_REPO_ID}" \
  POLICY_REPO_ID="${NEXT_POLICY_REPO_ID}" \
  JOB_NAME="${JOB_NAME}" \
  STEPS="${STEPS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  ./scripts/train_on_modal.sh
