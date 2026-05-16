#!/usr/bin/env bash
set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-ofcourseistillloveyou/so-101-feed-me}"
POLICY_REPO_ID="${POLICY_REPO_ID:-ofcourseistillloveyou/so-101-feed-me-act}"
PRETRAINED_POLICY_REPO_ID="${PRETRAINED_POLICY_REPO_ID:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
JOB_NAME="${JOB_NAME:-so101-feed-me-act-${RUN_ID}}"
POLICY_TYPE="${POLICY_TYPE:-act}"
STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-500}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
RESUME="${RESUME:-false}"

args=(
  modal run scripts/modal_train_lerobot.py
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --policy-repo-id "${POLICY_REPO_ID}" \
  --pretrained-policy-repo-id "${PRETRAINED_POLICY_REPO_ID}" \
  --job-name "${JOB_NAME}" \
  --policy-type "${POLICY_TYPE}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-freq "${SAVE_FREQ}"
)

if [[ "${WANDB_ENABLE}" == "true" ]]; then
  args+=(--wandb-enable)
fi

if [[ "${RESUME}" == "true" ]]; then
  args+=(--resume)
fi

exec "${args[@]}"
