#!/usr/bin/env bash
set -euo pipefail

# Collect expert correction episodes into the initialized aggregate DAgger dataset.
# Use this after watching a policy failure and resetting the scene to a similar state.

DATASET_REPO_ID="${DATASET_REPO_ID:-cdomkam/so101_feed_me_dagger_r1}"
TASK="${TASK:-Pick up the tape and put it on the pink post-it.}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-15}"
RESET_TIME_S="${RESET_TIME_S:-10}"
FPS="${FPS:-30}"
RESUME="${RESUME:-true}"

exec env \
  DATASET_REPO_ID="${DATASET_REPO_ID}" \
  TASK="${TASK}" \
  NUM_EPISODES="${NUM_EPISODES}" \
  EPISODE_TIME_S="${EPISODE_TIME_S}" \
  RESET_TIME_S="${RESET_TIME_S}" \
  FPS="${FPS}" \
  RESUME="${RESUME}" \
  ./scripts/record_data.sh
