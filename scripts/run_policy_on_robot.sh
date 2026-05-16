#!/usr/bin/env bash
set -euo pipefail

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
POLICY_REPO_ID="${POLICY_REPO_ID:-ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1}"
POLICY_TYPE="${POLICY_TYPE:-act}"
POLICY_DEVICE="${POLICY_DEVICE:-mps}"
UV_PYTHON="${UV_PYTHON:-3.12}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATASET_REPO_ID="${DATASET_REPO_ID:-cdomkam/eval_so101_policy_rollout_${RUN_ID}}"
TASK="${TASK:-Pick up the tape and put it on the pink post-it.}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-15}"
RESET_TIME_S="${RESET_TIME_S:-10}"
FPS="${FPS:-30}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-0}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-1}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

CAMERAS="{ front: {type: opencv, index_or_path: ${CAMERA_FRONT_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}, side: {type: opencv, index_or_path: ${CAMERA_SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}} }"

echo "Policy repo:     ${POLICY_REPO_ID}"
echo "Policy device:   ${POLICY_DEVICE}"
echo "Rollout dataset: ${DATASET_REPO_ID}"
echo "Robot port:      ${ROBOT_PORT}"
echo "Python:          ${UV_PYTHON}"
echo
echo "Keep one hand near power/USB. Press Ctrl-C to stop the policy."

exec uvx --python "${UV_PYTHON}" --from 'lerobot[feetech]' lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERAS}" \
  --policy.type="${POLICY_TYPE}" \
  --policy.pretrained_path="${POLICY_REPO_ID}" \
  --policy.device="${POLICY_DEVICE}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps="${FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --display_data="${DISPLAY_DATA}"
