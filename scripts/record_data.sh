#!/usr/bin/env bash
set -euo pipefail

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-/dev/cu.usbmodem5C4C1255491}"
TELEOP_ID="${TELEOP_ID:-so101_leader}"
UV_PYTHON="${UV_PYTHON:-3.12}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATASET_REPO_ID="${DATASET_REPO_ID:-cdomkam/so101_recording_${RUN_ID}}"
LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
DATASET_ROOT="${DATASET_ROOT:-${LEROBOT_HOME}/${DATASET_REPO_ID}}"
TASK="${TASK:-Teleoperate the SO-101 arm.}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-15}"
RESET_TIME_S="${RESET_TIME_S:-10}"
FPS="${FPS:-30}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
RESUME="${RESUME:-false}"

CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-1}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

CAMERAS="{ front: {type: opencv, index_or_path: ${CAMERA_FRONT_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}, side: {type: opencv, index_or_path: ${CAMERA_SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}} }"

echo "Recording dataset: ${DATASET_REPO_ID}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Python: ${UV_PYTHON}"

exec uvx --python "${UV_PYTHON}" --from 'lerobot[feetech]' lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERAS}" \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id="${TELEOP_ID}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps="${FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --display_data="${DISPLAY_DATA}" \
  --resume="${RESUME}"
