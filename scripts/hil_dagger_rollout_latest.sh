#!/usr/bin/env bash
set -euo pipefail

LEROBOT_MAIN_DIR="${LEROBOT_MAIN_DIR:-vendor/lerobot-main}"
ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-/dev/cu.usbmodem5C4C1255491}"
TELEOP_ID="${TELEOP_ID:-so101_leader}"
POLICY_REPO_ID="${POLICY_REPO_ID:-ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1}"
POLICY_DEVICE="${POLICY_DEVICE:-mps}"
UV_PYTHON="${UV_PYTHON:-3.12}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATASET_REPO_ID="${DATASET_REPO_ID:-cdomkam/rollout_hil_so101_feed_me_${RUN_ID}}"
TASK="${TASK:-Pick up the tape and put it on the pink post-it.}"
NUM_EPISODES="${NUM_EPISODES:-5}"
FPS="${FPS:-30}"
DURATION="${DURATION:-0}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
RECORD_AUTONOMOUS="${RECORD_AUTONOMOUS:-false}"

CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-1}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

if [[ ! -d "${LEROBOT_MAIN_DIR}" ]]; then
  echo "Missing latest LeRobot checkout: ${LEROBOT_MAIN_DIR}" >&2
  echo "Expected it at vendor/lerobot-main." >&2
  exit 1
fi

CAMERAS="{ front: {type: opencv, index_or_path: ${CAMERA_FRONT_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}, side: {type: opencv, index_or_path: ${CAMERA_SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}} }"

echo "Latest LeRobot:  ${LEROBOT_MAIN_DIR}"
echo "Policy repo:     ${POLICY_REPO_ID}"
echo "Dataset repo:    ${DATASET_REPO_ID}"
echo "FPS:             ${FPS}"
echo "Python:          ${UV_PYTHON}"
echo
echo "DAgger controls in v0.5.2:"
echo "  space  pause/resume policy"
echo "  tab    toggle human correction"
echo "  enter  upload/push on demand"
echo
echo "Keep one hand near power/USB. Press Ctrl-C to stop."

cd "${LEROBOT_MAIN_DIR}"

exec uv run --python "${UV_PYTHON}" --extra core_scripts --extra feetech lerobot-rollout \
  --strategy.type=dagger \
  --strategy.num_episodes="${NUM_EPISODES}" \
  --strategy.record_autonomous="${RECORD_AUTONOMOUS}" \
  --policy.path="${POLICY_REPO_ID}" \
  --device="${POLICY_DEVICE}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERAS}" \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id="${TELEOP_ID}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps="${FPS}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --fps="${FPS}" \
  --duration="${DURATION}" \
  --display_data="${DISPLAY_DATA}"
