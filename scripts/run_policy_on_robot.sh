#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
LEROBOT_MAIN_DIR="${LEROBOT_MAIN_DIR:-${ROOT_DIR}/vendor/lerobot-main}"
if [[ "${LEROBOT_MAIN_DIR}" != /* ]]; then
  LEROBOT_MAIN_DIR="${ROOT_DIR}/${LEROBOT_MAIN_DIR}"
fi

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
POLICY_REPO_ID="${POLICY_REPO_ID:-ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1}"
POLICY_TYPE="${POLICY_TYPE:-act}"
POLICY_DEVICE="${POLICY_DEVICE:-mps}"
POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-50}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-50}"
UV_PYTHON="${UV_PYTHON:-3.12}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DATASET_REPO_ID="${DATASET_REPO_ID:-cdomkam/eval_so101_policy_rollout_${RUN_ID}}"
TASK="${TASK:-Pick up the tape and put it on the pink post-it.}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-45}"
RESET_TIME_S="${RESET_TIME_S:-10}"
FPS="${FPS:-30}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
DEBUG_ACTIONS="${DEBUG_ACTIONS:-false}"
DEBUG_SEND_ACTIONS="${DEBUG_SEND_ACTIONS:-false}"
DEBUG_ACTION_LOG="${DEBUG_ACTION_LOG:-${ROOT_DIR}/logs/policy_action_debug_${RUN_ID}.jsonl}"
DEBUG_LOG_EVERY_N="${DEBUG_LOG_EVERY_N:-1}"

CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-1}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

CAMERAS="{ front: {type: opencv, index_or_path: ${CAMERA_FRONT_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}, side: {type: opencv, index_or_path: ${CAMERA_SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}} }"

echo "Policy repo:     ${POLICY_REPO_ID}"
echo "Policy device:   ${POLICY_DEVICE}"
echo "Policy chunk:    ${POLICY_CHUNK_SIZE}"
echo "Action steps:    ${POLICY_N_ACTION_STEPS}"
echo "Rollout dataset: ${DATASET_REPO_ID}"
echo "Robot port:      ${ROBOT_PORT}"
echo "Debug actions:   ${DEBUG_ACTIONS}"
if [[ "${DEBUG_ACTIONS}" == "true" ]]; then
  echo "Debug output:    ${DEBUG_ACTION_LOG}"
  echo "Debug sends:     ${DEBUG_SEND_ACTIONS}"
fi
echo "Python:          ${UV_PYTHON}"
echo
echo "Keep one hand near power/USB. Press Ctrl-C to stop the policy."

POLICY_DEBUG_ARGS=(
  --policy.path="${POLICY_REPO_ID}"
  --policy.device="${POLICY_DEVICE}"
  --policy.n_action_steps="${POLICY_N_ACTION_STEPS}"
)
POLICY_RECORD_ARGS=(
  --policy.type="${POLICY_TYPE}"
  --policy.pretrained_path="${POLICY_REPO_ID}"
  --policy.device="${POLICY_DEVICE}"
  --policy.n_action_steps="${POLICY_N_ACTION_STEPS}"
)
UVX_FROM="lerobot[feetech]"
UV_RUN_EXTRAS=(--extra dataset --extra hardware --extra viz --extra feetech)
if [[ "${POLICY_TYPE}" == "diffusion" ]]; then
  UVX_FROM="lerobot[feetech,diffusion]"
  UV_RUN_EXTRAS+=(--extra diffusion)
fi
if [[ "${POLICY_TYPE}" != "diffusion" ]]; then
  POLICY_DEBUG_ARGS+=(--policy.chunk_size="${POLICY_CHUNK_SIZE}")
  POLICY_RECORD_ARGS+=(--policy.chunk_size="${POLICY_CHUNK_SIZE}")
fi

if [[ "${DEBUG_ACTIONS}" == "true" ]]; then
  if [[ ! -d "${LEROBOT_MAIN_DIR}" ]]; then
    echo "Missing latest LeRobot checkout: ${LEROBOT_MAIN_DIR}" >&2
    echo "Expected it at vendor/lerobot-main." >&2
    exit 1
  fi

  cd "${LEROBOT_MAIN_DIR}"
  exec uv run --project "${LEROBOT_MAIN_DIR}" --python "${UV_PYTHON}" \
    "${UV_RUN_EXTRAS[@]}" \
    python "${ROOT_DIR}/scripts/debug_policy_actions.py" \
    --strategy.type=base \
    "${POLICY_DEBUG_ARGS[@]}" \
    --device="${POLICY_DEVICE}" \
    --robot.type=so101_follower \
    --robot.port="${ROBOT_PORT}" \
    --robot.id="${ROBOT_ID}" \
    --robot.cameras="${CAMERAS}" \
    --task="${TASK}" \
    --fps="${FPS}" \
    --duration="${EPISODE_TIME_S}" \
    --output_path="${DEBUG_ACTION_LOG}" \
    --send_actions="${DEBUG_SEND_ACTIONS}" \
    --log_every_n="${DEBUG_LOG_EVERY_N}"
fi

exec uvx --python "${UV_PYTHON}" --from "${UVX_FROM}" lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERAS}" \
  "${POLICY_RECORD_ARGS[@]}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.single_task="${TASK}" \
  --dataset.fps="${FPS}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --display_data="${DISPLAY_DATA}"
