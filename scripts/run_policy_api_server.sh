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
API_URL="${API_URL:-http://154.54.100.64:8080/infer}"
FPS="${FPS:-50}"
DURATION="${DURATION:-45}"
PREFETCH_AT_ACTIONS="${PREFETCH_AT_ACTIONS:-20}"
ACTION_LOG_EVERY_N_CHUNKS="${ACTION_LOG_EVERY_N_CHUNKS:-1}"
API_STATE_UNITS="${API_STATE_UNITS:-radians}"
API_ACTION_UNITS="${API_ACTION_UNITS:-robot}"
GRIPPER_ACTION_UNITS="${GRIPPER_ACTION_UNITS:-robot}"
MAX_JOINT_STEP_DEG="${MAX_JOINT_STEP_DEG:-0.5}"
MAX_GRIPPER_STEP="${MAX_GRIPPER_STEP:-1.0}"
UV_PYTHON="${UV_PYTHON:-3.12}"

CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-1}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

if [[ -z "${TOKEN:-${API_TOKEN:-}}" ]]; then
  echo "Missing API token. Set TOKEN=... or API_TOKEN=..." >&2
  exit 1
fi

if [[ ! -d "${LEROBOT_MAIN_DIR}" ]]; then
  echo "Missing latest LeRobot checkout: ${LEROBOT_MAIN_DIR}" >&2
  echo "Expected it at vendor/lerobot-main." >&2
  exit 1
fi

CAMERAS="{ front: {type: opencv, index_or_path: ${CAMERA_FRONT_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, side: {type: opencv, index_or_path: ${CAMERA_SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}} }"

echo "API URL:     ${API_URL}"
echo "Robot port:  ${ROBOT_PORT}"
echo "Control FPS: ${FPS}"
echo "Camera FPS:  ${CAMERA_FPS}"
echo "Duration:    ${DURATION}s"
echo "Prefetch at: ${PREFETCH_AT_ACTIONS} queued actions"
echo "Action log:  every ${ACTION_LOG_EVERY_N_CHUNKS} chunk(s)"
echo "API state:   ${API_STATE_UNITS}"
echo "API action:  ${API_ACTION_UNITS}"
echo "Gripper:     ${GRIPPER_ACTION_UNITS}"
echo "Joint limit: ${MAX_JOINT_STEP_DEG} deg/step"
echo "Grip limit:  ${MAX_GRIPPER_STEP}/step"
echo "Python:      ${UV_PYTHON}"
echo
echo "Keep one hand near power/USB. Press Ctrl-C to stop."

cd "${LEROBOT_MAIN_DIR}"

exec uv run --project "${LEROBOT_MAIN_DIR}" --python "${UV_PYTHON}" \
  --extra hardware --extra feetech \
  --with requests --with pillow \
  python "${ROOT_DIR}/scripts/run_policy_api_server.py" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERAS}" \
  --api_url="${API_URL}" \
  --fps="${FPS}" \
  --duration="${DURATION}" \
  --prefetch_at_actions="${PREFETCH_AT_ACTIONS}" \
  --action_log_every_n_chunks="${ACTION_LOG_EVERY_N_CHUNKS}" \
  --api_state_units="${API_STATE_UNITS}" \
  --api_action_units="${API_ACTION_UNITS}" \
  --gripper_action_units="${GRIPPER_ACTION_UNITS}" \
  --max_joint_step_deg="${MAX_JOINT_STEP_DEG}" \
  --max_gripper_step="${MAX_GRIPPER_STEP}"
