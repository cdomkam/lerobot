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
UV_PYTHON="${UV_PYTHON:-3.12}"

CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-0}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-1}"
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
  --duration="${DURATION}"
