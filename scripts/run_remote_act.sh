#!/usr/bin/env bash
set -euo pipefail

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
URL="${URL:-http://154.54.100.64:8080/infer}"
FPS="${FPS:-30}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-5.0}"
LOG_PATH="${LOG_PATH:-/tmp/remote-act.jsonl}"
RUN_TAG="${RUN_TAG:-remote}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
# Remote inference takes ~300 ms over HTTP, so blocking would freeze the loop for ~300 ms every
# chunk. Default leaves enough lead time to cover that. Cost: each new chunk's action[0] was
# computed from an observation PREFETCH_AT_ACTIONS/FPS seconds ago (stale obs). Lower this to
# reduce staleness; raise it to absorb HTTP tail latency. Set to 0 to block (worst smoothness).
PREFETCH_AT_ACTIONS="${PREFETCH_AT_ACTIONS:-15}"
TIMEOUT_S="${TIMEOUT_S:-5.0}"
COMM_RETRIES="${COMM_RETRIES:-3}"
COMM_RETRY_SLEEP_S="${COMM_RETRY_SLEEP_S:-0.02}"
ON_REFILL_FAILURE="${ON_REFILL_FAILURE:-hold}"
AUTH_TOKEN="${AUTH_TOKEN:-${TOKEN:-}}"

CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-0}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-1}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

CAMERAS="{front: {type: opencv, index_or_path: ${CAMERA_FRONT_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}, side: {type: opencv, index_or_path: ${CAMERA_SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}}"

UV_PYTHON="${UV_PYTHON:-3.12}"

echo "Remote ACT URL:  ${URL}"
echo "Robot port:      ${ROBOT_PORT}"
echo "Robot id:        ${ROBOT_ID}"
echo "Cameras:         front=${CAMERA_FRONT_INDEX}, side=${CAMERA_SIDE_INDEX}, ${CAMERA_WIDTH}x${CAMERA_HEIGHT}@${FPS}"
echo "Max relative:    ${MAX_RELATIVE_TARGET}"
echo "Chunk/actions:   ${CHUNK_SIZE}/${N_ACTION_STEPS}"
echo "Prefetch at:     ${PREFETCH_AT_ACTIONS} queued actions"
echo "Comm retries:    ${COMM_RETRIES} x ${COMM_RETRY_SLEEP_S}s"
echo "Log path:        ${LOG_PATH}"
echo
echo "Keep one hand near power/USB. Press Ctrl-C to stop."

args=(
  uvx --python "${UV_PYTHON}" --from 'lerobot[feetech]' python scripts/serving/clients/run_remote_act.py
  --url "${URL}"
  --robot.port "${ROBOT_PORT}"
  --robot.id "${ROBOT_ID}"
  --robot.cameras="${CAMERAS}"
  --robot.max_relative_target "${MAX_RELATIVE_TARGET}"
  --chunk_size "${CHUNK_SIZE}"
  --n_action_steps "${N_ACTION_STEPS}"
  --prefetch_at_actions "${PREFETCH_AT_ACTIONS}"
  --timeout_s "${TIMEOUT_S}"
  --comm_retries "${COMM_RETRIES}"
  --comm_retry_sleep_s "${COMM_RETRY_SLEEP_S}"
  --on_refill_failure "${ON_REFILL_FAILURE}"
  --fps "${FPS}"
  --log_path "${LOG_PATH}"
  --run_tag "${RUN_TAG}"
)

if [[ -n "${AUTH_TOKEN}" ]]; then
  args+=(--auth_token "${AUTH_TOKEN}")
fi

exec "${args[@]}"
