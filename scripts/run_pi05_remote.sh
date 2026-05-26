#!/usr/bin/env bash
# Drive the SO-101 with a remote pi0.5 inference server.
# Same control loop as run_remote_act.sh — only differences are the URL,
# the model size (pi0.5 inference is slower than ACT), and the `--task`
# language prompt that pi0.5 expects.
set -euo pipefail

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
URL="${URL:-http://198.145.127.220:8080/infer}"
TASK="${TASK:-pick up the red lid}"
FPS="${FPS:-30}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-5.0}"
LOG_PATH="${LOG_PATH:-/tmp/pi05-remote.jsonl}"
RUN_TAG="${RUN_TAG:-pi05-remote}"
# pi0.5 typically uses a 50-step chunk. Override if your fine-tune used a
# different chunk_size.
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
# pi0.5 inference takes longer than ACT (~500-800 ms on an A100 for a 3B-param
# VLA). Increase prefetch lead so the queue doesn't dry up while the next
# chunk is computing. Trade-off: higher = more stale obs at chunk seams.
PREFETCH_AT_ACTIONS="${PREFETCH_AT_ACTIONS:-25}"
# Larger timeout for the bigger model. First call after the server warms up
# can be several seconds.
TIMEOUT_S="${TIMEOUT_S:-15.0}"
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

echo "pi0.5 URL:       ${URL}"
echo "Task prompt:     ${TASK}"
echo "Robot port:      ${ROBOT_PORT}"
echo "Robot id:        ${ROBOT_ID}"
echo "Cameras:         front=${CAMERA_FRONT_INDEX}, side=${CAMERA_SIDE_INDEX}, ${CAMERA_WIDTH}x${CAMERA_HEIGHT}@${FPS}"
echo "Max relative:    ${MAX_RELATIVE_TARGET}"
echo "Chunk/actions:   ${CHUNK_SIZE}/${N_ACTION_STEPS}"
echo "Prefetch at:     ${PREFETCH_AT_ACTIONS} queued actions"
echo "Timeout:         ${TIMEOUT_S}s"
echo "Log path:        ${LOG_PATH}"
echo
echo "Keep one hand near power/USB. Press Ctrl-C to stop."

args=(
  uvx --python "${UV_PYTHON}" --from 'lerobot[feetech]' python scripts/serving/clients/run_remote_act.py
  --url "${URL}"
  --task "${TASK}"
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
