#!/usr/bin/env bash
# Local equivalent of run_remote_act.sh — same loop, same JSONL schema,
# inference runs on MPS in-process. Use to compare apples-to-apples against
# the remote HTTP path.
set -euo pipefail

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
PRETRAINED_PATH="${PRETRAINED_PATH:-ofcourseistillloveyou/pick_up_red_lid_ac_50}"
DEVICE="${DEVICE:-mps}"
FPS="${FPS:-30}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-5.0}"
LOG_PATH="${LOG_PATH:-/tmp/local-act.jsonl}"
RUN_TAG="${RUN_TAG:-local}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"
# PREFETCH_AT_ACTIONS=0 → block at queue empty (matches lerobot-record / run_policy_on_robot.sh).
# >0 → kick off the next inference in the background when this many actions remain in the queue.
# Locally inference is ~90 ms on MPS, so blocking is fine and the chunk is computed from a fresh
# observation. Set higher only if you want to test the "stale observation" effect that the
# remote path inherently has.
PREFETCH_AT_ACTIONS="${PREFETCH_AT_ACTIONS:-0}"
COMM_RETRIES="${COMM_RETRIES:-3}"
COMM_RETRY_SLEEP_S="${COMM_RETRY_SLEEP_S:-0.02}"
ON_REFILL_FAILURE="${ON_REFILL_FAILURE:-hold}"

CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-0}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-1}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

CAMERAS="{front: {type: opencv, index_or_path: ${CAMERA_FRONT_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}, side: {type: opencv, index_or_path: ${CAMERA_SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}}"

UV_PYTHON="${UV_PYTHON:-3.12}"

echo "Pretrained:      ${PRETRAINED_PATH}"
echo "Device:          ${DEVICE}"
echo "Robot port:      ${ROBOT_PORT}"
echo "Robot id:        ${ROBOT_ID}"
echo "Cameras:         front=${CAMERA_FRONT_INDEX}, side=${CAMERA_SIDE_INDEX}, ${CAMERA_WIDTH}x${CAMERA_HEIGHT}@${FPS}"
echo "Max relative:    ${MAX_RELATIVE_TARGET}"
echo "Chunk/actions:   ${CHUNK_SIZE}/${N_ACTION_STEPS}"
echo "Prefetch at:     ${PREFETCH_AT_ACTIONS} queued actions"
echo "Log path:        ${LOG_PATH}"
echo
echo "Keep one hand near power/USB. Press Ctrl-C to stop."

exec uvx --python "${UV_PYTHON}" --from 'lerobot[feetech]' \
  python scripts/serving/clients/run_local_act.py \
  --pretrained_path "${PRETRAINED_PATH}" \
  --device "${DEVICE}" \
  --robot.port "${ROBOT_PORT}" \
  --robot.id "${ROBOT_ID}" \
  --robot.cameras="${CAMERAS}" \
  --robot.max_relative_target "${MAX_RELATIVE_TARGET}" \
  --chunk_size "${CHUNK_SIZE}" \
  --n_action_steps "${N_ACTION_STEPS}" \
  --prefetch_at_actions "${PREFETCH_AT_ACTIONS}" \
  --comm_retries "${COMM_RETRIES}" \
  --comm_retry_sleep_s "${COMM_RETRY_SLEEP_S}" \
  --on_refill_failure "${ON_REFILL_FAILURE}" \
  --fps "${FPS}" \
  --log_path "${LOG_PATH}" \
  --run_tag "${RUN_TAG}"
