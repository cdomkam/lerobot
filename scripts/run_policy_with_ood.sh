#!/usr/bin/env bash
set -euo pipefail

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
POLICY_REPO_ID="${POLICY_REPO_ID:-ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1}"
POLICY_DEVICE="${POLICY_DEVICE:-mps}"
UV_PYTHON="${UV_PYTHON:-3.12}"

OOD_DETECTOR_PATH="${OOD_DETECTOR_PATH:-models/ood_detector.npz}"
OOD_CAMERA="${OOD_CAMERA:-front}"
OOD_ENCODER="${OOD_ENCODER:-dinov2_vits14}"
OOD_LOG_IN_DIST_EVERY_N="${OOD_LOG_IN_DIST_EVERY_N:-0}"

TASK="${TASK:-Pick up the tape and put it on the pink post-it.}"
FPS="${FPS:-30}"
DURATION="${DURATION:-30}"

CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-0}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-1}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

CAMERAS="{ front: {type: opencv, index_or_path: ${CAMERA_FRONT_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}, side: {type: opencv, index_or_path: ${CAMERA_SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}} }"

if [[ ! -f "${OOD_DETECTOR_PATH}" ]]; then
  echo "Missing OOD detector at: ${OOD_DETECTOR_PATH}" >&2
  echo "Run scripts/fit_ood_detector.py first." >&2
  exit 1
fi

echo "Policy repo:     ${POLICY_REPO_ID}"
echo "Policy device:   ${POLICY_DEVICE}"
echo "OOD detector:    ${OOD_DETECTOR_PATH}"
echo "OOD encoder:     ${OOD_ENCODER}"
echo "OOD camera:      ${OOD_CAMERA}"
echo "Duration:        ${DURATION}s"
echo "Python:          ${UV_PYTHON}"
echo
echo "Keep one hand near power/USB. Press Ctrl-C to stop."

exec uv run --python "${UV_PYTHON}" python scripts/run_policy_with_ood.py \
  --strategy.type=base \
  --policy.path="${POLICY_REPO_ID}" \
  --device="${POLICY_DEVICE}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERAS}" \
  --task="${TASK}" \
  --fps="${FPS}" \
  --duration="${DURATION}" \
  --ood_detector_path="${OOD_DETECTOR_PATH}" \
  --ood_camera="${OOD_CAMERA}" \
  --ood_encoder="${OOD_ENCODER}" \
  --ood_log_in_dist_every_n="${OOD_LOG_IN_DIST_EVERY_N}"
