#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
LEROBOT_MAIN_DIR="${LEROBOT_MAIN_DIR:-${ROOT_DIR}/vendor/lerobot-main}"
if [[ "${LEROBOT_MAIN_DIR}" != /* ]]; then
  LEROBOT_MAIN_DIR="${ROOT_DIR}/${LEROBOT_MAIN_DIR}"
fi

usage() {
  cat >&2 <<'USAGE'
Usage: ./scripts/run_policy_with_ood.sh [--no-voice]

Runs the SO-101 OOD policy wrapper. ElevenLabs voice alerts are enabled by
default. Use --no-voice for testing without TTS credentials or speaker output.
USAGE
}

NO_VOICE=false
for arg in "$@"; do
  case "${arg}" in
    --no-voice)
      NO_VOICE=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      echo "Unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
POLICY_REPO_ID="${POLICY_REPO_ID:-ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1}"
POLICY_DEVICE="${POLICY_DEVICE:-mps}"
UV_PYTHON="${UV_PYTHON:-3.12}"

OOD_DETECTOR_PATH="${OOD_DETECTOR_PATH:-models/ood_detector.npz}"
OOD_CAMERA="${OOD_CAMERA:-front}"
OOD_ENCODER="${OOD_ENCODER:-act_backbone}"
OOD_LOG_IN_DIST_EVERY_N="${OOD_LOG_IN_DIST_EVERY_N:-0}"
OOD_TTS_ENABLED="${OOD_TTS_ENABLED:-true}"
OOD_TTS_CONFIG_PATH="${OOD_TTS_CONFIG_PATH:-.env}"
OOD_TTS_EVERY_N="${OOD_TTS_EVERY_N:-30}"
OOD_TTS_QUEUE_MAX="${OOD_TTS_QUEUE_MAX:-25}"

if [[ "${NO_VOICE}" == "true" ]]; then
  OOD_TTS_ENABLED=false
fi

TASK="${TASK:-Pick up the tape and put it on the pink post-it.}"
FPS="${FPS:-30}"
DURATION="${DURATION:-30}"

CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-0}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-1}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

if [[ "${OOD_DETECTOR_PATH}" != /* ]]; then
  OOD_DETECTOR_PATH="${ROOT_DIR}/${OOD_DETECTOR_PATH}"
fi
if [[ "${OOD_TTS_CONFIG_PATH}" != /* ]]; then
  OOD_TTS_CONFIG_PATH="${ROOT_DIR}/${OOD_TTS_CONFIG_PATH}"
fi

CAMERAS="{ front: {type: opencv, index_or_path: ${CAMERA_FRONT_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}, side: {type: opencv, index_or_path: ${CAMERA_SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}} }"

if [[ "${OOD_TTS_ENABLED}" == "true" ]]; then
  OOD_TTS_CONFIG_PATH="${OOD_TTS_CONFIG_PATH}" ROOT_DIR="${ROOT_DIR}" python3 - <<'PY'
import importlib.util
import os
import sys
from pathlib import Path

path = Path(os.environ["ROOT_DIR"]) / "src" / "lerobot_ood" / "tts.py"
spec = importlib.util.spec_from_file_location("lerobot_ood_tts_preflight", path)
tts = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tts
assert spec.loader is not None
spec.loader.exec_module(tts)

try:
    tts.load_elevenlabs_tts_config(os.environ["OOD_TTS_CONFIG_PATH"])
except Exception as exc:
    print(str(exc), file=sys.stderr)
    sys.exit(1)
PY
fi

if [[ ! -d "${LEROBOT_MAIN_DIR}" ]]; then
  echo "Missing latest LeRobot checkout: ${LEROBOT_MAIN_DIR}" >&2
  echo "Expected it at vendor/lerobot-main." >&2
  exit 1
fi

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
echo "OOD TTS:         ${OOD_TTS_ENABLED}"
echo "Duration:        ${DURATION}s"
echo "Python:          ${UV_PYTHON}"
echo
echo "Keep one hand near power/USB. Press Ctrl-C to stop."

cd "${LEROBOT_MAIN_DIR}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec uv run --project "${LEROBOT_MAIN_DIR}" --python "${UV_PYTHON}" \
  --extra dataset --extra hardware --extra viz --extra feetech \
  --with numpy --with scikit-learn --with torchvision \
  python "${ROOT_DIR}/scripts/run_policy_with_ood.py" \
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
  --ood_log_in_dist_every_n="${OOD_LOG_IN_DIST_EVERY_N}" \
  --ood_tts_enabled="${OOD_TTS_ENABLED}" \
  --ood_tts_config_path="${OOD_TTS_CONFIG_PATH}" \
  --ood_tts_every_n="${OOD_TTS_EVERY_N}" \
  --ood_tts_queue_max="${OOD_TTS_QUEUE_MAX}"
