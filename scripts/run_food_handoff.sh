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
Usage: ./scripts/run_food_handoff.sh [--no-voice] [--no-stt --target strawberry|oreo|marshmallow]
       ./scripts/run_food_handoff.sh --test-mode --test-audio recordings/voice_requests/strawberry_request.wav --no-voice

Loops over handoff cycles: record speech or use --target, classify/select the
food policy, run it immediately, detect placement/OOD, speak the outcome, then
pause before the next cycle. Use --max-cycles N to stop after N cycles; 0 means
unlimited.
USAGE
}

NO_VOICE=false
NO_STT=false
TEST_MODE=false
TEST_AUDIO_PATH="${TEST_AUDIO_PATH:-}"
TARGET="${TARGET:-}"
MAX_CYCLES="${MAX_CYCLES:-}"
RESET_PAUSE_S="${RESET_PAUSE_S:-7}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-voice)
      NO_VOICE=true
      shift
      ;;
    --no-stt)
      NO_STT=true
      shift
      ;;
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --test-mode)
      TEST_MODE=true
      shift
      ;;
    --test-audio)
      TEST_AUDIO_PATH="${2:-}"
      shift 2
      ;;
    --max-cycles)
      MAX_CYCLES="${2:-}"
      shift 2
      ;;
    --reset-pause-s)
      RESET_PAUSE_S="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
POLICY_DEVICE="${POLICY_DEVICE:-mps}"
UV_PYTHON="${UV_PYTHON:-3.12}"

FOOD_POLICY_CONFIG="${FOOD_POLICY_CONFIG:-config/food_policies.json}"
VISION_CONFIG="${VISION_CONFIG:-config/food_handoff_vision.json}"
OOD_DETECTOR_PATH="${OOD_DETECTOR_PATH:-models/ood_detector.npz}"
OOD_ENABLED="${OOD_ENABLED:-true}"
OOD_CAMERA="${OOD_CAMERA:-front}"
OOD_ENCODER="${OOD_ENCODER:-act_backbone}"
OOD_EVERY_N="${OOD_EVERY_N:-5}"
OOD_LOG_IN_DIST_EVERY_N="${OOD_LOG_IN_DIST_EVERY_N:-0}"
OOD_FAILURE_AFTER_N="${OOD_FAILURE_AFTER_N:-3}"
OOD_TTS_ENABLED="${OOD_TTS_ENABLED:-true}"
OOD_TTS_CONFIG_PATH="${OOD_TTS_CONFIG_PATH:-.env}"
OOD_TTS_EVERY_N="${OOD_TTS_EVERY_N:-3}"
OOD_TTS_QUEUE_MAX="${OOD_TTS_QUEUE_MAX:-25}"
OPENAI_SUCCESS_ENABLED="${OPENAI_SUCCESS_ENABLED:-true}"
OPENAI_SUCCESS_CONFIG_PATH="${OPENAI_SUCCESS_CONFIG_PATH:-.env}"
OPENAI_SUCCESS_EVERY_N="${OPENAI_SUCCESS_EVERY_N:-15}"
OPENAI_SUCCESS_SEQUENCE_FRAMES="${OPENAI_SUCCESS_SEQUENCE_FRAMES:-5}"
OPENAI_SUCCESS_SEQUENCE_STRIDE="${OPENAI_SUCCESS_SEQUENCE_STRIDE:-5}"
OPENAI_SUCCESS_GRACE_S="${OPENAI_SUCCESS_GRACE_S:-15}"
ROBOT_NEUTRAL_RESET_ENABLED="${ROBOT_NEUTRAL_RESET_ENABLED:-true}"
ROBOT_NEUTRAL_CONFIG="${ROBOT_NEUTRAL_CONFIG:-config/robot_neutral.json}"
ROBOT_NEUTRAL_RESET_DURATION_S="${ROBOT_NEUTRAL_RESET_DURATION_S:-3.0}"
ROBOT_NEUTRAL_RESET_FPS="${ROBOT_NEUTRAL_RESET_FPS:-30}"
RESULT_JSON_PATH="${RESULT_JSON_PATH:-}"
STT_ENABLED="${STT_ENABLED:-true}"
REQUEST_AUDIO_SECONDS="${REQUEST_AUDIO_SECONDS:-3}"
REQUEST_AUDIO_SAMPLE_RATE="${REQUEST_AUDIO_SAMPLE_RATE:-16000}"
REQUEST_RECORDER_COMMAND="${REQUEST_RECORDER_COMMAND:-}"
TEST_POLICY_STEPS="${TEST_POLICY_STEPS:-5}"
TEST_SUCCESS_AFTER_STEPS="${TEST_SUCCESS_AFTER_STEPS:-3}"

if [[ "${MAX_CYCLES}" == "" ]]; then
  if [[ "${TEST_MODE}" == "true" ]]; then
    MAX_CYCLES=1
  else
    MAX_CYCLES=0
  fi
fi

if [[ "${TEST_MODE}" == "true" ]]; then
  if [[ "${FOOD_POLICY_CONFIG}" == "config/food_policies.json" ]]; then
    FOOD_POLICY_CONFIG="config/food_policies.test.json"
  fi
fi

if [[ "${NO_VOICE}" == "true" ]]; then
  OOD_TTS_ENABLED=false
fi
if [[ "${NO_STT}" == "true" ]]; then
  STT_ENABLED=false
fi

FPS="${FPS:-30}"
DURATION="${DURATION:-30}"
CAMERA_FRONT_INDEX="${CAMERA_FRONT_INDEX:-1}"
CAMERA_SIDE_INDEX="${CAMERA_SIDE_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"

for path_var in FOOD_POLICY_CONFIG VISION_CONFIG OOD_DETECTOR_PATH OOD_TTS_CONFIG_PATH OPENAI_SUCCESS_CONFIG_PATH; do
  value="${!path_var}"
  if [[ "${value}" != /* ]]; then
    printf -v "${path_var}" '%s/%s' "${ROOT_DIR}" "${value}"
  fi
done

if [[ "${TEST_MODE}" != "true" && ! -d "${LEROBOT_MAIN_DIR}" ]]; then
  echo "Missing latest LeRobot checkout: ${LEROBOT_MAIN_DIR}" >&2
  echo "Expected it at vendor/lerobot-main." >&2
  exit 1
fi

if [[ ! -f "${FOOD_POLICY_CONFIG}" ]]; then
  echo "Missing food policy config: ${FOOD_POLICY_CONFIG}" >&2
  echo "Expected config/food_policies.json with filled policy repo ids." >&2
  exit 1
fi

if [[ ! -f "${VISION_CONFIG}" ]]; then
  echo "Missing vision config: ${VISION_CONFIG}" >&2
  echo "Expected config/food_handoff_vision.json with tuned scene-camera ROIs." >&2
  exit 1
fi

if [[ "${TEST_MODE}" != "true" && ! -f "${OOD_DETECTOR_PATH}" ]]; then
  echo "Global OOD detector not found at: ${OOD_DETECTOR_PATH}" >&2
  echo "Continuing; a per-target ood_detector_path in the food config may be used instead." >&2
fi

if [[ "${TEST_AUDIO_PATH}" != "" && "${TEST_AUDIO_PATH}" != /* ]]; then
  TEST_AUDIO_PATH="${ROOT_DIR}/${TEST_AUDIO_PATH}"
fi

BOOTSTRAP_TARGET="${TARGET:-strawberry}"
BOOTSTRAP_POLICY_REPO_ID="$(
  FOOD_POLICY_CONFIG="${FOOD_POLICY_CONFIG}" BOOTSTRAP_TARGET="${BOOTSTRAP_TARGET}" python3 - <<'PY'
import json
import os
import sys

config_path = os.environ["FOOD_POLICY_CONFIG"]
target = os.environ["BOOTSTRAP_TARGET"]
with open(config_path) as f:
    targets = json.load(f)["targets"]

candidate = targets.get(target) or targets.get("strawberry") or next(iter(targets.values()))
repo_id = candidate.get("policy_repo_id", "")
if not repo_id or repo_id.startswith("TODO_"):
    for candidate in targets.values():
        repo_id = candidate.get("policy_repo_id", "")
        if repo_id and not repo_id.startswith("TODO_"):
            break

if not repo_id or repo_id.startswith("TODO_"):
    print("No non-TODO policy_repo_id found in food policy config", file=sys.stderr)
    sys.exit(1)
print(repo_id)
PY
)"

CAMERAS="{ front: {type: opencv, index_or_path: ${CAMERA_FRONT_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}}, side: {type: opencv, index_or_path: ${CAMERA_SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}} }"

echo "Food config:     ${FOOD_POLICY_CONFIG}"
echo "Vision config:   ${VISION_CONFIG}"
echo "Policy device:   ${POLICY_DEVICE}"
echo "Bootstrap policy:${BOOTSTRAP_POLICY_REPO_ID}"
echo "OOD detector:    ${OOD_DETECTOR_PATH}"
echo "OOD enabled:     ${OOD_ENABLED}"
echo "OOD every N:     ${OOD_EVERY_N}"
echo "OOD fail after:  ${OOD_FAILURE_AFTER_N}"
echo "OpenAI success:  ${OPENAI_SUCCESS_ENABLED}"
echo "OpenAI sequence: ${OPENAI_SUCCESS_SEQUENCE_FRAMES} frame(s), stride ${OPENAI_SUCCESS_SEQUENCE_STRIDE}"
echo "OpenAI grace:    ${OPENAI_SUCCESS_GRACE_S}s"
echo "Neutral reset:   ${ROBOT_NEUTRAL_RESET_ENABLED}"
echo "Neutral config:  ${ROBOT_NEUTRAL_CONFIG}"
echo "Voice enabled:   ${OOD_TTS_ENABLED}"
echo "STT enabled:     ${STT_ENABLED}"
echo "Target override: ${TARGET:-none}"
echo "Test mode:       ${TEST_MODE}"
echo "Test audio:      ${TEST_AUDIO_PATH:-none}"
echo "Front camera:    index=${CAMERA_FRONT_INDEX} ${CAMERA_WIDTH}x${CAMERA_HEIGHT}@${FPS}"
echo "Side camera:     index=${CAMERA_SIDE_INDEX} ${CAMERA_WIDTH}x${CAMERA_HEIGHT}@${FPS}"
echo "Max cycles:      ${MAX_CYCLES} (0=unlimited)"
echo "Reset pause:     ${RESET_PAUSE_S}s"
echo "Duration:        ${DURATION}s"
echo "Result JSON:     ${RESULT_JSON_PATH:-none}"
echo
if [[ "${TEST_MODE}" == "true" ]]; then
  echo "Test mode uses mocked policy, action, success, and OOD paths."
else
  echo "Keep power/USB within reach. Press Ctrl-C to stop."
fi

if [[ "${TEST_MODE}" == "true" ]]; then
  exec uv run --project "${ROOT_DIR}" --python "${UV_PYTHON}" \
    --with numpy --with scikit-learn --with torchvision --with opencv-python --with sounddevice \
    python "${ROOT_DIR}/scripts/run_food_handoff_test.py" \
    --food-policy-config="${FOOD_POLICY_CONFIG}" \
    --target="${TARGET}" \
    --stt-enabled="${STT_ENABLED}" \
    --test-audio-path="${TEST_AUDIO_PATH}" \
    --tts-enabled="${OOD_TTS_ENABLED}" \
    --tts-config-path="${OOD_TTS_CONFIG_PATH}" \
    --fps="${FPS}" \
    --max-cycles="${MAX_CYCLES}" \
    --reset-pause-s="${RESET_PAUSE_S}" \
    --result-json-path="${RESULT_JSON_PATH}" \
    --test-policy-steps="${TEST_POLICY_STEPS}" \
    --test-success-after-steps="${TEST_SUCCESS_AFTER_STEPS}"
fi

cd "${LEROBOT_MAIN_DIR}"
UV_PROJECT_DIR="${LEROBOT_MAIN_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

UV_RUN=(uv run --project "${UV_PROJECT_DIR}" --python "${UV_PYTHON}")
UV_RUN+=(--extra dataset --extra hardware --extra viz --extra feetech)
UV_RUN+=(--with numpy --with scikit-learn --with torchvision --with opencv-python --with sounddevice)

exec "${UV_RUN[@]}" python "${ROOT_DIR}/scripts/run_food_handoff.py" \
  --strategy.type=base \
  --policy.path="${BOOTSTRAP_POLICY_REPO_ID}" \
  --device="${POLICY_DEVICE}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="${CAMERAS}" \
  --task="placeholder selected after request" \
  --fps="${FPS}" \
  --duration="${DURATION}" \
  --food_policy_config="${FOOD_POLICY_CONFIG}" \
  --vision_config="${VISION_CONFIG}" \
  --target="${TARGET}" \
  --stt_enabled="${STT_ENABLED}" \
  --request_audio_seconds="${REQUEST_AUDIO_SECONDS}" \
  --request_audio_sample_rate="${REQUEST_AUDIO_SAMPLE_RATE}" \
  --request_recorder_command="${REQUEST_RECORDER_COMMAND}" \
  --ood_enabled="${OOD_ENABLED}" \
  --ood_detector_path="${OOD_DETECTOR_PATH}" \
  --ood_camera="${OOD_CAMERA}" \
  --ood_encoder="${OOD_ENCODER}" \
  --ood_every_n="${OOD_EVERY_N}" \
  --ood_log_in_dist_every_n="${OOD_LOG_IN_DIST_EVERY_N}" \
  --ood_failure_after_n="${OOD_FAILURE_AFTER_N}" \
  --ood_tts_enabled="${OOD_TTS_ENABLED}" \
  --ood_tts_config_path="${OOD_TTS_CONFIG_PATH}" \
  --ood_tts_every_n="${OOD_TTS_EVERY_N}" \
  --ood_tts_queue_max="${OOD_TTS_QUEUE_MAX}" \
  --openai_success_enabled="${OPENAI_SUCCESS_ENABLED}" \
  --openai_success_config_path="${OPENAI_SUCCESS_CONFIG_PATH}" \
  --openai_success_every_n="${OPENAI_SUCCESS_EVERY_N}" \
  --openai_success_sequence_frames="${OPENAI_SUCCESS_SEQUENCE_FRAMES}" \
  --openai_success_sequence_stride="${OPENAI_SUCCESS_SEQUENCE_STRIDE}" \
  --openai_success_grace_s="${OPENAI_SUCCESS_GRACE_S}" \
  --robot_neutral_reset_enabled="${ROBOT_NEUTRAL_RESET_ENABLED}" \
  --robot_neutral_config="${ROBOT_NEUTRAL_CONFIG}" \
  --robot_neutral_reset_duration_s="${ROBOT_NEUTRAL_RESET_DURATION_S}" \
  --robot_neutral_reset_fps="${ROBOT_NEUTRAL_RESET_FPS}" \
  --test_mode="${TEST_MODE}" \
  --test_audio_path="${TEST_AUDIO_PATH}" \
  --max_cycles="${MAX_CYCLES}" \
  --reset_pause_s="${RESET_PAUSE_S}" \
  --result_json_path="${RESULT_JSON_PATH}" \
  --test_policy_steps="${TEST_POLICY_STEPS}" \
  --test_success_after_steps="${TEST_SUCCESS_AFTER_STEPS}"
