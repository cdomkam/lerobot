#!/usr/bin/env bash
set -euo pipefail

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
UV_PYTHON="${UV_PYTHON:-3.12}"

exec uvx --python "${UV_PYTHON}" --from 'lerobot[feetech]' lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}"
