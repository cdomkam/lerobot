#!/usr/bin/env bash
set -euo pipefail

ROBOT_PORT="${ROBOT_PORT:-/dev/cu.usbmodem5C4C1268491}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-/dev/cu.usbmodem5C4C1255491}"
TELEOP_ID="${TELEOP_ID:-so101_leader}"
UV_PYTHON="${UV_PYTHON:-3.12}"

exec uvx --python "${UV_PYTHON}" --from 'lerobot[feetech]' lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id="${TELEOP_ID}"
