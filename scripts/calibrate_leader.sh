#!/usr/bin/env bash
set -euo pipefail

TELEOP_PORT="${TELEOP_PORT:-/dev/cu.usbmodem5C4C1255491}"
TELEOP_ID="${TELEOP_ID:-so101_leader}"
UV_PYTHON="${UV_PYTHON:-3.12}"

exec uvx --python "${UV_PYTHON}" --from 'lerobot[feetech]' lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}" \
  --teleop.id="${TELEOP_ID}"
