"""Runnable: lerobot SO-101 + RemoteACTPolicy at fixed FPS.

Connects to a local SO-101 follower via lerobot, captures observations, runs
each one through a remote ACT inference server, and commands the robot at
the same FPS used during training (default 30 Hz).

The remote service returns 25-step action chunks; this script consumes them
sequentially via RemoteACTPolicy so the robot follows a smooth trajectory
between refreshes — same behavior as `lerobot-record` or `lerobot-eval`
running locally.

Safety is delegated to lerobot's standard `max_relative_target` on the
robot config (see `ensure_safe_goal_position` in `lerobot/robots/utils.py`).
The model's per-joint requested delta is clamped, never rejected, so the
robot always moves toward the target.

Example
-------

    python serving/clients/run_remote_act.py \\
        --url http://lerobot-act:8080/infer \\
        --robot.port /dev/ttyACM0 \\
        --robot.id my-so101 \\
        --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}' \\
        --robot.max_relative_target 5.0 \\
        --fps 30 \\
        --log_path /tmp/remote-act.jsonl

The robot config block (--robot.*) is parsed by lerobot's draccus. Note
that RunConfig.robot is typed concretely as SOFollowerRobotConfig, so
do NOT pass `--robot.type` — pass `--robot.port`, `--robot.cameras`,
etc. directly. ACT is vision + proprioception only and takes no
language input, so there is no --task flag either.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import signal
import sys
import time
from pathlib import Path

import draccus

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

# Register concrete CameraConfig subclasses with draccus's choice registry so
# `--robot.cameras='{cam: {type: opencv, ...}}'` parses. Matches the set that
# `lerobot/scripts/lerobot_record.py` imports for the same reason.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401

try:
    # When invoked as `python -m serving.clients.run_remote_act`.
    from .remote_act_policy import RemoteACTPolicy, SO101_JOINT_ORDER
except ImportError:
    # When invoked as `python run_remote_act.py` directly.
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from remote_act_policy import RemoteACTPolicy, SO101_JOINT_ORDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("vla.client")


@dataclasses.dataclass
class RunConfig:
    url: str
    robot: SOFollowerRobotConfig

    # Inference / consumption.
    chunk_size: int = 25
    n_action_steps: int = 25
    auth_token: str | None = None
    timeout_s: float = 5.0
    jpeg_quality: int = 90
    on_refill_failure: str = "hold"  # "hold" | "raise"

    # Loop.
    fps: int = 30
    log_path: str | None = None
    max_ticks: int | None = None


def _make_log_record(
    tick: int,
    t0: float,
    obs: dict,
    requested: dict[str, float],
    sent: dict[str, float],
    refill_calls: int,
    slow_tick_ms: float | None,
) -> dict:
    state = {k: float(obs[k]) for k in SO101_JOINT_ORDER}
    max_abs_delta = max(abs(requested[k] - state[k]) for k in SO101_JOINT_ORDER)
    return {
        "frame": tick,
        "time_s": t0,
        "state_robot": state,
        "local_action": requested,
        "processed_action": sent,
        "max_abs_delta": max_abs_delta,
        "sent": True,
        "refill_calls_total": refill_calls,
        "slow_tick_ms": slow_tick_ms,
    }


def main(cfg: RunConfig) -> int:
    if cfg.on_refill_failure not in ("hold", "raise"):
        raise SystemExit(f"on_refill_failure must be 'hold' or 'raise'")

    robot = SOFollower(cfg.robot)
    logger.info("connecting to robot on %s ...", cfg.robot.port)
    robot.connect()
    logger.info("connected. cameras=%s", list(robot.cameras.keys()))

    policy = RemoteACTPolicy(
        url=cfg.url,
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.n_action_steps,
        camera_keys=tuple(robot.cameras.keys()),
        auth_token=cfg.auth_token,
        timeout_s=cfg.timeout_s,
        jpeg_quality=cfg.jpeg_quality,
        on_refill_failure=cfg.on_refill_failure,
    )

    log_f = open(cfg.log_path, "w") if cfg.log_path else None
    if log_f:
        logger.info("logging per-tick JSONL to %s", cfg.log_path)

    stopping = False

    def _handle_sigint(_signum, _frame) -> None:
        nonlocal stopping
        if not stopping:
            logger.info("SIGINT received — stopping at next tick boundary")
            stopping = True
        else:
            logger.warning("second SIGINT — exiting immediately")
            raise SystemExit(130)

    signal.signal(signal.SIGINT, _handle_sigint)

    tick_period = 1.0 / cfg.fps
    next_tick = time.perf_counter() + tick_period
    tick = 0

    try:
        while not stopping and (cfg.max_ticks is None or tick < cfg.max_ticks):
            t0 = time.perf_counter()

            obs = robot.get_observation()
            requested = policy.select_action(obs)
            sent = robot.send_action(requested)

            elapsed = time.perf_counter() - t0
            slow_tick_ms: float | None = None
            if elapsed > tick_period:
                slow_tick_ms = elapsed * 1000.0
                logger.warning(
                    "tick %d took %.1f ms (>%.1f ms budget at %d Hz)",
                    tick, elapsed * 1000.0, tick_period * 1000.0, cfg.fps,
                )

            if log_f:
                rec = _make_log_record(
                    tick, t0, obs, requested, sent,
                    policy.refill_calls, slow_tick_ms,
                )
                log_f.write(json.dumps(rec) + "\n")
                log_f.flush()

            tick += 1
            # Drift-free sleep: pace to absolute next_tick, not relative wait.
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Behind schedule — skip ahead to catch up rather than fall further.
                next_tick = time.perf_counter()
            next_tick += tick_period
    finally:
        logger.info(
            "stopping: %d ticks, %d /infer calls (%d failed)",
            tick, policy.refill_calls, policy.refill_failures,
        )
        try:
            robot.disconnect()
        except Exception:
            logger.exception("disconnect failed")
        if log_f:
            log_f.close()
        policy.close()
    return 0


if __name__ == "__main__":
    cfg = draccus.parse(config_class=RunConfig)
    raise SystemExit(main(cfg))
