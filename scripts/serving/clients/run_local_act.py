"""Local equivalent of run_remote_act.py — same loop, same JSONL schema.

Loads an ACT checkpoint into the SOFollower process and drives the robot
at fixed FPS, writing the same per-tick and per-refill JSONL as the remote
runner so the two traces can be compared directly by
`scripts/analyze_act_traces.py`. The local "refill" event records MPS
inference time in `inference_ms` (no HTTP, no JPEG).
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

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401

try:
    from .local_act_policy import LocalACTPolicy, SO101_JOINT_ORDER
except ImportError:
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from local_act_policy import LocalACTPolicy, SO101_JOINT_ORDER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("vla.local")


@dataclasses.dataclass
class RunConfig:
    pretrained_path: str
    robot: SOFollowerRobotConfig

    device: str = "mps"
    chunk_size: int = 50
    n_action_steps: int = 50
    on_refill_failure: str = "hold"
    prefetch_at_actions: int = 20
    comm_retries: int = 3
    comm_retry_sleep_s: float = 0.02

    fps: int = 30
    log_path: str | None = None
    max_ticks: int | None = None
    run_tag: str = "local"


def _as_float_or_none(value) -> float | None:
    return None if value is None else float(value)


def _safe_action_from_observation(
    requested: dict[str, float],
    obs: dict,
    max_relative_target: float | None,
) -> tuple[dict[str, float], bool, float]:
    if max_relative_target is None:
        return requested, False, 0.0
    safe: dict[str, float] = {}
    engaged = False
    worst = 0.0
    for key, goal in requested.items():
        present = float(obs[key])
        delta = float(goal) - present
        clamped = max(min(delta, max_relative_target), -max_relative_target)
        if clamped != delta:
            engaged = True
            worst = max(worst, abs(delta) - max_relative_target)
        safe[key] = present + clamped
    return safe, engaged, worst


def _retry_io(label: str, fn, retries: int, sleep_s: float):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except (ConnectionError, TimeoutError) as exc:
            last_exc = exc
            if attempt >= retries:
                break
            logger.warning("%s failed (%s); retrying %d/%d", label, exc, attempt + 1, retries)
            time.sleep(sleep_s)
    raise last_exc


def _open_log(path: str | None, suffix: str = "") -> tuple[object | None, str | None]:
    if not path:
        return None, None
    p = Path(path)
    if suffix:
        p = p.with_suffix(suffix + p.suffix) if p.suffix else p.with_name(p.name + suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    return open(p, "w"), str(p)


def main(cfg: RunConfig) -> int:
    max_relative_target = _as_float_or_none(cfg.robot.max_relative_target)
    cfg.robot.max_relative_target = None
    robot = SOFollower(cfg.robot)
    logger.info("connecting to robot on %s ...", cfg.robot.port)
    robot.connect()
    logger.info("connected. cameras=%s", list(robot.cameras.keys()))

    policy = LocalACTPolicy(
        pretrained_path=cfg.pretrained_path,
        device=cfg.device,
        chunk_size=cfg.chunk_size,
        n_action_steps=cfg.n_action_steps,
        camera_keys=tuple(robot.cameras.keys()),
        on_refill_failure=cfg.on_refill_failure,
        prefetch_at_actions=cfg.prefetch_at_actions,
    )

    tick_log, tick_log_path = _open_log(cfg.log_path)
    refill_log, refill_log_path = _open_log(cfg.log_path, suffix=".refills")
    if tick_log:
        logger.info("logging per-tick JSONL to %s", tick_log_path)
        logger.info("logging refill JSONL to %s", refill_log_path)

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
    prev_tick_start: float | None = None

    try:
        while not stopping and (cfg.max_ticks is None or tick < cfg.max_ticks):
            t_tick_start = time.perf_counter()
            wall_tick_s = time.time()
            loop_dt_ms = (t_tick_start - prev_tick_start) * 1000.0 if prev_tick_start else None
            prev_tick_start = t_tick_start

            t0 = time.perf_counter()
            try:
                obs = _retry_io(
                    "get_observation",
                    robot.get_observation,
                    cfg.comm_retries,
                    cfg.comm_retry_sleep_s,
                )
            except (ConnectionError, TimeoutError) as exc:
                logger.error("get_observation failed after retries; skipping tick %d: %s", tick, exc)
                tick += 1
                time.sleep(tick_period)
                continue
            obs_read_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            requested = policy.select_action(obs)
            policy_select_ms = (time.perf_counter() - t0) * 1000.0
            queue_depth_after = policy.queue_depth
            was_holding = policy.last_was_holding
            chunk_seam_delta = policy.last_chunk_seam_delta
            prefetch_inflight = policy.prefetch_inflight
            obs_used_age_ms = policy.last_obs_age_ms

            safe_requested, clamp_engaged, clamp_worst_deg = _safe_action_from_observation(
                requested, obs, max_relative_target
            )

            t0 = time.perf_counter()
            try:
                sent = _retry_io(
                    "send_action",
                    lambda: robot.send_action(safe_requested),
                    cfg.comm_retries,
                    cfg.comm_retry_sleep_s,
                )
            except (ConnectionError, TimeoutError) as exc:
                logger.error("send_action failed after retries; skipping tick %d: %s", tick, exc)
                sent = {k: float(obs[k]) for k in SO101_JOINT_ORDER}
            motor_write_ms = (time.perf_counter() - t0) * 1000.0

            tick_total_ms = (time.perf_counter() - t_tick_start) * 1000.0
            slow_tick = tick_total_ms > tick_period * 1000.0
            if slow_tick:
                logger.warning(
                    "tick %d took %.1f ms (>%.1f ms budget at %d Hz)",
                    tick, tick_total_ms, tick_period * 1000.0, cfg.fps,
                )

            if refill_log:
                for ev in policy.pop_refill_events():
                    ev["tick"] = tick
                    ev["run_tag"] = cfg.run_tag
                    refill_log.write(json.dumps(ev) + "\n")
                refill_log.flush()

            if tick_log:
                state = {k: float(obs[k]) for k in SO101_JOINT_ORDER}
                max_abs_delta = max(abs(requested[k] - state[k]) for k in SO101_JOINT_ORDER)
                rec = {
                    "run_tag": cfg.run_tag,
                    "tick": tick,
                    "wall_s": wall_tick_s,
                    "loop_dt_ms": loop_dt_ms,
                    "obs_read_ms": obs_read_ms,
                    "policy_select_ms": policy_select_ms,
                    "motor_write_ms": motor_write_ms,
                    "tick_total_ms": tick_total_ms,
                    "slow_tick": slow_tick,
                    "queue_depth": queue_depth_after,
                    "was_holding": was_holding,
                    "prefetch_inflight": prefetch_inflight,
                    "chunk_seam_delta": chunk_seam_delta,
                    "obs_used_age_ms": obs_used_age_ms,
                    "max_abs_delta_deg": max_abs_delta,
                    "clamp_engaged": clamp_engaged,
                    "clamp_worst_deg": clamp_worst_deg,
                    "state_robot": state,
                    "action_requested": requested,
                    "action_sent": sent,
                    "action_safe": safe_requested,
                    "refill_calls_total": policy.refill_calls,
                    "refill_failures_total": policy.refill_failures,
                }
                tick_log.write(json.dumps(rec) + "\n")
                tick_log.flush()

            tick += 1
            sleep_for = next_tick - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.perf_counter()
            next_tick += tick_period
    finally:
        logger.info(
            "stopping: %d ticks, %d inferences (%d failed)",
            tick, policy.refill_calls, policy.refill_failures,
        )
        try:
            robot.disconnect()
        except Exception:
            logger.exception("disconnect failed")
        if tick_log:
            tick_log.close()
        if refill_log:
            refill_log.close()
        policy.close()
    return 0


if __name__ == "__main__":
    cfg = draccus.parse(config_class=RunConfig)
    raise SystemExit(main(cfg))
