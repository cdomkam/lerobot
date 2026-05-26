#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Log local LeRobot policy actions for comparison with a remote inference server."""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import so_follower  # noqa: F401
from lerobot.rollout.configs import BaseStrategyConfig, RolloutConfig
from lerobot.rollout.context import build_rollout_context
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


logger = logging.getLogger("debug_policy_actions")


@dataclass
class DebugPolicyActionsConfig(RolloutConfig):
    output_path: str = "logs/policy_action_debug.jsonl"
    send_actions: bool = False
    log_every_n: int = 1


def _state_from_observation(obs: dict) -> dict[str, float]:
    return {key: float(value) for key, value in obs.items() if key.endswith(".pos")}


def _max_abs_delta(action: dict[str, float], state: dict[str, float]) -> float | None:
    deltas = [abs(action[key] - state[key]) for key in action if key in state]
    return max(deltas) if deltas else None


@parser.wrap()
def main(cfg: DebugPolicyActionsConfig) -> None:
    init_logging()
    if not isinstance(cfg.strategy, BaseStrategyConfig):
        raise ValueError(f"only --strategy.type=base is supported; got {cfg.strategy.type!r}")
    if cfg.log_every_n <= 0:
        raise ValueError("--log_every_n must be positive")

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event
    logger.info("Building rollout context (loads policy and connects robot)...")
    ctx = build_rollout_context(cfg, shutdown_event)

    robot = ctx.hardware.robot_wrapper
    processors = ctx.processors
    engine = ctx.policy.inference
    control_interval = 1.0 / cfg.fps

    engine.reset()
    engine.start()
    engine.resume()

    logger.info(
        "Debug action loop starting (policy=%s, fps=%d, duration=%ss, send_actions=%s, output=%s)",
        cfg.policy.pretrained_path,
        cfg.fps,
        cfg.duration,
        cfg.send_actions,
        output_path,
    )

    frames = 0
    actions = 0
    t_start = time.perf_counter()
    try:
        with output_path.open("a") as f:
            while not shutdown_event.is_set():
                loop_start = time.perf_counter()
                if cfg.duration > 0 and (loop_start - t_start) >= cfg.duration:
                    logger.info("Duration limit reached (%.1fs)", cfg.duration)
                    break

                t_obs_start = time.perf_counter()
                obs_raw = robot.get_observation()
                t_obs_ms = (time.perf_counter() - t_obs_start) * 1000
                state = _state_from_observation(obs_raw)
                obs_processed = processors.robot_observation_processor(obs_raw)
                engine.notify_observation(obs_processed)
                obs_frame = build_dataset_frame(ctx.data.dataset_features, obs_processed, prefix=OBS_STR)
                t_infer_start = time.perf_counter()
                action_tensor = engine.get_action(obs_frame)
                t_infer_ms = (time.perf_counter() - t_infer_start) * 1000

                frames += 1
                if action_tensor is not None:
                    ordered_keys = ctx.data.ordered_action_keys
                    action_dict = {k: float(action_tensor[i].item()) for i, k in enumerate(ordered_keys)}
                    processed_action = processors.robot_action_processor((action_dict, obs_raw))
                    processed_action = {k: float(v) for k, v in processed_action.items()}
                    max_delta = _max_abs_delta(processed_action, state)
                    actions += 1

                    record = {
                        "frame": frames,
                        "time_s": time.perf_counter() - t_start,
                        "policy_repo_id": str(cfg.policy.pretrained_path),
                        "task": cfg.task,
                        "state_robot": state,
                        "local_action": action_dict,
                        "processed_action": processed_action,
                        "max_abs_delta": max_delta,
                        "sent": cfg.send_actions,
                    }
                    f.write(json.dumps(record) + "\n")
                    f.flush()

                    t_send_ms = 0.0
                    if cfg.send_actions:
                        t_send_start = time.perf_counter()
                        robot.send_action(processed_action)
                        t_send_ms = (time.perf_counter() - t_send_start) * 1000

                    if actions % cfg.log_every_n == 0:
                        logger.info(
                            "frame=%d obs=%.0fms infer=%.0fms send=%.0fms max_abs_delta=%s sent=%s",
                            frames,
                            t_obs_ms,
                            t_infer_ms,
                            t_send_ms,
                            f"{max_delta:.3f}" if max_delta is not None else "n/a",
                            cfg.send_actions,
                        )

                dt = time.perf_counter() - loop_start
                sleep_t = control_interval - dt
                if sleep_t > 0:
                    precise_sleep(sleep_t)
                else:
                    logger.warning("loop running slow (%.1f Hz < target %.0f Hz)", 1.0 / dt, cfg.fps)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        engine.stop()
        inner_robot = robot.inner
        if inner_robot.is_connected:
            inner_robot.disconnect()
        teleop = ctx.hardware.teleop
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        logger.info("Debug complete: frames=%d actions=%d output=%s", frames, actions, output_path)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
