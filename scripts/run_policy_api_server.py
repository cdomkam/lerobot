#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Run an SO-101 policy served by a remote HTTP inference API."""

import json
import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests
from PIL import Image

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging


logger = logging.getLogger("run_policy_api_server")


@dataclass
class ApiPolicyConfig:
    robot: RobotConfig
    api_url: str = "http://154.54.100.64:8080/infer"
    token_env: str = "TOKEN"
    fallback_token_env: str = "API_TOKEN"
    fps: int = 50
    duration: float = 45.0
    request_timeout_s: float = 10.0
    # Start fetching the next 25-action chunk while this many actions remain.
    # At 50 Hz, 20 actions leaves ~400ms for the HTTP request to complete.
    prefetch_at_actions: int = 20
    action_log_every_n_chunks: int = 1
    # The local SO-101 follower uses degrees for the first 5 joints and 0..100 for gripper.
    # Most external VLA servers use radians for arm joints; convert by default.
    api_state_units: str = "radians"  # "robot" or "radians"
    api_action_units: str = "robot"  # "robot" or "radians"
    gripper_action_units: str = "robot"  # "robot" or "minus1_1"
    max_joint_step_deg: float = 0.5
    max_gripper_step: float = 1.0
    front_camera: str = "front"
    side_camera: str = "side"
    state_keys: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
    )


def _get_token(cfg: ApiPolicyConfig) -> str:
    token = os.environ.get(cfg.token_env) or os.environ.get(cfg.fallback_token_env)
    if not token:
        raise RuntimeError(
            f"Missing API token. Set {cfg.token_env}=... or {cfg.fallback_token_env}=..."
        )
    return token


def _camera_frame(obs: dict[str, Any], camera_name: str) -> np.ndarray:
    if camera_name not in obs:
        image_like = [k for k in obs if "image" in k.lower() or "camera" in k.lower()]
        raise KeyError(
            f"camera '{camera_name}' not found in observation. "
            f"Available image-like keys: {image_like}"
        )
    frame = np.asarray(obs[camera_name])
    if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"unexpected frame shape for '{camera_name}': {frame.shape}")
    return frame.astype(np.uint8, copy=False)


def _jpeg_bytes(frame: np.ndarray) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _state(obs: dict[str, Any], state_keys: list[str]) -> list[float]:
    missing = [key for key in state_keys if key not in obs]
    if missing:
        raise KeyError(f"missing state keys in robot observation: {missing}")
    return [float(obs[key]) for key in state_keys]


def _state_for_api(cfg: ApiPolicyConfig, obs: dict[str, Any]) -> list[float]:
    state = _state(obs, cfg.state_keys)
    if cfg.api_state_units == "robot":
        return state
    if cfg.api_state_units == "radians":
        converted = state.copy()
        converted[:5] = np.deg2rad(converted[:5]).tolist()
        return converted
    raise ValueError("--api_state_units must be 'robot' or 'radians'")


def _action_for_robot(cfg: ApiPolicyConfig, action: list[float]) -> list[float]:
    converted = [float(value) for value in action]
    if cfg.api_action_units == "robot":
        pass
    elif cfg.api_action_units == "radians":
        converted[:5] = np.rad2deg(converted[:5]).tolist()
    else:
        raise ValueError("--api_action_units must be 'robot' or 'radians'")

    if cfg.gripper_action_units == "robot":
        pass
    elif cfg.gripper_action_units == "minus1_1":
        converted[5] = float(np.clip((converted[5] + 1.0) * 50.0, 0.0, 100.0))
    else:
        raise ValueError("--gripper_action_units must be 'robot' or 'minus1_1'")
    return converted


def _limit_action_step(
    target: list[float],
    previous: list[float],
    max_joint_step_deg: float,
    max_gripper_step: float,
) -> list[float]:
    limited = previous.copy()
    for i in range(5):
        delta = target[i] - previous[i]
        limited[i] = previous[i] + float(np.clip(delta, -max_joint_step_deg, max_joint_step_deg))
    gripper_delta = target[5] - previous[5]
    limited[5] = previous[5] + float(np.clip(gripper_delta, -max_gripper_step, max_gripper_step))
    return limited


def _summarize_vector(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:.2f}" for value in values) + "]"


def _log_chunk_diagnostics(
    chunk_index: int,
    cfg: ApiPolicyConfig,
    state: list[float],
    actions: list[list[float]],
    request_ms: float,
    prefetched: bool,
) -> None:
    if cfg.action_log_every_n_chunks <= 0:
        return
    if chunk_index % cfg.action_log_every_n_chunks != 0:
        return
    if not actions:
        logger.warning("chunk %d returned no actions", chunk_index)
        return

    first = actions[0]
    last = actions[-1]
    first_robot = _action_for_robot(cfg, first)
    last_robot = _action_for_robot(cfg, last)
    first_delta = [action - current for action, current in zip(first_robot, state, strict=True)]
    max_abs_delta = max(abs(delta) for delta in first_delta)
    source = "prefetched" if prefetched else "direct"
    logger.info(
        "chunk %d (%s, %.1fms): state_robot=%s first_api=%s first_robot=%s last_robot=%s first_delta=%s max_abs_delta=%.2f",
        chunk_index,
        source,
        request_ms,
        _summarize_vector(state),
        _summarize_vector(first),
        _summarize_vector(first_robot),
        _summarize_vector(last_robot),
        _summarize_vector(first_delta),
        max_abs_delta,
    )


def _decode_action(response: requests.Response) -> list[list[float]]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"inference API returned non-JSON response: {response.text[:500]}") from exc

    action = payload.get("action", payload) if isinstance(payload, dict) else payload
    arr = np.asarray(action, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected action shape [6] or [T, 6], got {arr.shape}")
    if arr.shape[1] != 6:
        raise ValueError(f"expected 6 action dimensions, got {arr.shape[1]}")
    return arr.tolist()


def _request_actions(
    cfg: ApiPolicyConfig,
    token: str,
    session: requests.Session,
    obs: dict[str, Any],
) -> list[list[float]]:
    front = _camera_frame(obs, cfg.front_camera)
    side = _camera_frame(obs, cfg.side_camera)
    state = _state_for_api(cfg, obs)

    files = {
        "front": ("front.jpg", _jpeg_bytes(front), "image/jpeg"),
        "side": ("side.jpg", _jpeg_bytes(side), "image/jpeg"),
    }
    data = {"state": json.dumps(state)}
    headers = {"Authorization": f"Bearer {token}"}
    response = session.post(
        cfg.api_url,
        headers=headers,
        files=files,
        data=data,
        timeout=cfg.request_timeout_s,
    )
    response.raise_for_status()
    return _decode_action(response)


def _timed_request_actions(
    cfg: ApiPolicyConfig,
    token: str,
    session: requests.Session,
    obs: dict[str, Any],
) -> tuple[list[list[float]], float]:
    start = time.perf_counter()
    actions = _request_actions(cfg, token, session, obs)
    elapsed_ms = (time.perf_counter() - start) * 1e3
    return actions, elapsed_ms


@parser.wrap()
def main(cfg: ApiPolicyConfig) -> None:
    init_logging()
    token = _get_token(cfg)
    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

    robot = make_robot_from_config(cfg.robot)
    action_keys = list(cfg.state_keys)
    action_queue: list[list[float]] = []
    pending_actions: Future[tuple[list[list[float]], float]] | None = None
    pending_state: list[float] | None = None
    current_target: list[float] | None = None
    control_interval = 1.0 / cfg.fps
    t_start = time.perf_counter()
    chunk_index = 0

    logger.info("Connecting robot and cameras...")
    robot.connect()
    session = requests.Session()
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        logger.info("API inference loop starting (url=%s, fps=%d)", cfg.api_url, cfg.fps)
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()
            if cfg.duration > 0 and (loop_start - t_start) >= cfg.duration:
                logger.info("Duration limit reached (%.1fs)", cfg.duration)
                break

            if not action_queue:
                if pending_actions is not None:
                    action_queue, request_ms = pending_actions.result()
                    pending_actions = None
                    chunk_index += 1
                    logger.info("received %d prefetched API actions in %.1fms", len(action_queue), request_ms)
                    if pending_state is not None:
                        _log_chunk_diagnostics(
                            chunk_index, cfg, pending_state, action_queue, request_ms, prefetched=True
                        )
                    pending_state = None
                else:
                    obs = robot.get_observation()
                    state = _state(obs, cfg.state_keys)
                    if current_target is None:
                        current_target = state
                    action_queue, request_ms = _timed_request_actions(cfg, token, session, obs)
                    chunk_index += 1
                    logger.info("received %d API actions in %.1fms", len(action_queue), request_ms)
                    _log_chunk_diagnostics(
                        chunk_index, cfg, state, action_queue, request_ms, prefetched=False
                    )

            action_values = action_queue.pop(0)
            action_values = _action_for_robot(cfg, action_values)
            if current_target is None:
                obs = robot.get_observation()
                current_target = _state(obs, cfg.state_keys)
            action_values = _limit_action_step(
                action_values,
                current_target,
                max_joint_step_deg=cfg.max_joint_step_deg,
                max_gripper_step=cfg.max_gripper_step,
            )
            action = {key: float(value) for key, value in zip(action_keys, action_values, strict=True)}
            robot.send_action(action)
            current_target = action_values

            if (
                cfg.prefetch_at_actions > 0
                and pending_actions is None
                and 0 < len(action_queue) <= cfg.prefetch_at_actions
            ):
                obs = robot.get_observation()
                pending_state = _state(obs, cfg.state_keys)
                pending_actions = executor.submit(_timed_request_actions, cfg, token, session, obs)
                logger.info("started API prefetch with %d queued actions remaining", len(action_queue))

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            if sleep_t > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning("loop running slow (%.1f Hz < target %.0f Hz)", 1.0 / dt, cfg.fps)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if robot.is_connected:
            robot.disconnect()
        session.close()
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
