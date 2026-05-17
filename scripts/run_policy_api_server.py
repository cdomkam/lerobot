#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Run an SO-101 policy served by a remote HTTP inference API."""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests
from PIL import Image

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import make_default_processors
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
    state = _state(obs, cfg.state_keys)

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


@parser.wrap()
def main(cfg: ApiPolicyConfig) -> None:
    init_logging()
    token = _get_token(cfg)
    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

    robot = make_robot_from_config(cfg.robot)
    _, robot_action_processor, _ = make_default_processors()
    action_keys = list(cfg.state_keys)
    action_queue: list[list[float]] = []
    control_interval = 1.0 / cfg.fps
    t_start = time.perf_counter()

    logger.info("Connecting robot and cameras...")
    robot.connect()
    session = requests.Session()

    try:
        logger.info("API inference loop starting (url=%s, fps=%d)", cfg.api_url, cfg.fps)
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()
            if cfg.duration > 0 and (loop_start - t_start) >= cfg.duration:
                logger.info("Duration limit reached (%.1fs)", cfg.duration)
                break

            obs = robot.get_observation()
            if not action_queue:
                action_queue = _request_actions(cfg, token, session, obs)

            action_values = action_queue.pop(0)
            action = {key: float(value) for key, value in zip(action_keys, action_values, strict=True)}
            processed_action = robot_action_processor((action, obs))
            robot.send_action(processed_action)

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


if __name__ == "__main__":
    register_third_party_plugins()
    main()
