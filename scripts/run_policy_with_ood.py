#!/usr/bin/env python
"""Run a lerobot policy with per-frame OOD detection.

A direct port of ``lerobot-record``'s policy-only control loop, minus the
dataset writer and plus an OOD score between observation and action. The
OOD path is purely additive: every score is logged, but the policy
action is always executed (log-only mode).

Targets ``lerobot==0.5.1`` (the stable PyPI release). The ``lerobot.rollout``
package was added on ``main`` after 0.5.1, so we wire the policy, robot,
and processors ourselves using the helpers that 0.5.1 already ships.

Usage
-----

::

    uv run python scripts/run_policy_with_ood.py \\
        --policy.path=ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1 \\
        --robot.type=so101_follower \\
        --robot.port=/dev/cu.usbmodem5C4C1268491 \\
        --robot.id=so101_follower \\
        --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}' \\
        --task='Pick up the tape and put it on the pink post-it.' \\
        --fps=30 --duration=30 \\
        --device=mps \\
        --ood_detector_path=models/ood_detector.npz \\
        --ood_camera=front
"""

import logging
import signal
import time
from dataclasses import dataclass
from threading import Event

import torch

# Importing camera configs registers them with draccus so they parse from CLI.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import (
    build_dataset_frame,
    combine_feature_dicts,
)
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots import (  # noqa: F401  — registers robot subclasses with draccus
    Robot,
    RobotConfig,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
)
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

from lerobot_ood import ACTBackboneEncoder, DinoV2Encoder, OODDetector, extract_camera_frame

logger = logging.getLogger("run_policy_with_ood")


@dataclass
class OODRunConfig:
    """CLI config — mirrors the subset of `lerobot-record` we need."""

    robot: RobotConfig | None = None
    policy: PreTrainedConfig | None = None
    fps: int = 30
    duration: float = 30.0
    task: str = ""
    device: str | None = None
    display_data: bool = False

    ood_detector_path: str = ""
    ood_camera: str = "front"
    # "act_backbone" (default) reuses the loaded policy's vision backbone.
    # "dinov2_vits14" / "dinov2_vitb14" / "dinov2_vitl14" run a separate model.
    ood_encoder: str = "act_backbone"
    # Print every Nth OOD detection (1 = every one).
    ood_log_every_n: int = 1
    # Periodically print in-dist scores too — useful for sanity-checking the threshold.
    ood_log_in_dist_every_n: int = 0

    def __post_init__(self):
        if self.robot is None:
            raise ValueError("--robot.type is required")
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        if self.policy is None:
            raise ValueError("--policy.path is required")
        if not self.ood_detector_path:
            raise ValueError("--ood_detector_path is required")
        if self.device is None:
            self.device = self.policy.device or ("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        # Enables `--policy.path=<repo_or_dir>` to load the saved config.
        return ["policy"]


def _build_ood_encoder(name: str, policy, device: str):
    if name == "act_backbone":
        return ACTBackboneEncoder(policy=policy, device=device)
    if name.startswith("dinov2_"):
        return DinoV2Encoder(model=name, device=device)
    raise ValueError(
        f"unknown --ood_encoder '{name}'. Expected 'act_backbone' or 'dinov2_<size>'."
    )


def _build_dataset_features(robot, teleop_action_processor, robot_observation_processor):
    """Reconstruct the dataset feature dict the same way lerobot-record does.

    Needed by ``build_dataset_frame`` to format the observation tensor
    that ``predict_action`` consumes.
    """
    # Only `.pos` joint features go to the policy as state/action.
    obs_features_hw = {
        k: v
        for k, v in robot.observation_features.items()
        if isinstance(v, tuple) or (v is float and k.endswith(".pos"))
    }
    action_features_hw = {
        k: v for k, v in robot.action_features.items() if k.endswith(".pos")
    }
    action_ds_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=True,
    )
    obs_ds_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=obs_features_hw),
        use_videos=True,
    )
    dataset_features = combine_feature_dicts(action_ds_features, obs_ds_features)
    ordered_action_keys = list(action_features_hw.keys())
    return dataset_features, ordered_action_keys


@parser.wrap()
def main(cfg: OODRunConfig) -> None:
    init_logging()

    # --- Detector + Ctrl-C handler set up before we touch the robot. ---
    detector = OODDetector.load(cfg.ood_detector_path)
    logger.info(
        "OOD detector loaded from %s (threshold=%.3f, pca_components=%s)",
        cfg.ood_detector_path,
        detector.threshold,
        detector.pca_components,
    )
    shutdown = Event()
    signal.signal(signal.SIGINT, lambda *_: shutdown.set())
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())

    # --- Policy (heavy: download + load weights). ---
    logger.info("Loading policy '%s' on %s ...", cfg.policy.pretrained_path, cfg.device)
    policy_class = get_policy_class(cfg.policy.type)
    policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=cfg.policy)
    policy.to(cfg.device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )

    # --- Robot processors + robot. ---
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    logger.info("Connecting robot (%s) ...", cfg.robot.type)
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot connected: %s", robot.name)

    dataset_features, ordered_action_keys = _build_dataset_features(
        robot, teleop_action_processor, robot_observation_processor
    )

    # --- OOD encoder needs the loaded policy (for act_backbone). ---
    encoder = _build_ood_encoder(cfg.ood_encoder, policy=policy, device=cfg.device)
    logger.info("OOD encoder: %s", cfg.ood_encoder)

    # --- Control loop. ---
    device = get_safe_torch_device(cfg.device)
    control_interval = 1.0 / cfg.fps
    n_seen, n_ood, sum_score = 0, 0, 0.0
    t_start = time.perf_counter()
    logger.info(
        "Inference loop starting (fps=%d, duration=%.0fs, ood_camera=%s)",
        cfg.fps,
        cfg.duration,
        cfg.ood_camera,
    )

    try:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

        while not shutdown.is_set():
            loop_start = time.perf_counter()
            if cfg.duration > 0 and (loop_start - t_start) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs_raw = robot.get_observation()

            # --- OOD scoring on the raw camera frame (never blocks the action). ---
            try:
                frame = extract_camera_frame(obs_raw, cfg.ood_camera)
            except KeyError as e:
                logger.error("OOD camera frame extraction failed: %s", e)
                shutdown.set()
                continue
            embedding = encoder(frame)
            result = detector.score(embedding)
            n_seen += 1
            sum_score += result.score
            if result.is_ood:
                n_ood += 1
                if cfg.ood_log_every_n > 0 and (n_ood % cfg.ood_log_every_n == 0):
                    logger.warning(
                        "[OOD] frame=%d  score=%.3f  threshold=%.3f  (%d/%d so far, %.1f%%)",
                        n_seen,
                        result.score,
                        result.threshold,
                        n_ood,
                        n_seen,
                        100.0 * n_ood / n_seen,
                    )
            elif cfg.ood_log_in_dist_every_n > 0 and n_seen % cfg.ood_log_in_dist_every_n == 0:
                logger.info(
                    "[in-dist] frame=%d  score=%.3f  threshold=%.3f",
                    n_seen,
                    result.score,
                    result.threshold,
                )

            # --- Normal inference path. ---
            obs_processed = robot_observation_processor(obs_raw)
            obs_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            action_tensor = predict_action(
                observation=obs_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=cfg.task,
                robot_type=robot.robot_type,
            )
            action_dict = make_robot_action(action_tensor, dataset_features)
            robot_action_to_send = robot_action_processor((action_dict, obs_raw))
            robot.send_action(robot_action_to_send)

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            if sleep_t > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    "loop running slow (%.1f Hz < target %d Hz); "
                    "OOD path adds encoder latency.",
                    1.0 / dt,
                    cfg.fps,
                )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if robot.is_connected:
            robot.disconnect()
        mean_score = sum_score / max(n_seen, 1)
        logger.info(
            "Run complete: %d frames, %d flagged OOD (%.1f%%), mean score=%.3f, threshold=%.3f",
            n_seen,
            n_ood,
            100.0 * n_ood / max(n_seen, 1),
            mean_score,
            detector.threshold,
        )


if __name__ == "__main__":
    main()
