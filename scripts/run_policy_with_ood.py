#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Run a lerobot policy with per-frame OOD detection.

This is the same control loop pattern as ``lerobot-rollout --strategy.type=base``,
unrolled here so we can drop an OOD score between observation and action.
The OOD path is purely additive: every score is logged, but the policy
action is always executed regardless of the result (log-only mode).

Usage
-----

::

    uv run python scripts/run_policy_with_ood.py \\
        --strategy.type=base \\
        --policy.path=ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1 \\
        --device=mps \\
        --robot.type=so101_follower \\
        --robot.port=/dev/cu.usbmodem5C4C1268491 \\
        --robot.id=so101_follower \\
        --robot.cameras='{front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
        --task='Pick up the tape and put it on the pink post-it.' \\
        --fps=30 --duration=30 \\
        --ood_detector_path=models/ood_detector.npz \\
        --ood_camera=front
"""

import logging
import time
from dataclasses import dataclass

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

from lerobot_ood import (
    ACTBackboneEncoder,
    DinoV2Encoder,
    ElevenLabsTTSWorker,
    OODDetector,
    choose_strawberry_ood_phrase,
    extract_camera_frame,
    load_elevenlabs_tts_config,
)

logger = logging.getLogger("run_policy_with_ood")


@dataclass
class OODRolloutConfig(RolloutConfig):
    """Extends RolloutConfig with OOD-specific knobs."""

    ood_detector_path: str = ""
    ood_camera: str = "front"
    # "act_backbone" (default) reuses the loaded policy's vision backbone.
    # "dinov2_vits14" / "dinov2_vitb14" / "dinov2_vitl14" run a separate model.
    ood_encoder: str = "act_backbone"
    # Print every Nth OOD detection (1 = every one).
    ood_log_every_n: int = 1
    # Periodically print in-dist scores too — useful for sanity-checking threshold.
    ood_log_in_dist_every_n: int = 0
    # Optional ElevenLabs TTS alerts. Disable with --ood_tts_enabled=false or the shell --no-voice flag.
    ood_tts_enabled: bool = True
    ood_tts_config_path: str = ".env"
    ood_tts_every_n: int = 30
    ood_tts_queue_max: int = 25


@parser.wrap()
def main(cfg: OODRolloutConfig) -> None:
    init_logging()

    if not cfg.ood_detector_path:
        raise ValueError("--ood_detector_path is required")
    if not isinstance(cfg.strategy, BaseStrategyConfig):
        raise ValueError(
            f"only --strategy.type=base is supported here (no dataset writing); "
            f"got '{cfg.strategy.type}'"
        )

    tts_config = None
    tts_worker = None
    if cfg.ood_tts_enabled:
        if cfg.ood_tts_every_n <= 0:
            raise ValueError("--ood_tts_every_n must be positive when TTS is enabled")
        if cfg.ood_tts_queue_max <= 0:
            raise ValueError("--ood_tts_queue_max must be positive when TTS is enabled")
        tts_config = load_elevenlabs_tts_config(cfg.ood_tts_config_path)

    detector = OODDetector.load(cfg.ood_detector_path)
    logger.info(
        "OOD detector loaded from %s (threshold=%.3f, pca_components=%s)",
        cfg.ood_detector_path,
        detector.threshold,
        detector.pca_components,
    )

    if tts_config is not None:
        tts_worker = ElevenLabsTTSWorker(tts_config, max_queue_size=cfg.ood_tts_queue_max)
        tts_worker.start()
        logger.info(
            "ElevenLabs OOD TTS enabled (voice_id=%s, model_id=%s, config=%s)",
            tts_config.voice_id,
            tts_config.model_id,
            cfg.ood_tts_config_path,
        )

    shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

    logger.info("Building rollout context (this loads the policy and connects the robot)...")
    ctx = build_rollout_context(cfg, shutdown_event)

    # Build the encoder *after* the policy is loaded so act_backbone can
    # share the policy's vision module.
    if cfg.ood_encoder == "act_backbone":
        encoder = ACTBackboneEncoder(policy=ctx.policy.policy, device=cfg.device)
    elif cfg.ood_encoder.startswith("dinov2_"):
        encoder = DinoV2Encoder(model=cfg.ood_encoder, device=cfg.device)
    else:
        raise ValueError(
            f"unknown --ood_encoder '{cfg.ood_encoder}'. "
            f"Expected 'act_backbone' or 'dinov2_<size>'."
        )
    logger.info("OOD encoder: %s", cfg.ood_encoder)

    robot = ctx.hardware.robot_wrapper
    processors = ctx.processors
    engine = ctx.policy.inference

    fps = cfg.fps
    control_interval = 1.0 / fps

    engine.reset()
    engine.start()
    engine.resume()
    logger.info(
        "Inference loop starting (fps=%d, duration=%ds, ood_camera=%s)",
        fps,
        cfg.duration,
        cfg.ood_camera,
    )

    n_seen = 0
    n_ood = 0
    sum_score = 0.0
    t_start = time.perf_counter()
    try:
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()

            if cfg.duration > 0 and (loop_start - t_start) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs_raw = robot.get_observation()

            # --- OOD scoring on the raw camera frame (additive, never blocks). ---
            try:
                frame = extract_camera_frame(obs_raw, cfg.ood_camera)
            except KeyError as e:
                logger.error("OOD camera frame extraction failed: %s", e)
                shutdown_event.set()
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
                    if (
                        tts_worker is not None
                        and cfg.ood_tts_every_n > 0
                        and n_ood % cfg.ood_tts_every_n == 0
                    ):
                        phrase = choose_strawberry_ood_phrase()
                        if not tts_worker.speak(phrase):
                            logger.warning("OOD TTS queue full; dropping voice alert")
            elif cfg.ood_log_in_dist_every_n > 0 and n_seen % cfg.ood_log_in_dist_every_n == 0:
                logger.info(
                    "[in-dist] frame=%d  score=%.3f  threshold=%.3f",
                    n_seen,
                    result.score,
                    result.threshold,
                )

            # --- Normal inference path: same flow as upstream BaseStrategy. ---
            obs_processed = processors.robot_observation_processor(obs_raw)
            engine.notify_observation(obs_processed)
            obs_frame = build_dataset_frame(
                ctx.data.dataset_features, obs_processed, prefix=OBS_STR
            )
            action_tensor = engine.get_action(obs_frame)
            if action_tensor is not None:
                ordered_keys = ctx.data.ordered_action_keys
                action_dict = {k: action_tensor[i].item() for i, k in enumerate(ordered_keys)}
                processed_action = processors.robot_action_processor((action_dict, obs_raw))
                robot.send_action(processed_action)

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            if sleep_t > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    "loop running slow (%.1f Hz < target %.0f Hz); "
                    "OOD path adds encoder latency — consider dinov2_vits14 on MPS/CUDA.",
                    1.0 / dt,
                    fps,
                )
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
        if tts_worker is not None:
            tts_worker.close()

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
    register_third_party_plugins()
    main()
