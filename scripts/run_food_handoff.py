#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Voice-gated multi-policy food handoff runtime for SO-101."""

from __future__ import annotations

import logging
import tempfile
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

from lerobot_ood import (
    ACTBackboneEncoder,
    DinoV2Encoder,
    ElevenLabsTTSWorker,
    HandPresenceDetector,
    OODDetector,
    TargetSuccessDetector,
    canonicalize_target,
    choose_food_handoff_ood_phrase,
    choose_success_phrase,
    classify_food_request,
    extract_camera_frame,
    load_elevenlabs_stt_config,
    load_elevenlabs_tts_config,
    load_food_policy_config,
    load_vision_config,
    transcribe_audio_file,
)
from lerobot_ood.audio import record_wav

logger = logging.getLogger("run_food_handoff")


@dataclass
class FoodHandoffConfig(RolloutConfig):
    """Extends RolloutConfig with food handoff, STT, OOD, and TTS knobs."""

    food_policy_config: str = "config/food_policies.json"
    vision_config: str = "config/food_handoff_vision.json"
    target: str = ""
    stt_enabled: bool = True
    request_audio_seconds: float = 3.0
    request_audio_sample_rate: int = 16000
    request_recorder_command: str = ""
    hand_wait_timeout_s: float = 0.0
    ood_detector_path: str = ""
    ood_camera: str = "front"
    ood_encoder: str = "act_backbone"
    ood_log_every_n: int = 1
    ood_log_in_dist_every_n: int = 0
    ood_tts_enabled: bool = True
    ood_tts_config_path: str = ".env"
    ood_tts_every_n: int = 30
    ood_tts_queue_max: int = 25
    test_mode: bool = False
    test_audio_path: str = ""
    max_cycles: int = 0
    reset_pause_s: float = 7.0
    test_policy_steps: int = 5
    test_success_after_steps: int = 3


@parser.wrap()
def main(cfg: FoodHandoffConfig) -> None:
    init_logging()
    if not isinstance(cfg.strategy, BaseStrategyConfig):
        raise ValueError(
            f"only --strategy.type=base is supported for food handoff; got {cfg.strategy.type!r}"
        )
    if cfg.max_cycles < 0:
        raise ValueError("--max_cycles must be >= 0")
    if cfg.reset_pause_s < 0:
        raise ValueError("--reset_pause_s must be >= 0")

    policy_config = load_food_policy_config(cfg.food_policy_config)
    vision_config = load_vision_config(cfg.vision_config)

    tts_worker = None
    tts_config = None
    if cfg.ood_tts_enabled:
        if cfg.ood_tts_queue_max <= 0:
            raise ValueError("--ood_tts_queue_max must be positive when voice is enabled")
        if cfg.ood_tts_every_n <= 0:
            raise ValueError("--ood_tts_every_n must be positive when voice is enabled")
        tts_config = load_elevenlabs_tts_config(cfg.ood_tts_config_path)
        tts_worker = ElevenLabsTTSWorker(tts_config, max_queue_size=cfg.ood_tts_queue_max)
        tts_worker.start()
        logger.info(
            "ElevenLabs TTS enabled (voice_id=%s, model_id=%s)",
            tts_config.voice_id,
            tts_config.model_id,
        )

    stt_config = None
    if cfg.stt_enabled:
        stt_config = load_elevenlabs_stt_config(cfg.ood_tts_config_path)
        logger.info("ElevenLabs STT enabled (model_id=scribe_v2)")

    target_override = canonicalize_target(cfg.target)
    if cfg.target and target_override is None:
        raise ValueError("--target must be one of: strawberry, oreo, marshmallow")

    base_ood_detector_path = cfg.ood_detector_path
    cycle = 0
    logger.info(
        "Starting handoff loop (max_cycles=%s, reset_pause_s=%.1f)",
        cfg.max_cycles if cfg.max_cycles > 0 else "unlimited",
        cfg.reset_pause_s,
    )
    try:
        while cfg.max_cycles == 0 or cycle < cfg.max_cycles:
            cycle += 1
            logger.info("[CYCLE] start cycle=%d", cycle)
            selected_target = target_override

            logger.info("Waiting for hand in camera frame...")
            if cfg.test_mode:
                logger.info("[HAND] mocked present")
            else:
                wait_for_hand(vision_config.hand, timeout_s=cfg.hand_wait_timeout_s)
            logger.info("[HAND] present")

            if selected_target is None:
                if stt_config is None:
                    raise ValueError("--no-stt requires --target strawberry|oreo|marshmallow")
                selected_target = listen_and_classify_request(cfg, stt_config)
                if selected_target is None:
                    logger.warning("[REQUEST] could not classify target")
                    speak(tts_worker, choose_food_handoff_ood_phrase(), wait=True)
                    logger.info("[CYCLE] complete cycle=%d outcome=ood_unclassified", cycle)
                    reset_between_cycles(cfg, cycle)
                    continue

            selected_policy = policy_config.require(selected_target)
            logger.info(
                "[REQUEST] target=%s policy=%s task=%r",
                selected_policy.target,
                selected_policy.policy_repo_id,
                selected_policy.task,
            )

            if cfg.test_mode:
                outcome = run_mock_policy_flow(cfg, selected_policy, tts_worker)
                logger.info("[CYCLE] complete cycle=%d outcome=%s", cycle, outcome)
                reset_between_cycles(cfg, cycle)
                continue

            setattr(cfg.policy, "path", selected_policy.policy_repo_id)
            setattr(cfg, "task", selected_policy.task)
            ood_detector_path = base_ood_detector_path
            if selected_policy.ood_detector_path:
                ood_detector_path = resolve_config_path(
                    selected_policy.ood_detector_path,
                    cfg.food_policy_config,
                )
            if not ood_detector_path:
                raise ValueError(
                    "No OOD detector configured. Set --ood_detector_path or target.ood_detector_path "
                    "in the food policy config."
                )

            shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event
            detector = OODDetector.load(ood_detector_path)
            logger.info(
                "OOD detector loaded from %s (threshold=%.3f, pca_components=%s)",
                ood_detector_path,
                detector.threshold,
                detector.pca_components,
            )

            logger.info("Building rollout context for selected policy...")
            ctx = build_rollout_context(cfg, shutdown_event)
            if cfg.ood_encoder == "act_backbone":
                encoder = ACTBackboneEncoder(policy=ctx.policy.policy, device=cfg.device)
            elif cfg.ood_encoder.startswith("dinov2_"):
                encoder = DinoV2Encoder(model=cfg.ood_encoder, device=cfg.device)
            else:
                raise ValueError(
                    f"unknown --ood_encoder {cfg.ood_encoder!r}; expected 'act_backbone' or 'dinov2_<size>'"
                )

            success_detector = TargetSuccessDetector(vision_config.success)
            outcome = run_selected_policy(
                cfg=cfg,
                ctx=ctx,
                detector=detector,
                encoder=encoder,
                success_detector=success_detector,
                selected_policy=selected_policy,
                tts_worker=tts_worker,
                shutdown_event=shutdown_event,
                vision_config=vision_config,
            )
            logger.info("[CYCLE] complete cycle=%d outcome=%s", cycle, outcome)
            if outcome == "interrupted":
                break
            reset_between_cycles(cfg, cycle)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if tts_worker is not None:
            tts_worker.close(timeout=10.0)
        logger.info("Handoff loop stopped after %d cycle(s)", cycle)


def listen_and_classify_request(cfg: FoodHandoffConfig, stt_config) -> str | None:
    if cfg.test_mode and cfg.test_audio_path:
        transcript = transcribe_audio_file(
            stt_config,
            cfg.test_audio_path,
            keyterms=["Strawberry", "Oreo", "Marshmallow", "Marshmellow"],
        )
        target = classify_food_request(transcript)
        logger.info("[REQUEST] test_audio=%s transcript=%r target=%s", cfg.test_audio_path, transcript, target)
        return target

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = Path(f.name)
    try:
        logger.info("[REQUEST] recording %.1fs microphone clip", cfg.request_audio_seconds)
        record_wav(
            audio_path,
            duration_s=cfg.request_audio_seconds,
            sample_rate=cfg.request_audio_sample_rate,
            channels=1,
            recorder_command=cfg.request_recorder_command,
        )
        transcript = transcribe_audio_file(
            stt_config,
            audio_path,
            keyterms=["Strawberry", "Oreo", "Marshmallow", "Marshmellow"],
        )
        target = classify_food_request(transcript)
        logger.info("[REQUEST] transcript=%r target=%s", transcript, target)
        return target
    finally:
        try:
            audio_path.unlink()
        except FileNotFoundError:
            pass


def resolve_config_path(value: str, config_path: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    config_parent = Path(config_path).resolve().parent
    base = config_parent.parent if config_parent.name == "config" else config_parent
    return str(base / path)


def reset_between_cycles(cfg: FoodHandoffConfig, cycle: int) -> None:
    if cfg.max_cycles > 0 and cycle >= cfg.max_cycles:
        return
    if cfg.reset_pause_s <= 0:
        return
    logger.info(
        "[CYCLE] reset pause %.1fs; move the hand out of frame before the next cycle",
        cfg.reset_pause_s,
    )
    time.sleep(cfg.reset_pause_s)


def wait_for_hand(hand_config, timeout_s: float = 0.0) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Hand-triggered recording requires opencv-python in the runtime environment."
        ) from exc

    cap = cv2.VideoCapture(hand_config.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open hand camera index {hand_config.camera_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, hand_config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hand_config.height)
    cap.set(cv2.CAP_PROP_FPS, hand_config.fps)

    detector = HandPresenceDetector(hand_config)
    start = time.perf_counter()
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read camera index {hand_config.camera_index}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = detector.update(frame_rgb)
            if result.detected:
                logger.info("[HAND] score=%.3f %s", result.score, result.reason)
                return
            if timeout_s > 0 and (time.perf_counter() - start) >= timeout_s:
                raise TimeoutError(f"hand did not enter frame within {timeout_s:.1f}s")
            time.sleep(1.0 / max(hand_config.fps, 1))
    finally:
        cap.release()


def run_selected_policy(
    cfg: FoodHandoffConfig,
    ctx,
    detector: OODDetector,
    encoder,
    success_detector: TargetSuccessDetector,
    selected_policy,
    tts_worker: ElevenLabsTTSWorker | None,
    shutdown_event,
    vision_config,
) -> str:
    robot = ctx.hardware.robot_wrapper
    processors = ctx.processors
    engine = ctx.policy.inference
    control_interval = 1.0 / cfg.fps

    n_seen = 0
    n_ood = 0
    n_actions = 0
    sum_score = 0.0
    success = False
    success_frame = 0
    outcome = "timeout"

    engine.reset()
    engine.start()
    engine.resume()
    t_start = time.perf_counter()
    logger.info(
        "[POLICY] starting target=%s fps=%d duration=%ds",
        selected_policy.target,
        cfg.fps,
        cfg.duration,
    )
    try:
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()
            if cfg.duration > 0 and (loop_start - t_start) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs_raw = robot.get_observation()
            try:
                frame = extract_camera_frame(obs_raw, cfg.ood_camera)
            except KeyError as e:
                logger.error("Camera frame extraction failed: %s", e)
                shutdown_event.set()
                continue

            embedding = encoder(frame)
            result = detector.score(embedding)
            n_seen += 1
            sum_score += result.score
            if result.is_ood:
                n_ood += 1
                if cfg.ood_log_every_n > 0 and n_ood % cfg.ood_log_every_n == 0:
                    logger.warning(
                        "[OOD] frame=%d score=%.3f threshold=%.3f (%d/%d so far, %.1f%%)",
                        n_seen,
                        result.score,
                        result.threshold,
                        n_ood,
                        n_seen,
                        100.0 * n_ood / n_seen,
                    )
                if tts_worker is not None and n_ood % cfg.ood_tts_every_n == 0:
                    speak(tts_worker, choose_food_handoff_ood_phrase())
            elif cfg.ood_log_in_dist_every_n > 0 and n_seen % cfg.ood_log_in_dist_every_n == 0:
                logger.info("[in-dist] frame=%d score=%.3f threshold=%.3f", n_seen, result.score, result.threshold)

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
                n_actions += 1

            if vision_config.success.camera_name == cfg.ood_camera:
                success_frame_raw = frame
            else:
                success_frame_raw = extract_camera_frame(obs_raw, vision_config.success.camera_name)
            success_result = success_detector.update(success_frame_raw, selected_policy.target)
            if success_result.detected:
                success = True
                success_frame = n_seen
                outcome = "success"
                logger.info(
                    "[TASK_SUCCESS] target=%s frame=%d confidence=%.3f %s",
                    selected_policy.target,
                    success_frame,
                    success_result.score,
                    success_result.reason,
                )
                phrase = choose_success_phrase(selected_policy.display_name)
                speak(tts_worker, phrase, wait=True)
                break

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            if sleep_t > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning("loop running slow (%.1f Hz < target %.0f Hz)", 1.0 / dt, cfg.fps)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        outcome = "interrupted"
    finally:
        engine.stop()
        inner_robot = robot.inner
        if inner_robot.is_connected:
            inner_robot.disconnect()
        teleop = ctx.hardware.teleop
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        mean_score = sum_score / max(n_seen, 1)
        logger.info(
            "Run complete: target=%s success=%s success_frame=%s frames=%d actions=%d "
            "ood=%d (%.1f%%) mean_score=%.3f threshold=%.3f",
            selected_policy.target,
            success,
            success_frame if success else "none",
            n_seen,
            n_actions,
            n_ood,
            100.0 * n_ood / max(n_seen, 1),
            mean_score,
            detector.threshold,
        )
    return outcome


def run_mock_policy_flow(
    cfg: FoodHandoffConfig,
    selected_policy,
    tts_worker: ElevenLabsTTSWorker | None,
) -> str:
    if cfg.test_policy_steps <= 0:
        raise ValueError("--test_policy_steps must be positive")
    if cfg.test_success_after_steps <= 0:
        raise ValueError("--test_success_after_steps must be positive")

    logger.info(
        "[POLICY] mocked starting target=%s policy=%s task=%r",
        selected_policy.target,
        selected_policy.policy_repo_id,
        selected_policy.task,
    )
    success = False
    success_frame = "none"
    n_actions = 0
    for step in range(1, cfg.test_policy_steps + 1):
        logger.info("[MOCK_ACTION] frame=%d target=%s", step, selected_policy.target)
        n_actions += 1
        if step >= cfg.test_success_after_steps:
            success = True
            success_frame = step
            logger.info(
                "[TASK_SUCCESS] target=%s frame=%d confidence=1.000 mocked_success=true",
                selected_policy.target,
                step,
            )
            phrase = choose_success_phrase(selected_policy.display_name)
            speak(tts_worker, phrase, wait=True)
            break
        time.sleep(1.0 / max(cfg.fps, 1))

    logger.info(
        "Run complete: target=%s success=%s success_frame=%s frames=%d actions=%d "
        "ood=0 (0.0%%) mean_score=0.000 threshold=0.000 test_mode=true",
        selected_policy.target,
        success,
        success_frame,
        int(success_frame) if success else cfg.test_policy_steps,
        n_actions,
    )
    return "success" if success else "timeout"


def speak(worker: ElevenLabsTTSWorker | None, phrase: str, wait: bool = False) -> None:
    if worker is None:
        return
    if not worker.speak(phrase):
        logger.warning("TTS queue full; dropping voice phrase")
        return
    if wait and not worker.wait_until_idle(timeout=30.0):
        logger.warning("Timed out waiting for TTS phrase to finish")


if __name__ == "__main__":
    register_third_party_plugins()
    main()
