#!/usr/bin/env python
# pyright: reportMissingImports=false
"""Voice-gated multi-policy food handoff runtime for SO-101."""

import logging
import json
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
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
    TargetSuccessDetector,
    canonicalize_target,
    choose_unsupported_item_phrase,
    choose_food_handoff_ood_phrase,
    choose_fetching_phrase,
    choose_success_phrase,
    classify_food_request,
    extract_camera_frame,
    confirm_food_handoff_success,
    interpolate_neutral_actions,
    is_probable_unsupported_food_request,
    load_elevenlabs_stt_config,
    load_elevenlabs_tts_config,
    load_food_policy_config,
    load_neutral_reset_config,
    load_openai_vision_config,
    load_vision_config,
    SO101_NEUTRAL_ACTION_KEYS,
    transcribe_audio_file,
    unsupported_item_label,
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
    ood_enabled: bool = False
    ood_detector_path: str = ""
    ood_camera: str = "front"
    ood_encoder: str = "act_backbone"
    ood_every_n: int = 5
    ood_log_every_n: int = 1
    ood_log_in_dist_every_n: int = 0
    ood_failure_after_n: int = 3
    ood_initial_window_s: float = 5.0
    ood_tts_enabled: bool = True
    ood_tts_config_path: str = ".env"
    ood_tts_every_n: int = 3
    ood_tts_queue_max: int = 25
    openai_success_enabled: bool = False
    openai_success_config_path: str = ".env"
    openai_success_every_n: int = 60
    openai_success_sequence_frames: int = 1
    openai_success_sequence_stride: int = 1
    openai_success_max_in_flight: int = 2
    openai_success_grace_s: float = 15.0
    openai_success_frame_dir: str = ""
    loop_timing_log_every_n: int = 30
    test_mode: bool = False
    test_audio_path: str = ""
    max_cycles: int = 0
    reset_pause_s: float = 7.0
    robot_neutral_reset_enabled: bool = True
    robot_neutral_config: str = "config/robot_neutral.json"
    robot_neutral_reset_duration_s: float = 3.0
    robot_neutral_reset_fps: int = 30
    result_json_path: str = ""
    test_policy_steps: int = 5
    test_success_after_steps: int = 3


@dataclass(frozen=True)
class ClassifiedRequest:
    target: str | None
    transcript: str
    unsupported_item: bool = False
    unsupported_item_label: str = "that item"


@dataclass
class OpenAISuccessCheck:
    frame_id: int
    trigger: str
    opencv_candidate: bool
    opencv_score: float
    opencv_reason: str
    sequence_len: int
    frames: list
    submitted_at_s: float = 0.0


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
    if cfg.robot_neutral_reset_duration_s < 0:
        raise ValueError("--robot_neutral_reset_duration_s must be >= 0")
    if cfg.robot_neutral_reset_fps <= 0:
        raise ValueError("--robot_neutral_reset_fps must be positive")
    if cfg.openai_success_every_n <= 0:
        raise ValueError("--openai_success_every_n must be positive")
    if cfg.openai_success_sequence_frames <= 0:
        raise ValueError("--openai_success_sequence_frames must be positive")
    if cfg.openai_success_sequence_stride <= 0:
        raise ValueError("--openai_success_sequence_stride must be positive")
    if cfg.openai_success_max_in_flight <= 0:
        raise ValueError("--openai_success_max_in_flight must be positive")
    if cfg.openai_success_grace_s < 0:
        raise ValueError("--openai_success_grace_s must be >= 0")
    if cfg.loop_timing_log_every_n < 0:
        raise ValueError("--loop_timing_log_every_n must be >= 0")
    if cfg.ood_every_n <= 0:
        raise ValueError("--ood_every_n must be positive")
    if cfg.ood_failure_after_n < 0:
        raise ValueError("--ood_failure_after_n must be non-negative")
    if cfg.ood_initial_window_s < 0:
        raise ValueError("--ood_initial_window_s must be >= 0")

    policy_config = load_food_policy_config(cfg.food_policy_config)
    vision_config = load_vision_config(cfg.vision_config)

    openai_success_config = None
    if cfg.openai_success_enabled:
        openai_success_config = load_openai_vision_config(cfg.openai_success_config_path)
        logger.info(
            "OpenAI success confirmation enabled (model=%s, min_confidence=%.2f)",
            openai_success_config.model,
            openai_success_config.min_confidence,
        )

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
    bootstrap_policy_repo_id = policy_repo_id_from_config(cfg.policy)
    if not cfg.test_mode and not bootstrap_policy_repo_id:
        raise ValueError("A bootstrap policy is required for LeRobot configuration.")

    base_ood_detector_path = cfg.ood_detector_path
    cycle = 0
    cycle_results = []
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

            if selected_target is None:
                if stt_config is None:
                    raise ValueError("--no-stt requires --target strawberry|oreo|marshmallow")
                request = listen_and_classify_request(cfg, stt_config)
                selected_target = request.target
                if selected_target is None:
                    logger.warning("[REQUEST] could not classify target")
                    if request.unsupported_item:
                        phrase = choose_unsupported_item_phrase(request.unsupported_item_label)
                        outcome = "unsupported_item"
                    else:
                        phrase = choose_food_handoff_ood_phrase()
                        outcome = "ood_unclassified"
                    speak(tts_worker, phrase, wait=True)
                    logger.info("[CYCLE] complete cycle=%d outcome=%s", cycle, outcome)
                    cycle_results.append(
                        handoff_result(
                            cycle=cycle,
                            target=selected_target,
                            policy=None,
                            outcome=outcome,
                        )
                    )
                    write_result_json(cfg.result_json_path, cycle_results)
                    reset_between_cycles(cfg, cycle)
                    continue

            selected_policy = policy_config.require(selected_target)
            logger.info(
                "[REQUEST] target=%s policy=%s task=%r",
                selected_policy.target,
                selected_policy.policy_repo_id,
                selected_policy.task,
            )
            speak(tts_worker, choose_fetching_phrase(selected_policy.display_name), wait=True)

            if cfg.test_mode:
                outcome = run_mock_policy_flow(cfg, selected_policy, tts_worker)
                logger.info("[CYCLE] complete cycle=%d outcome=%s", cycle, outcome)
                cycle_results.append(
                    handoff_result(
                        cycle=cycle,
                        target=selected_policy.target,
                        policy=selected_policy,
                        outcome=outcome,
                    )
                )
                write_result_json(cfg.result_json_path, cycle_results)
                reset_between_cycles(cfg, cycle)
                continue

            setattr(cfg, "task", selected_policy.task)
            ood_detector_path = base_ood_detector_path
            if cfg.ood_enabled and selected_policy.ood_detector_path:
                ood_detector_path = resolve_config_path(
                    selected_policy.ood_detector_path,
                    cfg.food_policy_config,
                )
            if not cfg.ood_enabled:
                detector = None
                logger.info("OOD scoring disabled.")
            elif not ood_detector_path:
                detector = None
                logger.info("No OOD detector configured; OOD scoring disabled for this cycle.")
            else:
                detector = OODDetector.load(ood_detector_path)
                logger.info(
                    "OOD detector loaded from %s (threshold=%.3f, pca_components=%s)",
                    ood_detector_path,
                    detector.threshold,
                    detector.pca_components,
                )

            shutdown_event = ProcessSignalHandler(use_threads=True, display_pid=False).shutdown_event

            logger.info("Building rollout context for selected policy...")
            set_policy_config(cfg, selected_policy.policy_repo_id)
            ctx = build_rollout_context(cfg, shutdown_event)
            encoder = None
            if detector is not None:
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
                openai_success_config=openai_success_config,
            )
            logger.info("[CYCLE] complete cycle=%d outcome=%s", cycle, outcome)
            cycle_results.append(
                handoff_result(
                    cycle=cycle,
                    target=selected_policy.target,
                    policy=selected_policy,
                    outcome=outcome,
                )
            )
            write_result_json(cfg.result_json_path, cycle_results)
            if outcome == "interrupted":
                break
            reset_between_cycles(cfg, cycle)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if tts_worker is not None:
            tts_worker.close(timeout=10.0)
        write_result_json(cfg.result_json_path, cycle_results)
        logger.info("Handoff loop stopped after %d cycle(s)", cycle)


def listen_and_classify_request(cfg: FoodHandoffConfig, stt_config) -> ClassifiedRequest:
    if cfg.test_mode and cfg.test_audio_path:
        transcript = transcribe_audio_file(
            stt_config,
            cfg.test_audio_path,
            keyterms=["Strawberry", "Oreo", "Marshmallow", "Marshmellow"],
        )
        target = classify_food_request(transcript)
        logger.info("[REQUEST] test_audio=%s transcript=%r target=%s", cfg.test_audio_path, transcript, target)
        return ClassifiedRequest(
            target=target,
            transcript=transcript,
            unsupported_item=target is None and is_probable_unsupported_food_request(transcript),
            unsupported_item_label=unsupported_item_label(transcript),
        )

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
        return ClassifiedRequest(
            target=target,
            transcript=transcript,
            unsupported_item=target is None and is_probable_unsupported_food_request(transcript),
            unsupported_item_label=unsupported_item_label(transcript),
        )
    finally:
        try:
            audio_path.unlink()
        except FileNotFoundError:
            pass


def policy_repo_id_from_config(policy_config) -> str:
    for attr in ("pretrained_path", "path", "repo_id"):
        value = getattr(policy_config, attr, None)
        if value:
            return str(value)
    return ""


def set_policy_config(cfg: FoodHandoffConfig, policy_repo_id: str) -> None:
    cfg.policy = PreTrainedConfig.from_pretrained(policy_repo_id)
    cfg.policy.pretrained_path = policy_repo_id
    cfg.policy.device = cfg.device


def handoff_result(cycle: int, target: str | None, policy, outcome: str) -> dict:
    result = {
        "cycle": cycle,
        "target": target,
        "outcome": outcome,
        "success": outcome == "success",
    }
    if policy is not None:
        result.update(
            {
                "display_name": policy.display_name,
                "policy_repo_id": policy.policy_repo_id,
                "task": policy.task,
            }
        )
    return result


def write_result_json(path: str, cycle_results: list[dict]) -> None:
    if not path:
        return
    payload = {
        "status": cycle_results[-1]["outcome"] if cycle_results else "not_started",
        "success": bool(cycle_results and cycle_results[-1]["success"]),
        "cycles": cycle_results,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


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
        "[CYCLE] reset pause %.1fs; reset the scene before the next cycle",
        cfg.reset_pause_s,
    )
    time.sleep(cfg.reset_pause_s)


def reset_robot_to_neutral(cfg: FoodHandoffConfig, robot, processors, outcome: str, n_actions: int) -> None:
    if not cfg.robot_neutral_reset_enabled:
        logger.info("[NEUTRAL_RESET] disabled")
        return
    if outcome == "interrupted":
        logger.info("[NEUTRAL_RESET] skipped after interrupted run")
        return
    if n_actions <= 0:
        logger.info("[NEUTRAL_RESET] skipped; no robot actions were sent")
        return

    config_path = resolve_config_path(cfg.robot_neutral_config, cfg.food_policy_config)
    logger.info("[NEUTRAL_RESET] starting outcome=%s config=%s", outcome, config_path)
    try:
        neutral_config = load_neutral_reset_config(config_path)
        obs_raw = robot.get_observation()
        current = {key: float(obs_raw[key]) for key in SO101_NEUTRAL_ACTION_KEYS}
        steps = max(
            1,
            int(round(cfg.robot_neutral_reset_duration_s * cfg.robot_neutral_reset_fps)),
        )
        control_interval = 1.0 / cfg.robot_neutral_reset_fps
        for action_dict in interpolate_neutral_actions(current, neutral_config.action, steps):
            loop_start = time.perf_counter()
            obs_raw = robot.get_observation()
            processed_action = processors.robot_action_processor((action_dict, obs_raw))
            robot.send_action(processed_action)
            sleep_t = control_interval - (time.perf_counter() - loop_start)
            if sleep_t > 0:
                precise_sleep(sleep_t)
        logger.info("[NEUTRAL_RESET] complete steps=%d", steps)
    except Exception as exc:
        logger.warning("[NEUTRAL_RESET] failed: %s", exc)


def run_cleanup_step(name: str, fn) -> None:
    try:
        fn()
    except Exception as exc:
        logger.warning("[CLEANUP] %s failed: %s", name, exc)


def score_ood_frame(detector: OODDetector, encoder, frame) -> object:
    embedding = encoder(frame)
    return detector.score(embedding)


def confirm_success_frame(openai_success_config, frames, display_name: str):
    return confirm_food_handoff_success(openai_success_config, frames, display_name)


def select_success_sequence(
    frame_buffer,
    sequence_frames: int,
    sequence_stride: int,
) -> list:
    frames = list(frame_buffer)
    selected = frames[::-sequence_stride][:sequence_frames]
    return list(reversed(selected))


def build_openai_success_check(
    frame_buffer,
    sequence_frames: int,
    sequence_stride: int,
    frame_id: int,
    trigger: str,
    success_result,
) -> OpenAISuccessCheck:
    success_sequence = list(frame_buffer)[-sequence_frames:]
    return OpenAISuccessCheck(
        frame_id=frame_id,
        trigger=trigger,
        opencv_candidate=bool(success_result.detected),
        opencv_score=float(success_result.score),
        opencv_reason=str(success_result.reason),
        sequence_len=len(success_sequence),
        frames=success_sequence,
    )


def save_openai_success_frames(check: OpenAISuccessCheck, output_dir: str | Path) -> list[Path]:
    if not output_dir:
        return []
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Saving OpenAI success frames requires opencv-python and numpy.") from exc

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, frame in enumerate(check.frames, 1):
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"unexpected OpenAI success frame shape: {arr.shape}")
        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        frame_path = path / f"openai_success_frame_{check.frame_id:06d}_{i:02d}.jpg"
        if not cv2.imwrite(str(frame_path), frame_bgr):
            raise RuntimeError(f"failed to write OpenAI success frame: {frame_path}")
        saved.append(frame_path)
    return saved


def run_selected_policy(
    cfg: FoodHandoffConfig,
    ctx,
    detector: OODDetector | None,
    encoder,
    success_detector: TargetSuccessDetector,
    selected_policy,
    tts_worker: ElevenLabsTTSWorker | None,
    shutdown_event,
    vision_config,
    openai_success_config=None,
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
    success_source = ""
    success_opencv_score = 0.0
    success_opencv_reason = ""
    last_openai_success_check_frame = 0
    last_ood_check_frame = 0
    pending_openai_success: dict[Future, OpenAISuccessCheck] = {}
    queued_openai_success: OpenAISuccessCheck | None = None
    pending_ood: Future | None = None
    pending_ood_frame = 0
    outcome = "timeout"
    success_frame_buffer = deque(maxlen=1)
    in_success_grace = False
    pending_grace_wait_logged = False
    ood_window_closed_logged = False

    engine.reset()
    engine.start()
    engine.resume()
    t_start = time.perf_counter()
    openai_executor = (
        ThreadPoolExecutor(max_workers=cfg.openai_success_max_in_flight)
        if openai_success_config is not None
        else None
    )
    ood_executor = ThreadPoolExecutor(max_workers=1) if detector is not None else None
    logger.info(
        "[POLICY] starting target=%s fps=%d duration=%ds",
        selected_policy.target,
        cfg.fps,
        cfg.duration,
    )
    try:
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()
            obs_dt = 0.0
            ood_dt = 0.0
            policy_dt = 0.0
            success_dt = 0.0
            openai_dt = 0.0
            elapsed_s = loop_start - t_start
            policy_active = cfg.duration <= 0 or elapsed_s < cfg.duration
            grace_elapsed_s = max(0.0, elapsed_s - cfg.duration) if cfg.duration > 0 else 0.0
            openai_grace_available = (
                cfg.duration > 0
                and openai_success_config is not None
                and cfg.openai_success_grace_s > 0
            )
            if not policy_active:
                if not openai_grace_available:
                    logger.info("Duration limit reached (%.0fs)", cfg.duration)
                    break
                if not in_success_grace:
                    in_success_grace = True
                    logger.info(
                        "Duration limit reached (%.0fs); stopping policy actions and entering "
                        "OpenAI success grace for %.1fs",
                        cfg.duration,
                        cfg.openai_success_grace_s,
                    )
                if grace_elapsed_s >= cfg.openai_success_grace_s:
                    if not pending_openai_success:
                        logger.info(
                            "OpenAI success grace limit reached (%.1fs)",
                            cfg.openai_success_grace_s,
                        )
                        break
                    if not pending_grace_wait_logged:
                        logger.info(
                            "OpenAI success grace limit reached; waiting for %d pending success check(s)",
                            len(pending_openai_success),
                        )
                        pending_grace_wait_logged = True
            allow_openai_submit = policy_active or (
                openai_grace_available and grace_elapsed_s < cfg.openai_success_grace_s
            )

            section_start = time.perf_counter()
            obs_raw = robot.get_observation()
            obs_dt += time.perf_counter() - section_start
            frame = None
            ood_window_active = (
                policy_active
                and detector is not None
                and cfg.ood_initial_window_s > 0
                and elapsed_s < cfg.ood_initial_window_s
            )
            if policy_active and (ood_window_active or vision_config.success.camera_name == cfg.ood_camera):
                try:
                    frame = extract_camera_frame(obs_raw, cfg.ood_camera)
                except KeyError as e:
                    logger.error("Camera frame extraction failed: %s", e)
                    shutdown_event.set()
                    continue
            if (
                policy_active
                and detector is not None
                and not ood_window_active
                and not ood_window_closed_logged
            ):
                logger.info(
                    "[OOD] initial window closed at %.2fs; skipping further OOD checks",
                    elapsed_s,
                )
                ood_window_closed_logged = True

            n_seen += 1
            if policy_active and pending_ood is not None and pending_ood.done():
                section_start = time.perf_counter()
                try:
                    result = pending_ood.result()
                except Exception as exc:
                    logger.warning("[OOD] async score failed on frame=%d: %s", pending_ood_frame, exc)
                else:
                    sum_score += result.score
                    if result.is_ood:
                        n_ood += 1
                        if cfg.ood_log_every_n > 0 and n_ood % cfg.ood_log_every_n == 0:
                            logger.warning(
                                "[OOD] frame=%d score=%.3f threshold=%.3f (%d/%d sampled so far)",
                                pending_ood_frame,
                                result.score,
                                result.threshold,
                                n_ood,
                                max((pending_ood_frame + cfg.ood_every_n - 1) // cfg.ood_every_n, 1),
                            )
                        if cfg.ood_failure_after_n > 0 and n_ood >= cfg.ood_failure_after_n:
                            outcome = "ood_failure"
                            logger.warning(
                                "[OOD_FAILURE] target=%s frame=%d ood_detections=%d threshold=%d",
                                selected_policy.target,
                                pending_ood_frame,
                                n_ood,
                                cfg.ood_failure_after_n,
                            )
                            speak(tts_worker, choose_food_handoff_ood_phrase(), wait=True)
                            break
                        if tts_worker is not None and n_ood % cfg.ood_tts_every_n == 0:
                            speak(tts_worker, choose_food_handoff_ood_phrase())
                    elif cfg.ood_log_in_dist_every_n > 0 and pending_ood_frame % cfg.ood_log_in_dist_every_n == 0:
                        logger.info(
                            "[in-dist] frame=%d score=%.3f threshold=%.3f",
                            pending_ood_frame,
                            result.score,
                            result.threshold,
                        )
                pending_ood = None
                pending_ood_frame = 0
                ood_dt += time.perf_counter() - section_start

            if (
                ood_window_active
                and
                detector is not None
                and ood_executor is not None
                and pending_ood is None
                and n_seen - last_ood_check_frame >= cfg.ood_every_n
            ):
                assert frame is not None
                section_start = time.perf_counter()
                pending_ood = ood_executor.submit(score_ood_frame, detector, encoder, frame.copy())
                pending_ood_frame = n_seen
                last_ood_check_frame = n_seen
                ood_dt += time.perf_counter() - section_start

            if policy_active:
                section_start = time.perf_counter()
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
                policy_dt += time.perf_counter() - section_start

            success_confirmed = False
            section_start = time.perf_counter()
            completed_openai = [future for future in list(pending_openai_success) if future.done()]
            for future in completed_openai:
                check = pending_openai_success.pop(future)
                latency_s = loop_start - check.submitted_at_s
                age_frames = max(0, n_seen - check.frame_id)
                try:
                    openai_result = future.result()
                except Exception as exc:
                    logger.warning(
                        "[OPENAI_SUCCESS] async check failed frame=%d latency=%.2fs age_frames=%d "
                        "trigger=%s: %s",
                        check.frame_id,
                        latency_s,
                        age_frames,
                        check.trigger,
                        exc,
                    )
                else:
                    logger.info(
                        "[OPENAI_SUCCESS] result frame=%d success=%s confidence=%.3f "
                        "latency=%.2fs age_frames=%d reason=%r",
                        check.frame_id,
                        openai_result.success,
                        openai_result.confidence,
                        latency_s,
                        age_frames,
                        openai_result.reason,
                    )
                    success_confirmed = openai_result.success
                    if success_confirmed:
                        success_source = "openai"
                        success_frame = check.frame_id
                        success_opencv_score = check.opencv_score
                        success_opencv_reason = check.opencv_reason
                        break
            openai_dt += time.perf_counter() - section_start

            openai_enabled = openai_success_config is not None and openai_executor is not None
            openai_due = n_seen - last_openai_success_check_frame >= cfg.openai_success_every_n
            openai_sample_due = openai_enabled and allow_openai_submit and not success_confirmed and openai_due
            success_result = None
            if openai_sample_due or not openai_enabled:
                section_start = time.perf_counter()
                if policy_active and vision_config.success.camera_name == cfg.ood_camera:
                    assert frame is not None
                    success_frame_raw = frame
                else:
                    success_frame_raw = extract_camera_frame(obs_raw, vision_config.success.camera_name)
                success_result = success_detector.update(success_frame_raw, selected_policy.target)
                if openai_sample_due:
                    success_frame_buffer.append(success_frame_raw.copy())
                success_dt += time.perf_counter() - section_start

            if openai_enabled:
                section_start = time.perf_counter()
                openai_due = n_seen - last_openai_success_check_frame >= cfg.openai_success_every_n
                if allow_openai_submit and openai_due and not success_confirmed:
                    if success_result is not None:
                        queued_openai_success = build_openai_success_check(
                            success_frame_buffer,
                            cfg.openai_success_sequence_frames,
                            cfg.openai_success_sequence_stride,
                            n_seen,
                            "cadence",
                            success_result,
                        )
                        last_openai_success_check_frame = n_seen

                while (
                    allow_openai_submit
                    and queued_openai_success is not None
                    and len(pending_openai_success) < cfg.openai_success_max_in_flight
                    and not success_confirmed
                ):
                    check = queued_openai_success
                    queued_openai_success = None
                    check.submitted_at_s = loop_start
                    if cfg.openai_success_frame_dir:
                        try:
                            saved_frames = save_openai_success_frames(
                                check,
                                cfg.openai_success_frame_dir,
                            )
                        except Exception as exc:
                            logger.warning("[OPENAI_SUCCESS] failed to save sampled frame(s): %s", exc)
                        else:
                            logger.info(
                                "[OPENAI_SUCCESS] saved %d sampled frame(s) to %s",
                                len(saved_frames),
                                cfg.openai_success_frame_dir,
                            )
                    future = openai_executor.submit(
                        confirm_success_frame,
                        openai_success_config,
                        check.frames,
                        selected_policy.display_name,
                    )
                    pending_openai_success[future] = check
                    logger.info(
                        "[OPENAI_SUCCESS] evaluate frame=%d sequence_frames=%d in_flight=%d/%d",
                        check.frame_id,
                        check.sequence_len,
                        len(pending_openai_success),
                        cfg.openai_success_max_in_flight,
                    )
                openai_dt += time.perf_counter() - section_start
            elif success_result is not None and success_result.detected:
                logger.info(
                    "[TASK_SUCCESS_CANDIDATE_LOCAL_ONLY] target=%s frame=%d "
                    "opencv_score=%.3f openai_disabled=true; ignoring local detector for success",
                    selected_policy.target,
                    n_seen,
                    success_result.score,
                )

            if success_confirmed:
                success = True
                outcome = "success"
                logger.info(
                    "[TASK_SUCCESS] target=%s frame=%d source=%s opencv_score=%.3f %s",
                    selected_policy.target,
                    success_frame,
                    success_source,
                    success_opencv_score,
                    success_opencv_reason,
                )
                phrase = choose_success_phrase(selected_policy.display_name)
                speak(tts_worker, phrase, wait=True, priority=True)
                break

            dt = time.perf_counter() - loop_start
            sleep_t = control_interval - dt
            if cfg.loop_timing_log_every_n > 0 and n_seen % cfg.loop_timing_log_every_n == 0:
                logger.info(
                    "[LOOP_TIMING] frame=%d total_ms=%.1f obs_ms=%.1f policy_ms=%.1f "
                    "success_ms=%.1f openai_ms=%.1f ood_ms=%.1f sleep_ms=%.1f "
                    "openai_in_flight=%d openai_samples=%d",
                    n_seen,
                    dt * 1000.0,
                    obs_dt * 1000.0,
                    policy_dt * 1000.0,
                    success_dt * 1000.0,
                    openai_dt * 1000.0,
                    ood_dt * 1000.0,
                    max(0.0, sleep_t) * 1000.0,
                    len(pending_openai_success),
                    len(success_frame_buffer),
                )
            if sleep_t > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning("loop running slow (%.1f Hz < target %.0f Hz)", 1.0 / dt, cfg.fps)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        outcome = "interrupted"
    finally:
        if openai_executor is not None:
            openai_executor.shutdown(wait=False, cancel_futures=True)
        if ood_executor is not None:
            ood_executor.shutdown(wait=False, cancel_futures=True)
        run_cleanup_step("inference engine stop", engine.stop)
        run_cleanup_step(
            "neutral reset",
            lambda: reset_robot_to_neutral(cfg, robot, processors, outcome, n_actions),
        )
        try:
            inner_robot = robot.inner
            if inner_robot.is_connected:
                run_cleanup_step("robot disconnect", inner_robot.disconnect)
        except Exception as exc:
            logger.warning("[CLEANUP] robot disconnect precheck failed: %s", exc)
        try:
            teleop = ctx.hardware.teleop
            if teleop is not None and teleop.is_connected:
                run_cleanup_step("teleop disconnect", teleop.disconnect)
        except Exception as exc:
            logger.warning("[CLEANUP] teleop disconnect precheck failed: %s", exc)
        mean_score = sum_score / max(n_seen, 1) if detector is not None else 0.0
        threshold = detector.threshold if detector is not None else 0.0
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
            threshold,
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


def speak(
    worker: ElevenLabsTTSWorker | None,
    phrase: str,
    wait: bool = False,
    priority: bool = False,
) -> None:
    if worker is None:
        logger.info("[TTS] disabled; skipping phrase=%r", phrase)
        return
    if priority:
        dropped = worker.clear_pending()
        if dropped:
            logger.info("[TTS] dropped %d queued phrase(s) before priority phrase", dropped)
    logger.info("[TTS] queue wait=%s priority=%s phrase=%r", wait, priority, phrase)
    if not worker.speak(phrase, block=wait, timeout=30.0 if wait else None):
        logger.warning("TTS queue full; dropping voice phrase")
        return
    if wait and not worker.wait_until_idle(timeout=30.0):
        logger.warning("Timed out waiting for TTS phrase to finish")
    elif wait and worker.last_error is not None:
        logger.warning("TTS phrase finished with error: %s", worker.last_error)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
