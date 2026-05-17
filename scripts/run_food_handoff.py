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
    is_probable_unsupported_food_request,
    load_elevenlabs_stt_config,
    load_elevenlabs_tts_config,
    load_food_policy_config,
    load_openai_vision_config,
    load_vision_config,
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
    ood_enabled: bool = True
    ood_detector_path: str = ""
    ood_camera: str = "front"
    ood_encoder: str = "act_backbone"
    ood_every_n: int = 5
    ood_log_every_n: int = 1
    ood_log_in_dist_every_n: int = 0
    ood_failure_after_n: int = 3
    ood_tts_enabled: bool = True
    ood_tts_config_path: str = ".env"
    ood_tts_every_n: int = 3
    ood_tts_queue_max: int = 25
    openai_success_enabled: bool = False
    openai_success_config_path: str = ".env"
    openai_success_every_n: int = 15
    openai_success_sequence_frames: int = 5
    openai_success_sequence_stride: int = 5
    openai_success_grace_s: float = 15.0
    test_mode: bool = False
    test_audio_path: str = ""
    max_cycles: int = 0
    reset_pause_s: float = 7.0
    result_json_path: str = ""
    test_policy_steps: int = 5
    test_success_after_steps: int = 3


@dataclass(frozen=True)
class ClassifiedRequest:
    target: str | None
    transcript: str
    unsupported_item: bool = False
    unsupported_item_label: str = "that item"


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
    if cfg.openai_success_every_n <= 0:
        raise ValueError("--openai_success_every_n must be positive")
    if cfg.openai_success_sequence_frames <= 0:
        raise ValueError("--openai_success_sequence_frames must be positive")
    if cfg.openai_success_sequence_stride <= 0:
        raise ValueError("--openai_success_sequence_stride must be positive")
    if cfg.openai_success_grace_s < 0:
        raise ValueError("--openai_success_grace_s must be >= 0")
    if cfg.ood_every_n <= 0:
        raise ValueError("--ood_every_n must be positive")
    if cfg.ood_failure_after_n < 0:
        raise ValueError("--ood_failure_after_n must be non-negative")

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
    last_openai_success_check_frame = 0
    last_ood_check_frame = 0
    pending_openai_success: Future | None = None
    pending_openai_meta = None
    pending_ood: Future | None = None
    pending_ood_frame = 0
    outcome = "timeout"
    success_frame_buffer = deque(
        maxlen=max(1, cfg.openai_success_sequence_frames * cfg.openai_success_sequence_stride)
    )
    in_success_grace = False
    pending_grace_wait_logged = False

    engine.reset()
    engine.start()
    engine.resume()
    t_start = time.perf_counter()
    openai_executor = ThreadPoolExecutor(max_workers=1) if openai_success_config is not None else None
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
                    if pending_openai_success is None:
                        logger.info(
                            "OpenAI success grace limit reached (%.1fs)",
                            cfg.openai_success_grace_s,
                        )
                        break
                    if not pending_openai_success.done():
                        if not pending_grace_wait_logged:
                            logger.info(
                                "OpenAI success grace limit reached; waiting for pending success check"
                            )
                            pending_grace_wait_logged = True
            allow_openai_submit = policy_active or (
                openai_grace_available and grace_elapsed_s < cfg.openai_success_grace_s
            )

            obs_raw = robot.get_observation()
            frame = None
            if policy_active:
                try:
                    frame = extract_camera_frame(obs_raw, cfg.ood_camera)
                except KeyError as e:
                    logger.error("Camera frame extraction failed: %s", e)
                    shutdown_event.set()
                    continue

            n_seen += 1
            if policy_active and pending_ood is not None and pending_ood.done():
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

            if (
                policy_active
                and
                detector is not None
                and ood_executor is not None
                and pending_ood is None
                and n_seen - last_ood_check_frame >= cfg.ood_every_n
            ):
                assert frame is not None
                pending_ood = ood_executor.submit(score_ood_frame, detector, encoder, frame.copy())
                pending_ood_frame = n_seen
                last_ood_check_frame = n_seen

            if policy_active:
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

            if policy_active and vision_config.success.camera_name == cfg.ood_camera:
                assert frame is not None
                success_frame_raw = frame
            else:
                success_frame_raw = extract_camera_frame(obs_raw, vision_config.success.camera_name)
            success_frame_buffer.append(success_frame_raw.copy())
            success_result = success_detector.update(success_frame_raw, selected_policy.target)
            local_success_candidate = success_result.detected
            success_confirmed = False
            success_source = ""
            if pending_openai_success is not None and pending_openai_success.done():
                frame_id, opencv_candidate, opencv_score, sequence_len = pending_openai_meta
                try:
                    openai_result = pending_openai_success.result()
                except Exception as exc:
                    logger.warning(
                        "[OPENAI_SUCCESS] async check failed; not accepting OpenCV candidate "
                        "while OpenAI success is enabled: %s",
                        exc,
                    )
                else:
                    logger.info(
                        "[OPENAI_SUCCESS] frame=%d sequence_frames=%d success=%s confidence=%.3f hand=%s "
                        "robot=%s gripper_near_hand=%s robot_placing=%s visible=%s "
                        "in_user_hand=%s in_gripper=%s on_tray=%s correct_food=%s "
                        "user_grabbed=%s opencv_candidate=%s opencv_score=%.3f reason=%r",
                        frame_id,
                        sequence_len,
                        openai_result.success,
                        openai_result.confidence,
                        openai_result.user_hand_present,
                        openai_result.robot_visible,
                        openai_result.robot_gripper_near_user_hand,
                        openai_result.robot_placing_target_in_user_hand,
                        openai_result.target_food_visible,
                        openai_result.target_food_in_user_hand,
                        openai_result.target_food_in_robot_gripper,
                        openai_result.target_food_on_tray,
                        openai_result.correct_target_food,
                        openai_result.user_grabbing_without_robot_placement,
                        opencv_candidate,
                        opencv_score,
                        openai_result.reason,
                    )
                    success_confirmed = openai_result.success
                    success_source = "openai" if success_confirmed else ""
                    if opencv_candidate and not success_confirmed:
                        logger.info(
                            "[TASK_SUCCESS_CANDIDATE_REJECTED] target=%s frame=%d "
                            "opencv_score=%.3f openai_confidence=%.3f",
                            selected_policy.target,
                            frame_id,
                            opencv_score,
                            openai_result.confidence,
                        )
                pending_openai_success = None
                pending_openai_meta = None

            if openai_success_config is not None and openai_executor is not None:
                openai_due = n_seen - last_openai_success_check_frame >= cfg.openai_success_every_n
                if (
                    allow_openai_submit
                    and pending_openai_success is None
                    and (local_success_candidate or openai_due)
                ):
                    success_sequence = select_success_sequence(
                        success_frame_buffer,
                        cfg.openai_success_sequence_frames,
                        cfg.openai_success_sequence_stride,
                    )
                    last_openai_success_check_frame = n_seen
                    pending_openai_meta = (
                        n_seen,
                        local_success_candidate,
                        success_result.score,
                        len(success_sequence),
                    )
                    pending_openai_success = openai_executor.submit(
                        confirm_success_frame,
                        openai_success_config,
                        success_sequence,
                        selected_policy.display_name,
                    )
                    logger.info(
                        "[OPENAI_SUCCESS] submitted async check frame=%d sequence_frames=%d "
                        "opencv_candidate=%s opencv_score=%.3f",
                        n_seen,
                        len(success_sequence),
                        local_success_candidate,
                        success_result.score,
                    )
            elif local_success_candidate:
                logger.info(
                    "[TASK_SUCCESS_CANDIDATE_LOCAL_ONLY] target=%s frame=%d "
                    "opencv_score=%.3f openai_disabled=true; not accepting as success",
                    selected_policy.target,
                    n_seen,
                    success_result.score,
                )

            if success_confirmed:
                success = True
                success_frame = n_seen
                outcome = "success"
                logger.info(
                    "[TASK_SUCCESS] target=%s frame=%d source=%s opencv_score=%.3f %s",
                    selected_policy.target,
                    success_frame,
                    success_source,
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
        if openai_executor is not None:
            openai_executor.shutdown(wait=False, cancel_futures=True)
        if ood_executor is not None:
            ood_executor.shutdown(wait=False, cancel_futures=True)
        engine.stop()
        inner_robot = robot.inner
        if inner_robot.is_connected:
            inner_robot.disconnect()
        teleop = ctx.hardware.teleop
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
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


def speak(worker: ElevenLabsTTSWorker | None, phrase: str, wait: bool = False) -> None:
    if worker is None:
        logger.info("[TTS] disabled; skipping phrase=%r", phrase)
        return
    logger.info("[TTS] queue wait=%s phrase=%r", wait, phrase)
    if not worker.speak(phrase):
        logger.warning("TTS queue full; dropping voice phrase")
        return
    if wait and not worker.wait_until_idle(timeout=30.0):
        logger.warning("Timed out waiting for TTS phrase to finish")
    elif wait and worker.last_error is not None:
        logger.warning("TTS phrase finished with error: %s", worker.last_error)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
