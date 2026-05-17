#!/usr/bin/env python3
"""No-robot end-to-end test path for the food handoff flow."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from lerobot_ood import (
    ElevenLabsTTSWorker,
    choose_fetching_phrase,
    choose_food_handoff_ood_phrase,
    choose_success_phrase,
    choose_unsupported_item_phrase,
    classify_food_request,
    is_probable_unsupported_food_request,
    load_elevenlabs_stt_config,
    load_elevenlabs_tts_config,
    load_food_policy_config,
    transcribe_audio_file,
    unsupported_item_label,
)

logger = logging.getLogger("run_food_handoff_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--food-policy-config", required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--stt-enabled", default="true")
    parser.add_argument("--test-audio-path", default="")
    parser.add_argument("--tts-enabled", default="true")
    parser.add_argument("--tts-config-path", default=".env")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--reset-pause-s", type=float, default=7.0)
    parser.add_argument("--result-json-path", default="")
    parser.add_argument("--test-policy-steps", type=int, default=5)
    parser.add_argument("--test-success-after-steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    stt_enabled = parse_bool(args.stt_enabled)
    tts_enabled = parse_bool(args.tts_enabled)
    if args.max_cycles < 0:
        raise ValueError("--max-cycles must be >= 0")
    if args.reset_pause_s < 0:
        raise ValueError("--reset-pause-s must be >= 0")

    policy_config = load_food_policy_config(args.food_policy_config)
    cycle_results = []

    tts_worker = None
    if tts_enabled:
        tts_config = load_elevenlabs_tts_config(args.tts_config_path)
        tts_worker = ElevenLabsTTSWorker(tts_config)
        tts_worker.start()
        logger.info(
            "ElevenLabs TTS enabled (voice_id=%s, model_id=%s)",
            tts_config.voice_id,
            tts_config.model_id,
        )

    stt_config = None
    if stt_enabled and not args.target.strip():
        if not args.test_audio_path:
            raise ValueError("--test-audio-path is required when STT is enabled in test mode")
        stt_config = load_elevenlabs_stt_config(args.tts_config_path)

    cycle = 0
    logger.info(
        "Starting mocked handoff loop (max_cycles=%s, reset_pause_s=%.1f)",
        args.max_cycles if args.max_cycles > 0 else "unlimited",
        args.reset_pause_s,
    )
    try:
        while args.max_cycles == 0 or cycle < args.max_cycles:
            cycle += 1
            logger.info("[CYCLE] start cycle=%d", cycle)

            target = args.target.strip().lower() or None
            if target is None:
                if not stt_enabled:
                    raise ValueError("--stt-enabled=false requires --target")
                assert stt_config is not None
                transcript = transcribe_audio_file(
                    stt_config,
                    args.test_audio_path,
                    keyterms=["Strawberry", "Oreo", "Marshmallow", "Marshmellow"],
                )
                target = classify_food_request(transcript)
                logger.info(
                    "[REQUEST] test_audio=%s transcript=%r target=%s",
                    args.test_audio_path,
                    transcript,
                    target,
                )
                if target is None:
                    logger.warning("[REQUEST] could not classify target")
                    if is_probable_unsupported_food_request(transcript):
                        phrase = choose_unsupported_item_phrase(unsupported_item_label(transcript))
                        outcome = "unsupported_item"
                    else:
                        phrase = choose_food_handoff_ood_phrase()
                        outcome = "ood_unclassified"
                    speak(tts_worker, phrase, wait=True)
                    logger.info("[CYCLE] complete cycle=%d outcome=%s", cycle, outcome)
                    cycle_results.append(
                        {
                            "cycle": cycle,
                            "target": target,
                            "outcome": outcome,
                            "success": False,
                        }
                    )
                    write_result_json(args.result_json_path, cycle_results)
                    reset_between_cycles(args, cycle)
                    continue

            selected_policy = policy_config.require(target)
            logger.info(
                "[REQUEST] target=%s policy=%s task=%r",
                selected_policy.target,
                selected_policy.policy_repo_id,
                selected_policy.task,
            )
            speak(tts_worker, choose_fetching_phrase(selected_policy.display_name), wait=True)
            outcome = run_mock_policy_cycle(args, selected_policy, tts_worker)
            logger.info("[CYCLE] complete cycle=%d outcome=%s", cycle, outcome)
            cycle_results.append(
                {
                    "cycle": cycle,
                    "target": selected_policy.target,
                    "display_name": selected_policy.display_name,
                    "policy_repo_id": selected_policy.policy_repo_id,
                    "task": selected_policy.task,
                    "outcome": outcome,
                    "success": outcome == "success",
                }
            )
            write_result_json(args.result_json_path, cycle_results)
            reset_between_cycles(args, cycle)
    finally:
        close_tts(tts_worker)
        write_result_json(args.result_json_path, cycle_results)
        logger.info("Mocked handoff loop stopped after %d cycle(s)", cycle)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def close_tts(worker: ElevenLabsTTSWorker | None) -> None:
    if worker is not None:
        worker.close(timeout=10.0)


def write_result_json(path: str, cycle_results: list[dict]) -> None:
    if not path:
        return
    payload = {
        "status": cycle_results[-1]["outcome"] if cycle_results else "not_started",
        "success": bool(cycle_results and cycle_results[-1]["success"]),
        "cycles": cycle_results,
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def reset_between_cycles(args: argparse.Namespace, cycle: int) -> None:
    if args.max_cycles > 0 and cycle >= args.max_cycles:
        return
    if args.reset_pause_s <= 0:
        return
    logger.info(
        "[CYCLE] reset pause %.1fs; reset the scene before the next cycle",
        args.reset_pause_s,
    )
    time.sleep(args.reset_pause_s)


def run_mock_policy_cycle(
    args: argparse.Namespace,
    selected_policy,
    tts_worker: ElevenLabsTTSWorker | None,
) -> str:
    logger.info(
        "[POLICY] mocked starting target=%s policy=%s task=%r",
        selected_policy.target,
        selected_policy.policy_repo_id,
        selected_policy.task,
    )

    success = False
    success_frame = "none"
    n_actions = 0
    for step in range(1, args.test_policy_steps + 1):
        logger.info("[MOCK_ACTION] frame=%d target=%s", step, selected_policy.target)
        n_actions += 1
        if step >= args.test_success_after_steps:
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
        time.sleep(1.0 / max(args.fps, 1))

    logger.info(
        "Run complete: target=%s success=%s success_frame=%s frames=%d actions=%d "
        "ood=0 (0.0%%) mean_score=0.000 threshold=0.000 test_mode=true",
        selected_policy.target,
        success,
        success_frame,
        int(success_frame) if success else args.test_policy_steps,
        n_actions,
    )
    return "success" if success else "timeout"


if __name__ == "__main__":
    main()
