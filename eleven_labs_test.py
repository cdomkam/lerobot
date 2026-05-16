#!/usr/bin/env python3
"""Smoke-test ElevenLabs TTS credentials and local speaker playback.

This does not connect to the robot. It loads the local `.env`, calls
ElevenLabs text-to-speech with the configured voice, and plays the result
through the same worker used by the OOD runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_tts_module(repo_root: Path):
    path = repo_root / "src" / "lerobot_ood" / "tts.py"
    spec = importlib.util.spec_from_file_location("lerobot_ood_tts_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load TTS module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test ElevenLabs TTS credentials and laptop speaker playback."
    )
    parser.add_argument("--env", default=".env", help="Path to ElevenLabs .env file.")
    parser.add_argument(
        "--text",
        default=None,
        help="Text to speak. Defaults to a random Cheeto/OOD phrase.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    tts = _load_tts_module(repo_root)

    config = tts.load_elevenlabs_tts_config(args.env)
    phrase = args.text or tts.choose_cheeto_ood_phrase()
    worker = tts.ElevenLabsTTSWorker(config, max_queue_size=2)

    print(f"voice_id={config.voice_id}")
    print(f"model_id={config.model_id}")
    print(f"phrase={phrase}")

    if not worker.speak(phrase):
        raise RuntimeError("TTS queue rejected the alert")
    worker._queue.join()
    worker.close()
    print("elevenlabs_tts_smoke_ok")


if __name__ == "__main__":
    main()
