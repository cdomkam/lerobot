#!/usr/bin/env python3
"""Smoke-test ElevenLabs TTS credentials and local speaker playback.

This does not connect to the robot. It loads the local `.env`, calls
ElevenLabs text-to-speech with the configured voice, and plays the result
through the same worker used by the OOD runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError

PLACEHOLDER_VALUES = {
    "your_elevenlabs_api_key",
    "your_elevenlabs_voice_id",
}


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
        help="Text to speak. Defaults to a random Strawberry/OOD phrase.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    tts = _load_tts_module(repo_root)

    try:
        config = tts.load_elevenlabs_tts_config(args.env)
    except Exception as exc:
        print(f"ElevenLabs config error: {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("Expected local setup:", file=sys.stderr)
        print("  cp .env.example .env", file=sys.stderr)
        print("  # edit .env and add ELEVENLABS_API_KEY=your_key_here", file=sys.stderr)
        raise SystemExit(1) from exc
    if config.api_key in PLACEHOLDER_VALUES or config.voice_id in PLACEHOLDER_VALUES:
        print(f"ElevenLabs config error: {args.env} still contains placeholder values.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Expected local setup:", file=sys.stderr)
        print("  cp .env.example .env", file=sys.stderr)
        print("  # edit .env and replace ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID", file=sys.stderr)
        raise SystemExit(1)

    if platform.system() == "Darwin" and not Path(config.player).exists():
        print(f"Audio player not found: {config.player}", file=sys.stderr)
        raise SystemExit(1)
    if platform.system() != "Darwin" and shutil.which(config.player) is None:
        print(f"Audio player not found on PATH: {config.player}", file=sys.stderr)
        raise SystemExit(1)

    phrase = args.text or tts.choose_strawberry_ood_phrase()
    worker = tts.ElevenLabsTTSWorker(config, max_queue_size=2, log_errors=False)

    print(f"voice_id={config.voice_id}")
    print(f"model_id={config.model_id}")
    print(f"phrase={phrase}")

    try:
        if not worker.speak(phrase):
            print("TTS queue rejected the alert", file=sys.stderr)
            raise SystemExit(1)
        worker._queue.join()
        if worker.last_error is not None:
            raise worker.last_error
    except HTTPError as exc:
        print(f"ElevenLabs HTTP error: {exc.code} {exc.reason}", file=sys.stderr)
        print("Check that the API key belongs to the workspace and can access the voice id.", file=sys.stderr)
        raise SystemExit(1) from exc
    except URLError as exc:
        print(f"ElevenLabs network error: {exc.reason}", file=sys.stderr)
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        print(f"ElevenLabs TTS error: {exc}", file=sys.stderr)
        print("Check that the API key belongs to the workspace and can access the voice id.", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        worker.close()
    print("elevenlabs_tts_smoke_ok")


if __name__ == "__main__":
    main()
