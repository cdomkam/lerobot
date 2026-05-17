#!/usr/bin/env python3
"""Smoke-test ElevenLabs STT credentials with a local audio file."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path


def _load_module(repo_root: Path, name: str):
    package = sys.modules.get("lerobot_ood")
    if package is None:
        package = types.ModuleType("lerobot_ood")
        package.__path__ = [str(repo_root / "src" / "lerobot_ood")]
        sys.modules["lerobot_ood"] = package
    path = repo_root / "src" / "lerobot_ood" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"lerobot_ood.{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test ElevenLabs speech-to-text credentials.")
    parser.add_argument("audio", help="Path to a short WAV/MP3/M4A request clip.")
    parser.add_argument("--env", default=".env", help="Path to ElevenLabs .env file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    tts = _load_module(repo_root, "tts")
    stt = _load_module(repo_root, "stt")
    targets = _load_module(repo_root, "targets")

    try:
        config = stt.load_elevenlabs_stt_config(args.env)
        transcript = stt.transcribe_audio_file(
            config,
            args.audio,
            keyterms=["Strawberry", "Oreo", "Marshmallow", "Marshmellow"],
        )
    except Exception as exc:
        print(f"ElevenLabs STT error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    target = targets.classify_food_request(transcript)
    print("model_id=scribe_v2")
    print(f"transcript={transcript}")
    print(f"target={target or 'unclassified'}")
    print("elevenlabs_stt_smoke_ok")


if __name__ == "__main__":
    main()
