#!/usr/bin/env python3
"""Replay local handoff fixtures through non-robot parts of the flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot_ood import (
    SuccessVisionConfig,
    TargetColorConfig,
    TargetSuccessDetector,
    classify_food_request,
    load_elevenlabs_stt_config,
    transcribe_audio_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/food_handoff_fixtures/manifest.json")
    parser.add_argument("--audio-dir", default="recordings/voice_requests")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--use-stt", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    print(f"fixture_manifest={args.manifest}")
    print(f"repo_id={manifest.get('repo_id')}")
    print(f"task={manifest.get('task')}")

    run_transcript_checks()
    if args.use_stt:
        run_stt_checks(Path(args.audio_dir), args.env)
    run_success_checks(manifest)


def run_transcript_checks() -> None:
    print("\n[transcript_classifier]")
    phrases = {
        "Please put the Strawberry in my hand.": "strawberry",
        "Please put the Oreo in my hand.": "oreo",
        "Please put the Marshmellow in my hand.": "marshmallow",
        "Please put the apple in my hand.": None,
    }
    for text, expected in phrases.items():
        actual = classify_food_request(text)
        status = "ok" if actual == expected else "mismatch"
        print(f"{status}: {text!r} -> {actual!r} expected={expected!r}")


def run_stt_checks(audio_dir: Path, env_path: str) -> None:
    print("\n[elevenlabs_stt]")
    config = load_elevenlabs_stt_config(env_path)
    for audio_path in sorted(audio_dir.glob("*_request.wav")):
        try:
            transcript = transcribe_audio_file(
                config,
                audio_path,
                keyterms=["Strawberry", "Oreo", "Marshmallow", "Marshmellow"],
            )
            target = classify_food_request(transcript)
            print(f"{audio_path.name}: transcript={transcript!r} target={target!r}")
        except Exception as exc:
            print(f"{audio_path.name}: error={exc}")


def run_success_checks(manifest: dict) -> None:
    print("\n[success_detector]")
    detector = TargetSuccessDetector(
        SuccessVisionConfig(
            camera_name="side",
            roi=(0.40, 0.32, 0.50, 0.45),
            min_blob_fraction=0.008,
            min_context_bright_fraction=0.15,
            context_bright_threshold=210,
            debounce_frames=1,
            targets={
                "strawberry": TargetColorConfig(
                    rgb_min=(95, 0, 0),
                    rgb_max=(255, 95, 95),
                )
            },
        )
    )
    rows = [
        row
        for row in manifest["fixtures"]
        if row["camera"] == "side" and row["target"] == "strawberry"
    ]
    for row in sorted(rows, key=lambda r: r["frame_index"]):
        frame = np.asarray(Image.open(row["path"]).convert("RGB"))
        result = detector.update(frame, "strawberry")
        print(
            f"frame={row['frame_index']:03d} expected_success={row['success']} "
            f"detected={result.detected} score={result.score:.3f} {result.reason}"
        )


if __name__ == "__main__":
    main()
