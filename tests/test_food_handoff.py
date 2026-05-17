from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "lerobot_ood"


def load_module(name: str):
    package = sys.modules.get("lerobot_ood")
    if package is None:
        package = types.ModuleType("lerobot_ood")
        package.__path__ = [str(SRC_DIR)]
        sys.modules["lerobot_ood"] = package
    spec = importlib.util.spec_from_file_location(f"lerobot_ood.{name}", SRC_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


targets = load_module("targets")
tts = load_module("tts")
stt = load_module("stt")
vision = load_module("vision")


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FoodHandoffTest(unittest.TestCase):
    def test_transcript_classifier_maps_food_targets(self):
        self.assertEqual(targets.classify_food_request("Please bring me the Strawberry."), "strawberry")
        self.assertEqual(targets.classify_food_request("I would like an Oreo"), "oreo")
        self.assertEqual(targets.classify_food_request("Can I have the Marshmellow?"), "marshmallow")
        self.assertIsNone(targets.classify_food_request("Either the Oreo or the Strawberry is fine"))
        self.assertIsNone(targets.classify_food_request("Bring me something"))

    def test_policy_config_requires_all_three_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "food.json"
            path.write_text(
                json.dumps(
                    {
                        "targets": {
                            "strawberry": {
                                "policy_repo_id": "org/strawberry",
                                "task": "Pick up Strawberry",
                            },
                            "oreo": {"policy_repo_id": "org/oreo", "task": "Pick up Oreo"},
                            "marshmellow": {
                                "policy_repo_id": "org/marshmallow",
                                "task": "Pick up Marshmallow",
                            },
                        }
                    }
                )
            )

            config = targets.load_food_policy_config(path)

        self.assertEqual(config.require("Marshmellow").target, "marshmallow")
        self.assertEqual(config.require("oreo").policy_repo_id, "org/oreo")

    def test_stt_request_uses_speech_to_text_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "request.wav"
            audio.write_bytes(b"fake-wav")
            opener = Mock(return_value=FakeResponse({"text": "Bring me the Oreo"}))
            config = stt.ElevenLabsSTTConfig(
                api_key="key",
                model_id="scribe_v2",
                api_base_url="https://example.test",
            )

            transcript = stt.transcribe_audio_file(
                config,
                audio,
                keyterms=["Strawberry", "Oreo"],
                opener=opener,
            )

        self.assertEqual(transcript, "Bring me the Oreo")
        request = opener.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/v1/speech-to-text")
        self.assertEqual(request.headers["Xi-api-key"], "key")
        self.assertIn("multipart/form-data", request.headers["Content-type"])
        self.assertIn(b'name="model_id"', request.data)
        self.assertIn(b"scribe_v2", request.data)
        self.assertIn(b'name="keyterms"', request.data)

    def test_hand_presence_debounces_roi_change(self):
        config = vision.HandVisionConfig(
            roi=(0, 0, 1, 1),
            baseline_frames=2,
            debounce_frames=2,
            min_mean_abs_diff=10,
            min_changed_fraction=0.1,
            changed_pixel_threshold=10,
        )
        detector = vision.HandPresenceDetector(config)
        empty = np.zeros((10, 10, 3), dtype=np.uint8)
        changed = np.full((10, 10, 3), 80, dtype=np.uint8)

        self.assertFalse(detector.update(empty).ready)
        self.assertFalse(detector.update(empty).ready)
        self.assertFalse(detector.update(changed).detected)
        self.assertTrue(detector.update(changed).detected)

    def test_success_detector_debounces_target_color(self):
        config = vision.SuccessVisionConfig(
            roi=(0, 0, 1, 1),
            min_blob_fraction=0.05,
            debounce_frames=2,
            targets={
                "strawberry": vision.TargetColorConfig(
                    rgb_min=(120, 0, 0),
                    rgb_max=(255, 80, 80),
                )
            },
        )
        detector = vision.TargetSuccessDetector(config)
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        frame[:10, :10] = np.array([200, 20, 20], dtype=np.uint8)

        self.assertFalse(detector.update(frame, "strawberry").detected)
        self.assertTrue(detector.update(frame, "strawberry").detected)


if __name__ == "__main__":
    unittest.main()
