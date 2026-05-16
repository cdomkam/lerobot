from __future__ import annotations

import tempfile
import unittest
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import Mock

TTS_PATH = Path(__file__).resolve().parents[1] / "src" / "lerobot_ood" / "tts.py"
spec = importlib.util.spec_from_file_location("lerobot_ood_tts_test", TTS_PATH)
tts = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = tts
spec.loader.exec_module(tts)

(
    CHEETO_OOD_PHRASES,
    ElevenLabsTTSConfig,
    ElevenLabsTTSWorker,
    choose_cheeto_ood_phrase,
    load_elevenlabs_tts_config,
) = (
    tts.CHEETO_OOD_PHRASES,
    tts.ElevenLabsTTSConfig,
    tts.ElevenLabsTTSWorker,
    tts.choose_cheeto_ood_phrase,
    tts.load_elevenlabs_tts_config,
)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return b"mp3-bytes"


class TTSTest(unittest.TestCase):
    def test_phrase_inventory_has_twenty_non_empty_phrases(self):
        self.assertEqual(len(CHEETO_OOD_PHRASES), 20)
        self.assertTrue(all("Cheeto" in phrase for phrase in CHEETO_OOD_PHRASES))
        self.assertIn(choose_cheeto_ood_phrase(), CHEETO_OOD_PHRASES)

    def test_load_config_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "ELEVENLABS_API_KEY=test-key",
                        "ELEVENLABS_VOICE_ID=test-voice",
                        "ELEVENLABS_MODEL_ID=test-model",
                    ]
                )
            )

            config = load_elevenlabs_tts_config(path)

        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.voice_id, "test-voice")
        self.assertEqual(config.model_id, "test-model")

    def test_env_file_values_override_shell_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "ELEVENLABS_API_KEY=file-key",
                        "ELEVENLABS_VOICE_ID=file-voice",
                        "ELEVENLABS_MODEL_ID=file-model",
                    ]
                )
            )
            old_values = {
                key: os.environ.get(key)
                for key in ("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "ELEVENLABS_MODEL_ID")
            }
            os.environ["ELEVENLABS_API_KEY"] = "shell-key"
            os.environ["ELEVENLABS_VOICE_ID"] = "shell-voice"
            os.environ["ELEVENLABS_MODEL_ID"] = "shell-model"
            try:
                config = load_elevenlabs_tts_config(path)
            finally:
                for key, value in old_values.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(config.api_key, "file-key")
        self.assertEqual(config.voice_id, "file-voice")
        self.assertEqual(config.model_id, "file-model")

    def test_missing_required_config_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("ELEVENLABS_API_KEY=test-key\n")
            with self.assertRaisesRegex(ValueError, "--no-voice"):
                load_elevenlabs_tts_config(path)

    def test_missing_config_file_mentions_no_voice(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            with self.assertRaisesRegex(FileNotFoundError, "--no-voice"):
                load_elevenlabs_tts_config(path)

    def test_fetch_audio_uses_elevenlabs_stream_endpoint(self):
        opener = Mock(return_value=FakeResponse())
        player = Mock()
        config = ElevenLabsTTSConfig(
            api_key="test-key",
            voice_id="voice/id",
            model_id="test-model",
            api_base_url="https://example.test",
            player="/bin/echo",
        )
        worker = ElevenLabsTTSWorker(config, opener=opener, player=player)

        audio = worker._fetch_audio("hello")

        self.assertEqual(audio, b"mp3-bytes")
        request = opener.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://example.test/v1/text-to-speech/voice%2Fid/stream?output_format=mp3_44100_128",
        )
        self.assertEqual(request.headers["Xi-api-key"], "test-key")
        self.assertEqual(request.headers["Content-type"], "application/json")

    def test_speak_returns_false_when_queue_is_full(self):
        worker = ElevenLabsTTSWorker(
            ElevenLabsTTSConfig(api_key="k", voice_id="v"),
            max_queue_size=1,
            opener=Mock(return_value=FakeResponse()),
            player=Mock(),
        )

        worker._queue.put_nowait("already queued")
        worker._started = True

        self.assertFalse(worker.speak("new alert"))


if __name__ == "__main__":
    unittest.main()
