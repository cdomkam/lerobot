"""ElevenLabs text-to-speech alerts for OOD events."""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CHEETO_OOD_PHRASES = (
    "I say, that does not look like the Cheeto I was trained to find.",
    "Pardon me, but this Cheeto situation appears unfamiliar.",
    "Goodness me, I cannot make out the Cheeto in this scene.",
    "I am afraid this does not resemble my usual Cheeto arrangement.",
    "Steady on, this view is rather unlike the expected Cheeto.",
    "I beg your pardon, but the Cheeto appears to have gone off-script.",
    "This is most irregular; I do not recognise the Cheeto before me.",
    "By my reckoning, this is not the Cheeto I was looking for.",
    "I am not entirely convinced that I see the proper Cheeto here.",
    "How curious; the scene does not match my Cheeto training.",
    "I say, this Cheeto business looks distinctly out of distribution.",
    "Forgive me, but the Cheeto target is not presenting as expected.",
    "This appears to be a rather unfamiliar Cheeto predicament.",
    "I must report that the Cheeto does not look quite right.",
    "Dear me, I am seeing something rather unlike the expected Cheeto.",
    "The Cheeto, if present, is not in a form I confidently recognise.",
    "This view is a touch improper for a well-behaved Cheeto search.",
    "I am afraid the Cheeto evidence is not up to my usual standard.",
    "Rather odd, this does not look like my trained Cheeto scene.",
    "I should like a human to inspect this Cheeto situation, please.",
)


@dataclass(frozen=True)
class ElevenLabsTTSConfig:
    api_key: str
    voice_id: str
    model_id: str = "eleven_flash_v2_5"
    output_format: str = "mp3_44100_128"
    api_base_url: str = "https://api.elevenlabs.io"
    player: str = "/usr/bin/afplay"
    timeout_s: float = 20.0


def choose_cheeto_ood_phrase(rng: random.Random | None = None) -> str:
    """Return one British-English Cheeto/OOD alert phrase."""
    rng = rng or random
    return rng.choice(CHEETO_OOD_PHRASES)


def load_env_file(path: str | Path) -> dict[str, str]:
    """Parse a small dotenv-style file without mutating ``os.environ``."""
    env_path = Path(path)
    if not env_path.is_file():
        raise FileNotFoundError(f"TTS config file not found: {env_path}")

    values: dict[str, str] = {}
    for lineno, raw_line in enumerate(env_path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {lineno} in {env_path}: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid empty .env key on line {lineno} in {env_path}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_elevenlabs_tts_config(path: str | Path) -> ElevenLabsTTSConfig:
    """Load ElevenLabs config from process env plus a local .env file."""
    try:
        file_values = load_env_file(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"ElevenLabs voice alerts are enabled, but the TTS config file was not found: {path}. "
            "Create it with ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID, or run with --no-voice."
        ) from exc
    values = {**file_values, **os.environ}

    api_key = values.get("ELEVENLABS_API_KEY", "").strip()
    voice_id = values.get("ELEVENLABS_VOICE_ID", "").strip()
    if not api_key:
        raise ValueError(
            f"ElevenLabs voice alerts are enabled, but ELEVENLABS_API_KEY is missing in {path}. "
            "Add the key or run with --no-voice."
        )
    if not voice_id:
        raise ValueError(
            f"ElevenLabs voice alerts are enabled, but ELEVENLABS_VOICE_ID is missing in {path}. "
            "Add the voice id or run with --no-voice."
        )

    return ElevenLabsTTSConfig(
        api_key=api_key,
        voice_id=voice_id,
        model_id=values.get("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5").strip()
        or "eleven_flash_v2_5",
        output_format=values.get("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128").strip()
        or "mp3_44100_128",
        api_base_url=values.get("ELEVENLABS_API_BASE_URL", "https://api.elevenlabs.io").rstrip("/"),
        player=values.get("OOD_TTS_PLAYER", "/usr/bin/afplay").strip() or "/usr/bin/afplay",
        timeout_s=float(values.get("ELEVENLABS_TIMEOUT_S", "20")),
    )


class ElevenLabsTTSWorker:
    """Background worker that turns queued alert text into local speaker audio."""

    def __init__(
        self,
        config: ElevenLabsTTSConfig,
        max_queue_size: int = 25,
        opener=urllib.request.urlopen,
        player=subprocess.run,
        log_errors: bool = True,
    ):
        self.config = config
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue_size)
        self._opener = opener
        self._player = player
        self._log_errors = log_errors
        self._thread = threading.Thread(target=self._run, name="elevenlabs-tts", daemon=True)
        self._started = False
        self.last_error: BaseException | None = None

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def speak(self, text: str) -> bool:
        """Queue text for speech. Returns ``False`` if the queue is full."""
        if not self._started:
            self.start()
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            return False
        return True

    def close(self, timeout: float = 2.0) -> None:
        if not self._started:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                self._speak_now(text)
                self.last_error = None
            except Exception as exc:
                self.last_error = exc
                if self._log_errors:
                    logger.exception("ElevenLabs TTS alert failed")
            finally:
                self._queue.task_done()

    def _speak_now(self, text: str) -> None:
        audio = self._fetch_audio(text)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio)
            audio_path = f.name
        try:
            self._player([self.config.player, audio_path], check=True)
        finally:
            try:
                Path(audio_path).unlink()
            except FileNotFoundError:
                pass

    def _fetch_audio(self, text: str) -> bytes:
        query = urllib.parse.urlencode({"output_format": self.config.output_format})
        url = (
            f"{self.config.api_base_url}/v1/text-to-speech/"
            f"{urllib.parse.quote(self.config.voice_id, safe='')}/stream?{query}"
        )
        payload = json.dumps({"text": text, "model_id": self.config.model_id}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "xi-api-key": self.config.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.config.timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ElevenLabs TTS request failed with HTTP {exc.code} {exc.reason}: {body}"
            ) from exc
