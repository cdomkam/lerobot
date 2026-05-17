"""ElevenLabs speech-to-text for short food request clips."""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .tts import load_env_file


@dataclass(frozen=True)
class ElevenLabsSTTConfig:
    api_key: str
    api_base_url: str = "https://api.elevenlabs.io"
    timeout_s: float = 30.0
    language_code: str = "en"


def load_elevenlabs_stt_config(path: str | Path) -> ElevenLabsSTTConfig:
    """Load ElevenLabs STT config, preferring local .env values over shell env."""
    try:
        file_values = load_env_file(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"ElevenLabs speech-to-text is enabled, but the config file was not found: {path}. "
            "Create it with ELEVENLABS_API_KEY, or run with --no-stt and --target."
        ) from exc
    values = {**os.environ, **file_values}
    api_key = values.get("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            f"ElevenLabs speech-to-text is enabled, but ELEVENLABS_API_KEY is missing in {path}. "
            "Add the key, or run with --no-stt and --target."
        )
    return ElevenLabsSTTConfig(
        api_key=api_key,
        api_base_url=values.get("ELEVENLABS_API_BASE_URL", "https://api.elevenlabs.io").rstrip("/"),
        timeout_s=float(values.get("ELEVENLABS_TIMEOUT_S", "30")),
        language_code=values.get("ELEVENLABS_STT_LANGUAGE_CODE", "en").strip() or "en",
    )


def transcribe_audio_file(
    config: ElevenLabsSTTConfig,
    audio_path: str | Path,
    keyterms: list[str] | None = None,
    opener=urllib.request.urlopen,
) -> str:
    audio = Path(audio_path)
    fields: list[tuple[str, str]] = [
        ("model_id", "scribe_v2"),
        ("tag_audio_events", "false"),
        ("diarize", "false"),
        ("timestamps_granularity", "none"),
    ]
    if config.language_code:
        fields.append(("language_code", config.language_code))
    for keyterm in keyterms or []:
        fields.append(("keyterms", keyterm))

    body, content_type = _multipart_body(fields, "file", audio)
    url = f"{config.api_base_url}/v1/speech-to-text"
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "xi-api-key": config.api_key,
            "Content-Type": content_type,
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=config.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ElevenLabs STT request failed with HTTP {exc.code} {exc.reason}: {body_text}"
        ) from exc
    transcript = str(payload.get("text", "")).strip()
    if not transcript:
        raise RuntimeError(f"ElevenLabs STT response did not contain transcript text: {payload}")
    return transcript


def _multipart_body(
    fields: list[tuple[str, str]],
    file_field: str,
    file_path: Path,
) -> tuple[bytes, str]:
    boundary = f"----lerobot-ood-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"
