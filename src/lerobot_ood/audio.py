"""Local microphone recording helpers."""

from __future__ import annotations

import shlex
import subprocess
import wave
from pathlib import Path

import numpy as np


def record_wav(
    output_path: str | Path,
    duration_s: float,
    sample_rate: int = 16000,
    channels: int = 1,
    recorder_command: str = "",
) -> Path:
    """Record a short WAV clip from the default microphone."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if recorder_command:
        command = [
            part.format(
                path=str(path),
                duration_s=str(duration_s),
                sample_rate=str(sample_rate),
                channels=str(channels),
            )
            for part in shlex.split(recorder_command)
        ]
        subprocess.run(command, check=True)
        return path

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Microphone recording requires the 'sounddevice' package. "
            "Install it in the runtime environment or set REQUEST_RECORDER_COMMAND."
        ) from exc

    n_samples = max(1, int(duration_s * sample_rate))
    data = sd.rec(n_samples, samplerate=sample_rate, channels=channels, dtype="float32")
    sd.wait()
    pcm = np.clip(data, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return path
