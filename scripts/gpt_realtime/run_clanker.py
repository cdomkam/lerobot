#!/usr/bin/env python
"""Realtime speech-to-tool butler that dispatches full food handoff cycles."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import math
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SAMPLE_RATE = 24_000
MIC_SUPPRESS_AFTER_ASSISTANT_AUDIO_S = 1.5
DEFAULT_DRY_RUN_SECONDS = 15.0
ELEVATOR_MUSIC_SAMPLE_RATE = 16_000
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
HANDOFF_SCRIPT = ROOT_DIR / "scripts" / "run_food_handoff.sh"
DEFAULT_ENV_FILE = ROOT_DIR / ".env"
DEFAULT_ELEVATOR_MUSIC_PATH = SCRIPT_DIR / "assets" / "elevator_music.mp3"
DEFAULT_FOOD_POLICY_CONFIG = ROOT_DIR / "config" / "food_policies.json"
TARGET_ALIASES = {
    "strawberry": "strawberry",
    "strawberries": "strawberry",
    "oreo": "oreo",
    "oreos": "oreo",
    "cookie": "oreo",
    "cookies": "oreo",
    "marshmallow": "marshmallow",
    "marshmallows": "marshmallow",
    "marshmellow": "marshmallow",
    "marshmellows": "marshmallow",
}

_TTS_SPEC = importlib.util.spec_from_file_location(
    "realtime_handoff_tts",
    ROOT_DIR / "src" / "lerobot_ood" / "tts.py",
)
if _TTS_SPEC is None or _TTS_SPEC.loader is None:
    raise RuntimeError("Could not load lerobot_ood.tts")
_tts = importlib.util.module_from_spec(_TTS_SPEC)
sys.modules[_TTS_SPEC.name] = _tts
_TTS_SPEC.loader.exec_module(_tts)
ElevenLabsTTSWorker = _tts.ElevenLabsTTSWorker
load_elevenlabs_tts_config = _tts.load_elevenlabs_tts_config
choose_unsupported_item_phrase = _tts.choose_unsupported_item_phrase

SYSTEM_PROMPT = """\
You are Alfred, a polite, funny robot butler. You listen to the user and orchestrate
exactly one snack handoff at a time for strawberry, marshmallow, or oreo.

You do not produce audio yourself. Any text you emit is converted to speech locally
with ElevenLabs, so keep replies crisp and suitable for speaking aloud. Use British
phrasing such as "very good", "right you are", "splendid", "I say", and "shall".

When the user clearly asks for one of the three snacks, call run_handoff with that
target. If the request is ambiguous, ask a short clarifying question instead of
calling the tool. If the user asks for anything outside strawberry, marshmallow, or
oreo, call unsupported_item_requested with the requested item. Do not improvise
your own unavailable-item line; the local control loop will speak it.

The run_handoff tool speaks the fetching line through ElevenLabs, immediately runs
the requested robot policy, checks side-camera success using GPT-5.4-nano in the
background, speaks the final success phrase through ElevenLabs, and returns
structured status. While the tool call is pending, the microphone is muted. After a
successful tool result, remain silent because the control loop has already spoken
the outcome. After an unsupported_item_requested result, remain silent because the
control loop has already spoken the unavailable-item line. If the tool returns
failure or error, briefly apologise and name the problem.
"""


@dataclass(frozen=True)
class PolicyConfig:
    target: str
    path: Path
    display_name: str
    policy_repo_id: str
    task: str


@dataclass
class ActivePolicyRun:
    target: str
    started_at: float
    kind: str
    process: subprocess.Popen | None = None
    thread: threading.Thread | None = None


class ElevatorMusic:
    def __init__(self, music_path: Path = DEFAULT_ELEVATOR_MUSIC_PATH) -> None:
        self.music_path = music_path
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.process: subprocess.Popen | None = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._play_loop, name="clanker-elevator-music", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._stop_process()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)

    def _play_loop(self) -> None:
        if self.music_path.is_file() and Path("/usr/bin/afplay").is_file():
            self._play_file_loop()
            return
        try:
            import sounddevice as sd
        except ImportError:
            return
        try:
            with sd.RawOutputStream(
                samplerate=ELEVATOR_MUSIC_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=1600,
            ) as stream:
                phase = 0.0
                notes = (261.63, 329.63, 392.0, 523.25, 392.0, 329.63)
                note_index = 0
                while not self.stop_event.is_set():
                    freq = notes[note_index % len(notes)]
                    note_index += 1
                    chunk, phase = self._tone_chunk(freq, phase, duration_s=0.35)
                    stream.write(chunk)
                    if self.stop_event.wait(0.03):
                        break
        except Exception as exc:
            print(f"[music] Elevator music unavailable: {exc}", file=sys.stderr)

    def _play_file_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.process = subprocess.Popen(
                    ["/usr/bin/afplay", "-v", "0.25", str(self.music_path)]
                )
                while self.process.poll() is None:
                    if self.stop_event.wait(0.2):
                        self._stop_process()
                        break
            except Exception as exc:
                print(f"[music] Elevator music file unavailable: {exc}", file=sys.stderr)
                return
            finally:
                self.process = None

    def _stop_process(self) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)

    def _tone_chunk(self, freq: float, phase: float, duration_s: float) -> tuple[bytes, float]:
        n_samples = int(ELEVATOR_MUSIC_SAMPLE_RATE * duration_s)
        frames = bytearray()
        for i in range(n_samples):
            envelope = min(1.0, i / 800, (n_samples - i) / 800)
            value = (
                math.sin(phase) * 0.13
                + math.sin(phase * 1.5) * 0.035
                + math.sin(phase * 2.0) * 0.02
            ) * envelope
            sample = int(max(-1.0, min(1.0, value)) * 32767)
            frames.extend(sample.to_bytes(2, "little", signed=True))
            phase += 2 * math.pi * freq / ELEVATOR_MUSIC_SAMPLE_RATE
            if phase > 2 * math.pi:
                phase -= 2 * math.pi
        return bytes(frames), phase


def canonical_target(value: str) -> str:
    normalized = "".join(ch for ch in value.lower() if ch.isalnum())
    target = TARGET_ALIASES.get(normalized)
    if target is None:
        expected = ", ".join(sorted({"strawberry", "marshmallow", "oreo"}))
        raise ValueError(f"unknown food target {value!r}; expected one of: {expected}")
    return target


def load_policy_configs(config_path: Path) -> dict[str, PolicyConfig]:
    if not config_path.is_file():
        raise FileNotFoundError(f"missing food policy config: {config_path}")
    raw = json.loads(config_path.read_text())
    targets = raw.get("targets", {})
    if not isinstance(targets, dict):
        raise ValueError(f"{config_path} must contain a targets object")
    configs: dict[str, PolicyConfig] = {}
    for target in ("strawberry", "oreo", "marshmallow"):
        values = targets.get(target)
        if not isinstance(values, dict):
            raise ValueError(f"{config_path} is missing target {target!r}")
        policy_repo_id = str(values.get("policy_repo_id", "")).strip()
        task = str(values.get("task", "")).strip()
        if not policy_repo_id or not task:
            raise ValueError(f"{config_path} target {target!r} needs policy_repo_id and task")
        configs[target] = PolicyConfig(
            target=target,
            path=config_path,
            display_name=str(values.get("display_name", target.title())).strip() or target.title(),
            policy_repo_id=policy_repo_id,
            task=task,
        )
    return configs


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


class PolicyRunner:
    def __init__(
        self,
        configs: dict[str, PolicyConfig],
        dry_run: bool = False,
        dry_run_seconds: float = DEFAULT_DRY_RUN_SECONDS,
    ) -> None:
        self.configs = configs
        self.dry_run = dry_run
        self.dry_run_seconds = max(0.0, dry_run_seconds)
        self.run_lock = threading.Lock()
        self.lock = threading.Lock()
        self.active: ActivePolicyRun | None = None
        self.stop_requested = threading.Event()

    def run(self, target_value: str) -> dict[str, Any]:
        target = canonical_target(target_value)
        config = self.configs[target]
        command = [
            str(HANDOFF_SCRIPT),
            "--no-stt",
            "--target",
            target,
            "--max-cycles",
            "1",
            "--reset-pause-s",
            "0",
        ]
        env = os.environ.copy()
        env.setdefault("RUN_ID", f"gpt_realtime_{target}_{time.strftime('%Y%m%d_%H%M%S')}")
        env.setdefault("OPENAI_SUCCESS_ENABLED", "true")
        env.setdefault("STT_ENABLED", "false")

        with self.run_lock:
            result_path = None
            with self.lock:
                if self.dry_run:
                    self.active = ActivePolicyRun(
                        target=target,
                        started_at=time.monotonic(),
                        kind="dry_run",
                        thread=threading.current_thread(),
                    )
                    self.stop_requested.clear()
                    active = self.active
                else:
                    if not HANDOFF_SCRIPT.is_file():
                        return {
                            "status": "error",
                            "ok": False,
                            "target": target,
                            "error": f"handoff runner not found: {HANDOFF_SCRIPT}",
                        }

                    result_file = tempfile.NamedTemporaryFile(
                        prefix=f"realtime_handoff_{target}_",
                        suffix=".json",
                        delete=False,
                    )
                    result_path = Path(result_file.name)
                    result_file.close()
                    env["RESULT_JSON_PATH"] = str(result_path)
                    try:
                        process = subprocess.Popen(command, cwd=ROOT_DIR, env=env)
                    except Exception as exc:
                        try:
                            result_path.unlink()
                        except FileNotFoundError:
                            pass
                        return {
                            "status": "error",
                            "ok": False,
                            "target": target,
                            "error": str(exc),
                            "policy_repo_id": config.policy_repo_id,
                            "task": config.task,
                        }

                    self.active = ActivePolicyRun(
                        target=target,
                        started_at=time.monotonic(),
                        kind="robot",
                        process=process,
                        thread=threading.current_thread(),
                    )
                    self.stop_requested.clear()
                    active = self.active

            if self.dry_run:
                return self._run_fake_policy(active, config, command)
            assert result_path is not None
            return self._wait_for_handoff(active, config, command, result_path)

    def stop_active(self) -> None:
        with self.lock:
            active = self.active
        if active is None:
            return
        self.stop_requested.set()
        process = active.process
        if process is not None and process.poll() is None:
            print(f"[handoff] Stopping active {active.target} handoff...", flush=True)
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        if active.thread is not None and active.thread is not threading.current_thread():
            active.thread.join(timeout=6.0)
        with self.lock:
            if self.active is active:
                self.active = None

    def _run_fake_policy(
        self,
        active: ActivePolicyRun,
        config: PolicyConfig,
        command: list[str],
    ) -> dict[str, Any]:
        print(
            f"\n[handoff] Dry run started {active.target}: "
            f"pretending to run {config.policy_repo_id} for {self.dry_run_seconds:.1f}s",
            flush=True,
        )
        deadline = time.monotonic() + self.dry_run_seconds
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            if self.stop_requested.wait(timeout=min(0.25, remaining)):
                break
        if self.stop_requested.is_set():
            print(f"[handoff] Dry run stopped {active.target}", flush=True)
            result = self._failure_result(active.target, config, "dry run stopped before completion")
        else:
            print(f"[handoff] Dry run finished {active.target}", flush=True)
            result = self._success_result(active.target, config, command, dry_run=True)
        self._clear_active(active)
        return result

    def _wait_for_handoff(
        self,
        active: ActivePolicyRun,
        config: PolicyConfig,
        command: list[str],
        result_path: Path,
    ) -> dict[str, Any]:
        assert active.process is not None
        print(f"\n[handoff] Started {active.target}: {config.policy_repo_id}", flush=True)
        exit_code = active.process.wait()
        print(
            f"[handoff] Finished {active.target} exit_code={exit_code} "
            f"repo={config.policy_repo_id}",
            flush=True,
        )
        self._clear_active(active)
        try:
            payload = json.loads(result_path.read_text()) if result_path.is_file() else {}
        except Exception as exc:
            payload = {"status": "error", "success": False, "error": f"invalid result json: {exc}"}
        finally:
            try:
                result_path.unlink()
            except FileNotFoundError:
                pass

        if exit_code != 0:
            result = self._error_result(
                active.target,
                config,
                f"handoff runner exited with code {exit_code}",
                exit_code=exit_code,
            )
            result["handoff_result"] = payload
            return result
        return self._result_from_handoff_payload(
            active.target,
            config,
            command,
            payload,
            pid=active.process.pid,
            exit_code=exit_code,
        )

    def _clear_active(self, active: ActivePolicyRun) -> None:
        with self.lock:
            if self.active is active:
                self.active = None

    def _success_result(
        self,
        target: str,
        config: PolicyConfig,
        command: list[str],
        dry_run: bool,
        pid: int | None = None,
        exit_code: int | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "success",
            "ok": True,
            "target": target,
            "success": True,
            "spoken_by_control_loop": True,
            "dry_run": dry_run,
            "policy_repo_id": config.policy_repo_id,
            "task": config.task,
            "command": command,
        }
        if dry_run:
            result["fake_process_seconds"] = self.dry_run_seconds
        if pid is not None:
            result["pid"] = pid
        if exit_code is not None:
            result["exit_code"] = exit_code
        return result

    def _failure_result(self, target: str, config: PolicyConfig, reason: str) -> dict[str, Any]:
        return {
            "status": "failure",
            "ok": False,
            "success": False,
            "target": target,
            "reason": reason,
            "policy_repo_id": config.policy_repo_id,
            "task": config.task,
        }

    def _error_result(
        self,
        target: str,
        config: PolicyConfig,
        error: str,
        exit_code: int | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "error",
            "ok": False,
            "success": False,
            "target": target,
            "error": error,
            "policy_repo_id": config.policy_repo_id,
            "task": config.task,
        }
        if exit_code is not None:
            result["exit_code"] = exit_code
        return result


    def _result_from_handoff_payload(
        self,
        target: str,
        config: PolicyConfig,
        command: list[str],
        payload: dict[str, Any],
        pid: int,
        exit_code: int,
    ) -> dict[str, Any]:
        success = bool(payload.get("success"))
        status = "success" if success else "failure"
        result: dict[str, Any] = {
            "status": status,
            "ok": success,
            "success": success,
            "target": target,
            "policy_repo_id": config.policy_repo_id,
            "task": config.task,
            "command": command,
            "pid": pid,
            "exit_code": exit_code,
            "spoken_by_control_loop": True,
            "handoff_result": payload,
        }
        cycles = payload.get("cycles")
        if isinstance(cycles, list) and cycles:
            last = cycles[-1]
            result["outcome"] = last.get("outcome")
            result["cycle"] = last.get("cycle")
            result["reason"] = last.get("outcome")
        else:
            result["reason"] = payload.get("status", "missing_result")
        return result


class AudioIO:
    def __init__(self, send_audio, stop_event: threading.Event) -> None:
        self.send_audio = send_audio
        self.stop_event = stop_event
        self.input_queue: queue.Queue[bytes] = queue.Queue(maxsize=100)
        self.output_queue: queue.Queue[tuple[int, str, bytes]] = queue.Queue(maxsize=200)
        self.threads: list[threading.Thread] = []
        self.input_stream = None
        self.output_stream = None
        self.output_generation = 0
        self.output_lock = threading.Lock()
        self.played_ms_by_item: dict[str, float] = {}
        self.last_output_playback_at = 0.0

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "run_clanker.py needs sounddevice for microphone audio. "
                "Install project dependencies first with `uv sync`, or install sounddevice."
            ) from exc

        self.input_stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=1200,
            callback=self._on_input,
        )
        self.input_stream.start()
        self.threads = [
            threading.Thread(target=self._input_sender, name="clanker-audio-input", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def close(self) -> None:
        self.stop_event.set()
        for stream in (self.input_stream, self.output_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass

    def play(self, item_id: str, audio_bytes: bytes) -> None:
        if not audio_bytes:
            return
        with self.output_lock:
            generation = self.output_generation
            self.played_ms_by_item.setdefault(item_id, 0.0)
        # Keep chunks short so an interruption stops audible playback quickly.
        for start in range(0, len(audio_bytes), 2400):
            try:
                self.output_queue.put_nowait((generation, item_id, audio_bytes[start : start + 2400]))
            except queue.Full:
                print("[audio] Output queue full; dropping assistant audio chunk", file=sys.stderr)
                return

    def clear_output(self) -> None:
        with self.output_lock:
            self.output_generation += 1
        try:
            while True:
                self.output_queue.get_nowait()
        except queue.Empty:
            pass

    def played_ms(self, item_id: str) -> int:
        with self.output_lock:
            return max(0, int(self.played_ms_by_item.get(item_id, 0.0)))

    def output_active(self) -> bool:
        with self.output_lock:
            recently_played = (
                self.last_output_playback_at > 0
                and time.monotonic() - self.last_output_playback_at
                < MIC_SUPPRESS_AFTER_ASSISTANT_AUDIO_S
            )
        return recently_played or not self.output_queue.empty()

    def _on_input(self, indata, frames, time_info, status) -> None:
        del frames, time_info
        if status:
            print(f"[audio] Input status: {status}", file=sys.stderr)
        if self.stop_event.is_set():
            return
        try:
            self.input_queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    def _input_sender(self) -> None:
        while not self.stop_event.is_set():
            try:
                chunk = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.send_audio(chunk)

    def _output_player(self) -> None:
        assert self.output_stream is not None
        while not self.stop_event.is_set():
            try:
                generation, item_id, chunk = self.output_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self.output_lock:
                if generation != self.output_generation:
                    continue
            self.output_stream.write(chunk)
            duration_ms = len(chunk) / 2 / SAMPLE_RATE * 1000
            with self.output_lock:
                self.last_output_playback_at = time.monotonic()
                self.played_ms_by_item[item_id] = self.played_ms_by_item.get(item_id, 0.0) + duration_ms


class RealtimeClanker:
    def __init__(
        self,
        api_key: str,
        model: str,
        voice: str,
        runner: PolicyRunner,
        tts_worker,
        greeting: bool,
        barge_in: bool,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.runner = runner
        self.tts_worker = tts_worker
        self.greeting = greeting
        self.barge_in = barge_in
        self.stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.send_lock = threading.Lock()
        self.playback_lock = threading.Lock()
        self.mic_suppressed_until = 0.0
        self.handled_call_ids: set[str] = set()
        self.response_in_progress = False
        self.current_audio_item_id = ""
        self.current_audio_content_index = 0
        self.tool_lock = threading.Lock()
        self.pending_tool_calls = 0
        self.elevator_music = ElevatorMusic()
        self.ws = None
        self.audio = AudioIO(self.send_audio, self.stop_event)
        self.audio_started = False

    def run(self) -> None:
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError(
                "run_clanker.py needs the websocket-client package. "
                "Run it with `uv run --with websocket-client scripts/gpt_realtime/run_clanker.py`."
            ) from exc

        url = f"wss://api.openai.com/v1/realtime?model={self.model}"
        headers = [f"Authorization: Bearer {self.api_key}", "OpenAI-Safety-Identifier: clanker-local"]
        self.ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        if self.stop_event.is_set():
            return
        print("[clanker] Connecting to OpenAI Realtime...", flush=True)
        self.ws.run_forever()

    def start_enter_to_quit(self) -> None:
        if not sys.stdin.isatty():
            return
        thread = threading.Thread(target=self._wait_for_enter, name="clanker-enter-to-quit", daemon=True)
        thread.start()
        print("[clanker] Press Enter to stop.", flush=True)

    def close(self) -> None:
        self.stop_event.set()
        self.elevator_music.stop()
        self.runner.stop_active()
        self.audio.close()
        if self.tts_worker is not None:
            self.tts_worker.close(timeout=10.0)
        if self.ws is not None:
            self.ws.close()

    def _wait_for_enter(self) -> None:
        try:
            input()
        except EOFError:
            return
        if self.stop_event.is_set():
            return
        print("[clanker] Enter pressed; shutting down.", flush=True)
        self.close()

    def send_event(self, event: dict[str, Any]) -> None:
        if self.ws is None or not self.connected_event.is_set():
            return
        with self.send_lock:
            try:
                self.ws.send(json.dumps(event))
            except Exception as exc:
                if not self.stop_event.is_set():
                    event_type = event.get("type", "event")
                    print(f"[websocket:error] failed to send {event_type}: {exc}", file=sys.stderr)
                self.connected_event.clear()
                self.stop_event.set()

    def send_audio(self, chunk: bytes) -> None:
        if self.stop_event.is_set() or not self.connected_event.is_set():
            return
        if self.tool_call_pending():
            return
        if self.should_suppress_mic():
            return
        audio = base64.b64encode(chunk).decode("ascii")
        self.send_event({"type": "input_audio_buffer.append", "audio": audio})

    def tool_call_pending(self) -> bool:
        with self.tool_lock:
            return self.pending_tool_calls > 0

    def mark_tool_pending(self) -> None:
        should_start_music = False
        with self.tool_lock:
            should_start_music = self.pending_tool_calls == 0
            self.pending_tool_calls += 1
        if should_start_music:
            self.elevator_music.start()

    def mark_tool_done(self) -> None:
        should_stop_music = False
        with self.tool_lock:
            self.pending_tool_calls = max(0, self.pending_tool_calls - 1)
            should_stop_music = self.pending_tool_calls == 0
        if should_stop_music:
            self.elevator_music.stop()

    def should_suppress_mic(self) -> bool:
        if self.barge_in:
            return False
        if self.audio.output_active():
            return True
        with self.playback_lock:
            return time.monotonic() < self.mic_suppressed_until

    def suppress_mic_for_playback(self) -> None:
        if self.barge_in:
            return
        with self.playback_lock:
            self.mic_suppressed_until = max(
                self.mic_suppressed_until,
                time.monotonic() + MIC_SUPPRESS_AFTER_ASSISTANT_AUDIO_S,
            )

    def on_open(self, ws) -> None:
        del ws
        self.connected_event.set()
        self.send_event({"type": "session.update", "session": self.session_config()})
        if not self.audio_started:
            self.audio.start()
            self.audio_started = True
        print("[clanker] Listening. Ask Alfred for a strawberry, marshmallow, or oreo.", flush=True)
        if self.greeting:
            self.send_event(
                {
                    "type": "response.create",
                    "response": {
                        "instructions": (
                            "Greet the user as Alfred in one sentence and say you can fetch "
                            "a strawberry, marshmallow, or oreo."
                        )
                    },
                }
            )

    def on_message(self, ws, message: str) -> None:
        del ws
        event = json.loads(message)
        event_type = event.get("type")

        if event_type == "session.updated":
            print("[clanker] Realtime session configured.", flush=True)
        elif event_type == "response.created":
            self.response_in_progress = True
        elif event_type == "input_audio_buffer.speech_started":
            print("[user] speaking...", flush=True)
            self.handle_user_speech_started()
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript:
                print(f"[user] {transcript}", flush=True)
        elif event_type in {"response.output_audio.delta", "response.audio.delta"}:
            # Production audio output is ElevenLabs-only. If Realtime ever emits
            # audio despite the text-only session config, discard it.
            pass
        elif event_type == "response.output_audio_transcript.done":
            transcript = event.get("transcript", "")
            if transcript:
                print(f"[alfred] {transcript}", flush=True)
        elif event_type in {"response.output_text.done", "response.text.done"}:
            text = event.get("text") or event.get("content") or event.get("transcript") or ""
            if text:
                self.speak_text(str(text))
        elif event_type == "response.function_call_arguments.done":
            # The Realtime docs recommend acting on complete function calls in
            # response.done. This earlier event can arrive before the response
            # lifecycle has fully settled, which can create overlapping replies.
            pass
        elif event_type == "response.done":
            self.response_in_progress = False
            self.handle_response_done(event)
        elif event_type == "error":
            print(f"[openai:error] {json.dumps(event, indent=2)}", file=sys.stderr, flush=True)

    def on_error(self, ws, error) -> None:
        del ws
        self.connected_event.clear()
        print(f"[websocket:error] {error}", file=sys.stderr, flush=True)

    def on_close(self, ws, close_status_code, close_msg) -> None:
        del ws
        print(f"[clanker] Realtime session closed: {close_status_code} {close_msg}", flush=True)
        self.connected_event.clear()
        self.stop_event.set()

    def handle_user_speech_started(self) -> None:
        if not self.barge_in:
            return
        item_id = self.current_audio_item_id
        content_index = self.current_audio_content_index
        audio_end_ms = self.audio.played_ms(item_id) if item_id else 0
        self.audio.clear_output()
        if item_id and audio_end_ms > 0:
            self.send_event(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": content_index,
                    "audio_end_ms": audio_end_ms,
                }
            )

    def handle_response_done(self, event: dict[str, Any]) -> None:
        response = event.get("response") or {}
        if response.get("status") != "completed":
            return
        for item in response.get("output") or []:
            if item.get("type") != "function_call":
                continue
            self.handle_function_call(
                name=str(item.get("name", "")),
                call_id=str(item.get("call_id", "")),
                arguments=str(item.get("arguments", "{}")),
            )

    def handle_tool_event(self, event: dict[str, Any]) -> None:
        self.handle_function_call(
            name=str(event.get("name", "")),
            call_id=str(event.get("call_id", "")),
            arguments=str(event.get("arguments", "{}")),
        )

    def handle_function_call(self, name: str, call_id: str, arguments: str) -> None:
        if name not in {"run_handoff", "unsupported_item_requested"} or not call_id:
            return
        with self.send_lock:
            if call_id in self.handled_call_ids:
                return
            self.handled_call_ids.add(call_id)
        self.mark_tool_pending()
        threading.Thread(
            target=self._run_tool_and_respond,
            args=(name, call_id, arguments),
            name=f"clanker-tool-{call_id}",
            daemon=True,
        ).start()

    def speak_text(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        print(f"[alfred] {text}", flush=True)
        self.suppress_mic_for_text(text)
        if self.tts_worker is None:
            return
        if not self.tts_worker.speak(text):
            print("[tts] ElevenLabs queue full; dropping assistant text", file=sys.stderr)

    def suppress_mic_for_text(self, text: str) -> None:
        if self.barge_in:
            return
        estimated_s = max(2.0, min(8.0, len(text) / 12.0))
        with self.playback_lock:
            self.mic_suppressed_until = max(
                self.mic_suppressed_until,
                time.monotonic() + estimated_s,
            )

    def _run_tool_and_respond(self, name: str, call_id: str, arguments: str) -> None:
        try:
            payload = json.loads(arguments or "{}")
            if name == "run_handoff":
                target = str(payload.get("target", ""))
                result = self.runner.run(target)
            else:
                item = str(payload.get("item", "")).strip() or "that item"
                phrase = choose_unsupported_item_phrase(item)
                self.speak_text(phrase)
                result = {
                    "status": "unsupported_item",
                    "ok": False,
                    "success": False,
                    "item": item,
                    "spoken_by_control_loop": True,
                }
        except Exception as exc:
            result = {"status": "error", "ok": False, "error": str(exc)}
        try:
            self.send_event(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    },
                }
            )
            self.send_event({"type": "response.create"})
        finally:
            self.mark_tool_done()

    def session_config(self) -> dict[str, Any]:
        return {
            "type": "realtime",
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "output_modalities": ["text"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "noise_reduction": {"type": "far_field"},
                    "transcription": {
                        "model": "gpt-4o-mini-transcribe",
                        "language": "en",
                        "prompt": "Snack requests may mention strawberry, oreo, marshmallow, or marshmellow.",
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.7,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 850,
                        "create_response": True,
                        "interrupt_response": self.barge_in,
                    },
                },
            },
            "tools": [
                {
                    "type": "function",
                    "name": "run_handoff",
                    "description": (
                        "Run exactly one local SO-101 food handoff cycle for the requested snack. "
                        "The control loop speaks status through ElevenLabs, immediately runs the "
                        "policy, checks side-camera success with GPT-5.4-nano, and returns structured "
                        "success, failure, or error status."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {
                                "type": "string",
                                "description": "The snack handoff target to run.",
                                "enum": ["strawberry", "marshmallow", "oreo"],
                            }
                        },
                        "required": ["target"],
                        "additionalProperties": False,
                    },
                },
                {
                    "type": "function",
                    "name": "unsupported_item_requested",
                    "description": (
                        "Speak one local Queen's English unavailable-item phrase through "
                        "ElevenLabs when the user asks for a snack outside strawberry, "
                        "marshmallow, or oreo. Does not run the robot."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "item": {
                                "type": "string",
                                "description": "The unsupported item the user asked for.",
                            }
                        },
                        "required": ["item"],
                        "additionalProperties": False,
                    },
                }
            ],
            "tool_choice": "auto",
            "tracing": "auto",
        }


def api_key_from_env(env_name: str) -> str:
    api_key = os.environ.get(env_name) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            f"Missing OpenAI API key. Set {env_name}=sk-... in your shell "
            "or set OPENAI_API_KEY."
        )
    return api_key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Alfred, the GPT Realtime snack butler.")
    parser.add_argument("--food-policy-config", type=Path, default=DEFAULT_FOOD_POLICY_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default=os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-2"))
    parser.add_argument("--voice", default=os.environ.get("OPENAI_REALTIME_VOICE", "ballad"))
    parser.add_argument("--api-key-env", default="OAI_KEY")
    parser.add_argument("--tts-enabled", default=os.environ.get("OOD_TTS_ENABLED", "true"))
    parser.add_argument("--tts-config-path", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument(
        "--tts-queue-max",
        type=int,
        default=int(os.environ.get("OOD_TTS_QUEUE_MAX", "25")),
    )
    barge_group = parser.add_mutually_exclusive_group()
    barge_group.add_argument(
        "--barge-in",
        dest="barge_in",
        action="store_true",
        default=False,
        help="Allow user speech to interrupt Alfred while he is talking.",
    )
    barge_group.add_argument(
        "--no-barge-in",
        dest="barge_in",
        action="store_false",
        help="Do not let user speech interrupt Alfred while he is talking.",
    )
    parser.set_defaults(greeting=False)
    parser.add_argument("--greeting", dest="greeting", action="store_true")
    parser.add_argument("--no-greeting", dest="greeting", action="store_false")
    parser.add_argument(
        "--dry-run-policy",
        action="store_true",
        help="Mock robot policy runs with a background fake process instead of moving the robot.",
    )
    parser.add_argument(
        "--dry-run-seconds",
        type=float,
        default=float(os.environ.get("CLANKER_DRY_RUN_SECONDS", str(DEFAULT_DRY_RUN_SECONDS))),
        help="Seconds that each fake policy process should stay active.",
    )
    parser.add_argument(
        "--dry-run-tool",
        choices=["strawberry", "marshmellow", "marshmallow", "oreo"],
        help="Load configs and print the policy command without connecting to OpenAI.",
    )
    parser.add_argument("--list-policies", action="store_true")
    return parser


def parse_env_file_arg(argv: list[str] | None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    args, _ = parser.parse_known_args(argv)
    return args.env_file


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main(argv: list[str] | None = None) -> int:
    load_dotenv(parse_env_file_arg(argv))
    args = build_parser().parse_args(argv)
    configs = load_policy_configs(args.food_policy_config)
    runner = PolicyRunner(
        configs,
        dry_run=args.dry_run_policy,
        dry_run_seconds=args.dry_run_seconds,
    )

    if args.list_policies:
        for target, config in configs.items():
            print(f"{target}: {config.policy_repo_id} ({config.path})")
        return 0

    if args.dry_run_tool:
        target = canonical_target(args.dry_run_tool)
        config = configs[target]
        print(
            json.dumps(
                {
                    "status": "success",
                    "ok": True,
                    "success": True,
                    "target": target,
                    "spoken_by_control_loop": True,
                    "dry_run": True,
                    "fake_process_seconds": 0.0,
                    "policy_repo_id": config.policy_repo_id,
                    "task": config.task,
                    "command": [
                        str(HANDOFF_SCRIPT),
                        "--no-stt",
                        "--target",
                        target,
                        "--max-cycles",
                        "1",
                        "--reset-pause-s",
                        "0",
                    ],
                },
                indent=2,
            )
        )
        return 0

    tts_worker = None
    if parse_bool(args.tts_enabled):
        tts_config = load_elevenlabs_tts_config(args.tts_config_path)
        tts_worker = ElevenLabsTTSWorker(tts_config, max_queue_size=args.tts_queue_max)
        tts_worker.start()
        print(
            f"[tts] ElevenLabs enabled voice_id={tts_config.voice_id} model_id={tts_config.model_id}",
            flush=True,
        )

    api_key = api_key_from_env(args.api_key_env)
    clanker = RealtimeClanker(
        api_key=api_key,
        model=args.model,
        voice=args.voice,
        runner=runner,
        tts_worker=tts_worker,
        greeting=args.greeting,
        barge_in=args.barge_in,
    )
    clanker.start_enter_to_quit()

    def shutdown(signum, frame) -> None:
        del signum, frame
        clanker.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        clanker.run()
    finally:
        clanker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
