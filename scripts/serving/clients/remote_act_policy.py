"""Drop-in HTTP client for the ACT inference server.

Mirrors lerobot's `policy.select_action(observation)` semantics: returns ONE
action per call from an internal queue. The queue is refilled by a single
POST to `/infer` every `n_action_steps` ticks, so the robot follows a smooth
trajectory between refreshes — matches what `lerobot/policies/act` does
locally via its built-in `_action_queue`. Without this buffering the client
would re-request a fresh chunk every tick and use only `chunk[0]`, which
produces a permanent "first-step catch-up" delta and either jerks the robot
or trips the safety clamp on every tick.

Safety is intentionally NOT done here. Use lerobot's `SOFollowerConfig.
max_relative_target` (calls `ensure_safe_goal_position` in `send_action`) —
that's where lerobot itself puts per-step position clamping for SO-101.
"""
from __future__ import annotations

import io
import json
import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Sequence

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)

# SO-101 joint order as trained — matches the column order in the LeRobot
# SO-101 datasets and the action vector the model emits. Override if a
# different ordering was used during training.
SO101_JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


class RemoteACTPolicy:
    def __init__(
        self,
        url: str,
        *,
        chunk_size: int = 25,
        n_action_steps: int | None = None,
        joint_order: Sequence[str] = SO101_JOINT_ORDER,
        camera_keys: Sequence[str] = ("front", "side"),
        auth_token: str | None = None,
        timeout_s: float = 5.0,
        jpeg_quality: int = 90,
        on_refill_failure: str = "hold",
        prefetch_at_actions: int = 20,
        task: str | None = None,
    ) -> None:
        if on_refill_failure not in ("hold", "raise"):
            raise ValueError("on_refill_failure must be 'hold' or 'raise'")
        if n_action_steps is None:
            n_action_steps = chunk_size
        if not 1 <= n_action_steps <= chunk_size:
            raise ValueError("1 <= n_action_steps <= chunk_size required")

        self._url = url
        self._chunk_size = chunk_size
        self._n_action_steps = n_action_steps
        self._joint_order = tuple(joint_order)
        self._camera_keys = tuple(camera_keys)
        self._auth_token = auth_token
        self._timeout_s = timeout_s
        self._jpeg_quality = jpeg_quality
        self._on_refill_failure = on_refill_failure
        self._prefetch_at_actions = prefetch_at_actions
        # Optional language prompt for VLA endpoints (e.g. pi0.5). ACT endpoints
        # ignore unknown form fields, so leaving this set is harmless for ACT too.
        self._task = task

        self._queue: deque[np.ndarray] = deque(maxlen=n_action_steps)
        self._session = requests.Session()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_refill: Future[dict] | None = None
        self._prefetched_actions: list[np.ndarray] | None = None
        self._prefetched_obs_taken_at: float | None = None
        # Track the very last popped action so we can measure the chunk-seam
        # discontinuity when a new chunk gets installed.
        self._last_popped: np.ndarray | None = None
        # Time (perf_counter) when the observation that produced the currently
        # executing chunk was captured. Every pop measures its own age from this.
        self._current_chunk_obs_taken_at: float | None = None
        # Telemetry counters — useful for tests and for logging from the runner.
        self.refill_calls = 0
        self.refill_failures = 0
        self.refill_latency_ms: float | None = None
        # Buffered per-refill records. The runner drains via pop_refill_events()
        # each tick and writes to a separate JSONL file.
        self._refill_events: list[dict] = []
        # Per-tick observables (overwritten each call to select_action).
        self.last_was_holding: bool = False
        self.last_chunk_seam_delta: float | None = None
        self.last_obs_age_ms: float | None = None

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._session.close()

    def reset(self) -> None:
        """Clear the action queue. Call between episodes."""
        self._queue.clear()
        if self._pending_refill is not None:
            self._pending_refill.cancel()
            self._pending_refill = None
        self._prefetched_actions = None
        self._prefetched_obs_taken_at = None
        self._last_popped = None
        self._current_chunk_obs_taken_at = None

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def prefetch_inflight(self) -> bool:
        return self._pending_refill is not None

    def pop_refill_events(self) -> list[dict]:
        """Return + clear all refill events buffered since the last call."""
        events = self._refill_events
        self._refill_events = []
        return events

    def select_action(self, observation: dict) -> dict[str, float]:
        """Return one action. Refills queue from server when empty."""
        self.last_was_holding = False
        self.last_chunk_seam_delta = None
        self.last_obs_age_ms = None

        self._finish_pending_refill_if_ready()
        if not self._queue:
            if self._prefetched_actions is not None:
                self._install_chunk(self._prefetched_actions, self._prefetched_obs_taken_at)
                self._prefetched_actions = None
                self._prefetched_obs_taken_at = None
                logger.debug("installed prefetched remote ACT chunk (%d actions)", len(self._queue))
            else:
                self._refill_blocking(observation)
        if not self._queue:
            # Refill failed and on_refill_failure="hold" — return current state
            # as the goal so the safety clamp on the robot produces no motion.
            self.last_was_holding = True
            return self._hold_action(observation)
        action_arr = self._queue.popleft()
        self._last_popped = action_arr
        if self._current_chunk_obs_taken_at is not None:
            self.last_obs_age_ms = (time.perf_counter() - self._current_chunk_obs_taken_at) * 1000.0
        if (
            self._prefetch_at_actions > 0
            and self._pending_refill is None
            and self._prefetched_actions is None
            and 0 < len(self._queue) <= self._prefetch_at_actions
        ):
            queue_depth_at_request = len(self._queue)
            obs_taken_at = time.perf_counter()
            self._pending_refill = self._executor.submit(
                self._request_actions, observation, queue_depth_at_request, False, obs_taken_at
            )
            logger.debug("started remote ACT prefetch with %d queued actions", queue_depth_at_request)
        return dict(zip(self._joint_order, (float(v) for v in action_arr)))

    def _install_chunk(self, actions: list[np.ndarray], obs_taken_at: float | None) -> None:
        """Replace the queue contents with a fresh chunk; record the seam delta + obs timestamp."""
        if self._last_popped is not None and len(actions) > 0:
            seam = float(np.max(np.abs(actions[0] - self._last_popped)))
            self.last_chunk_seam_delta = seam
        self._queue.clear()
        self._queue.extend(actions)
        self._current_chunk_obs_taken_at = obs_taken_at

    def _finish_pending_refill_if_ready(self) -> None:
        if self._pending_refill is None or not self._pending_refill.done():
            return
        try:
            event = self._pending_refill.result()
        except Exception as e:
            self.refill_failures += 1
            msg = f"/infer prefetch failed: {type(e).__name__}: {e}"
            self._refill_events.append({
                "trigger": "prefetch",
                "outcome": "error",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:200],
                "wall_end_s": time.time(),
            })
            if self._on_refill_failure == "raise":
                raise RuntimeError(msg) from e
            logger.warning("%s — keeping existing queue", msg)
        else:
            self._prefetched_actions = event["actions"]
            self._prefetched_obs_taken_at = event.get("obs_taken_at")
            event["queue_depth_at_landing"] = len(self._queue)
            event.pop("actions", None)
            self._refill_events.append(event)
            logger.debug("prefetched remote ACT chunk ready (%d actions)", len(self._prefetched_actions))
        finally:
            self._pending_refill = None

    def _refill_blocking(self, observation: dict) -> None:
        obs_taken_at = time.perf_counter()
        try:
            event = self._request_actions(
                observation, queue_depth_at_request=0, was_blocking=True,
                obs_taken_at=obs_taken_at,
            )
        except Exception as e:
            self.refill_failures += 1
            msg = f"/infer call failed: {type(e).__name__}: {e}"
            self._refill_events.append({
                "trigger": "blocking",
                "outcome": "error",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:200],
                "wall_end_s": time.time(),
            })
            if self._on_refill_failure == "raise":
                raise RuntimeError(msg) from e
            logger.warning("%s — holding position", msg)
            return
        actions = event["actions"]
        event["queue_depth_at_landing"] = len(self._queue)
        event.pop("actions", None)
        self._refill_events.append(event)
        self._install_chunk(actions, event.get("obs_taken_at"))

    def _request_actions(
        self,
        observation: dict,
        queue_depth_at_request: int,
        was_blocking: bool,
        obs_taken_at: float,
    ) -> dict:
        try:
            state_arr = np.fromiter(
                (float(observation[k]) for k in self._joint_order),
                dtype=np.float32,
                count=len(self._joint_order),
            )
        except KeyError as e:
            raise KeyError(
                f"observation missing joint key {e!r}; "
                f"expected keys: {self._joint_order}"
            ) from None

        wall_start_s = time.time()
        enc_start = time.perf_counter()
        files = []
        jpeg_bytes: dict[str, int] = {}
        for cam_name in self._camera_keys:
            if cam_name not in observation:
                raise KeyError(
                    f"observation missing camera {cam_name!r}; "
                    f"available keys: {sorted(observation)}"
                )
            img = observation[cam_name]
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"camera {cam_name!r} must be uint8 HxWx3 RGB, "
                    f"got dtype={img.dtype} shape={img.shape}"
                )
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=self._jpeg_quality)
            payload = buf.getvalue()
            jpeg_bytes[cam_name] = len(payload)
            files.append((cam_name, (f"{cam_name}.jpg", payload, "image/jpeg")))
        jpeg_encode_ms = (time.perf_counter() - enc_start) * 1000.0

        data = {"state": json.dumps(state_arr.tolist())}
        if self._task is not None:
            data["task"] = self._task
        headers = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        self.refill_calls += 1
        http_start = time.perf_counter()
        outcome = "success"
        server_inference_us = None
        try:
            resp = self._session.post(
                self._url, files=files, data=data,
                headers=headers, timeout=self._timeout_s,
            )
            resp.raise_for_status()
            body = resp.json()
            actions = body.get("action")
            server_inference_us = int(body.get("inference_us", 0)) or None
            if not isinstance(actions, list) or len(actions) < self._n_action_steps:
                outcome = "malformed"
                raise RuntimeError(
                    f"server returned malformed action (got "
                    f"{len(actions) if isinstance(actions, list) else type(actions).__name__}, "
                    f"expected list of {self._n_action_steps}+)"
                )
            actions_np = [np.asarray(a, dtype=np.float32) for a in actions[: self._n_action_steps]]
        except requests.Timeout:
            outcome = "timeout"
            raise
        except requests.HTTPError as e:
            outcome = f"http_{e.response.status_code if e.response is not None else 'error'}"
            raise
        finally:
            http_total_ms = (time.perf_counter() - http_start) * 1000.0
            self.refill_latency_ms = http_total_ms

        return {
            "actions": actions_np,
            "obs_taken_at": obs_taken_at,
            "trigger": "blocking" if was_blocking else "prefetch",
            "outcome": outcome,
            "wall_start_s": wall_start_s,
            "wall_end_s": time.time(),
            "jpeg_encode_ms": jpeg_encode_ms,
            "jpeg_bytes_total": sum(jpeg_bytes.values()),
            "jpeg_bytes_per_cam": jpeg_bytes,
            "http_total_ms": http_total_ms,
            "server_inference_us": server_inference_us,
            "network_ms": (http_total_ms - server_inference_us / 1000.0)
            if server_inference_us is not None
            else None,
            "queue_depth_at_request": queue_depth_at_request,
            "refill_call_idx": self.refill_calls,
        }

    def _hold_action(self, observation: dict) -> dict[str, float]:
        return {k: float(observation[k]) for k in self._joint_order}
