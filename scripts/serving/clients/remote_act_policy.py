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

        self._queue: deque[np.ndarray] = deque(maxlen=n_action_steps)
        self._session = requests.Session()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_refill: Future[list[np.ndarray]] | None = None
        # Telemetry counters — useful for tests and for logging from the runner.
        self.refill_calls = 0
        self.refill_failures = 0
        self.refill_latency_ms: float | None = None

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._session.close()

    def reset(self) -> None:
        """Clear the action queue. Call between episodes."""
        self._queue.clear()
        if self._pending_refill is not None:
            self._pending_refill.cancel()
            self._pending_refill = None

    def select_action(self, observation: dict) -> dict[str, float]:
        """Return one action. Refills queue from server when empty."""
        self._finish_pending_refill_if_ready()
        if not self._queue:
            self._refill_blocking(observation)
        if not self._queue:
            # Refill failed and on_refill_failure="hold" — return current state
            # as the goal so the safety clamp on the robot produces no motion.
            return self._hold_action(observation)
        action_arr = self._queue.popleft()
        if (
            self._prefetch_at_actions > 0
            and self._pending_refill is None
            and 0 < len(self._queue) <= self._prefetch_at_actions
        ):
            self._pending_refill = self._executor.submit(self._request_actions, observation)
            logger.debug("started remote ACT prefetch with %d queued actions", len(self._queue))
        return dict(zip(self._joint_order, (float(v) for v in action_arr)))

    def _finish_pending_refill_if_ready(self) -> None:
        if self._pending_refill is None or not self._pending_refill.done():
            return
        try:
            actions = self._pending_refill.result()
        except Exception as e:
            self.refill_failures += 1
            msg = f"/infer prefetch failed: {type(e).__name__}: {e}"
            if self._on_refill_failure == "raise":
                raise RuntimeError(msg) from e
            logger.warning("%s — keeping existing queue", msg)
        else:
            self._queue.clear()
            self._queue.extend(actions)
            logger.debug("installed prefetched remote ACT chunk (%d actions)", len(actions))
        finally:
            self._pending_refill = None

    def _refill_blocking(self, observation: dict) -> None:
        try:
            actions = self._request_actions(observation)
        except Exception as e:
            self.refill_failures += 1
            msg = f"/infer call failed: {type(e).__name__}: {e}"
            if self._on_refill_failure == "raise":
                raise RuntimeError(msg) from e
            logger.warning("%s — holding position", msg)
            return
        self._queue.clear()
        self._queue.extend(actions)

    def _request_actions(self, observation: dict) -> list[np.ndarray]:
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

        files = []
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
            files.append((cam_name, (f"{cam_name}.jpg", buf.getvalue(), "image/jpeg")))

        data = {"state": json.dumps(state_arr.tolist())}
        headers = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        self.refill_calls += 1
        start = time.perf_counter()
        resp = self._session.post(
            self._url, files=files, data=data,
            headers=headers, timeout=self._timeout_s,
        )
        self.refill_latency_ms = (time.perf_counter() - start) * 1000.0
        resp.raise_for_status()
        body = resp.json()

        actions = body.get("action")
        if not isinstance(actions, list) or len(actions) < self._n_action_steps:
            raise RuntimeError(
                f"server returned malformed action (got "
                f"{len(actions) if isinstance(actions, list) else type(actions).__name__}, "
                f"expected list of {self._n_action_steps}+)"
            )

        return [np.asarray(action_arr, dtype=np.float32) for action_arr in actions[: self._n_action_steps]]

    def _hold_action(self, observation: dict) -> dict[str, float]:
        return {k: float(observation[k]) for k in self._joint_order}
