"""Local in-process equivalent of RemoteACTPolicy for apples-to-apples profiling.

Loads an ACT checkpoint with the same pre/post processors lerobot trains
with, then exposes the same queue + prefetch interface as RemoteACTPolicy.
This lets `scripts/analyze_act_traces.py` compare local-MPS vs remote-HTTP
runs with identical JSONL schemas — only the latency source differs
(MPS inference time replaces HTTP roundtrip).

Inference runs in a background thread via ThreadPoolExecutor so prefetch
overlaps with the robot control loop, matching RemoteACTPolicy's behavior.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Sequence

import numpy as np
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference

logger = logging.getLogger(__name__)

SO101_JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


class LocalACTPolicy:
    def __init__(
        self,
        pretrained_path: str,
        *,
        device: str = "mps",
        chunk_size: int = 50,
        n_action_steps: int | None = None,
        joint_order: Sequence[str] = SO101_JOINT_ORDER,
        camera_keys: Sequence[str] = ("front", "side"),
        on_refill_failure: str = "hold",
        prefetch_at_actions: int = 20,
    ) -> None:
        if on_refill_failure not in ("hold", "raise"):
            raise ValueError("on_refill_failure must be 'hold' or 'raise'")
        if n_action_steps is None:
            n_action_steps = chunk_size
        if not 1 <= n_action_steps <= chunk_size:
            raise ValueError("1 <= n_action_steps <= chunk_size required")

        self._pretrained_path = pretrained_path
        self._device = torch.device(device)
        self._chunk_size = chunk_size
        self._n_action_steps = n_action_steps
        self._joint_order = tuple(joint_order)
        self._camera_keys = tuple(camera_keys)
        self._on_refill_failure = on_refill_failure
        self._prefetch_at_actions = prefetch_at_actions

        # Load policy via the same factory lerobot-train/eval use. ``act`` is
        # registered in `lerobot.policies.factory`.
        logger.info("loading ACT from %s onto %s ...", pretrained_path, self._device)
        policy_cls = get_policy_class("act")
        self._policy: PreTrainedPolicy = policy_cls.from_pretrained(pretrained_path)
        self._policy.config.device = str(self._device)
        # Match the deployed chunk size (the user trained chunk_size=50 but
        # may want to evaluate with a smaller n_action_steps).
        self._policy.config.chunk_size = chunk_size
        self._policy.config.n_action_steps = n_action_steps
        self._policy.to(self._device)
        self._policy.eval()
        # Pre/post processors live alongside the checkpoint on the Hub.
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            policy_cfg=self._policy.config,
            pretrained_path=pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(self._device)}},
        )
        logger.info("loaded ACT chunk_size=%d n_action_steps=%d device=%s",
                    chunk_size, n_action_steps, self._device)

        self._queue: deque[np.ndarray] = deque(maxlen=n_action_steps)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_refill: Future[dict] | None = None
        self._prefetched_actions: list[np.ndarray] | None = None
        self._prefetched_obs_taken_at: float | None = None
        self._last_popped: np.ndarray | None = None
        self._current_chunk_obs_taken_at: float | None = None

        # Public telemetry.
        self.refill_calls = 0
        self.refill_failures = 0
        self.refill_latency_ms: float | None = None
        self._refill_events: list[dict] = []
        self.last_was_holding: bool = False
        self.last_chunk_seam_delta: float | None = None
        self.last_obs_age_ms: float | None = None

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def reset(self) -> None:
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
        events = self._refill_events
        self._refill_events = []
        return events

    def select_action(self, observation: dict) -> dict[str, float]:
        self.last_was_holding = False
        self.last_chunk_seam_delta = None
        self.last_obs_age_ms = None

        self._finish_pending_refill_if_ready()
        if not self._queue:
            if self._prefetched_actions is not None:
                self._install_chunk(self._prefetched_actions, self._prefetched_obs_taken_at)
                self._prefetched_actions = None
                self._prefetched_obs_taken_at = None
            else:
                self._refill_blocking(observation)
        if not self._queue:
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
            qd = len(self._queue)
            obs_taken_at = time.perf_counter()
            obs_snapshot = self._snapshot_obs(observation)
            self._pending_refill = self._executor.submit(
                self._infer_chunk, obs_snapshot, qd, False, obs_taken_at
            )
        return dict(zip(self._joint_order, (float(v) for v in action_arr)))

    def _install_chunk(self, actions: list[np.ndarray], obs_taken_at: float | None) -> None:
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
            self._refill_events.append({
                "trigger": "prefetch",
                "outcome": "error",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:200],
                "wall_end_s": time.time(),
            })
            if self._on_refill_failure == "raise":
                raise RuntimeError(str(e)) from e
            logger.warning("inference prefetch failed: %s — keeping existing queue", e)
        else:
            self._prefetched_actions = event["actions"]
            self._prefetched_obs_taken_at = event.get("obs_taken_at")
            event["queue_depth_at_landing"] = len(self._queue)
            event.pop("actions", None)
            self._refill_events.append(event)
        finally:
            self._pending_refill = None

    def _refill_blocking(self, observation: dict) -> None:
        obs_taken_at = time.perf_counter()
        obs_snapshot = self._snapshot_obs(observation)
        try:
            event = self._infer_chunk(obs_snapshot, 0, was_blocking=True, obs_taken_at=obs_taken_at)
        except Exception as e:
            self.refill_failures += 1
            self._refill_events.append({
                "trigger": "blocking",
                "outcome": "error",
                "error_type": type(e).__name__,
                "error_msg": str(e)[:200],
                "wall_end_s": time.time(),
            })
            if self._on_refill_failure == "raise":
                raise RuntimeError(str(e)) from e
            logger.warning("inference call failed: %s — holding position", e)
            return
        actions = event["actions"]
        event["queue_depth_at_landing"] = len(self._queue)
        event.pop("actions", None)
        self._refill_events.append(event)
        self._install_chunk(actions, event.get("obs_taken_at"))

    def _snapshot_obs(self, observation: dict) -> dict[str, np.ndarray]:
        """Copy out only what we need from the live robot obs — caller is on
        the control thread, inference happens later on the executor."""
        state = np.fromiter(
            (float(observation[k]) for k in self._joint_order),
            dtype=np.float32,
            count=len(self._joint_order),
        )
        snap: dict[str, np.ndarray] = {"observation.state": state}
        for cam in self._camera_keys:
            if cam not in observation:
                raise KeyError(f"observation missing camera {cam!r}; have {sorted(observation)}")
            img = observation[cam]
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"camera {cam!r} must be uint8 HxWx3 RGB, got {img.dtype} {img.shape}")
            # Copy because the camera buffer may be reused before inference runs.
            snap[f"observation.images.{cam}"] = img.copy()
        return snap

    def _infer_chunk(
        self,
        obs_snapshot: dict[str, np.ndarray],
        queue_depth_at_request: int,
        was_blocking: bool,
        obs_taken_at: float,
    ) -> dict:
        self.refill_calls += 1
        wall_start_s = time.time()
        outcome = "success"
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                batch = prepare_observation_for_inference(
                    dict(obs_snapshot),  # function mutates in place
                    device=self._device,
                )
                batch = self._preprocessor(batch)
                chunk = self._policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
                # Postprocess each step. ACT's postprocessor is action-only
                # denorm; applying once-per-action keeps shapes unambiguous.
                actions_np: list[np.ndarray] = []
                for i in range(min(self._n_action_steps, chunk.shape[1])):
                    a = chunk[:, i, :]  # (1, action_dim)
                    a = self._postprocessor(a)
                    actions_np.append(a.squeeze(0).detach().cpu().numpy().astype(np.float32))
        except Exception:
            outcome = "error"
            raise
        finally:
            inference_ms = (time.perf_counter() - t0) * 1000.0
            self.refill_latency_ms = inference_ms

        return {
            "actions": actions_np,
            "obs_taken_at": obs_taken_at,
            "trigger": "blocking" if was_blocking else "prefetch",
            "outcome": outcome,
            "wall_start_s": wall_start_s,
            "wall_end_s": time.time(),
            "inference_ms": inference_ms,
            "queue_depth_at_request": queue_depth_at_request,
            "refill_call_idx": self.refill_calls,
        }

    def _hold_action(self, observation: dict) -> dict[str, float]:
        return {k: float(observation[k]) for k in self._joint_order}
