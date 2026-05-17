"""ROI-based hand and placement detectors for the food handoff flow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    detected: bool
    score: float
    ready: bool = True
    reason: str = ""


@dataclass(frozen=True)
class HandVisionConfig:
    camera_name: str = "side"
    fps: int = 30
    roi: tuple[float, float, float, float] = (0.35, 0.20, 0.60, 0.75)
    baseline_frames: int = 15
    min_mean_abs_diff: float = 18.0
    min_changed_fraction: float = 0.08
    changed_pixel_threshold: float = 28.0
    debounce_frames: int = 10


@dataclass(frozen=True)
class TargetColorConfig:
    rgb_min: tuple[int, int, int]
    rgb_max: tuple[int, int, int]


@dataclass(frozen=True)
class SuccessVisionConfig:
    camera_name: str = "side"
    roi: tuple[float, float, float, float] = (0.35, 0.20, 0.60, 0.75)
    min_blob_fraction: float = 0.012
    min_context_bright_fraction: float = 0.0
    context_bright_threshold: int = 210
    debounce_frames: int = 15
    targets: dict[str, TargetColorConfig] = field(default_factory=dict)


@dataclass(frozen=True)
class FoodHandoffVisionConfig:
    hand: HandVisionConfig
    success: SuccessVisionConfig


class HandPresenceDetector:
    def __init__(self, config: HandVisionConfig):
        self.config = config
        self._baseline_frames: list[np.ndarray] = []
        self._baseline: np.ndarray | None = None
        self._streak = 0

    def update(self, frame: np.ndarray) -> DetectionResult:
        crop = _crop_roi(frame, self.config.roi).astype(np.float32)
        if self._baseline is None:
            self._baseline_frames.append(crop)
            if len(self._baseline_frames) >= self.config.baseline_frames:
                self._baseline = np.mean(np.stack(self._baseline_frames), axis=0)
            return DetectionResult(False, 0.0, ready=False, reason="building_baseline")

        diff = np.abs(crop - self._baseline).mean(axis=2)
        mean_diff = float(diff.mean())
        changed_fraction = float((diff >= self.config.changed_pixel_threshold).mean())
        raw_detected = (
            mean_diff >= self.config.min_mean_abs_diff
            and changed_fraction >= self.config.min_changed_fraction
        )
        self._streak = self._streak + 1 if raw_detected else 0
        detected = self._streak >= self.config.debounce_frames
        return DetectionResult(
            detected=detected,
            score=changed_fraction,
            reason=f"mean_abs_diff={mean_diff:.2f}, changed_fraction={changed_fraction:.3f}",
        )


class TargetSuccessDetector:
    def __init__(self, config: SuccessVisionConfig):
        self.config = config
        self._streaks: dict[str, int] = {}

    def update(self, frame: np.ndarray, target: str) -> DetectionResult:
        color = self.config.targets.get(target)
        if color is None:
            raise ValueError(f"missing success color config for target {target!r}")
        crop = _crop_roi(frame, self.config.roi)
        lo = np.asarray(color.rgb_min, dtype=np.uint8)
        hi = np.asarray(color.rgb_max, dtype=np.uint8)
        mask = np.all((crop >= lo) & (crop <= hi), axis=2)
        fraction = float(mask.mean())
        bright_fraction = float(
            np.all(crop >= self.config.context_bright_threshold, axis=2).mean()
        )
        raw_detected = (
            fraction >= self.config.min_blob_fraction
            and bright_fraction >= self.config.min_context_bright_fraction
        )
        self._streaks[target] = self._streaks.get(target, 0) + 1 if raw_detected else 0
        detected = self._streaks[target] >= self.config.debounce_frames
        return DetectionResult(
            detected=detected,
            score=fraction,
            reason=(
                f"target_blob_fraction={fraction:.3f}, "
                f"context_bright_fraction={bright_fraction:.3f}"
            ),
        )


def load_vision_config(path: str | Path) -> FoodHandoffVisionConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Vision config not found: {config_path}. "
            "Expected config/food_handoff_vision.json with tuned scene-camera ROIs."
        )
    raw = json.loads(config_path.read_text())
    hand_raw = raw.get("hand", {})
    success_raw = raw.get("success", {})
    target_raw = success_raw.get("targets", {})
    hand = HandVisionConfig(
        camera_name=str(hand_raw.get("camera_name", "side")).strip() or "side",
        fps=int(hand_raw.get("fps", 30)),
        roi=_tuple4(hand_raw.get("roi", (0.35, 0.20, 0.60, 0.75))),
        baseline_frames=int(hand_raw.get("baseline_frames", 15)),
        min_mean_abs_diff=float(hand_raw.get("min_mean_abs_diff", 18.0)),
        min_changed_fraction=float(hand_raw.get("min_changed_fraction", 0.08)),
        changed_pixel_threshold=float(hand_raw.get("changed_pixel_threshold", 28.0)),
        debounce_frames=int(hand_raw.get("debounce_frames", 10)),
    )
    targets = {
        target: TargetColorConfig(
            rgb_min=_tuple3(values["rgb_min"]),
            rgb_max=_tuple3(values["rgb_max"]),
        )
        for target, values in target_raw.items()
    }
    success = SuccessVisionConfig(
        camera_name=str(success_raw.get("camera_name", "side")).strip() or "side",
        roi=_tuple4(success_raw.get("roi", hand.roi)),
        min_blob_fraction=float(success_raw.get("min_blob_fraction", 0.012)),
        min_context_bright_fraction=float(success_raw.get("min_context_bright_fraction", 0.0)),
        context_bright_threshold=int(success_raw.get("context_bright_threshold", 210)),
        debounce_frames=int(success_raw.get("debounce_frames", 15)),
        targets=targets,
    )
    return FoodHandoffVisionConfig(hand=hand, success=success)


def _crop_roi(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB frame, got shape {arr.shape}")
    h, w = arr.shape[:2]
    x, y, rw, rh = roi
    if max(abs(x), abs(y), abs(rw), abs(rh)) <= 1.0:
        x, rw = x * w, rw * w
        y, rh = y * h, rh * h
    x0 = max(0, min(w - 1, int(round(x))))
    y0 = max(0, min(h - 1, int(round(y))))
    x1 = max(x0 + 1, min(w, int(round(x + rw))))
    y1 = max(y0 + 1, min(h, int(round(y + rh))))
    return arr[y0:y1, x0:x1]


def _tuple4(value) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"expected 4-value ROI, got {value!r}")
    return tuple(float(v) for v in value)


def _tuple3(value) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"expected 3-value RGB threshold, got {value!r}")
    return tuple(int(v) for v in value)
