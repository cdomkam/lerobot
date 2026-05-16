"""Helpers for plucking a camera frame out of a lerobot observation dict."""

from __future__ import annotations

import numpy as np


def extract_camera_frame(obs: dict, camera_name: str) -> np.ndarray:
    """Return ``(H, W, 3)`` uint8 frame for ``camera_name``.

    Handles both raw robot observations (bare camera keys like
    ``"front"``) and processed observations
    (``"observation.images.front"``). Falls back to a CHW→HWC transpose
    when the channel axis is leading.
    """
    candidates = (camera_name, f"observation.images.{camera_name}")
    for key in candidates:
        if key in obs:
            frame = np.asarray(obs[key])
            break
    else:
        image_like = [k for k in obs.keys() if "image" in k.lower() or "camera" in k.lower()]
        raise KeyError(
            f"camera '{camera_name}' not found in observation. "
            f"Tried keys: {candidates}. Image-like keys present: {image_like}"
        )

    if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"unexpected frame shape for '{camera_name}': {frame.shape}")
    return frame
