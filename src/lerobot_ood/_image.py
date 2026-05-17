"""Image-format helpers shared by the encoder backends."""

from __future__ import annotations

import numpy as np


def to_uint8_rgb_hw3(image) -> np.ndarray:
    """Coerce ``image`` to ``(H, W, 3)`` uint8 RGB.

    Accepts numpy arrays or torch tensors, in HWC or CHW layout, with
    values in either ``[0, 1]`` floats or ``[0, 255]`` ints.
    """
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) image, got shape {img.shape}")
    return img
