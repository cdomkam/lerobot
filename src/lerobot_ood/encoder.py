"""DINOv2 vision encoder used by the OOD detector.

Why DINOv2: it has the strongest off-the-shelf self-supervised visual
representations and is available via ``torch.hub`` with no auth. The
ViT-S/14 variant runs in ~3 ms on MPS and ~10 ms on CPU, which fits
comfortably inside a 30 FPS control loop.

The encoder returns L2-normalised float32 vectors so the downstream
``OODDetector`` can treat embeddings as points on a unit hypersphere.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

logger = logging.getLogger(__name__)

DinoModel = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"]


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DinoV2Encoder:
    """Frozen DINOv2 encoder. Callable: ``(H,W,3) image -> (D,) np.ndarray``."""

    def __init__(
        self,
        model: DinoModel = "dinov2_vits14",
        device: str | None = None,
        input_size: int = 224,
    ):
        self.model_name = model
        self.device = _resolve_device(device)
        logger.info("Loading %s on %s ...", model, self.device)
        self._model = torch.hub.load("facebookresearch/dinov2", model)
        self._model.eval().to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)
        # DINOv2's published preprocessing — must match how the model was trained.
        self._tx = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(input_size, antialias=True),
                transforms.CenterCrop(input_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        x = self._tx(_to_uint8_rgb(image)).unsqueeze(0).to(self.device)
        z = self._model(x).squeeze(0)
        z = F.normalize(z, dim=0)
        return z.float().cpu().numpy()

    def encode_many(self, images, log_every: int = 100) -> np.ndarray:
        """Encode a list of images sequentially. Returns ``(N, D)``."""
        out = []
        for i, img in enumerate(images, 1):
            out.append(self(img))
            if log_every and i % log_every == 0:
                logger.info("encoded %d frames", i)
        return np.stack(out, axis=0)


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Coerce to ``(H, W, 3)`` uint8 RGB."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) image, got shape {img.shape}")
    return img
