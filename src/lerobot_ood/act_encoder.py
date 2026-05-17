"""Use ACT's own ResNet vision backbone as the OOD encoder.

ACT's ``model.backbone`` is a torchvision ResNet wrapped by
``IntermediateLayerGetter`` that returns ``{"feature_map": (B, C, H, W)}``
(C = 512 for ResNet18, ResNet's layer4 output).  We global-avg-pool that
feature map to get one vector per frame, L2-normalise it, and hand it
to the same Mahalanobis detector.

Why this is the default
-----------------------
The OOD signal becomes "what does *this policy* see as different from
training" rather than "what does a generic visual model see."  The
backbone is also one model fewer to load and ~3 ms cheaper per frame
than running DINOv2 separately.

What we don't do
----------------
We don't replicate ACT's full preprocessor pipeline. Images are passed
through as ``image / 255`` floats — no ImageNet or dataset-specific
normalisation. The Mahalanobis detector is fit on whatever feature
distribution the backbone produces under this preprocessing, so as
long as fit and runtime use the same code path the scores are
self-consistent. The cost is that an OOD frame in our encoder's space
isn't *exactly* an OOD frame in ACT's transformer's input space — it's
a close approximation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from ._image import to_uint8_rgb_hw3

logger = logging.getLogger(__name__)


def _find_backbone(policy) -> torch.nn.Module:
    """Locate the ResNet backbone module inside an ACT policy."""
    candidates = ("model.backbone", "backbone")
    for path in candidates:
        obj = policy
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            return obj
    raise AttributeError(
        f"could not find a 'backbone' module on the policy "
        f"(tried {candidates}). Is this an ACT policy?"
    )


class ACTBackboneEncoder:
    """ACT's vision backbone, exposed as ``image -> L2-normalised vector``."""

    def __init__(self, policy, device: str | None = None):
        self._backbone = _find_backbone(policy)
        self._backbone.eval()
        for p in self._backbone.parameters():
            p.requires_grad_(False)
        if device is None:
            device = str(next(self._backbone.parameters()).device)
        self.device = torch.device(device)
        # ``image / 255`` only — no ImageNet/dataset-specific normalisation.
        # See module docstring for the rationale.
        self._tx = transforms.ToTensor()

    @torch.inference_mode()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = to_uint8_rgb_hw3(image)
        x = self._tx(img).unsqueeze(0).to(self.device)
        feat = self._backbone(x)
        if isinstance(feat, dict):
            # IntermediateLayerGetter returns {"feature_map": tensor}.
            feat = next(iter(feat.values()))
        # (B, C, H, W) -> (B, C)
        if feat.ndim == 4:
            feat = feat.mean(dim=(-2, -1))
        feat = F.normalize(feat, dim=-1).squeeze(0)
        return feat.float().cpu().numpy()
