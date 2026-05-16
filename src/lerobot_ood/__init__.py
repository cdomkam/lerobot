"""OOD detection wrapped around a lerobot inference loop.

The package is intentionally small and self-contained:

- ``detector.OODDetector``  — Mahalanobis density model over image embeddings.
- ``encoder.DinoV2Encoder`` — frozen DINOv2 ViT-S/14 vision encoder.
- ``obs.extract_camera_frame`` — pull a camera frame from a lerobot observation.
"""

from .detector import OODDetector, OODResult
from .encoder import DinoV2Encoder
from .obs import extract_camera_frame

__all__ = ["OODDetector", "OODResult", "DinoV2Encoder", "extract_camera_frame"]
