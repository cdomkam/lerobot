"""OOD detection wrapped around a lerobot inference loop.

The package is intentionally small and self-contained:

- ``detector.OODDetector``      — Mahalanobis density model over image embeddings.
- ``act_encoder.ACTBackboneEncoder`` — uses ACT's own ResNet backbone (default).
- ``encoder.DinoV2Encoder``     — DINOv2 ViT-S/14 fallback encoder.
- ``obs.extract_camera_frame``  — pull a camera frame from a lerobot observation.
- ``tts.ElevenLabsTTSWorker``   — background voice alerts for OOD events.
"""

from .act_encoder import ACTBackboneEncoder
from .detector import OODDetector, OODResult
from .encoder import DinoV2Encoder
from .obs import extract_camera_frame
from .tts import (
    STRAWBERRY_OOD_PHRASES,
    ElevenLabsTTSConfig,
    ElevenLabsTTSWorker,
    choose_strawberry_ood_phrase,
    load_elevenlabs_tts_config,
)

__all__ = [
    "OODDetector",
    "OODResult",
    "ACTBackboneEncoder",
    "DinoV2Encoder",
    "extract_camera_frame",
    "STRAWBERRY_OOD_PHRASES",
    "ElevenLabsTTSConfig",
    "ElevenLabsTTSWorker",
    "choose_strawberry_ood_phrase",
    "load_elevenlabs_tts_config",
]
