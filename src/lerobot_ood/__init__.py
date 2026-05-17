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
from .openai_vision import (
    OpenAISuccessResult,
    OpenAIVisionConfig,
    confirm_food_handoff_success,
    load_openai_vision_config,
)
from .stt import ElevenLabsSTTConfig, load_elevenlabs_stt_config, transcribe_audio_file
from .targets import (
    TARGETS,
    FoodPolicy,
    FoodPolicyConfig,
    canonicalize_target,
    classify_food_request,
    load_food_policy_config,
)
from .tts import (
    FETCHING_PHRASES,
    FOOD_HANDOFF_OOD_PHRASES,
    SUCCESS_PHRASES,
    STRAWBERRY_OOD_PHRASES,
    ElevenLabsTTSConfig,
    ElevenLabsTTSWorker,
    choose_fetching_phrase,
    choose_food_handoff_ood_phrase,
    choose_strawberry_ood_phrase,
    choose_success_phrase,
    load_elevenlabs_tts_config,
)
from .vision import (
    FoodHandoffVisionConfig,
    HandPresenceDetector,
    HandVisionConfig,
    SuccessVisionConfig,
    TargetColorConfig,
    TargetSuccessDetector,
    load_vision_config,
)

__all__ = [
    "OODDetector",
    "OODResult",
    "ACTBackboneEncoder",
    "DinoV2Encoder",
    "extract_camera_frame",
    "OpenAISuccessResult",
    "OpenAIVisionConfig",
    "confirm_food_handoff_success",
    "load_openai_vision_config",
    "TARGETS",
    "FoodPolicy",
    "FoodPolicyConfig",
    "canonicalize_target",
    "classify_food_request",
    "load_food_policy_config",
    "STRAWBERRY_OOD_PHRASES",
    "FOOD_HANDOFF_OOD_PHRASES",
    "FETCHING_PHRASES",
    "SUCCESS_PHRASES",
    "ElevenLabsTTSConfig",
    "ElevenLabsTTSWorker",
    "ElevenLabsSTTConfig",
    "choose_fetching_phrase",
    "choose_food_handoff_ood_phrase",
    "choose_strawberry_ood_phrase",
    "choose_success_phrase",
    "load_elevenlabs_tts_config",
    "load_elevenlabs_stt_config",
    "transcribe_audio_file",
    "FoodHandoffVisionConfig",
    "HandPresenceDetector",
    "HandVisionConfig",
    "SuccessVisionConfig",
    "TargetColorConfig",
    "TargetSuccessDetector",
    "load_vision_config",
]
