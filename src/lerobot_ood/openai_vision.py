"""OpenAI vision confirmation for food handoff success."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tts import load_env_file


@dataclass(frozen=True)
class OpenAIVisionConfig:
    api_key: str
    model: str = "gpt-5.4-nano"
    api_base_url: str = "https://api.openai.com/v1"
    timeout_s: float = 12.0
    max_image_width: int = 320
    jpeg_quality: int = 72
    min_confidence: float = 0.70


@dataclass(frozen=True)
class OpenAISuccessResult:
    success: bool
    confidence: float
    user_hand_present: bool
    robot_visible: bool
    robot_gripper_near_user_hand: bool
    robot_placing_target_in_user_hand: bool
    target_food_visible: bool
    target_food_in_user_hand: bool
    target_food_in_robot_gripper: bool
    target_food_on_tray: bool
    correct_target_food: bool
    user_grabbing_without_robot_placement: bool
    reason: str
    raw_text: str


def load_openai_vision_config(path: str | Path) -> OpenAIVisionConfig:
    """Load OpenAI vision config, preferring local .env values over shell env."""
    try:
        file_values = load_env_file(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"OpenAI success confirmation is enabled, but the config file was not found: {path}. "
            "Create it with OPENAI_API_KEY or OAI_KEY, or disable OPENAI_SUCCESS_ENABLED."
        ) from exc
    values = {**os.environ, **file_values}
    api_key = (values.get("OPENAI_API_KEY") or values.get("OAI_KEY") or "").strip()
    if not api_key:
        raise ValueError(
            f"OpenAI success confirmation is enabled, but OPENAI_API_KEY/OAI_KEY is missing in {path}."
        )

    return OpenAIVisionConfig(
        api_key=api_key,
        model=values.get("OPENAI_VISION_MODEL", "gpt-5.4-nano").strip() or "gpt-5.4-nano",
        api_base_url=values.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        timeout_s=float(values.get("OPENAI_TIMEOUT_S", "12")),
        max_image_width=int(values.get("OPENAI_VISION_MAX_IMAGE_WIDTH", "320")),
        jpeg_quality=int(values.get("OPENAI_VISION_JPEG_QUALITY", "72")),
        min_confidence=float(values.get("OPENAI_SUCCESS_MIN_CONFIDENCE", "0.70")),
    )


def confirm_food_handoff_success(
    config: OpenAIVisionConfig,
    frame_rgb: np.ndarray,
    target_display_name: str,
    opener=urllib.request.urlopen,
) -> OpenAISuccessResult:
    """Ask a vision model whether the robot is placing the target food in the user's hand."""
    prompt = (
        "You are evaluating an SO-101 robot food handoff. Judge only the image. "
        "The user hand is a human hand, not the robot gripper. "
        f"The target food is: {target_display_name}. "
        "Be strict: success requires visible evidence that the robot is actively placing or has "
        "just released the correct target food into the human/user hand. The robot arm or gripper "
        "must be visible and near the user's hand, and the correct target food must be in or on "
        "the user's hand. It is not success if the robot is stalled, absent, too far away, holding "
        "nothing, holding the wrong item, or if the user appears to have grabbed the food from the "
        "tray, table, or robot without a robot placement. It is not success if the food is only in "
        "the robot gripper or only on the tray/table. If unsure, set success-relevant booleans false "
        "and use low confidence. "
        "Return strict JSON only with this shape: "
        '{"user_hand_present":true,"robot_visible":true,'
        '"robot_gripper_near_user_hand":true,"robot_placing_target_in_user_hand":true,'
        '"target_food_visible":true,"target_food_in_user_hand":true,'
        '"target_food_in_robot_gripper":false,"target_food_on_tray":false,'
        '"correct_target_food":true,"user_grabbing_without_robot_placement":false,'
        '"confidence":0.0,"reason":"short"}'
    )
    payload = {
        "model": config.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": _frame_to_data_url(
                            frame_rgb,
                            max_width=config.max_image_width,
                            jpeg_quality=config.jpeg_quality,
                        ),
                    },
                ],
            }
        ],
        "max_output_tokens": 300,
    }
    request = urllib.request.Request(
        f"{config.api_base_url}/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=config.timeout_s) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI vision request failed with HTTP {exc.code} {exc.reason}: {body_text}"
        ) from exc

    raw_text = _extract_response_text(response_payload)
    parsed = _parse_json_object(raw_text)
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    user_hand_present = bool(parsed.get("user_hand_present"))
    robot_visible = bool(parsed.get("robot_visible"))
    robot_gripper_near_user_hand = bool(parsed.get("robot_gripper_near_user_hand"))
    robot_placing_target_in_user_hand = bool(parsed.get("robot_placing_target_in_user_hand"))
    target_food_visible = bool(parsed.get("target_food_visible"))
    target_food_in_user_hand = bool(parsed.get("target_food_in_user_hand"))
    target_food_on_tray = bool(parsed.get("target_food_on_tray"))
    correct_target_food = bool(parsed.get("correct_target_food", target_food_visible))
    user_grabbing_without_robot_placement = bool(
        parsed.get("user_grabbing_without_robot_placement")
    )
    success = (
        user_hand_present
        and robot_visible
        and robot_gripper_near_user_hand
        and robot_placing_target_in_user_hand
        and target_food_visible
        and target_food_in_user_hand
        and correct_target_food
        and not target_food_on_tray
        and not user_grabbing_without_robot_placement
        and confidence >= config.min_confidence
    )
    return OpenAISuccessResult(
        success=success,
        confidence=confidence,
        user_hand_present=user_hand_present,
        robot_visible=robot_visible,
        robot_gripper_near_user_hand=robot_gripper_near_user_hand,
        robot_placing_target_in_user_hand=robot_placing_target_in_user_hand,
        target_food_visible=target_food_visible,
        target_food_in_user_hand=target_food_in_user_hand,
        target_food_in_robot_gripper=bool(parsed.get("target_food_in_robot_gripper")),
        target_food_on_tray=target_food_on_tray,
        correct_target_food=correct_target_food,
        user_grabbing_without_robot_placement=user_grabbing_without_robot_placement,
        reason=str(parsed.get("reason", "")).strip(),
        raw_text=raw_text,
    )


def _frame_to_data_url(frame_rgb: np.ndarray, max_width: int, jpeg_quality: int) -> str:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenAI vision confirmation requires opencv-python.") from exc

    arr = np.asarray(frame_rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB frame, got shape {arr.shape}")
    if max_width > 0 and arr.shape[1] > max_width:
        scale = max_width / arr.shape[1]
        size = (max_width, max(1, int(round(arr.shape[0] * scale))))
        arr = cv2.resize(arr, size, interpolation=cv2.INTER_AREA)
    frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(jpeg_quality)))],
    )
    if not ok:
        raise RuntimeError("failed to encode frame as JPEG")
    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{image_b64}"


def _extract_response_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"].strip()
    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    text = "\n".join(chunks).strip()
    if not text:
        raise RuntimeError(f"OpenAI response did not contain output text: {payload}")
    return text


def _parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"OpenAI vision response was not JSON: {text!r}") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise RuntimeError(f"OpenAI vision response must be a JSON object: {text!r}")
    return value
