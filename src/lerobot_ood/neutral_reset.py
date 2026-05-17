"""Neutral-pose reset helpers for SO-101 handoff cycles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SO101_NEUTRAL_ACTION_KEYS: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


@dataclass(frozen=True)
class NeutralResetConfig:
    action: dict[str, float]


def load_neutral_reset_config(path: str | Path) -> NeutralResetConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Neutral reset config not found: {config_path}. "
            "Create it with an 'action' object containing SO-101 joint targets."
        )

    raw = json.loads(config_path.read_text())
    action = raw.get("action")
    if not isinstance(action, dict):
        raise ValueError(f"{config_path} must contain an 'action' object")

    missing = [key for key in SO101_NEUTRAL_ACTION_KEYS if key not in action]
    if missing:
        raise ValueError(f"{config_path} neutral action is missing keys: {', '.join(missing)}")

    parsed: dict[str, float] = {}
    for key in SO101_NEUTRAL_ACTION_KEYS:
        try:
            parsed[key] = float(action[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{config_path} neutral action key {key!r} must be numeric") from exc
    return NeutralResetConfig(action=parsed)


def interpolate_neutral_actions(
    current: dict[str, float],
    target: dict[str, float],
    steps: int,
) -> list[dict[str, float]]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    missing_current = [key for key in SO101_NEUTRAL_ACTION_KEYS if key not in current]
    missing_target = [key for key in SO101_NEUTRAL_ACTION_KEYS if key not in target]
    if missing_current:
        raise ValueError(f"current action is missing keys: {', '.join(missing_current)}")
    if missing_target:
        raise ValueError(f"target action is missing keys: {', '.join(missing_target)}")

    actions: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        alpha = step / steps
        actions.append(
            {
                key: float(current[key]) + (float(target[key]) - float(current[key])) * alpha
                for key in SO101_NEUTRAL_ACTION_KEYS
            }
        )
    return actions
