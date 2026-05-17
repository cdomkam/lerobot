"""Food target config and transcript classification for handoff policies."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

TARGETS = ("strawberry", "oreo", "marshmallow")

ALIASES: dict[str, tuple[str, ...]] = {
    "strawberry": ("strawberry", "strawberries"),
    "oreo": ("oreo", "oreos", "cookie", "cookies"),
    "marshmallow": ("marshmallow", "marshmallows", "marshmellow", "marshmellows"),
}

REQUEST_WORDS = {
    "bring",
    "fetch",
    "get",
    "give",
    "grab",
    "have",
    "like",
    "please",
    "snack",
    "food",
    "item",
    "eat",
    "want",
}

UNSUPPORTED_ITEM_FILLER_WORDS = REQUEST_WORDS | {
    "a",
    "an",
    "and",
    "can",
    "could",
    "i",
    "me",
    "my",
    "the",
    "to",
    "would",
    "you",
}


@dataclass(frozen=True)
class FoodPolicy:
    target: str
    display_name: str
    policy_repo_id: str
    task: str
    ood_detector_path: str = ""


@dataclass(frozen=True)
class FoodPolicyConfig:
    targets: dict[str, FoodPolicy]

    def require(self, target: str) -> FoodPolicy:
        canonical = canonicalize_target(target)
        if canonical is None or canonical not in self.targets:
            raise ValueError(
                f"unknown food target {target!r}; expected one of: {', '.join(TARGETS)}"
            )
        return self.targets[canonical]


def canonicalize_target(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "", value.lower())
    for target, aliases in ALIASES.items():
        if normalized == target or normalized in {re.sub(r'[^a-z0-9]+', '', a) for a in aliases}:
            return target
    return None


def classify_food_request(transcript: str) -> str | None:
    """Map a short user request transcript to a single food enum."""
    matches = match_food_targets(transcript)
    if len(matches) == 1:
        return next(iter(matches))
    return None


def match_food_targets(transcript: str) -> set[str]:
    """Return every supported food target mentioned in a transcript."""
    tokens = set(re.findall(r"[a-z0-9]+", transcript.lower()))
    return {
        target
        for target, aliases in ALIASES.items()
        if any(alias.lower() in tokens for alias in aliases)
    }


def is_probable_unsupported_food_request(transcript: str) -> bool:
    """Heuristic for requests that ask for food outside the supported set."""
    tokens = set(re.findall(r"[a-z0-9]+", transcript.lower()))
    if not tokens or match_food_targets(transcript):
        return False
    return bool(tokens & REQUEST_WORDS)


def unsupported_item_label(transcript: str) -> str:
    """Extract a short display label for an unsupported requested item."""
    words = re.findall(r"[a-z0-9]+", transcript.lower())
    candidates = [word for word in words if word not in UNSUPPORTED_ITEM_FILLER_WORDS]
    if not candidates:
        return "that item"
    return " ".join(candidates[-3:])


def load_food_policy_config(path: str | Path) -> FoodPolicyConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Food policy config not found: {config_path}. "
            "Expected config/food_policies.json with filled policy repo ids."
        )

    raw = json.loads(config_path.read_text())
    target_map = raw.get("targets")
    if not isinstance(target_map, dict):
        raise ValueError(f"{config_path} must contain a 'targets' object")

    policies: dict[str, FoodPolicy] = {}
    for key, value in target_map.items():
        canonical = canonicalize_target(key)
        if canonical is None:
            raise ValueError(f"unknown target key in {config_path}: {key!r}")
        if not isinstance(value, dict):
            raise ValueError(f"target {key!r} in {config_path} must be an object")
        policy_repo_id = str(value.get("policy_repo_id", "")).strip()
        task = str(value.get("task", "")).strip()
        if not policy_repo_id or policy_repo_id.startswith("TODO_"):
            raise ValueError(f"target {canonical!r} is missing a real policy_repo_id")
        if not task:
            raise ValueError(f"target {canonical!r} is missing a task prompt")
        policies[canonical] = FoodPolicy(
            target=canonical,
            display_name=str(value.get("display_name", canonical.title())).strip()
            or canonical.title(),
            policy_repo_id=policy_repo_id,
            task=task,
            ood_detector_path=str(value.get("ood_detector_path", "")).strip(),
        )

    missing = [target for target in TARGETS if target not in policies]
    if missing:
        raise ValueError(f"{config_path} is missing target configs: {', '.join(missing)}")
    return FoodPolicyConfig(targets=policies)
