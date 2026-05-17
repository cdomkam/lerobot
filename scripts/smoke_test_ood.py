#!/usr/bin/env python
"""Smoke-test a fitted OOD detector against one or more datasets.

Encodes a subsample of frames from each dataset, scores them, and
prints OOD rate + score percentiles. Useful for sanity-checking
the threshold without the robot: an in-dist dataset should sit
near 5% OOD (the p95 fit baseline), an OOD dataset should be much
higher.

Example
-------

::

    uv run python scripts/smoke_test_ood.py \\
        --detector_path models/ood_detector.npz \\
        --policy_path   ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1 \\
        --camera_name   front \\
        --datasets \\
            ofcourseistillloveyou/so-101-feed-me \\
            ofcourseistillloveyou/so101_recording_20260516_105727
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import init_logging

from lerobot_ood import ACTBackboneEncoder, DinoV2Encoder, OODDetector
from lerobot_ood._image import to_uint8_rgb_hw3

logger = logging.getLogger("smoke_test_ood")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--detector_path", default="models/ood_detector.npz")
    p.add_argument("--datasets", nargs="+", required=True,
                   help="One or more LeRobotDataset repo ids (HF or local).")
    p.add_argument("--camera_name", default="front")
    p.add_argument("--encoder", default="act_backbone")
    p.add_argument("--policy_path", default=None,
                   help="Required when --encoder=act_backbone.")
    p.add_argument("--device", default=None)
    p.add_argument("--max_frames", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _build_encoder(args):
    if args.encoder == "act_backbone":
        if not args.policy_path:
            raise ValueError("--policy_path is required for --encoder=act_backbone")
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        logger.info("Loading ACT policy from '%s' ...", args.policy_path)
        cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        cfg.pretrained_path = args.policy_path
        cls = get_policy_class(cfg.type)
        policy = cls.from_pretrained(args.policy_path, config=cfg)
        if args.device:
            policy = policy.to(args.device)
        policy.eval()
        return ACTBackboneEncoder(policy=policy, device=args.device)
    if args.encoder.startswith("dinov2_"):
        return DinoV2Encoder(model=args.encoder, device=args.device)
    raise ValueError(f"unknown --encoder '{args.encoder}'")


def _score_dataset(ds, encoder, detector, image_key, idx, label):
    scores = np.empty(len(idx), dtype=np.float32)
    for i, j in enumerate(idx):
        sample = ds[int(j)]
        emb = encoder(to_uint8_rgb_hw3(sample[image_key]))
        scores[i] = detector.score(emb).score
        if (i + 1) % 100 == 0:
            logger.info("  [%s] scored %d / %d", label, i + 1, len(idx))
    return scores


def main() -> None:
    init_logging()
    args = parse_args()

    detector = OODDetector.load(args.detector_path)
    tau = detector.threshold
    logger.info("Detector threshold τ = %.3f", tau)
    encoder = _build_encoder(args)

    rng = np.random.default_rng(args.seed)
    image_key = f"observation.images.{args.camera_name}"

    results = []
    for repo_id in args.datasets:
        logger.info("Loading dataset '%s' ...", repo_id)
        ds = LeRobotDataset(repo_id)
        if image_key not in ds.features:
            image_keys = [k for k in ds.features if "images" in k]
            raise KeyError(
                f"'{image_key}' not in dataset '{repo_id}'. Image keys: {image_keys}"
            )
        n = len(ds)
        if args.max_frames > 0 and n > args.max_frames:
            idx = rng.choice(n, size=args.max_frames, replace=False)
        else:
            idx = np.arange(n)
        label = repo_id.split("/")[-1]
        logger.info("Scoring %d / %d frames from '%s' ...", len(idx), n, label)
        scores = _score_dataset(ds, encoder, detector, image_key, idx, label)
        results.append((repo_id, scores))

    print()
    print(f"Detector threshold τ = {tau:.3f}  (p95 of training in-dist scores)")
    print()
    header = (
        f"{'dataset':<55s} {'n':>5s} {'mean':>8s} {'med':>8s} "
        f"{'p90':>8s} {'p99':>8s} {'max':>8s} {'%OOD':>7s}"
    )
    print(header)
    print("-" * len(header))
    for repo_id, scores in results:
        pct_ood = 100.0 * (scores >= tau).mean()
        print(
            f"{repo_id:<55s} {len(scores):>5d} "
            f"{scores.mean():>8.2f} {np.median(scores):>8.2f} "
            f"{np.percentile(scores, 90):>8.2f} {np.percentile(scores, 99):>8.2f} "
            f"{scores.max():>8.2f} {pct_ood:>6.1f}%"
        )


if __name__ == "__main__":
    main()
