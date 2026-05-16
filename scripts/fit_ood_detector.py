#!/usr/bin/env python
"""Fit the OOD detector on frames from a lerobot dataset.

Pulls a ``LeRobotDataset`` (from local cache or HuggingFace), encodes a
subsample of frames, and saves a Mahalanobis detector.

By default the encoder is ACT's own ResNet backbone — pass
``--policy_path <repo_or_dir>`` so we can load it. If you'd rather use
a generic visual encoder (e.g. while evaluating a policy you don't
own), pass ``--encoder dinov2_vits14``.

Examples
--------

::

    # Default: reuse ACT's backbone.
    uv run python scripts/fit_ood_detector.py \\
        --dataset_repo_id ofcourseistillloveyou/so-101-feed-me \\
        --policy_path  ofcourseistillloveyou/act-so101-feed-me-vai-10ep-run1 \\
        --camera_name  front \\
        --output_path  models/ood_detector.npz

    # Fallback: DINOv2.
    uv run python scripts/fit_ood_detector.py \\
        --dataset_repo_id ofcourseistillloveyou/so-101-feed-me \\
        --encoder dinov2_vits14 \\
        --camera_name front
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import init_logging

from lerobot_ood import ACTBackboneEncoder, DinoV2Encoder, OODDetector
from lerobot_ood._image import to_uint8_rgb_hw3

logger = logging.getLogger("fit_ood_detector")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_repo_id", required=True)
    p.add_argument("--camera_name", default="front",
                   help="Camera to use (default 'front'). Must match the runtime --ood_camera.")
    p.add_argument("--output_path", default="models/ood_detector.npz")
    p.add_argument("--max_frames", type=int, default=2000,
                   help="Random subsample size. Set to 0 to use all frames.")
    p.add_argument(
        "--encoder",
        default="act_backbone",
        help="'act_backbone' (default; requires --policy_path) or 'dinov2_<size>'.",
    )
    p.add_argument("--policy_path", default=None,
                   help="HF repo id or local path of the ACT policy (required for act_backbone).")
    p.add_argument("--device", default=None)
    p.add_argument("--pca_components", type=int, default=32)
    p.add_argument("--threshold_percentile", type=float, default=95.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _build_encoder(args):
    if args.encoder == "act_backbone":
        if not args.policy_path:
            raise ValueError("--policy_path is required when --encoder=act_backbone")
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        logger.info("Loading ACT policy from '%s'...", args.policy_path)
        policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        policy_cfg.pretrained_path = args.policy_path
        policy_class = get_policy_class(policy_cfg.type)
        policy = policy_class.from_pretrained(args.policy_path, config=policy_cfg)
        if args.device:
            policy = policy.to(args.device)
        policy.eval()
        return ACTBackboneEncoder(policy=policy, device=args.device)

    if args.encoder.startswith("dinov2_"):
        return DinoV2Encoder(model=args.encoder, device=args.device)

    raise ValueError(
        f"unknown --encoder '{args.encoder}'. "
        f"Expected 'act_backbone' or 'dinov2_<size>'."
    )


def main() -> None:
    init_logging()
    args = parse_args()

    logger.info("Loading dataset '%s'...", args.dataset_repo_id)
    ds = LeRobotDataset(args.dataset_repo_id)
    image_key = f"observation.images.{args.camera_name}"
    if image_key not in ds.features:
        image_keys = [k for k in ds.features if "images" in k]
        raise KeyError(
            f"camera '{args.camera_name}' not in dataset; available image keys: {image_keys}"
        )

    n_total = len(ds)
    rng = np.random.default_rng(args.seed)
    if args.max_frames > 0 and n_total > args.max_frames:
        idx = rng.choice(n_total, size=args.max_frames, replace=False)
    else:
        idx = np.arange(n_total)
    logger.info("Encoding %d / %d frames from key '%s' with encoder '%s'",
                len(idx), n_total, image_key, args.encoder)

    encoder = _build_encoder(args)

    out = []
    for i, j in enumerate(idx, 1):
        sample = ds[int(j)]
        out.append(encoder(to_uint8_rgb_hw3(sample[image_key])))
        if i % 100 == 0:
            logger.info("  encoded %d / %d", i, len(idx))
    embeddings = np.stack(out, axis=0)
    logger.info("Embedding matrix: %s", embeddings.shape)

    detector = OODDetector(
        threshold_percentile=args.threshold_percentile,
        pca_components=args.pca_components,
    ).fit(embeddings)
    logger.info(
        "Detector fitted: threshold=%.3f (p%.0f of in-dist scores)",
        detector.threshold,
        args.threshold_percentile,
    )

    detector.save(args.output_path)
    logger.info("Saved detector to %s", args.output_path)


if __name__ == "__main__":
    main()
