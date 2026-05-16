#!/usr/bin/env python
"""Fit the OOD detector on frames from a lerobot dataset.

Pulls a ``LeRobotDataset`` (from local cache or HuggingFace), encodes a
subsample of frames with DINOv2, and saves a Mahalanobis detector.

Example
-------

::

    uv run python scripts/fit_ood_detector.py \\
        --dataset_repo_id ofcourseistillloveyou/so-101-feed-me \\
        --camera_name front \\
        --output_path models/ood_detector.npz
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import init_logging

from lerobot_ood import DinoV2Encoder, OODDetector

logger = logging.getLogger("fit_ood_detector")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_repo_id", required=True)
    p.add_argument("--camera_name", default="front",
                   help="Camera to use (default 'front'). Must match the runtime --ood_camera.")
    p.add_argument("--output_path", default="models/ood_detector.npz")
    p.add_argument("--max_frames", type=int, default=2000,
                   help="Random subsample size. Set to 0 to use all frames.")
    p.add_argument("--encoder", default="dinov2_vits14")
    p.add_argument("--device", default=None)
    p.add_argument("--pca_components", type=int, default=32)
    p.add_argument("--threshold_percentile", type=float, default=95.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _to_uint8_rgb_hw3(img) -> np.ndarray:
    """LeRobotDataset gives back torch tensors in CHW float [0,1]; convert."""
    if torch.is_tensor(img):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


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
    logger.info("Encoding %d / %d frames from key '%s'", len(idx), n_total, image_key)

    encoder = DinoV2Encoder(model=args.encoder, device=args.device)

    embeddings = np.empty((len(idx), 0), dtype=np.float32)
    out = []
    for i, j in enumerate(idx, 1):
        sample = ds[int(j)]
        out.append(encoder(_to_uint8_rgb_hw3(sample[image_key])))
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
