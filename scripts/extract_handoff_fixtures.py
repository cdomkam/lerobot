#!/usr/bin/env python3
"""Extract local food-handoff mock fixtures from a LeRobot/HF dataset.

The generated images/manifests are local artifacts. By default they are written
under ``outputs/``, which is gitignored in this repo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import pyarrow.parquet as pq

DEFAULT_REPO_ID = "ofcourseistillloveyou/so101_recording_strawberry_num40_20260516_161110"
DEFAULT_LOCAL_ROOT = Path("data/hf/so101_recording_strawberry_num40_20260516_161110")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--dataset-root", default=str(DEFAULT_LOCAL_ROOT))
    parser.add_argument("--output-dir", default="outputs/food_handoff_fixtures")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--frames",
        default="0,30,90,180,270,360,449",
        help="Comma-separated frame indexes within the episode.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset into --dataset-root before extracting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    if args.download:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=dataset_root,
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}. Run with --download or use `hf download` first."
        )

    frames = [int(value) for value in args.frames.split(",") if value.strip()]
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    fps = int(info["fps"])
    episode = load_episode(dataset_root, args.episode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "repo_id": args.repo_id,
        "dataset_root": str(dataset_root),
        "episode": args.episode,
        "task": episode["tasks"][0],
        "fps": fps,
        "fixtures": [],
    }
    for camera in ("front", "side"):
        video_path = dataset_root / info["video_path"].format(
            video_key=f"observation.images.{camera}",
            chunk_index=episode[f"videos/observation.images.{camera}/chunk_index"],
            file_index=episode[f"videos/observation.images.{camera}/file_index"],
        )
        start_frame = round(
            episode[f"videos/observation.images.{camera}/from_timestamp"] * fps
        )
        extracted = extract_video_frames(video_path, [start_frame + f for f in frames])
        for frame_index, image in extracted.items():
            local_frame = frame_index - start_frame
            image_path = output_dir / f"{camera}_episode{args.episode:03d}_frame{local_frame:03d}.jpg"
            image.save(image_path, quality=92)
            manifest["fixtures"].append(
                {
                    "camera": camera,
                    "episode": args.episode,
                    "frame_index": local_frame,
                    "video_frame_index": frame_index,
                    "path": str(image_path),
                    "target": "strawberry",
                    "hand_present": True,
                    "success": bool(local_frame >= 270),
                }
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


def load_episode(dataset_root: Path, episode_index: int) -> dict:
    episode_tables = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    for path in episode_tables:
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            if row["episode_index"] == episode_index:
                return row
    raise ValueError(f"episode {episode_index} not found under {dataset_root}")


def extract_video_frames(video_path: Path, frame_indexes: list[int]):
    wanted = set(frame_indexes)
    extracted = {}
    with av.open(str(video_path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                extracted[i] = frame.to_image().convert("RGB")
                if wanted.issubset(extracted.keys()):
                    break
    missing = wanted - set(extracted.keys())
    if missing:
        raise RuntimeError(f"missing frames from {video_path}: {sorted(missing)}")
    return extracted


if __name__ == "__main__":
    main()
