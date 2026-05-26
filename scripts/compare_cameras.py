"""Compare local cameras 0/1 against the training dataset's front/side streams.

Downloads the first episode of each camera from the HF dataset, grabs frame 0,
captures one frame from local OpenCV indices 0 and 1, and writes everything to
outputs/camera_compare/ so you can eyeball which live index matches which.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download

DATASET_REPO = "ofcourseistillloveyou/so101_recording_marshmellow_40ep"
CAMERA_KEYS = ["front", "side"]
LOCAL_INDICES = [0, 1]
OUT_DIR = Path("outputs/camera_compare")


def first_frame_from_video(video_path: Path) -> "cv2.typing.MatLike":
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame from {video_path}")
    return frame


def grab_dataset_frames() -> None:
    for key in CAMERA_KEYS:
        rel = f"videos/observation.images.{key}/chunk-000/file-000.mp4"
        local = hf_hub_download(repo_id=DATASET_REPO, filename=rel, repo_type="dataset")
        frame = first_frame_from_video(Path(local))
        out = OUT_DIR / f"dataset_{key}.png"
        cv2.imwrite(str(out), frame)
        print(f"  dataset {key:<5} -> {out}  ({frame.shape[1]}x{frame.shape[0]})")


def grab_live_frames() -> None:
    for idx in LOCAL_INDICES:
        cap = cv2.VideoCapture(idx)
        # warm up — some cams return a black first frame
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"  live index {idx}: FAILED to capture", file=sys.stderr)
            continue
        out = OUT_DIR / f"live_{idx}.png"
        cv2.imwrite(str(out), frame)
        print(f"  live  {idx}     -> {out}  ({frame.shape[1]}x{frame.shape[0]})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Dataset frames:")
    grab_dataset_frames()
    print("Live frames:")
    grab_live_frames()
    print(f"\nOpen them with:  open {OUT_DIR}/*.png")


if __name__ == "__main__":
    main()
