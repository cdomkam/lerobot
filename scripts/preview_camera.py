"""Live OpenCV preview window for a camera index. Press 'q' to quit."""

from __future__ import annotations

import argparse
import sys

import cv2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("index", type=int, help="OpenCV camera index (e.g. 0 or 1)")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        print(f"Could not open camera {args.index}", file=sys.stderr)
        return 1
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    win = f"cam {args.index} — press 'q' to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imshow(win, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
