"""Train YOLOv8 pose on the 86-keypoint character set. From rozan.ipynb / rozan2.ipynb."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--pose", type=float, default=15.0)
    parser.add_argument("--name", default="character_pose")
    parser.add_argument("--project", default="yolov8_pose_run")
    parser.add_argument("--weights", default="yolov8n-pose.pt")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=0,
        pose=args.pose,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
