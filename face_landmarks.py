"""Draw MediaPipe Face Landmarker dots on every image in a folder."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def ensure_model(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name}...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Download complete.")
    return path


def make_detector(model_path: Path) -> vision.FaceLandmarker:
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def process_image(detector: vision.FaceLandmarker, src: Path, dest: Path) -> None:
    image_mp = mp.Image.create_from_file(str(src))
    result = detector.detect(image_mp)
    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(src)
    h, w, _ = img.shape
    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            for lm in face_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), img)


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Folder of images")
    parser.add_argument("--output", required=True, type=Path, help="Folder for annotated images")
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    args = parser.parse_args()

    detector = make_detector(ensure_model(args.model))
    images = collect_images(args.input)
    print(f"Found {len(images)} images to process...")
    for src in images:
        dest = args.output / f"annotated_{src.name}"
        try:
            process_image(detector, src, dest)
            print(f"Saved {dest}")
        except Exception as exc:
            print(f"Error processing {src.name}: {exc}")
    print(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()
