"""Draw face contours, Loomis construction, pose, and hands on a folder of images."""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from paths import TRAIN_IMAGES

MODELS = {
    "face": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    ),
    "pose": (
        "pose_landmarker_heavy.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    ),
    "hand": (
        "hand_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    ),
}

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 61]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
RED = (0, 0, 255)


def ensure_models(model_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, (filename, url) in MODELS.items():
        path = model_dir / filename
        if not path.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def make_detectors(paths: dict[str, Path]):
    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(paths["face"])),
            num_faces=1,
            min_face_detection_confidence=0.2,
            min_face_presence_confidence=0.2,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(paths["pose"])),
            num_poses=1,
            min_pose_detection_confidence=0.2,
            min_pose_presence_confidence=0.2,
        )
    )
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(paths["hand"])),
            num_hands=2,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
        )
    )
    return face, pose, hand


def process_single_image(img_path: Path, save_dir: Path, face_detector, pose_detector, hand_detector) -> None:
    img_cv = cv2.imread(str(img_path))
    if img_cv is None:
        return
    if len(img_cv.shape) == 2:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
    elif img_cv.shape[2] == 4:
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR)

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    h, w, _ = img_cv.shape
    face_res = face_detector.detect(image)
    pose_res = pose_detector.detect(image)
    hand_res = hand_detector.detect(image)

    if face_res.face_landmarks:
        f_pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_res.face_landmarks[0])}
        for group in [FACE_OVAL, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS]:
            g_pts = np.array([f_pts[i] for i in group if i in f_pts], dtype=np.int32)
            if len(g_pts) > 1:
                cv2.polylines(img_cv, [g_pts], isClosed=False, color=RED, thickness=1, lineType=cv2.LINE_AA)
        if 168 in f_pts and 152 in f_pts:
            top_a, bot_a = np.array(f_pts[168]), np.array(f_pts[152])
            vec = bot_a - top_a
            norm = np.linalg.norm(vec)
            if norm > 0:
                dir_v = vec / norm
                cv2.line(
                    img_cv,
                    tuple((top_a - dir_v * (norm * 0.4)).astype(int)),
                    tuple((bot_a + dir_v * (norm * 0.2)).astype(int)),
                    RED,
                    2,
                    cv2.LINE_AA,
                )
                center = tuple((top_a + dir_v * (norm * 0.1)).astype(int))
                angle = np.degrees(np.arctan2(dir_v[1], dir_v[0])) - 90
                cv2.ellipse(
                    img_cv,
                    center,
                    (int(norm * 1.0) // 2, int(norm * 1.3) // 2),
                    angle,
                    0,
                    360,
                    RED,
                    2,
                    cv2.LINE_AA,
                )

    if pose_res.pose_landmarks:
        p_pts = {
            i: (int(lm.x * w), int(lm.y * h))
            for i, lm in enumerate(pose_res.pose_landmarks[0])
            if lm.visibility > 0.2
        }
        if all(k in p_pts for k in [11, 12, 23, 24]):
            torso_poly = np.array([p_pts[11], p_pts[12], p_pts[24], p_pts[23]], dtype=np.int32)
            cv2.polylines(img_cv, [torso_poly], isClosed=True, color=RED, thickness=3, lineType=cv2.LINE_AA)
        limbs = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]
        for p1, p2 in limbs:
            if p1 in p_pts and p2 in p_pts:
                cv2.line(img_cv, p_pts[p1], p_pts[p2], RED, 2, cv2.LINE_AA)
        for hip_idx in [23, 24]:
            if hip_idx in p_pts:
                cv2.circle(img_cv, p_pts[hip_idx], 14, RED, 2, cv2.LINE_AA)

    if hand_res.hand_landmarks:
        for hand_lms in hand_res.hand_landmarks:
            h_pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(hand_lms)}
            palm_indices = [0, 5, 9, 13, 17]
            if all(i in h_pts for i in palm_indices):
                palm_poly = np.array([h_pts[i] for i in palm_indices], dtype=np.int32)
                cv2.polylines(img_cv, [palm_poly], isClosed=True, color=RED, thickness=2, lineType=cv2.LINE_AA)
            finger_chains = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
            for chain in finger_chains:
                for i in range(len(chain) - 1):
                    if chain[i] in h_pts and chain[i + 1] in h_pts:
                        cv2.line(img_cv, h_pts[chain[i]], h_pts[chain[i + 1]], RED, 2, cv2.LINE_AA)

    save_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_dir / img_path.name), img_cv)


def collect_images(folder: Path) -> list[Path]:
    paths: list[Path] = []
    for root, _, files in os.walk(folder):
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() in IMAGE_EXTS:
                paths.append(path)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TRAIN_IMAGES, help="Folder of images")
    parser.add_argument("--output", type=Path, default=Path("out/anatomy"), help="Folder for overlay images")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images (0 = all)")
    args = parser.parse_args()

    face, pose, hand = make_detectors(ensure_models(args.models))
    images = collect_images(args.input)
    if args.limit:
        images = images[: args.limit]
    print(f"Processing {len(images)} images...")
    for idx, src in enumerate(images, 1):
        try:
            process_single_image(src, args.output, face, pose, hand)
        except Exception as exc:
            print(f"Skipping {src.name}: {exc}")
        if idx % 10 == 0 or idx == len(images):
            print(f"Processed {idx}/{len(images)}")
    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
