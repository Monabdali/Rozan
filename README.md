# Rozan

Construction overlays for drawing. Face landmarks, Loomis, pose, and hands.

![Rozan Loomis overlay](docs/result.png)

Colab PDFs mapped to Python. Images come from local `D:\project\rozan-dataset` (same files as Drive).

| Colab | Script | What it did |
| --- | --- | --- |
| `rozan-m.ipynb` | `anatomy_overlay.py` | MediaPipe face + pose + hands. This is the red mannequin (torso box, hip circles, hand cards). |
| `rozan-f1.ipynb` | `face_landmarks.py` | Face Landmarker dots on test images |
| `rozan.ipynb` | `fix_labels.py`, `train_pose.py` | YOLO 86-keypoint train v1–v4, label cleanup |
| `rozan2.ipynb` | `train_pose.py` | Later YOLO runs + Loomis drawn from YOLO keypoints |

```bash
pip install -r requirements.txt
```

Mannequin overlay (rozan-m) on local train photos:

```bash
python anatomy_overlay.py
```

Face dots (rozan-f1) on local test photos:

```bash
python face_landmarks.py
```

Rebuild boxes, then train YOLO (needs GPU):

```bash
python fix_labels.py
python train_pose.py
```

Models download on first MediaPipe run into `models/`. The Flutter app is not in this repo.
