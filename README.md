# Rozan

Construction overlays for drawing. Face landmarks, Loomis, pose, and hands.

![Rozan Loomis overlay](docs/result.png)

## Scripts

```bash
pip install -r requirements.txt
```

Face Landmarker dots:

```bash
python face_landmarks.py --input path/to/images --output path/to/out
```

Face contours, Loomis, pose, and hands:

```bash
python anatomy_overlay.py --input path/to/images --output path/to/out
```

Models download on first run into `models/`. The dataset and mobile app are not in this repo.
