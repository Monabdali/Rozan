"""Local dataset roots. Same files as the Colab Drive folder `/content/drive/MyDrive/dataset`."""

from pathlib import Path

DATASET = Path(r"D:\project\rozan-dataset")
IMAGES = DATASET / "images"
TRAIN_IMAGES = IMAGES / "train"
VAL_IMAGES = IMAGES / "val"
TEST_IMAGES = IMAGES / "test"
LABELS = DATASET / "dataset" / "labels"
TRAIN_LABELS = LABELS / "train"
VAL_LABELS = LABELS / "val"
