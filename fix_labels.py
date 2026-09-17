"""Rebuild YOLO boxes from visible keypoints. From rozan.ipynb / rozan2.ipynb."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path


def fix_labels(folder: Path, pad: float = 0.05) -> int:
    count = 0
    for file_path in glob.glob(os.path.join(folder, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = parts[0]
            kpts = [float(x) for x in parts[5:]]
            xs = [kpts[i] for i in range(0, len(kpts), 3) if i + 2 < len(kpts) and kpts[i + 2] > 0]
            ys = [kpts[i + 1] for i in range(0, len(kpts), 3) if i + 2 < len(kpts) and kpts[i + 2] > 0]
            if xs and ys:
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                pad_x = (max_x - min_x) * pad
                pad_y = (max_y - min_y) * pad
                x1 = max(0.0, min_x - pad_x)
                y1 = max(0.0, min_y - pad_y)
                x2 = min(1.0, max_x + pad_x)
                y2 = min(1.0, max_y + pad_y)
                box_w = max(x2 - x1, 0.01)
                box_h = max(y2 - y1, 0.01)
                center_x = x1 + box_w / 2.0
                center_y = y1 + box_h / 2.0
                new_lines.append(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f} "
                    + " ".join(parts[5:])
                    + "\n"
                )
            else:
                new_lines.append(line if line.endswith("\n") else line + "\n")
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    args = parser.parse_args()
    n_train = fix_labels(args.train)
    n_val = fix_labels(args.val)
    print(f"Fixed {n_train} train and {n_val} val label files.")


if __name__ == "__main__":
    main()
