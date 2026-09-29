"""Generate artificial rectangles ONLY to exercise training; never a football dataset.

Run from the project root: python -m tests.smoke_training is not required;
use python tests/smoke_training.py, then the printed training command.
"""
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prepare_dataset import export_dataset


if __name__ == "__main__":
    root = ROOT / "outputs/synthetic_training_smoke"
    if root.exists():
        raise SystemExit(f"Refusing to overwrite {root}")
    rng = np.random.default_rng(42)
    for split in ("train", "val", "test"):
        for index in range(2):
            folder = root / "reviews" / split / str(index)
            folder.mkdir(parents=True)
            image = rng.integers(0, 100, (64, 64, 3), dtype=np.uint8)
            cv2.rectangle(image, (20, 20), (40, 40), (240, 240, 240), -1)
            cv2.imwrite(str(folder / "original.png"), image)
            (folder / "observations.json").write_text(json.dumps(dict(
                clip=str(root / f"synthetic_game_{split}/clip.mp4"), frame=index,
                review_target="helmet", review_complete=True,
                manual_labels=[dict(kind="helmet", box=[20, 20, 40, 40])],
                note="ARTIFICIAL RECTANGLES: pipeline smoke test, not real helmets")))
    export_dataset(root / "reviews/train", root / "reviews/val", root / "dataset",
                   "helmet", root / "reviews/test")
    print("Synthetic data ready. These weights must never be used as a football detector.")
    print("python -m tools.train_detector --data outputs/synthetic_training_smoke/dataset/dataset.yaml "
          "--output outputs/synthetic_training_smoke/run --epochs 1 --imgsz 64 --batch 2 --device cpu")
