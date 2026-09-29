"""Train an explicitly experimental helmet model when only one labeled clip exists.

The library's validation uses the TRAINING images for diagnostics only. It is
not a held-out accuracy estimate. Use python -m tools.train_detector for proper evaluation.
"""
import argparse
import json
from pathlib import Path
import shutil
import hashlib

from tools.prepare_dataset import read_labels

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolo11n.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/training/helmet_pilot")
    args = parser.parse_args()
    if args.epochs < 1 or args.imgsz < 32:
        parser.error("Positive epochs and imgsz >= 32 required")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output folder")
    rows = read_labels(ROOT / "data/annotations/game_01", "helmet")
    if not rows:
        parser.error("No reviewed helmet labels")
    dataset = output / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "labels").mkdir()
    for i,(file,data,lines) in enumerate(rows):
        shutil.copyfile(file.parent / "original.png", dataset / "images" / f"{i:04d}.png")
        (dataset / "labels" / f"{i:04d}.txt").write_text("\n".join(lines))
    config = dataset / "dataset.yaml"
    config.write_text(f"path: {json.dumps(dataset.as_posix())}\ntrain: images\nval: images\nnames:\n  0: helmet\n")
    report = dict(status="started", experimental=True, images=len(rows),
                  labels=sum(len(row[2]) for row in rows), epochs=args.epochs,
                  initial_model=str(args.model.resolve()),
                  initial_model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  note="Training and diagnostic validation use the same images. No held-out accuracy claim.",
                  reviews=[str(row[0].resolve()) for row in rows])
    (output / "pilot.json").write_text(json.dumps(report, indent=2))
    print(report["note"], flush=True)
    from ultralytics import YOLO
    model = YOLO(str(args.model.resolve()))
    model.train(data=str(config), epochs=args.epochs, imgsz=args.imgsz, batch=2,
                device="cpu", workers=0, amp=False, cache=False, seed=0,
                project=str(output), name="run", exist_ok=False, patience=0,
                mosaic=0, scale=.2, close_mosaic=0, plots=True,
                optimizer="AdamW", lr0=.001, nbs=2, warmup_epochs=0)
    report["status"] = "complete"
    report["weights"] = str(output / "run/weights/last.pt")
    (output / "pilot.json").write_text(json.dumps(report, indent=2))
    print(f"Pilot weights: {report['weights']}")


if __name__ == "__main__":
    main()
