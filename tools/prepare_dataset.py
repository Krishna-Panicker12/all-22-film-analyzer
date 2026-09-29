"""Export complete manually reviewed snapshots to a YOLO player dataset.

Use separate games for training and validation. Partial or uncertain labels are
rejected. Original detector boxes are never promoted to ground truth.
"""
import argparse
import json
from pathlib import Path
import shutil
import hashlib

import cv2


def read_labels(folder, target="player"):
    records = []
    for file in sorted(folder.rglob("observations.json")):
        data = json.loads(file.read_text())
        if not data.get("review_complete"):
            continue
        if data.get("review_target", "player") != target:
            continue
        labels = data.get("manual_labels", [])
        if any(item.get("kind") not in {target, "non_player", "uncertain"} for item in labels):
            raise ValueError(f"Unknown annotation type: {file}")
        if any(item["kind"] == "uncertain" for item in labels):
            raise ValueError(f"Uncertain regions need review before export: {file}")
        if not labels:
            raise ValueError(f"Complete frame has no manual labels: {file}")
        image = cv2.imread(str(file.parent / "original.png"))
        if image is None:
            raise ValueError(f"Missing original image: {file}")
        height, width = image.shape[:2]
        lines = []
        for item in labels:
            x1, y1, x2, y2 = item["box"]
            if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                raise ValueError(f"Invalid manual box: {file}")
            if item["kind"] == target:
                lines.append(f"0 {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} "
                             f"{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
        records.append((file, data, lines))
    return records


def export_dataset(train_folder, val_folder, output, target="player", test_folder=None):
    splits = {"train": read_labels(train_folder, target), "val": read_labels(val_folder, target)}
    if test_folder is not None:
        splits["test"] = read_labels(test_folder, target)
    if any(not records for records in splits.values()):
        raise ValueError("Every requested split needs complete manually labeled frames. No dataset exported.")
    seen_games, seen_images, seen_frames = {}, {}, set()
    for split, rows in splits.items():
        if not any(lines for _, _, lines in rows):
            raise ValueError(f"{split} needs at least one positive {target} label.")
        for file, data, _ in rows:
            clip = str(Path(data["clip"]).resolve()).casefold()
            game = str(Path(data["clip"]).resolve().parent).casefold()
            frame = (clip, data["frame"])
            digest = hashlib.sha256((file.parent / "original.png").read_bytes()).hexdigest()
            if game in seen_games and seen_games[game] != split:
                raise ValueError("Training, validation and test must come from different game folders.")
            if frame in seen_frames:
                raise ValueError(f"Duplicate reviewed frame: {clip}, frame {data['frame']}. Keep one review.")
            if digest in seen_images:
                raise ValueError(f"Duplicate image content: {file}. Do not copy frames between splits.")
            seen_games[game], seen_images[digest] = split, split
            seen_frames.add(frame)
    if output.exists():
        raise ValueError("Choose a new output directory to preserve existing labels.")
    manifest = {"version": 1, "target": target, "splits": {}}
    for split, rows in splits.items():
        manifest["splits"][split] = []
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)
        for index, (file, data, lines) in enumerate(rows):
            name = f"{index:06d}"
            shutil.copyfile(file.parent / "original.png", output / "images" / split / f"{name}.png")
            (output / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
            image = output / "images" / split / f"{name}.png"
            label = output / "labels" / split / f"{name}.txt"
            manifest["splits"][split].append(dict(
                image=image.relative_to(output).as_posix(), label=label.relative_to(output).as_posix(),
                image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
                clip=str(Path(data["clip"]).resolve()), frame=data["frame"],
                review=str(file.resolve())))
        (output / f"{split}_sources.json").write_text(json.dumps([str(row[0]) for row in rows], indent=2))
    (output / "dataset.yaml").write_text(
        f"path: {json.dumps(output.resolve().as_posix())}\ntrain: images/train\nval: images/val\n"
        + ("test: images/test\n" if test_folder else "") + f"names:\n  0: {target}\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=["player", "helmet"], default="player")
    parser.add_argument("--test", type=Path, help="Optional third game held out from model selection")
    args = parser.parse_args()
    try:
        export_dataset(args.train, args.val, args.output, args.target, args.test)
        print(f"Exported dataset to {args.output.resolve()}")
    except ValueError as error:
        parser.exit(1, f"{error}\n")
