"""Validate the local dataset and its provenance before training or evaluation."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import yaml
from src.data.catalog import game_identity


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_boxes(path, width=1, height=1):
    boxes = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label at {path}:{number}")
        cls, x, y, w, h = map(float, values)
        if (not all(math.isfinite(v) for v in (cls, x, y, w, h)) or cls != 0
                or not 0 < w <= 1 or not 0 < h <= 1
                or x - w / 2 < -1e-6 or x + w / 2 > 1.000001
                or y - h / 2 < -1e-6 or y + h / 2 > 1.000001):
            raise ValueError(f"Invalid normalized single-class box at {path}:{number}")
        boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
    return boxes


def validate_dataset(config):
    config = Path(config).resolve()
    if not config.is_file():
        raise ValueError(f"Dataset not found: {config}. Label frames, then run prepare_dataset.py first.")
    root = config.parent
    manifest_file = root / "manifest.json"
    if not manifest_file.is_file():
        raise ValueError("Dataset has no provenance manifest. Re-export with prepare_dataset.py.")
    manifest = json.loads(manifest_file.read_text())
    spec = yaml.safe_load(config.read_text())
    target = manifest.get("target")
    if manifest.get("version") != 1 or target not in ("player", "helmet"):
        raise ValueError("Unsupported dataset manifest")
    if spec.get("names") != {0: target} or Path(spec.get("path", "")).resolve() != root:
        raise ValueError("Dataset YAML does not match its manifest/location. Re-export it.")
    splits = manifest["splits"]
    if not {"train", "val"} <= splits.keys() or not splits.keys() <= {"train", "val", "test"}:
        raise ValueError("Dataset needs train and val splits; test is optional")
    games, images, identities, report = {}, set(), set(), {}
    for split, records in splits.items():
        if spec.get(split) != f"images/{split}" or not records:
            raise ValueError(f"Invalid or empty {split} split")
        expected_images, expected_labels = set(), set()
        positives = 0
        for record in records:
            image = (root / record["image"]).resolve()
            label = (root / record["label"]).resolve()
            if image.parent != root / "images" / split or label.parent != root / "labels" / split:
                raise ValueError("Manifest path is outside its split")
            if not image.is_file() or not label.is_file():
                raise ValueError(f"Missing image or label: {image}")
            if sha256(image) != record["image_sha256"] or sha256(label) != record["label_sha256"]:
                raise ValueError(f"Dataset changed after export: {image.name}. Re-export reviewed labels.")
            if record["image_sha256"] in images:
                raise ValueError("Duplicate image content in dataset")
            game = game_identity(record['clip'],record.get('game_id'))
            identity = (str(Path(record["clip"])).casefold(), record["frame"])
            if identity in identities or (game in games and games[game] != split):
                raise ValueError("Frame or game leakage between dataset splits")
            decoded = cv2.imread(str(image))
            if decoded is None:
                raise ValueError(f"Unreadable image: {image}")
            positives += len(read_boxes(label))
            games[game] = split
            identities.add(identity)
            images.add(record["image_sha256"])
            expected_images.add(image)
            expected_labels.add(label)
        # The trainer scans these directories: unlisted extra examples are unsafe.
        actual_images = {p.resolve() for p in (root / "images" / split).rglob("*") if p.is_file()}
        actual_labels = {p.resolve() for p in (root / "labels" / split).rglob("*.txt")}
        if actual_images != expected_images or actual_labels != expected_labels:
            raise ValueError(f"Unlisted image/label files in {split}; re-export a clean dataset")
        if positives == 0:
            raise ValueError(f"No positive {target} labels in {split}")
        report[split] = {"images": len(records), "objects": positives,
                         "games": len({game_identity(r['clip'],r.get('game_id')) for r in records})}
    return {"yaml": str(config), "root": root, "target": target, "manifest": manifest,
            "manifest_sha256": sha256(manifest_file), "counts": report}
