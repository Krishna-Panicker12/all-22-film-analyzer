"""Check reviewed helmet labels and create numbered overlays without changing labels."""
import argparse
import json
from pathlib import Path

import cv2

from tools.prepare_dataset import read_labels

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/annotations/game_01")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/helmet_label_audit")
    args = parser.parse_args()
    records = read_labels(args.input, "helmet")
    if not records:
        parser.error("No completed helmet frames found")
    if args.output.exists():
        parser.error("Choose a new output folder to preserve the previous audit")
    args.output.mkdir(parents=True)
    rows = []
    for number, (file, data, lines) in enumerate(records):
        image = cv2.imread(str(file.parent / "original.png"))
        for index, label in enumerate(data["manual_labels"], 1):
            x1, y1, x2, y2 = map(round, label["box"])
            color = (0, 255, 0) if label["kind"] == "helmet" else (0, 255, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
            cv2.putText(image, str(index), (x1, max(12, y1-3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        name = f"{number:03d}_frame_{data['frame']}.jpg"
        if not cv2.imwrite(str(args.output / name), image):
            raise OSError(f"Could not write {name}")
        rows.append(dict(review=str(file.resolve()), frame=data["frame"],
                         helmets=len(lines), overlay=name))
    report = dict(frames=len(rows), helmets=sum(r["helmets"] for r in rows),
                  note="Structural checks only. Visually check for missed helmets and incorrect boxes.",
                  records=rows)
    (args.output / "audit.json").write_text(json.dumps(report, indent=2))
    print(f"Checked {report['frames']} completed frames, {report['helmets']} helmet boxes.\n{args.output.resolve()}")


if __name__ == "__main__":
    main()
