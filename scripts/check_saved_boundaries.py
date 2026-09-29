"""Replay the current field filter on recent S snapshots, without inference.

This checks display filtering only; saved observations are already roster-limited.
It cannot measure detector recall or recover detections omitted from a snapshot.
"""
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.tracking.field import FieldFilter
from src.tracking.tracking import Observation, on_field


def main():
    output = ROOT / "outputs" / "boundary_review"
    output.mkdir(parents=True, exist_ok=True)
    snapshots = sorted((ROOT / "outputs" / "review").glob("*/observations.json"),
                       key=lambda path: path.stat().st_mtime, reverse=True)[:3]
    report = []
    for path in snapshots:
        data = json.loads(path.read_text())
        field = FieldFilter(False)
        if data.get("field"):
            field.select(data["field"])
        frame = cv2.imread(str(path.parent / "original.png"))
        kept, rejected = [], []
        for row in data["observed"]:
            item = Observation(**row)
            accepted = on_field(item, field)
            (kept if accepted else rejected).append(item.track_id)
            x1, y1, x2, y2 = map(int, item.box)
            color = (0, 220, 0) if accepted else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        destination = output / f"frame_{data['frame']}.jpg"
        if not cv2.imwrite(str(destination), frame):
            raise OSError(f"Could not save {destination}")
        report.append(dict(frame=data["frame"], before=len(data["observed"]),
                           kept=len(kept), rejected_ids=rejected))
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
