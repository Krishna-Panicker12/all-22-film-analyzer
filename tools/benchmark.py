"""Compare settings on consecutive frames. Counts are diagnostics, not accuracy.

python benchmark.py data/plays/game_01/pass_001.mp4 --start-frame 120 --end-frame 427
"""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2

from main import ROOT
from src.detection.detection import load_detector
from src.tracking.tracking import PlayerTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--start-frame", type=int, default=120)
    parser.add_argument("--end-frame", type=int, default=427)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/benchmark")
    args = parser.parse_args()
    if not 0 <= args.start_frame <= args.end_frame:
        parser.error("Frame interval must be nonnegative and ordered")
    if args.output.exists():
        parser.error("Use a new output folder to preserve previous comparisons")
    args.output.mkdir(parents=True)
    summaries = []
    variants = [("baseline", "yolo11n.pt", 0.25, 30),
                ("low_conf", "yolo11n.pt", 0.10, 30),
                ("larger", "yolo11s.pt", 0.25, 30),
                ("long_buffer", "yolo11n.pt", 0.25, 60)]
    for name, weights, confidence, buffer in variants:
        print(f"Testing {name}", flush=True)
        model, classes = load_detector(ROOT / "models" / weights)
        tracker = PlayerTracker(model, classes, confidence=confidence, buffer=buffer)
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        rows, seen = [], set()
        elapsed = 0
        try:
            for index in range(args.start_frame, args.end_frame + 1):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {index}")
                started = perf_counter()
                observations = tracker.update(frame)
                elapsed += perf_counter() - started
                ids = {p.track_id for p in observations}
                rows.append([index, len(tracker.raw), len(observations), len(ids - seen), len(tracker.missing)])
                seen |= ids
                if index in {135, 306, 378, 427, args.end_frame}:
                    for kind, items in [("raw", tracker.raw), ("tracked", observations)]:
                        cv2.imwrite(str(args.output / f"{name}_{index}_{kind}.jpg"),
                                    tracker.draw(frame, items))
                    (args.output / f"{name}_{index}.json").write_text(json.dumps({
                        "raw": [dict(box=p.box, confidence=p.confidence) for p in tracker.raw],
                        "tracks": [dict(id=p.track_id, box=p.box) for p in observations]}, default=float))
        finally:
            cap.release()
        with (args.output / f"{name}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["frame", "raw", "tracked", "new_ids", "missing"])
            writer.writerows(rows)
        summaries.append(dict(variant=name, unique_ids=len(seen), frames=len(rows),
                              processing_fps=len(rows) / elapsed,
                              mean_raw=sum(row[1] for row in rows) / len(rows),
                              mean_tracked=sum(row[2] for row in rows) / len(rows)))
        print(summaries[-1], flush=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
