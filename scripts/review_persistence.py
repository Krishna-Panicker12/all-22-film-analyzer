"""Summarize a saved tracking segment and extract frames for visual review."""
import argparse
import csv
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[379, 390])
    args = parser.parse_args()
    with args.csv.open(newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["state"] == "observed"]
    counts = {}
    for row in rows:
        index = int(row["frame_index"])
        counts[index] = counts.get(index, 0) + 1
    summary = dict(observed_frames=len(counts),
                   unique_ids=len({row["track_id"] for row in rows}),
                   mean_visible=sum(counts.values()) / max(1, len(counts)),
                   note="Counts exclude frames with zero observations; unique IDs are not an identity accuracy score.")
    capture = cv2.VideoCapture(str(args.video))
    for index in args.frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index-args.start)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read exported frame {index}")
        destination = args.video.with_name(f"{args.video.stem}_{index}.jpg")
        if not cv2.imwrite(str(destination), frame):
            raise OSError(destination)
    capture.release()
    args.csv.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
