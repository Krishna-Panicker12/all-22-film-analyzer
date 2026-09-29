"""Label helmets in selected video frames without loading a detector.

Run: python label_helmets.py
Z zooms around the mouse; N/P saves and changes frame; Q saves and exits.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.detection.annotations import FrameLabels

ROOT = Path(__file__).resolve().parents[1]
WINDOW = "All-22 | Helmet labels"
DEFAULT_FRAMES = [141, 148, 355, 378, 408, 414, 420, 426]


def save_frame(folder, video, frame, index, fps, labels):
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "original.png").exists():
        if not cv2.imwrite(str(folder / "original.png"), frame):
            raise OSError("Could not save the original frame")
    data = dict(clip=str(video.resolve()), frame=index, seconds=index / fps,
                review_target="helmet", review_complete=labels.complete,
                manual_labels=labels.boxes)
    # Preserve the last complete file if an interrupted write fails.
    temporary = folder / "observations.tmp"
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(folder / "observations.json")


class HelmetLabeler:
    header = 112

    def __init__(self, frame, labels):
        self.frame, self.labels = frame, labels
        self.height, self.width = frame.shape[:2]
        self.zoom = 1
        self.origin = (0, 0)
        self.pointer = (self.width // 2, self.height // 2)

    def original_point(self, x, y):
        left, top = self.origin
        return (int(np.clip(left + x / self.zoom, 0, self.width - 1)),
                int(np.clip(top + y / self.zoom, 0, self.height - 1)))

    def mouse(self, event, x, y, flags, param):
        if y < self.header:
            return
        point = self.original_point(x, y - self.header)
        self.pointer = point
        if event == cv2.EVENT_RBUTTONDOWN:
            # Remove the smallest enclosing label when crowded boxes overlap.
            candidates = [(i, box) for i, box in enumerate(self.labels.boxes)
                          if box["box"][0] <= point[0] <= box["box"][2]
                          and box["box"][1] <= point[1] <= box["box"][3]]
            if candidates:
                index, _ = min(candidates, key=lambda item:
                               (item[1]["box"][2]-item[1]["box"][0]) *
                               (item[1]["box"][3]-item[1]["box"][1]))
                self.labels.boxes.pop(index)
                self.labels.complete = False
            return
        self.labels.mouse(event, *point)

    def toggle_zoom(self):
        if self.labels.start is not None:
            return
        if self.zoom > 1:
            self.zoom, self.origin = 1, (0, 0)
        else:
            self.zoom = 3
            x, y = self.pointer
            left = int(np.clip(x - self.width / 6, 0, self.width - self.width // 3))
            top = int(np.clip(y - self.height / 6, 0, self.height - self.height // 3))
            self.origin = (left, top)

    def render(self, index, position, total):
        image = self.labels.draw(self.frame.copy())
        left, top = self.origin
        crop = image[top:top + self.height // self.zoom, left:left + self.width // self.zoom]
        image = cv2.resize(crop, (self.width, self.height))
        header = np.zeros((self.header, self.width, 3), dtype=np.uint8)
        lines = [f"Frame {index} | {position + 1}/{total} | {self.zoom}x | "
                 f"{len(self.labels.boxes)} labels | {'COMPLETE' if self.labels.complete else 'PARTIAL'} | {self.labels.kind}",
                 "Drag a tight box around each visible worn football helmet. Include sideline helmets.",
                 "Z zoom | 1 helmet / 2 background / 3 uncertain | Backspace undo | Right-click delete",
                 "S save | N/P save + next/previous | V complete (ALL helmets labeled) | Q save + quit"]
        for i, line in enumerate(lines):
            cv2.putText(header, line, (10, 22 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        .48, (230, 240, 230), 1, cv2.LINE_AA)
        return np.vstack([header, image])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=ROOT / "data/plays/game_01/pass_001.mp4")
    parser.add_argument("--frames", type=int, nargs="+", default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path, help="Folder for this clip's manual labels")
    parser.add_argument("--prepare-only", action="store_true", help="Extract clean frames without opening a window")
    args = parser.parse_args()
    if not args.video.is_file() or any(i < 0 for i in args.frames):
        parser.error("Provide an existing video and nonnegative frame numbers")
    destination = args.output or ROOT / "data/annotations" / args.video.parent.name / args.video.stem
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise ValueError(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Valid video FPS is required for frame labeling")
    position = 0
    try:
        if not args.prepare_only:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, 1280, 832)
        while position < len(args.frames):
            index = args.frames[position]
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise ValueError(f"Could not seek to frame {index}")
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Frame {index} is outside the clip or cannot be decoded")
            folder = destination / f"{index:06d}"
            labels = FrameLabels("helmet")
            previous = folder / "observations.json"
            if previous.exists():
                data = json.loads(previous.read_text(encoding="utf-8"))
                if (Path(data["clip"]).resolve() != args.video.resolve() or data["frame"] != index
                        or data.get("review_target") != "helmet"):
                    raise ValueError(f"Existing labels belong to a different source: {folder}")
                labels.boxes = data["manual_labels"]
                labels.complete = data["review_complete"]
            if args.prepare_only:
                save_frame(folder, args.video, frame, index, fps, labels)
                position += 1
                continue
            view = HelmetLabeler(frame, labels)
            cv2.setMouseCallback(WINDOW, view.mouse)
            while True:
                cv2.imshow(WINDOW, view.render(index, position, len(args.frames)))
                key = cv2.waitKey(30) & 0xff
                closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
                if closed or key in (27, ord('q'), ord('n'), ord('p'), ord('s')):
                    save_frame(folder, args.video, frame, index, fps, labels)
                    if closed or key in (27, ord('q')):
                        return
                    if key in (ord('n'), ord('p')):
                        position = max(0, position + (1 if key == ord('n') else -1))
                        break
                elif key == ord('z'):
                    view.toggle_zoom()
                elif key in (ord('1'), ord('2'), ord('3')):
                    labels.kind = {ord('1'): "helmet", ord('2'): "non_player", ord('3'): "uncertain"}[key]
                elif key in (8, 127) and labels.boxes:
                    labels.boxes.pop()
                    labels.complete = False
                elif key == ord('v'):
                    labels.complete = not labels.complete
        print(f"Prepared/saved {len(args.frames)} frames in {destination}")
    finally:
        cap.release()
        if not args.prepare_only:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
