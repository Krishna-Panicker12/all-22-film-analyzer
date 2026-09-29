"""OpenCV review controls. Rendering never runs model inference."""
import json
from dataclasses import asdict
from datetime import datetime

import cv2
import numpy as np
from src.detection.annotations import FrameLabels
from src.tracking.tracking import on_field


class ReviewControls:
    def __init__(self, tracker, field, label_target="player"):
        self.tracker, self.field = tracker, field
        self.raw = False
        self.trails = True
        self.confidence = False
        self.paused = False
        self.selecting = False
        self.selecting_trails = False
        self.points = []
        self.observations = []
        self.labels = FrameLabels(label_target)
        self.frame_shape = None
        self.header_height = 112
        self.notice = "M: select trail players | T: trails on/off | D: raw/tracked | right-click: exclude | S: save"

    def mouse(self, event, x, y, flags, param):
        y -= self.header_height
        if y < 0:
            return
        if self.frame_shape is not None:
            height, width = self.frame_shape[:2]
            x, y = max(0, min(x, width - 1)), max(0, min(y, height - 1))
        if self.labels.enabled:
            self.labels.mouse(event, x, y)
            return
        if self.selecting_trails and event == cv2.EVENT_LBUTTONDOWN:
            hits = [item for item in self.visible()
                    if item.box[0] <= x <= item.box[2] and item.box[1] <= y <= item.box[3]]
            if hits:
                item = min(hits, key=lambda p: (p.box[2]-p.box[0]) * (p.box[3]-p.box[1]))
                selected = self.tracker.manual_trail_ids
                if item.track_id in selected:
                    selected.remove(item.track_id)
                    self.notice = f"Trail cleared for ID {item.track_id}."
                else:
                    selected.add(item.track_id)
                    self.notice = f"Trail selected for ID {item.track_id}."
        elif self.selecting and event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif self.selecting and event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
        elif event == cv2.EVENT_RBUTTONDOWN and not self.raw:
            hits = [item for item in self.visible()
                    if item.box[0] <= x <= item.box[2] and item.box[1] <= y <= item.box[3]]
            if hits:
                item = min(hits, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
                self.tracker.excluded.add(item.track_id)
                self.notice = f"Excluded ID {item.track_id}; boundary changes preserve exclusions. U restores."
                self.paused = True

    def key(self, key):
        if key == ord('l'):
            self.labels.enabled = not self.labels.enabled
            self.selecting = False
            self.paused = True
            self.notice = f"LABEL: 1 {self.labels.target} / 2 background / 3 uncertain | drag | Backspace undo | V complete | S save"
        elif self.labels.enabled and key in (ord('1'), ord('2'), ord('3')):
            self.labels.kind = {ord('1'): self.labels.target, ord('2'): "non_player", ord('3'): "uncertain"}[key]
            self.notice = f"Label kind: {self.labels.kind}. Drag a box; S saves labels with original frame."
        elif self.labels.enabled and key in (8, 127):
            if self.labels.boxes:
                self.labels.boxes.pop()
                self.labels.complete = False
        elif self.labels.enabled and key == ord('v'):
            self.labels.complete = not self.labels.complete
            self.notice = (f"COMPLETE: every visible {self.labels.target} must be labeled (helmets include sidelines); S saves."
                           if self.labels.complete else "Labels marked partial; excluded from training export.")
        elif key == ord('a'):
            self.field.enable_auto()
            self.selecting = False
            self.notice = "Automatic field enabled. Needs both visible sidelines; detections run as playback advances."
        elif key == ord('f'):
            self.labels.enabled = False
            self.field.automatic = False
            self.field.candidates = {}
            self.selecting = True
            self.points = []
            self.paused = True
            self.notice = "Click field corners in order | right-click undo | Enter apply | F restart"
        elif key in (10, 13) and self.selecting:
            try:
                self.field.select(self.points)
            except ValueError as error:
                self.notice = str(error)
            else:
                self.selecting = False
                self.notice = "Field applied. Space resumes. Check boundary during camera movement."
        elif key == ord('c'):
            self.field.clear()
            self.field.automatic = False
            self.field.candidates = {}
            self.selecting = False
            self.points = []
        elif key == ord('u'):
            self.tracker.excluded.clear()
        elif key == ord('r'):
            self.tracker.reset()
            self.observations = []
            self.field.clear("Field cleared after tracker reset")
            self.notice = "Tracker reset; IDs restart on next frame."
        elif key == ord('d'):
            self.raw = not self.raw
        elif key == ord('m'):
            self.selecting_trails = not self.selecting_trails
            self.paused = True
            self.notice = ("Click a tracked player to toggle its trail; M cancels selection."
                           if self.selecting_trails else "Trail selection canceled.")
        elif key == ord('t'):
            self.trails = not self.trails
            self.notice = f"Trails {'ON' if self.trails else 'OFF'}: smoothed footpoints; short gaps are connected."
        elif key == ord('h'):
            self.confidence = not self.confidence
        elif key == ord(' ') and not self.selecting and not self.labels.enabled:
            self.paused = not self.paused

    def visible(self):
        items=[item for item in self.observations if item.track_id not in self.tracker.excluded
               and on_field(item, self.field)]
        return items

    def visible_missing(self):
        return [identity for identity in self.tracker.missing
                if identity not in self.tracker.excluded]

    def render(self, frame, context, index, fps):
        self.frame_shape = frame.shape
        items = self.tracker.raw if self.raw else self.visible()
        image = (frame.copy() if self.labels.enabled else
                 self.tracker.draw(frame, items, self.trails and not self.raw, self.confidence))
        self.labels.draw(image)
        polygon = self.points if self.selecting else self.field.points
        if polygon is not None and len(polygon):
            cv2.polylines(image, [np.array(polygon, dtype=np.int32)], not self.selecting, (0, 240, 255), 2)
        elif self.field.automatic:
            for line in self.field.candidates.values():
                cv2.line(image, tuple(line[0].astype(int)), tuple(line[1].astype(int)), (255, 200, 0), 1)
        header = np.zeros((self.header_height, image.shape[1], 3), dtype=np.uint8)
        missing = [str(i) for i in self.visible_missing()]
        lines = [context,
                 f"Frame {index} | {index / fps:.2f}s | {self.tracker.target.upper()} {'RAW' if self.raw else 'TRACKED'} {len(items)}"
                 f" | {'PAUSED' if self.paused else 'PLAY'} | {self.field.message}",
                 f"Missing IDs (no measured position): {', '.join(missing[:12]) or 'none'}"
                 + (f" (+{len(missing)-12} more)" if len(missing) > 12 else ""), self.notice]
        for number, line in enumerate(lines):
            cv2.putText(header, line, (10, 22 + number * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (220, 235, 220), 1, cv2.LINE_AA)
        return np.vstack([header, image])

    def snapshot(self, folder, video, frame, context, index, fps):
        folder = folder / f"{video.stem}_{index:06d}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        folder.mkdir(parents=True)
        if not cv2.imwrite(str(folder / "original.png"), frame):
            raise OSError("Could not save original frame")
        if not cv2.imwrite(str(folder / "annotated.png"), self.render(frame, context, index, fps)):
            raise OSError("Could not save annotated frame")
        data = dict(clip=str(video), frame=index, seconds=index / fps, segment=self.tracker.segment,
                    raw=[asdict(p) for p in self.tracker.raw],
                    observed=[asdict(p) for p in self.visible()], missing=self.visible_missing(),
                        trail_ids=sorted(self.tracker.trail_ids_to_draw()),
                        formation_roles={str(identity): role for identity, role
                                         in self.tracker.formation_roles.items()},
                        trails={str(identity): list(self.tracker.history[identity][0])
                            for identity in self.tracker.trail_ids_to_draw()
                            if identity in self.tracker.history},
                    excluded=sorted(self.tracker.excluded),
                    manual_labels=self.labels.boxes, review_complete=self.labels.complete,
                    review_target=self.labels.target, detection_target=self.tracker.target,
                    field_mode="auto" if self.field.automatic else "manual",
                    field=None if self.field.points is None else self.field.points.tolist())
        (folder / "observations.json").write_text(json.dumps(data, indent=2, default=float))
        self.notice = f"Saved frame {index} in {folder.parent}"
        return folder
