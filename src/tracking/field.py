"""Detect visible sidelines; follow automatic or manually selected boundaries."""
import cv2
import numpy as np


def detect_sidelines(frame):
    """Find long, mostly horizontal turf/non-turf edges on opposite sides.

    This targets the elevated sideline camera, not end-zone views. A candidate
    must have grass on the field side and less grass outside. Yard lines fail
    that test. Partial lines are extended across the image only after pairing.
    """
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (20, 35, 30), (95, 255, 255)) > 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=max(30, width // 20),
                            minLineLength=width * .14, maxLineGap=width * .035)
    candidates = {"far": [], "near": []}
    if lines is None or grass.mean() < .35:
        return {}
    for x1, y1, x2, y2 in lines[:, 0]:
        if abs(x2 - x1) < width * .14:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) > .35:
            continue
        intercept = y1 - slope * x1
        middle = slope * width / 2 + intercept
        side = "far" if middle < height * .40 else "near" if middle > height * .60 else None
        if side is None:
            continue
        xs = np.linspace(0, width - 1, 100).astype(int)
        ys = slope * xs + intercept
        inward = 1 if side == "far" else -1
        inside, outside = [], []
        for distance in (12, 22, 35):
            offset = distance * height / 720
            a = np.round(ys + inward * offset).astype(int)
            b = np.round(ys - inward * offset).astype(int)
            valid = (a >= 0) & (a < height) & (b >= 0) & (b < height)
            if valid.sum() >= 40:
                inside.extend(grass[a[valid], xs[valid]])
                outside.extend(grass[b[valid], xs[valid]])
        if not inside:
            continue
        inner, outer = np.mean(inside), np.mean(outside)
        if inner < .65 or inner - outer < .30:
            continue
        score = inner - outer + .1 * abs(x2 - x1) / width
        candidates[side].append((score, np.array([[0, intercept], [width - 1,
                                  slope * (width - 1) + intercept]], dtype=np.float32)))
    return {side: max(options, key=lambda option: option[0])[1]
            for side, options in candidates.items() if options}


class FieldFilter:
    def __init__(self, automatic=True):
        self.points = None
        self.previous = None
        self.motion = None
        self.automatic = automatic
        self.candidates = {}
        self.pending = None
        self.message = "Auto field: waiting for BOTH sidelines" if automatic else "Field: not selected"

    def select(self, points):
        polygon = np.asarray(points, dtype=np.float32)
        if len(polygon) < 3 or abs(cv2.contourArea(polygon)) < 100:
            raise ValueError("Choose at least 3 points around a non-empty field area.")
        if not cv2.isContourConvex(polygon.astype(np.int32)):
            raise ValueError("Choose a convex boundary in order around the field.")
        self.points = polygon
        self.automatic = False
        self.candidates = {}
        self.message = "Field: following camera (approximate)"

    def clear(self, message="Field: not selected"):
        self.motion = None
        self.points = None
        self.pending = None
        self.message = message

    def enable_auto(self):
        self.clear("Auto field: waiting for BOTH sidelines")
        self.automatic = True

    def detect(self, frame):
        self.candidates = detect_sidelines(frame)
        if len(self.candidates) != 2:
            self.pending = None
            if self.points is None:
                sides = ", ".join(self.candidates) or "none"
                self.message = f"Auto field: visible {sides}; waiting for BOTH sidelines"
            else:
                self.message = "Auto field: following previously detected sidelines"
            return
        far, near = self.candidates["far"], self.candidates["near"]
        if np.min(near[:, 1] - far[:, 1]) < frame.shape[0] * .3:
            self.pending = None
            return
        polygon = np.array([far[0], far[1], near[1], near[0]], dtype=np.float32)
        # Require agreement in two consecutive frames to reject one-frame clutter.
        if self.pending is not None and np.max(np.abs(polygon - self.pending)) < frame.shape[0] * .03:
            self.points = polygon
            self.message = "Auto field: BOTH sidelines detected"
        elif self.points is None:
            self.message = "Auto field: confirming BOTH sidelines on next frame"
        self.pending = polygon

    def contains(self, box):
        if self.points is None:
            return True
        x1, _, x2, y2 = box
        return cv2.pointPolygonTest(self.points, (float((x1 + x2) / 2), float(y2)), False) >= 0

    def advance(self, frame, previous_people=()):
        self._follow(frame, previous_people)
        if self.automatic:
            self.detect(frame)

    def _follow(self, frame, previous_people):
        self.motion = None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous, self.previous = self.previous, gray
        if previous is None:
            return
        mask = np.full_like(previous, 255)
        if self.points is not None:
            mask.fill(0)
            cv2.fillPoly(mask, [self.points.astype(np.int32)], 255)
        for item in previous_people:
            x1, y1, x2, y2 = map(int, item.box)
            cv2.rectangle(mask, (max(0, x1 - 5), max(0, y1 - 5)), (x2 + 5, y2 + 5), 0, -1)
        points = cv2.goodFeaturesToTrack(previous, 250, 0.01, 8, mask=mask)
        if points is None or len(points) < 12:
            if self.points is not None:
                self.clear("Field LOST: press F to redraw; filtering disabled")
            return
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        good = status.ravel() == 1
        if good.sum() < 12:
            if self.points is not None:
                self.clear("Field LOST: press F to redraw; filtering disabled")
            return
        transform, inliers = cv2.estimateAffinePartial2D(
            points[good], moved[good], method=cv2.RANSAC, ransacReprojThreshold=2)
        if transform is None or inliers.sum() < 12 or inliers.mean() < 0.6:
            if self.points is not None:
                self.clear("Field LOST: press F to redraw; filtering disabled")
            return
        scale = np.linalg.norm(transform[0, :2])
        if not 0.85 < scale < 1.18 or np.linalg.norm(transform[:, 2]) > 0.15 * gray.shape[1]:
            if self.points is not None:
                self.clear("Field LOST: camera jump; press F to redraw")
            return
        self.motion = transform
        if self.points is not None:
            self.points = cv2.transform(self.points[None], transform)[0]
