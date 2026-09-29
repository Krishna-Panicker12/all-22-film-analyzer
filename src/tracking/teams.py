"""Optional red/white jersey evidence, independent of player identity tracking.

This is a color baseline, not a trained team classifier. It never creates IDs.
Only enable it for a game whose two teams actually wear red and white jerseys.
"""
from collections import deque

import cv2
import numpy as np


def patch_color(frame, region):
    left,top,right,bottom = region
    height, width = frame.shape[:2]
    crop = frame[max(0,round(top)):min(height,round(bottom)),
                 max(0,round(left)):min(width,round(right))]
    if crop.size == 0 or min(crop.shape[:2]) < 3:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = np.mean(((hue <= 12) | (hue >= 165)) & (saturation >= 85) & (value >= 65))
    white = np.mean((saturation <= 85) & (value >= 145))
    if red >= .22 and red > white:
        return "red"
    if white >= .4 and red < .12:
        return "white"
    return "unknown"


def helmet_color(frame, helmet):
    """Classify the helmet shell for the configured red-versus-light matchup."""
    x1,y1,x2,y2 = helmet
    w,h = x2-x1,y2-y1
    # Avoid the lower facemask/face area and the box border.
    region = (x1+.12*w, y1+.08*h, x2-.12*w, y1+.72*h)
    left,top,right,bottom = region
    height,width = frame.shape[:2]
    crop = frame[max(0,round(top)):min(height,round(bottom)),
                 max(0,round(left)):min(width,round(right))]
    if crop.size == 0 or min(crop.shape[:2]) < 3:
        return "unknown"
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    hue,saturation,value = cv2.split(hsv)
    red = np.mean(((hue <= 12) | (hue >= 165)) & (saturation >= 70) & (value >= 55))
    light = np.mean((saturation <= 90) & (value >= 115))
    if red >= .18 and red > light*.55:
        return "red"
    if light >= .30 and red < .14:
        return "white"
    return "unknown"


def helmet_jersey_color(frame, helmet, other_helmets=()):
    """Estimate a small upper-jersey patch below a helmet, without requiring a body.

    This is image-space appearance evidence, not a body or ground-position estimate.
    Abstain if another helmet lies inside the patch (contact/overlap ambiguity).
    """
    x1,y1,x2,y2 = helmet
    w,h = x2-x1,y2-y1
    # Widen and deepen the patch for crouched linemen. Start below the facemask
    # and avoid the pants area, which is often the opposite team's jersey color.
    region = (x1-.65*w, y2+.15*h, x2+.65*w, y2+2.5*h)
    left,top,right,bottom = region
    for a,b,c,d in other_helmets:
        if left <= (a+c)/2 <= right and top <= (b+d)/2 <= bottom:
            return "unknown"
    return patch_color(frame, region)


def jersey_color(frame, box, other_boxes=()):
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    # Tall, upright body only: a tackle/merged body is too ambiguous.
    if w <= 0 or h < 1.2*w:
        return "unknown"
    left, right = x1+.2*w, x2-.2*w
    top, bottom = y1+.23*h, y1+.52*h
    area = (right-left)*(bottom-top)
    for a,b,c,d in other_boxes:
        overlap = max(0,min(right,c)-max(left,a))*max(0,min(bottom,d)-max(top,b))
        if overlap > .2*area:
            return "unknown"
    return patch_color(frame, (left,top,right,bottom))


class JerseyTeams:
    def __init__(self):
        self.votes = {}
        self.assignments = {}

    def update(self, frame, observations, frame_index):
        bodies = [item.body_box for item in observations if item.body_box is not None]
        self.votes = {identity: votes for identity, votes in self.votes.items()
                      if votes and frame_index-votes[-1][0] <= 12}
        for item in observations:
            if item.track_id is None:
                continue
            votes = self.votes.setdefault(item.track_id, deque(maxlen=12))
            while votes and frame_index-votes[0][0] > 12:
                votes.popleft()
            color = "unknown"
            if item.body_box is not None:
                others = [box for box in bodies if box is not item.body_box]
                color = jersey_color(frame, item.body_box, others)
            if item.source == "helmet":
                others = [p.box for p in observations if p is not item and p.source == "helmet"]
                upper = helmet_jersey_color(frame, item.box, others)
                shell = helmet_color(frame,item.box)
                evidence = [value for value in (shell,upper,color) if value != "unknown"]
                # Helmet shell leads for this matchup. The jersey sources break a
                # tie only when the shell is uncertain.
                color = shell if shell != "unknown" else (
                    max(set(evidence),key=evidence.count) if evidence else "unknown")
            votes.append((frame_index, color))
            known = [v for _,v in votes if v != "unknown"]
            counts = {name: known.count(name) for name in ("red", "white")}
            winner = max(counts, key=counts.get)
            # Two agreeing samples establish a team. Unknown frames retain that
            # assignment; a change requires a clear majority of recent evidence.
            if counts[winner] >= 2 and counts[winner] / max(1, len(known)) >= .67:
                self.assignments[item.track_id] = winner
            item.team = self.assignments.get(item.track_id, "unknown")

    def reset(self):
        self.votes.clear()
        self.assignments.clear()
