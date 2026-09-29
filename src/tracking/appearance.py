"""Conservative appearance evidence for reconnecting short helmet track gaps."""
import cv2
import numpy as np

from src.tracking.teams import helmet_color, helmet_jersey_color


def helmet_appearance(frame, box, observations):
    """Require agreeing helmet/jersey color; abstain for overlapping players.

    This is a color histogram, not a trained player re-identification network.
    It is supporting evidence only; teammates can look almost identical.
    """
    others = [item.box for item in observations if item.box != box]
    shell = helmet_color(frame, box)
    jersey = helmet_jersey_color(frame, box, others)
    if shell == "unknown" or shell != jersey:
        return None
    x1, y1, x2, y2 = box
    width, height = x2-x1, y2-y1
    left, right = max(0, int(x1-.4*width)), min(frame.shape[1], int(x2+.4*width))
    top, bottom = max(0, int(y1)), min(frame.shape[0], int(y2+2*height))
    if any(left <= (a+c)/2 <= right and top <= (b+d)/2 <= bottom for a,b,c,d in others):
        return None
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    histogram = cv2.normalize(histogram, None, norm_type=cv2.NORM_L1).astype(np.float32)
    return shell, histogram
