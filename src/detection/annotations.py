"""Manual frame labels, kept separate from model predictions and track IDs."""
import cv2


class FrameLabels:
    def __init__(self, target="player"):
        self.boxes = []
        self.enabled = False
        self.target = target
        self.kind = target
        self.start = None
        self.end = None
        self.complete = False

    def new_frame(self):
        self.boxes = []
        self.start = self.end = None
        self.complete = False
        self.enabled = False

    def mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            x1, x2 = sorted((self.start[0], x))
            y1, y2 = sorted((self.start[1], y))
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                self.boxes.append(dict(kind=self.kind, box=[x1, y1, x2, y2]))
                self.complete = False
            self.start = self.end = None

    def draw(self, image):
        for item in self.boxes:
            x1, y1, x2, y2 = item["box"]
            color = {"player": (0, 255, 0), "helmet": (255, 200, 0), "non_player": (0, 100, 255),
                     "uncertain": (0, 255, 255)}[item["kind"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, "LABEL " + item["kind"], (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
        if self.start is not None and self.end is not None:
            cv2.rectangle(image, self.start, self.end, (255, 255, 255), 1)
        return image
