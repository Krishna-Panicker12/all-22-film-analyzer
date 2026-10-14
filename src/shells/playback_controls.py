"""Mouse controls for OpenCV playback; releasing the speed button restores 1x."""
import cv2


class PlaybackControls:
    def __init__(self):
        self.fast = False
        self.skip_requested = False
        self.top = 0

    def mouse(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONUP:
            self.fast = False
        elif event == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_LBUTTON:
            self.fast = False
        elif event == cv2.EVENT_LBUTTONDOWN and self.top <= y < self.top + 52:
            if 12 <= x < 300:
                self.fast = True
            elif 320 <= x < 580:
                self.skip_requested = True

    def draw(self, frame):
        self.top = frame.shape[0]
        canvas = cv2.copyMakeBorder(frame, 0, 52, 0, 0, cv2.BORDER_CONSTANT, value=(20,24,30))
        for left, right, text in [(12,300,'HOLD: 2x speed'),(320,580,'Skip to snap [S]')]:
            cv2.rectangle(canvas,(left,self.top+6),(right,self.top+46),(70,80,90),-1)
            cv2.putText(canvas,text,(left+12,self.top+33),0,.7,(255,255,255),2,cv2.LINE_AA)
        return canvas

    def seek_target(self, key, index, snap):
        requested = self.skip_requested or key in (ord('s'),ord('S'))
        self.skip_requested = False
        return snap if requested and index <= snap else None
