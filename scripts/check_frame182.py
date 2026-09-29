"""Visual regression check on game_01/pass_001; not held-out evaluation."""
import argparse
import json
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.detection.detection import load_detector
from src.tracking.tracking import PlayerTracker, limit_roster


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/frame182_check")
    parser.add_argument("--conf", type=float, default=.25)
    parser.add_argument("--frame", type=int, default=182)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    body,body_classes = load_detector(ROOT/"models/yolo11n.pt")
    model,classes = load_detector(args.model,"helmet") if args.model else (body,body_classes)
    tracker = PlayerTracker(model,classes,confidence=args.conf,target="helmet" if args.model else "player",
                            body_model=body if args.model else None,body_classes=body_classes,
                            team_colors="red-white")
    tracker.team_names = {"red":"KC", "white":"NE"}
    cap = cv2.VideoCapture(str(ROOT/"data/plays/game_01/pass_001.mp4"))
    start = max(0,args.frame-12)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    for index in range(start,args.frame+1):
        ok,frame = cap.read()
        if not ok:
            raise ValueError("Missing frame")
        observations = tracker.update(frame)
    cap.release()
    observations=limit_roster(observations)
    cv2.imwrite(str(args.output/"original.png"),frame)
    cv2.imwrite(str(args.output/"tracked.jpg"),tracker.draw(frame,observations,confidence=True))
    boxes_image = frame.copy()
    for item in observations:
        x1,y1,x2,y2 = map(round,item.box)
        cv2.rectangle(boxes_image,(x1,y1),(x2,y2),(0,255,0),1)
    cv2.imwrite(str(args.output/"boxes.jpg"),boxes_image)
    from dataclasses import asdict
    report = dict(frame=args.frame, model=str(args.model), count=len(observations),
                  observed=[asdict(item) for item in observations])
    (args.output/"observations.json").write_text(json.dumps(report,indent=2,default=float))
    print(f"Frame {args.frame}: {len(observations)} tracked observations; {args.output}")


if __name__ == "__main__":
    main()
