"""Evaluate helmet detection, tracker duplicates, and team assignment on reviewed frames."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2

from tools.evaluate_detector import match_boxes
from src.detection.detection import load_detector
from src.tracking.tracking import PlayerTracker, limit_roster

ROOT = Path(__file__).resolve().parents[1]


def rates(tp,fp,fn):
    return dict(tp=tp,fp=fp,fn=fn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--reviews",type=Path,default=ROOT/"data/evaluation")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--conf",type=float,default=.25)
    parser.add_argument("--imgsz",type=int,default=1280)
    parser.add_argument("--helmet-match-scale",type=float,default=1.0)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output folder")
    files=sorted(args.reviews.rglob("observations.json"))
    files=[p for p in files if json.loads(p.read_text()).get("review_complete")]
    if not files:
        parser.error("No completed evaluation labels")
    model,classes=load_detector(args.model,"helmet")
    body,body_classes=load_detector(ROOT/"models/yolo11n.pt")
    totals={"raw":[0,0,0],"tracked":[0,0,0]}
    team_total=team_known=team_correct=duplicates=tentative=0
    records=[]
    args.output.mkdir(parents=True)
    for file in files:
        data=json.loads(file.read_text())
        image=cv2.imread(str(file.parent/"original.png"))
        truth=[label["box"] for label in data["manual_labels"] if label["kind"]=="helmet"]
        truth_teams=[label.get("team","unknown") for label in data["manual_labels"] if label["kind"]=="helmet"]
        raw_result=model.predict(image,classes=classes,conf=args.conf,imgsz=args.imgsz,verbose=False)[0]
        raw=[[*box,float(score)] for box,score in zip(raw_result.boxes.xyxy.cpu().tolist(),
                                                      raw_result.boxes.conf.cpu().tolist())]
        raw_pairs=match_boxes(truth,raw,.5)
        raw_counts=(len(raw_pairs),len(raw)-len(raw_pairs),len(truth)-len(raw_pairs))
        for i,value in enumerate(raw_counts): totals["raw"][i]+=value

        video=Path(data["clip"])
        tracker=PlayerTracker(model,classes,args.conf,args.imgsz,target="helmet",
                              body_model=body,body_classes=body_classes,team_colors="red-white",
                              helmet_match_scale=args.helmet_match_scale)
        cap=cv2.VideoCapture(str(video))
        start=max(0,data["frame"]-12)
        cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        observations=[]
        for _ in range(start,data["frame"]+1):
            ok,frame=cap.read()
            if not ok: raise ValueError(f"Could not read {video}")
            observations=tracker.update(frame)
            duplicates+=tracker.duplicates_suppressed
            tentative+=tracker.tentative_suppressed
        cap.release()
        observations=limit_roster(observations)
        predicted=[[*item.box,item.confidence] for item in observations]
        pairs=match_boxes(truth,predicted,.5)
        tracked_counts=(len(pairs),len(predicted)-len(pairs),len(truth)-len(pairs))
        for i,value in enumerate(tracked_counts): totals["tracked"][i]+=value
        for prediction,actual in pairs:
            team_total+=1
            value={"red":"KC","white":"NE"}.get(observations[prediction].team,"unknown")
            if value != "unknown":
                team_known+=1
                team_correct+=value==truth_teams[actual]
        records.append(dict(frame=data["frame"],clip=str(video),truth=len(truth),
                            raw=dict(zip(("tp","fp","fn"),raw_counts)),
                            tracked=dict(zip(("tp","fp","fn"),tracked_counts)),
                            observations=[asdict(item) for item in observations]))
    summary=dict(model=str(args.model.resolve()),helmet_match_scale=args.helmet_match_scale,
                 raw_confidence=args.conf,
                 frames=len(files),objects=sum(r["truth"] for r in records),
                 raw_detection=rates(*totals["raw"]),tracked_output=rates(*totals["tracked"]),
                 duplicate_tracks_suppressed=duplicates,
                 tentative_tracks_suppressed=tentative,
                 team=dict(matched=team_total,known=team_known,correct=team_correct,
                           coverage=team_known/team_total if team_total else None,
                           accuracy_when_known=team_correct/team_known if team_known else None,
                           overall_accuracy=team_correct/team_total if team_total else None),
                 limitations="Same game and one held-out frame; no identity ground truth, so ID-switch accuracy is not measured.",
                 records=records)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,default=float))
    print(json.dumps({k:v for k,v in summary.items() if k!="records"},indent=2))


if __name__=="__main__":
    main()
