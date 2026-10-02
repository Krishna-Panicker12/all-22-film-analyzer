"""Defense-only short-window analysis: python -m src.shells.run_defense --help."""
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from src.detection.detection import load_detector
from src.tracking.field import FieldFilter
from src.tracking.tracking import PlayerTracker
from .defense import (DefensiveDetector, reference_points, shell_from_snapshot,
                     compare_snapshots, coverage_hypotheses,
                     classify_corner_movement, zone_bubble_polygons,
                     load_coverage_reference, load_coverage_playbook,
                     classify_two_high_zone_structure)
from .following import receiver_following
from .display import banner, display_estimate, wait_for_manual_exit
from src.data.catalog import resolve_clip

ROOT = Path(__file__).resolve().parents[2]
COVERAGE_REFERENCE = ROOT / "data/metadata/cover_1_man_estimator.json"
COVERAGE_PLAYBOOK = ROOT / "data/metadata/coverage_reference_playbook.json"


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("start_frame", "pre_frame", "snap_frame"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not config["start_frame"] < config["pre_frame"] < config["snap_frame"]:
        raise ValueError("Require start_frame < pre_frame < snap_frame")
    if not 1 <= config.get("post_seconds", 2.3) <= 5:
        raise ValueError("Post snapshot must be 1–5 seconds after snap")
    if config["defense_color"] not in ("red", "white"):
        raise ValueError("Review a red/white uniform mapping first")
    geometry = config.get("geometry_at_pre", {})
    scales = geometry.get("pixels_per_yard")
    if not isinstance(scales, dict) or not all(
            isinstance(scales.get(axis), (int, float)) and
            math.isfinite(scales[axis]) and scales[axis] > 0
            for axis in ("depth", "lateral")):
        raise ValueError("geometry_at_pre.pixels_per_yard needs positive depth and lateral scales.")
    bubble_radius = config.get("corner_bubble_yards", 6.0)
    if not isinstance(bubble_radius, (int, float)) or not 5 <= bubble_radius <= 7:
        raise ValueError("corner_bubble_yards must be between 5 and 7.")
    return config


def observation_rows(observations, transform):
    rows = []
    for item in observations:
        body = item.body_box if item.body_box is not None else (item.box if item.source == "player" else None)
        box = body if body is not None else item.box
        anchor = [(box[0]+box[2])/2, box[3] if body is not None else (box[1]+box[3])/2]
        rows.append(dict(track_id=item.track_id, source=item.source, confidence=float(item.confidence),
                         box=[float(v) for v in item.box], anchor_kind="feet" if body is not None else "helmet_center",
                         image_point=[float(v) for v in anchor],
                         reference_point=reference_points([anchor], transform)[0].tolist() if transform is not None else None))
    return rows


def overlay(frame, rows, caption, bubbles=()):
    canvas = frame.copy()
    for identity, polygon in bubbles:
        cv2.polylines(canvas, [polygon], True, (50, 240, 110), 2, cv2.LINE_AA)
        center = tuple(np.mean(polygon, axis=0).astype(int))
        cv2.putText(canvas, f"D{identity} zone radius", center, 0, .4,
                    (50, 240, 110), 1, cv2.LINE_AA)
    for row in rows:
        x1, y1, x2, y2 = map(round, row["box"])
        cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,225,255), 2)
        cv2.putText(canvas, f'D{row["track_id"]} {row["source"]}', (x1,max(45,y1-6)), 0,.45,(0,225,255),1)
    cv2.rectangle(canvas, (0,0), (canvas.shape[1],35), (15,20,28), -1)
    cv2.putText(canvas, caption, (10,24), 0,.55,(255,255,255),1)
    return canvas


def run(config, output, target="hybrid", image_size=1280, tracker_name="botsort.yaml", buffer=30,
    show=False):
    video = resolve_clip(config['video'])
    cap = cv2.VideoCapture(str(video))
    writer = None
    window = f'Defense tracking | {config["play_id"]}'
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Reliable FPS required")
        pre, snap = config["pre_frame"], config["snap_frame"]
        post = snap + round(config.get("post_seconds",2.3)*fps)
        if post >= frame_count:
            raise ValueError("Post snapshot extends beyond clip")
        if output.exists():
            raise ValueError("Choose a new output directory; existing results are preserved")
        output.mkdir(parents=True)
        (output/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
        coverage_reference = load_coverage_reference(COVERAGE_REFERENCE)
        coverage_playbook = load_coverage_playbook(COVERAGE_PLAYBOOK)
        if show:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, min(1280, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
                             min(940, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))+220))
        weights = ROOT / ("models/football_helmet_pilot.pt" if target == "hybrid" else "models/yolo11n.pt")
        detector_target = "helmet" if target == "hybrid" else "player"
        model, classes = load_detector(weights, detector_target)
        primary = DefensiveDetector(model, config["defense_color"], detector_target)
        body, body_classes = None, None
        if target == "hybrid":
            body_model, body_classes = load_detector(ROOT/"models/yolo11n.pt")
            body = DefensiveDetector(body_model, config["defense_color"])
        tracker = PlayerTracker(primary, classes, image_size=image_size, tracker=tracker_name,
                                buffer=buffer, target=detector_target, body_model=body,
                                body_classes=body_classes, helmet_match_scale=2 if target == "hybrid" else 1)
        receiver_detector, receiver_tracker = None, None
        if config.get("receiver_following_enabled", True):
            receiver_model, receiver_classes = load_detector(ROOT/"models/yolo11n.pt")
            offense_color = "white" if config["defense_color"] == "red" else "red"
            receiver_detector = DefensiveDetector(receiver_model, offense_color)
            receiver_tracker = PlayerTracker(receiver_detector, receiver_classes,
                image_size=image_size, tracker=tracker_name, buffer=buffer, target="player")
        field = FieldFilter(automatic=False)
        field.select(config["field_polygon_at_start"])
        cumulative = np.eye(3)
        pre_camera = None
        motion_ok = True
        captures = {}
        corner_motion = None
        coverage = None
        current_shell = None
        live_coverage = None
        final_action = None
        track_frames = []
        detection_totals = {"primary": {}, "body": {}}
        previous_people = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, config["start_frame"])
        for index in range(config["start_frame"], post+1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Decode failed at frame {index}")
            field.advance(frame, previous_people)
            if index > config["start_frame"]:
                if field.motion is None:
                    motion_ok = False
                elif motion_ok:
                    cumulative = np.vstack([field.motion, [0,0,1]]) @ cumulative
            primary.polygon = field.points
            if body is not None:
                body.polygon = field.points
            observations = tracker.update(frame, field)
            receiver_observations = []
            if receiver_tracker is not None:
                receiver_detector.polygon = field.points
                receiver_observations = receiver_tracker.update(frame, field)
            previous_people = body.people if body is not None else primary.people
            frame_counts = {"primary": primary.counts.copy(),
                            "body": body.counts.copy() if body is not None else None}
            for source, counts in frame_counts.items():
                if counts is not None:
                    for key, value in counts.items():
                        detection_totals[source][key] = detection_totals[source].get(key, 0) + value
            if index == pre:
                pre_camera = cumulative.copy() if motion_ok else None
            transform = pre_camera @ np.linalg.inv(cumulative) if pre_camera is not None and motion_ok else None
            rows = observation_rows(observations, transform)
            receiver_rows = observation_rows(receiver_observations, transform)
            track_frames.append(dict(frame_index=index,time_seconds=index/fps,defenders=rows,
                                     receivers=receiver_rows,
                                     missing_ids=list(tracker.missing),registration_ok=motion_ok,
                                     detection_counts=frame_counts))
            caption = f'{config["defense"]} DEFENSE | {target} | frame {index} | snap {snap} | observed {len(rows)}/11'
            phase = None
            shell = None
            bubbles = []
            if index in (pre,post):
                phase = "pre_snap" if index == pre else "post_snap"
                shell = shell_from_snapshot(rows, config["geometry_at_pre"],
                                            config.get("secondary_visible",False), transform is not None)
                captures[phase] = dict(frame_index=index,time_seconds=index/fps,
                                       defenders=rows,shell=shell)
                if phase == "pre_snap":
                    caption += (f' | PRE shell candidate {shell["candidate"]}; '
                                f'confirmed {shell["shell"]}; '
                                f'high safeties {shell["high_safety_count"]}')
                else:
                    geometry = config["geometry_at_pre"]
                    corner_motion = classify_corner_movement(
                        captures["pre_snap"]["defenders"], rows, geometry,
                        geometry["pixels_per_yard"], config.get("corner_bubble_yards",6.0),
                        config.get("corner_ids"))
                    coverage = coverage_hypotheses(
                        shell["candidate"], corner_motion, shell["high_safety_count"],
                        captures["pre_snap"]["shell"]["candidate"], coverage_reference,
                        coverage_playbook,
                        classify_two_high_zone_structure(rows,geometry),
                        shell["completeness_confirmed"])
                    caption += (f' | POST shell candidate {shell["candidate"]}; '
                                f'confirmed {shell["shell"]}; '
                                f'high safeties {shell["high_safety_count"]}; '
                                f'corners {corner_motion["classification"]}; '
                                f'coverage {coverage["coverage"]}')
                    if transform is not None:
                        bubbles = zone_bubble_polygons(
                            rows, corner_motion, geometry, geometry["pixels_per_yard"],
                            np.linalg.inv(transform))
            annotated = overlay(frame, rows, caption, bubbles)
            for row in receiver_rows:
                x1,y1,x2,y2 = map(round,row["box"])
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,210,0),1)
                cv2.putText(annotated,f'O{row["track_id"]}',(x1,max(48,y1-5)),0,.4,(255,210,0),1)
            if index >= pre and (index == pre or index == post or index % 6 == 0):
                current_shell = shell or shell_from_snapshot(rows,config['geometry_at_pre'],
                    config.get('secondary_visible',False),transform is not None)
                if index > snap:
                    live_motion = classify_corner_movement(captures['pre_snap']['defenders'],rows,
                        config['geometry_at_pre'],config['geometry_at_pre']['pixels_per_yard'],
                        config.get('corner_bubble_yards',6),config.get('corner_ids'))
                    live_coverage = coverage_hypotheses(current_shell['candidate'],live_motion,
                        current_shell['high_safety_count'],captures['pre_snap']['shell']['candidate'],
                        coverage_reference,coverage_playbook,
                        classify_two_high_zone_structure(rows,config['geometry_at_pre']),
                        current_shell['completeness_confirmed'])
            # Film review is an explicitly labeled fallback, never inference input.
            # Don't reveal a saved post-snap review before its evidence frames.
            review = dict(config.get('review_context',{}))
            evidence_frame = max(review.get('coverage_evidence_frames',[post]))
            if index < evidence_frame and index != post:
                review.pop('coverage',None)
            displayed = display_estimate(current_shell,coverage or live_coverage,review,index>snap)
            annotated = banner(annotated,displayed,config['play_id'],(index-snap)/fps,index==post)
            if writer is None:
                writer = cv2.VideoWriter(str(output/"defense_tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (annotated.shape[1],annotated.shape[0]))
                if not writer.isOpened():
                    raise ValueError("Cannot open video writer")
            writer.write(annotated)
            if phase is not None:
                cv2.imwrite(str(output/f"{phase}_original.png"),frame)
                cv2.imwrite(str(output/f"{phase}_annotated.png"),annotated)
            if index % 30 == 0:
                print(f'Frame {index}/{post}: {len(rows)} defensive observations',flush=True)
            if show:
                cv2.imshow(window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    while True:
                        key = cv2.waitKey(100) & 0xFF
                        if key in (ord(' '),ord('q'),ord('Q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1: break
                if key in (27, ord('q'), ord('Q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    quit_batch = key == 27 or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1
                    if index == post:
                        final_action = 'quit' if quit_batch else 'next'
                        continue
                    (output/"run_status.json").write_text(json.dumps(
                        dict(status="stopped", last_frame=index, processed_frames=len(track_frames)),
                        indent=2), encoding="utf-8")
                    with (output/"tracks.jsonl").open("w",encoding="utf-8") as file:
                        for record in track_frames:
                            file.write(json.dumps(record,allow_nan=False)+"\n")
                    print(f"Stopped at frame {index}; partial video and tracks are in {output}", flush=True)
                    return dict(status="stopped", output=str(output), last_frame=index,playback_action='quit' if quit_batch else 'next')
        movement = compare_snapshots(captures["pre_snap"]["defenders"], captures["post_snap"]["defenders"])
        post_shell = captures["post_snap"]["shell"]
        from src.shells.coverage_stages import staged_coverage
        following = receiver_following(track_frames, snap, fps, config["geometry_at_pre"],
                                       corner_motion["selected_corner_ids"])
        stages = staged_coverage(captures["pre_snap"]["shell"], post_shell,
                                 corner_motion, coverage_playbook, following)
        candidate_coverage = coverage["candidates"]
        report = dict(play_id=config["play_id"],defense=config["defense"],target=target,
                      snap_frame=snap,snap_time_seconds=snap/fps,snap_source=config.get("snap_source","manual"),
                      post_delay_seconds=(post-snap)/fps,processed_frames=len(track_frames),
                      snapshots=captures,movement=movement,coverage=coverage,
                      corner_motion=corner_motion, coverage_stages=stages,
                      receiver_following=following,
                      display_estimate=displayed,
                      coverage_reference_source=coverage_playbook["source"],
                      detection_filter_totals=detection_totals,
                      provisional_shell_compatible_coverages=candidate_coverage,
                      coordinate_system="Approximate camera-compensated pre-snap pixels; not yards",
                      limitations=["Red/white uniform heuristic can miss or misclassify detections.",
                                   "Same track ID is an association estimate, not a verified player identity.",
                                   "Affine camera compensation and manually chosen image geometry need review.",
                                   "Two defense-only snapshots cannot establish receiver assignments or man versus zone."])
        (output/"result.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
        with (output/"tracks.jsonl").open("w",encoding="utf-8") as file:
            for record in track_frames:
                file.write(json.dumps(record,allow_nan=False)+"\n")
        with (output/"movement.csv").open("w",newline="",encoding="utf-8") as file:
            columns=["track_id","state","delta_reference_pixels","distance_reference_pixels"]
            w=csv.DictWriter(file,fieldnames=columns);w.writeheader();w.writerows(movement)
        comparison = np.hstack([cv2.imread(str(output/"pre_snap_annotated.png")),cv2.imread(str(output/"post_snap_annotated.png"))])
        cv2.imwrite(str(output/"comparison.png"),comparison)
        print(json.dumps(dict(output=str(output),
                      pre_shell=captures["pre_snap"]["shell"]["shell"],
                      pre_shell_candidate=captures["pre_snap"]["shell"]["candidate"],
                      pre_high_safeties=captures["pre_snap"]["shell"]["high_safety_count"],
                      post_shell=post_shell["shell"],
                      post_shell_candidate=post_shell["candidate"],
                      post_high_safeties=post_shell["high_safety_count"],
                      corner_motion=corner_motion["classification"],
                      coverage=coverage["coverage"],
                              matched=sum(r["state"]=="matched" for r in movement)),indent=2))
        # Flush outputs before holding the final frame indefinitely for review.
        if writer is not None:
            writer.release()
            writer = None
        if show:
            if final_action is None:
                cv2.imshow(window,annotated)
                final_action = wait_for_manual_exit(window)
            report['playback_action'] = final_action
        return report
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass


def run_batch(config_dir, output_root, target="hybrid", image_size=1280,
              tracker_name="botsort.yaml", buffer=30, show=False):
    """Run every reviewed defense_*.json config in sorted order."""
    configs = sorted(Path(config_dir).glob("defense_*.json"))
    if not configs:
        raise ValueError(f"No defense_*.json configs found in {config_dir}")
    reports = []
    for path in configs:
        config = read_config(path)
        output = Path(output_root) / config["play_id"]
        if output.exists():
            run_number = 2
            while output.with_name(f"{config['play_id']}_run{run_number:03d}").exists():
                run_number += 1
            output = output.with_name(f"{config['play_id']}_run{run_number:03d}")
        print(f"Starting {config['play_id']} from {path}", flush=True)
        reports.append(run(config, output, target, image_size, tracker_name, buffer, show))
        if reports[-1].get('playback_action') == 'quit':
            break
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--config-dir",type=Path,
                        help="Batch-run defense_*.json configs from this directory")
    parser.add_argument("--output-root",type=Path,
                        help="Parent directory for per-play batch output folders")
    parser.add_argument("--show",action="store_true",
                        help="Show live tracking and hold final frame; Space pauses, Q advances, Esc quits batch")
    parser.add_argument("--target",choices=["player","hybrid"],default="hybrid")
    parser.add_argument("--imgsz",type=int,default=1280)
    parser.add_argument("--tracker",choices=["botsort.yaml","bytetrack.yaml"],default="botsort.yaml")
    parser.add_argument("--buffer",type=int,default=30)
    args=parser.parse_args()
    if args.imgsz <= 0 or args.buffer <= 0:
        parser.error("imgsz and buffer must be positive")
    try:
        if args.config_dir is not None:
            if args.config is not None or args.output is not None or args.output_root is None:
                parser.error("Batch mode requires --config-dir and --output-root only")
            run_batch(args.config_dir,args.output_root,args.target,args.imgsz,
                      args.tracker,args.buffer,args.show)
        else:
            if args.config is None or args.output is None or args.output_root is not None:
                parser.error("Single-play mode requires --config and --output")
            run(read_config(args.config),args.output,args.target,args.imgsz,
                args.tracker,args.buffer,args.show)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1,f"Error: {error}\n")


if __name__ == "__main__":
    main()
