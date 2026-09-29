"""All-22 tracking viewer. Run with --help for clip and export options."""

import argparse
import csv
import math
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
WINDOW = "All-22 | Player tracking"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path,
                        help="Clip to open; otherwise a file picker opens.")
    parser.add_argument("--model", type=Path, help="Detector weights; helmet mode requires a trained helmet model")
    parser.add_argument("--target", choices=["auto", "player", "helmet"], default="player")
    parser.add_argument("--player-scope", choices=["manual", "formation", "eligible", "all"],
                        default="manual",
                        help="Show trails for mapped formation roles, manual selections, heuristic candidates, or everyone")
    parser.add_argument("--formation-map", type=Path,
                        help="JSON file with per-clip pre-snap slot points and positions")
    parser.add_argument("--label-target", choices=["player", "helmet"], default=None)
    parser.add_argument("--team-colors", choices=["red-white"],
                        help="Optional jersey color grouping for games with red/white uniforms")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--buffer", type=int, default=30, help="Frames to retain missing tracks")
    parser.add_argument("--helmet-match-scale", type=float, default=1.0,
                        help="Experimental helmet motion tolerance, 1–3; try 2 for fast small helmets, with more false matches")
    parser.add_argument("--pause", action="store_true", help="Start paused for field selection")
    parser.add_argument("--manual-field", action="store_true", help="Disable automatic sideline detection")
    parser.add_argument("--snapshots", type=Path, default=ROOT / "outputs/review")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Inference resolution; try 640 for faster CPU playback.")
    parser.add_argument("--device", default=None, help="cpu or CUDA device index, e.g. 0")
    parser.add_argument("--start", type=float, help="Start seconds into clip; default: metadata tracking marker, else 0")
    parser.add_argument("--frame", type=int, help="Start on an exact zero-based frame (for labeling)")
    parser.add_argument("--snap", type=float, help="Verified snap seconds into clip; process only the shell window")
    parser.add_argument("--pre-seconds", type=float, default=1.0)
    parser.add_argument("--post-seconds", type=float, default=2.5)
    parser.add_argument("--tracker", choices=["botsort.yaml", "bytetrack.yaml"],
                        default="botsort.yaml")
    parser.add_argument("--save-video", type=Path, help="Write an annotated .mp4")
    parser.add_argument("--save-csv", type=Path, help="Write per-frame person observations")
    parser.add_argument("--headless", action="store_true", help="Process without a window")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    args = parser.parse_args()
    if args.snap is not None:
        if not math.isfinite(args.snap) or args.snap <= 0:
            parser.error("--snap must be finite and positive")
        if args.start is not None or args.frame is not None:
            parser.error("--snap cannot be combined with --start or --frame")
    if not all(math.isfinite(v) and v > 0 for v in (args.pre_seconds, args.post_seconds)):
        parser.error("Window durations must be finite and positive")
    helmet_weights = ROOT / "models/football_helmet.pt"
    pilot_weights = ROOT / "models/football_helmet_pilot.pt"
    if args.target == "auto":
        args.target = "helmet" if args.model is None and (helmet_weights.exists() or pilot_weights.exists()) else "player"
    if args.model is None:
        args.model = ((helmet_weights if helmet_weights.exists() else pilot_weights)
                      if args.target == "helmet" else ROOT / "models/yolo11n.pt")
    args.label_target = args.label_target or args.target
    if not 0 < args.conf <= 1:
        parser.error("--conf must be greater than 0 and at most 1")
    if not 1 <= args.helmet_match_scale <= 3:
        parser.error("--helmet-match-scale must be between 1 and 3")
    if args.start is not None and (not math.isfinite(args.start) or args.start < 0):
        parser.error("--start must be a finite nonnegative number")
    if args.frame is not None and (args.frame < 0 or args.start is not None):
        parser.error("--frame must be nonnegative and cannot be combined with --start")
    if args.buffer <= 0 or args.imgsz <= 0 or (args.max_frames is not None and args.max_frames <= 0):
        parser.error("--imgsz and --max-frames must be positive")
    if args.headless and args.video is None:
        parser.error("--headless requires a video path")
    if args.save_video and args.save_video.suffix.lower() != ".mp4":
        parser.error("--save-video must end in .mp4")
    return args


def choose_video():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    root.update()
    try:
        print("Choose a clip in the file picker. If it is hidden, check Alt+Tab.", flush=True)
        selected = filedialog.askopenfilename(
            parent=root,
            title="Choose an All-22 clip", initialdir=str(ROOT / "data/plays"),
            filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")])
        return Path(selected) if selected else None
    finally:
        root.destroy()


def clip_context(video):
    """Only show pre-play context; do not reveal outcome labels in the overlay."""
    for path in sorted((ROOT / "data/metadata").glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                if row.get("clip_file") == video.name and path.stem == video.parent.name + "_md":
                    def seconds(value):
                        hours, minutes, seconds = map(float, value.split(":"))
                        return hours * 3600 + minutes * 60 + seconds
                    offset = max(0, seconds(row["tracking"]) - seconds(row["clip_start"]))
                    return ((f"{row.get('offense', '?')} vs {row.get('defense', '?')} | "
                            f"Q{row.get('quarter', '?')} {row.get('clock', '')} | "
                            f"Down {row.get('down', '?')} & {row.get('distance', '?')} | "
                            f"{row.get('yard_line', '')}"), offset)
    return video.name, 0.0


def run(args):
    print("Starting All-22 viewer...", flush=True)
    video = args.video if args.video is not None else choose_video()
    if video is None:
        print("No clip selected. Viewer closed.", flush=True)
        return 0
    video = video.resolve()
    if not video.is_file():
        raise ValueError(f"Video not found: {video}")
    outputs = [path.resolve() for path in [args.save_video, args.save_csv] if path]
    if len(set(outputs)) != len(outputs) or video in outputs:
        raise ValueError("Input and output paths must be different.")
    for path in outputs:
        if path.exists():
            raise ValueError(f"Output already exists; choose a new filename: {path}")

    print("Loading OpenCV and YOLO libraries; first startup may take a moment...", flush=True)
    import cv2
    from src.detection.detection import load_detector
    from src.tracking.tracking import PlayerTracker
    from src.tracking.formations import load_formation_map
    from src.tracking.field import FieldFilter
    from src.viewer import ReviewControls

    with ExitStack() as stack:
        cap = cv2.VideoCapture(str(video))
        stack.callback(cap.release)
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open {video}")
        ok, frame = cap.read()
        if not ok:
            raise ValueError("The clip has no decodable frames.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            if args.snap is not None:
                raise ValueError("A reliable clip FPS is required for a snap-based window.")
            fps = 30.0
            print("Warning: missing frame rate; using 30 FPS for timing/export.")
        height, width = frame.shape[:2]
        context, metadata_start = clip_context(video)
        formation = None
        if args.formation_map is not None:
            formations = load_formation_map(args.formation_map)
            formation = formations.get(video.stem)
            if formation is None:
                raise ValueError(f"No formation entry for clip {video.stem!r} in {args.formation_map}")
            if args.player_scope == "manual":
                args.player_scope = "formation"
        if args.player_scope == "formation" and formation is None:
            raise ValueError("--player-scope formation requires --formation-map")
        if formation is not None and args.target != "helmet":
            raise ValueError("Formation matching currently requires helmet + body mode.")
        known_uniforms = video.parent.name == "game_01" and context.split(" | ")[0] in ("NE vs KC", "KC vs NE")
        if args.target == "helmet" and args.team_colors is None and video.parent.name == "game_01":
            args.team_colors = "red-white"
        mode = "HELMET + BODY" if args.target == "helmet" else "BODY ONLY"
        if args.model.name == "football_helmet_pilot.pt":
            mode += " (PILOT)"
        context = mode + " | " + context
        print(f"Detection mode: {mode}\nWeights: {args.model}", flush=True)
        start_seconds = metadata_start if args.start is None else args.start
        start_index = args.frame if args.frame is not None else round(start_seconds * fps)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stop_index = frame_count
        if args.snap is not None:
            from src.shells.baseline import analysis_window
            start_index, stop_index = analysis_window(args.snap, fps, frame_count,
                                                     args.pre_seconds, args.post_seconds)
        if formation is not None and formation["anchor_frame"] < start_index:
            raise ValueError("Formation anchor_frame is before playback start; start earlier or update the map.")
        if formation is not None and formation["anchor_frame"] >= frame_count:
            raise ValueError("Formation anchor_frame is beyond the end of this clip.")
        if start_index:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_index):
                raise ValueError("Could not seek to the requested start frame.")
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Start time is beyond the clip or cannot be decoded.")
        print(f"Opening {video.name}: {width}x{height}, {fps:.2f} FPS")
        print(f"Starting {start_index / fps:.2f}s into the clip (frame {start_index}).")
        print("Loading YOLO (official weights download on first use)...", flush=True)
        model, person_ids = load_detector(args.model, args.target)
        body_model, body_classes = (load_detector(ROOT / "models/yolo11n.pt")
                                    if args.target == "helmet" else (None, None))
        tracker = PlayerTracker(model, person_ids, args.conf, args.imgsz,
                                args.device, args.tracker, args.buffer, args.target,
                                body_model, body_classes, args.team_colors,
                                helmet_match_scale=args.helmet_match_scale)
        tracker.trail_scope = args.player_scope
        if known_uniforms and args.team_colors == "red-white":
            tracker.team_names = {"red": "KC", "white": "NE"}
        field = FieldFilter(automatic=not args.manual_field)
        review = ReviewControls(tracker, field, args.label_target)
        review.paused = args.pause
        writer = None
        if args.save_video:
            args.save_video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height + review.header_height))
            stack.callback(writer.release)
            if not writer.isOpened():
                raise RuntimeError("Could not create MP4 output with the mp4v codec.")
        csv_writer = None
        if args.save_csv:
            args.save_csv.parent.mkdir(parents=True, exist_ok=True)
            file = stack.enter_context(args.save_csv.open("x", newline="", encoding="utf-8"))
            csv_writer = csv.writer(file)
            csv_writer.writerow(["clip_file", "frame_index", "time_seconds", "segment", "track_id",
                                 "state", "confidence", "x1", "y1", "x2", "y2",
                                 "detection_target", "ground_x", "ground_y", "team_color", "team_name",
                                 "formation_position"])
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            stack.callback(cv2.destroyAllWindows)
            cv2.resizeWindow(WINDOW, min(width, 1280), min(720, height))
            cv2.setMouseCallback(WINDOW, review.mouse)
        print("Space: pause | N: step | D: raw/tracked | A: auto field | F: manual | Enter: apply | C: clear\n"
              f"L: label frame | 1: {args.label_target} | 2: background | 3: uncertain | V: labels complete\n"
              "Right-click: exclude | U: restore IDs | S: snapshot | T: trails | H: confidence | R: reset | Q: quit",
              flush=True)
        index = start_index
        stopping = False
        while True:
            started = perf_counter()
            field.advance(frame, tracker.raw)
            review.observations = tracker.update(frame, field)
            if formation is not None and index == formation["anchor_frame"]:
                assignments, unmatched = tracker.apply_formation_slots(formation, frame.shape)
                review.notice = (f"Formation matched {len(assignments)}/{len(formation['slots'])} slots; "
                                 f"unmatched: {', '.join(slot['name'] for slot in unmatched) or 'none'}")
            if not args.headless:
                delay = max(1, int(1000 / fps - (perf_counter() - started) * 1000))
                while True:
                    cv2.imshow(WINDOW, review.render(frame, context, index, fps))
                    key = cv2.waitKey(30 if review.paused else delay) & 0xFF
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (27, ord('q')):
                        stopping = True
                        break
                    if key == ord('n') and not review.selecting and not review.labels.enabled:
                        review.paused = True
                        break
                    if key == ord('s'):
                        review.snapshot(args.snapshots, video, frame, context, index, fps)
                    else:
                        review.key(key)
                    if not review.paused:
                        break
            # Export once after any corrections made on this paused frame.
            if writer:
                writer.write(review.render(frame, context, index, fps))
            if csv_writer:
                for item in review.visible():
                    csv_writer.writerow([video.name, index, round(index / fps, 6),
                                         tracker.segment, item.track_id, "observed", item.confidence, *item.box,
                                         item.source, *(item.ground_anchor or ("", "")), item.team,
                                         tracker.team_names.get(item.team, "unknown"),
                                         tracker.formation_roles.get(item.track_id, "")])
                for identity in tracker.missing:
                    if identity not in tracker.excluded:
                        csv_writer.writerow([video.name, index, round(index / fps, 6),
                                             tracker.segment, identity, "missing", "", "", "", "", "",
                                             args.target, "", "", "unknown", "unknown",
                                             tracker.formation_roles.get(identity, "")])
            if stopping or index + 1 >= stop_index or (args.max_frames and index - start_index + 1 >= args.max_frames):
                break
            index += 1
            review.labels.new_frame()
            ok, frame = cap.read()
            if not ok:
                print(f"Finished {index - start_index} frames.")
                break
        print("Processing complete. Any requested exports have been written.")
    return 0


if __name__ == "__main__":
    args = parse_args()
    try:
        raise SystemExit(run(args))
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        raise SystemExit(f"Error: {error}") from error
