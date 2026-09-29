# All-22 player tracking

See [the comparison findings](docs/tracking-review.md) for measured results and limitations.

OpenCV reads the clip, a pilot YOLO model detects football helmets, and the
tracker combines helmet evidence with body detections across frames. For the
included NE/KC play, upper-jersey and helmet colors label the teams. Track IDs
are temporary identities, not jersey numbers. This remains a review tool rather
than a reliable scheme evaluator.

## Run

### Helmet labeling (new)

Eight clean frames from the reported pass play are prepared in
`data/annotations/game_01/pass_001/`. To label them without loading YOLO:

```powershell
.\.venv\Scripts\python.exe label_helmets.py
```

Drag tight boxes around visible **worn football helmets**, including both teams
and sideline helmets. Z zooms 3× around the mouse; Z again returns to full frame.
1 selects helmet, 2 background/non-helmet (e.g. the suspended camera), and 3 an
uncertain region. Backspace undoes the last box. N/P saves and moves to the
next/previous frame; S saves; Q saves and exits. Work resumes from saved labels
on the next run. Only press V when ALL visible helmets have been labeled. Partial
frames are preserved but not exported for training. The initial eight frames are
a labeling/diagnostic starter set, not a sufficient training or validation set.

For different footage:

```powershell
.\.venv\Scripts\python.exe label_helmets.py data/plays/game_02/pass_002.mp4 --frames 100 110 120
```

The normal viewer also accepts `--label-target helmet`. Complete helmet frames
are explicitly tagged and cannot be mixed with body labels. Export with:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --target helmet --train data/annotations/game_01 --val data/annotations/game_02 --output data/processed/helmet_dataset
```

The `game_02` paths above are examples, not files created by this change. Separate
games and enough reviewed examples are still needed before training/evaluation.

### Helmet detection status

The reviewed labels were used to train two pilot checkpoints. The better
checkpoint is installed as `models/football_helmet_pilot.pt`. Normal playback
selects it automatically and shows `HELMET + BODY (PILOT)` in the header:

```powershell
.\.venv\Scripts\python.exe main.py data/plays/game_01/pass_001.mp4
```

This mode tracks helmet detections as the primary identities and draws trails at
helmet centers. A second body detector provides estimated foot points only when a
helmet/body association is unambiguous. It does not require a body to track a
helmet, and it does not invent a body box when one is absent. `H?` means no reliable
body-derived ground anchor. With an active field boundary, these helmets must be
at least two helmet heights inside the polygon to appear in tracked view. This
conservative margin removes sideline heads that project onto the grass, but can
also hide actual players close to the boundary. Reliable body-derived feet take
priority over this margin. D still shows all raw detections for diagnosis.
Ground anchors are image pixels, **not field-yard coordinates**; calibration and
evaluation of association are still needed. Both detectors add processing cost.
An unambiguous body match can support an established helmet identity for up to two
frames during a short helmet occlusion. It is labeled `B`, while measured helmet
observations are labeled `H`. Crowded or ambiguous body matches are rejected.

Changing `--target` does not train a model. Missing weights or a person-only model
produce an explicit error, never fake helmet detections. Saving labels does not
update model weights. See [helmet performance](docs/helmet-performance.md) for
the measured pilot results and their limitations. The NFL helmet-assignment dataset is a potential future
source: https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment/data

### Player viewer

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Choose a clip in the file picker. The viewer opens in a separate window, not a
VS Code tab. To bypass the picker and begin paused on the example play:

```powershell
.\.venv\Scripts\python.exe main.py data/plays/game_01/pass_001.mp4 --start 4.5 --pause
```

Without `--start`, playback uses `tracking - clip_start` from matching metadata,
or zero if no metadata matches. Times and frame numbers always refer to the
original clip. Model weights download on first use. Startup messages distinguish
loading libraries from waiting for the picker; use Alt+Tab if the picker is hidden.

Automatic sidelines are now enabled by default. The viewer finds visible far/near
sideline candidates, waits for both to agree across two consecutive frames, then
filters to the field between them. A single blue line is only a candidate, not an
active filter. Yellow means an active boundary. Once established, the boundary
follows camera motion even when one side leaves the frame. The status distinguishes
fresh detection from following an earlier boundary. This is a geometric OpenCV
heuristic for elevated sideline footage, not a newly trained model or a guarantee.
Use F for manual correction, A to return to automatic mode, or `--manual-field`
to disable automatic detection at startup. No reliable pair means no automatic
filtering; the system does not invent the unseen sideline.

## Review controls

| Control | Action |
|---|---|
| Space | Pause/resume |
| N | Advance exactly one frame and pause |
| D | Switch between **raw detections** and **tracked observations** |
| A | Enable automatic sideline detection |
| F | Pause and start a field boundary; click corners in order |
| Enter | Apply the boundary (at least three points; convex polygon) |
| Right-click while drawing | Undo the last boundary point |
| C | Clear/cancel the field boundary |
| Right-click a tracked box | Exclude its ID and pause |
| U | Restore all manually excluded IDs |
| S | Save the original frame, current overlay, and observation JSON |
| T | Toggle trails (on by default; up to 120 points/break markers) |
| H | Toggle confidence labels (off by default) |
| R | Reset tracker and field at a camera cut; IDs restart |
| Q / Esc | Quit |
| L | Pause and toggle manual frame labeling (clean image) |
| 1 / 2 / 3 while labeling | Player / non-player (camera, referee, etc.) / uncertain |
| Left-drag while labeling | Draw a box for the selected label type |
| Backspace while labeling | Undo the last label |
| V while labeling | Mark/unmark the frame as fully reviewed |

Select a boundary around the playable field, not just the current cluster of
players: receivers move. The bottom-center of each box is the approximate foot
position used for filtering. Raw mode deliberately shows **all** detector output,
including rejected sideline detections, so you can diagnose failures.

The boundary follows approximate camera motion using background optical flow.
This handles modest pans/zoom but is not field calibration. Check the yellow
boundary regularly and redraw with F when it drifts. If motion estimation fails,
the viewer disables filtering and searches again in automatic mode; manual mode
shows **Field LOST** and requires a new boundary.
It never silently keeps using the old boundary. A referee inside the polygon
still requires manual exclusion. Exclusions are per ID and per session; if an
identity changes, exclude the new ID. The NE/KC color mapping is specific to the
included game and is not a general jersey-number or offense/defense classifier.

Boundary changes no longer remove detections from the underlying tracker. This
prevents boundary-induced ID loss from bringing excluded boxes back. Filtering
is applied to tracked display/export instead; excluded objects continue being
tracked internally. U restores them immediately. Actual detection gaps/ID switches
can still assign a fresh ID; exclusions are not permanent object recognition.
Raw mode intentionally shows every detector box, including exclusions.

Missing IDs are listed separately without a box: their position was not measured
in the current frame. They can include people who left the view,
not only occluded players. We intentionally do not invent predicted positions.
Tracks expire after the configured buffer. Fully hidden players remain uncertain.

Helmet-mode trails use smoothed body footpoints only; a missing body match
never switches the trail to a helmet center. The default `--player-scope manual`
shows trails only for players selected in the viewer: press M, click a player's
tracked box to toggle its trail, then press M to leave selection mode. This is
the reliable way to choose skill players and second-level defenders because the
available detectors have no position labels. `--player-scope eligible` enables
an experimental movement/density heuristic; `--player-scope all` shows every
confirmed trail. All tracked player boxes stay visible in each scope. If only a
body detection is available, the viewer draws an inferred head-level marker,
not a helmet detection. Confirmed
trails remain visible through temporary detector gaps and for the rest of the clip;
body detections can support an existing helmet ID
for up to 12 frames, and persistent body-only detections can establish an ID
that a later unambiguous helmet/body match adopts. Unavailable camera motion
leaves trails in their last image coordinates while spatial re-identification
memory is reset. Lines connect observed points across gaps; those connecting
segments are visual continuity cues, not measured or predicted locations.
Trails are image paths, not calibrated routes in yards. New IDs remain eligible:
freezing the first 22 would lock in false detections and prevent recovery of
missed players. The roster cap does not verify 11 per team.

### Formation maps

To assign trails automatically from a pre-snap alignment, provide a JSON map
keyed by clip stem. Each slot needs its football position and normalized head
center on a stable pre-snap frame. Coordinates are `[x / image_width,
y / image_height]`; `anchor_frame` is the zero-based frame in the original clip.
Only QB, RB, WR, TE, DB, and LB slots are accepted. Optional `team` values such
as `NE` or `KC` constrain the match. Each observation can match at most one slot;
ambiguous or distant slots are left unmatched. The formation name is descriptive;
the geometry comes from the slot points. The example points in
[formations.example.json](docs/formations.example.json) are placeholders, not
labels for `pass_001`.

```json
{
  "version": 1,
  "plays": {
    "pass_001": {
      "formation": "Shotgun Trips Right",
      "offense_direction": "right",
      "anchor_frame": 150,
      "max_distance": 0.06,
      "slots": [
        {"name": "X", "position": "WR", "team": "NE", "point": [0.20, 0.30]},
        {"name": "RB", "position": "RB", "team": "NE", "point": [0.45, 0.55]},
        {"name": "CB1", "position": "DB", "team": "KC", "point": [0.78, 0.32]}
      ]
    }
  }
}
```

Save the filled map and run from before its anchor frame:

```powershell
.\.venv\Scripts\python.exe main.py data/plays/game_01/pass_001.mp4 --start 0 --formation-map data/formations.json
```

The viewer reports matched and unmatched slots at the anchor frame, labels
matched players with their role, and trails only those identities. The project
does not currently infer slot points from a formation name alone; All-22 camera
perspective and formation variations make that guess unreliable without a
calibrated field map or roster labels.

Experimental `--helmet-match-scale 2` doubles the association box size internally
and admits scores down to 0.10 for the tracker's secondary association stage;
displayed boxes remain helmet-sized and starting a new identity still requires
the normal confidence threshold. It helps fast small helmets but can retain more
false positives. Default scale 1 keeps the original detection threshold and box
geometry. Near-duplicate helmet boxes are removed before tracking. ID memory
follows camera motion; reconnections beyond three frames require agreeing
helmet/jersey appearance and a unique close spatial match, up to `--buffer` frames.
This color histogram is not learned player ReID. It cannot reliably distinguish
teammates or recover a player who returns far from their last known location.

Snapshots go to `outputs/review/` in unique frame-specific folders. JSON includes
clip path, time, raw detections, observed tracks, missing IDs, exclusions, and the
current polygon. Model predictions are review records, not ground-truth labels.

## Label the failures for future detector training

Open an exact frame, such as the linemen at frame 148:

```powershell
.\.venv\Scripts\python.exe main.py data/plays/game_01/pass_001.mp4 --frame 148 --pause
```

Press L, then 1 and drag a separate box around each visible player. Press 2 to
mark non-player objects such as the camera. Press 3 for regions whose player
extent cannot be determined; never invent boxes for fully hidden players. S saves
your labels with the original image. Labels apply to this frame only, are separate
from predictions/track IDs, and do not propagate. Save before leaving the frame.
Press L again to leave labeling mode before N or Space.

Partial corrections are useful evidence. Only press V if **every visible active
player in the entire image** has been labeled; unlabeled people are treated as
background during training. Non-player labels identify deliberate background
examples; they are not a second class. Snapshots with uncertain regions cannot
be exported for training until reviewed. This conservative exporter has no
ignore-region support.

When complete examples from separate games are available:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --train outputs/labeled_game1 --val outputs/labeled_game2 --output data/processed/player_dataset
```

This produces YOLO images/labels and `dataset.yaml`, preserving source references.
It refuses partial-only input, uncertain examples, and same-game train/validation
splits. It never imports raw YOLO boxes as labels. It does not train the detector;
use the training and evaluation commands below once reviewed examples exist.
Neither field drawing nor right-click exclusion changes model weights. Future
custom models with a `player` class can be loaded using `--model`.

## Settings and exports

The conservative defaults remain `yolo11n.pt`, `--conf 0.25`, `--buffer 30`.
Try `--model models/yolo11s.pt` for the larger detector, `--conf 0.1` to admit weak
detections for association, or `--buffer 60` to retain missing tracks longer.
Lower confidence can increase false detections. Longer retention can preserve
wrong identities. These options are experiments, not guaranteed improvements.
Use `--imgsz 640` for faster CPU processing (default 1280), or `--device 0` with a
CUDA-enabled PyTorch installation. All frames are processed; playback may be slow.

```powershell
.\.venv\Scripts\python.exe main.py data/plays/game_01/pass_001.mp4 --save-video outputs/pass_review.mp4 --save-csv outputs/pass_review.csv
```

Outputs must not already exist. Video exports include the current diagnostic view
and header, and have no audio. VS Code may not decode their `mp4v` codec; use a
compatible external player. CSV always contains filtered tracked observations,
even when the display is in raw mode. Missing rows have `state=missing` and empty
coordinates/confidence. Observed rows contain original-image pixel coordinates.
Use `(segment, track_id)` as the identity: a manual reset increments the segment
because the tracker can reuse numeric IDs.
Each frame is exported once, after any paused corrections. Edits do not rewrite
earlier frames. An early quit exports only the processed portion. Frame/FPS time
assumes constant-frame-rate footage. Use `--headless --max-frames 30` for smoke tests.

## Code organization

- `main.py`: command options, metadata, video loop, and exports.
- `src/detection/detection.py`: load the person detector.
- `src/detection/annotations.py`: per-frame manual labels, separate from predictions.
- `src/tracking/tracking.py`: one prediction and one tracker update per frame;
  raw/observed/missing states and box drawing.
- `src/tracking/field.py`: boundary selection data and approximate camera motion.
- `src/viewer.py`: mouse/keyboard controls, overlays, and snapshots.
- `benchmark.py`: repeatable model/settings comparisons.
- `prepare_dataset.py`: export complete human labels with game-separated validation.
- `label_helmets.py`: resumable, zoomable labeling of selected frames without inference.
- `train_detector.py`: fine-tuning, checkpoint saving, and automatic validation.
- `evaluate_detector.py`: detection metrics and per-frame error reports.
- `src/detection/dataset.py`: dataset integrity and split checks.
- `tests/test_tracking.py`: synthetic occlusion, filtering, motion, and review tests.

Ultralytics is pinned because the explicit tracker adapter uses its Python tracker
classes. When upgrading, run these tests and a real-clip smoke test.

## Reproduce comparisons

```powershell
.\.venv\Scripts\python.exe benchmark.py data/plays/game_01/pass_001.mp4 --output outputs/new_comparison
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The benchmark processes frames 120–427 continuously and saves raw/tracked frames
at 135, 306, 378, and 427. It changes one setting at a time: confidence, model,
or track buffer. Frame counts and new-ID counts are **diagnostics, not accuracy**.
Fewer IDs may mean missed players or merged identities. Proper missed-player and
ID-switch rates require manually labeled identities and visibility. Keep a
separate game for eventual evaluation of football-specific training.

## Train and evaluate a helmet detector

### Optional jersey color grouping

Helmet mode now samples the upper jersey immediately beneath a detected helmet
even if the body detector misses that player. An unambiguous body crop provides
additional evidence; conflicting colors or overlapping helmets remain unknown.
This sampled region is appearance evidence, not a detected body or ground point.

For the current red-versus-white game, add `--team-colors red-white` to the viewer:

```powershell
.\.venv\Scripts\python.exe main.py data/plays/game_01/pass_001.mp4 --team-colors red-white
```

This works with the current body detector and, once trained weights are available,
with helmet mode. Helmet mode still gets identity from helmets; unambiguous body
matches supply torso crops. Red/white/unknown appears beside the ID and in the
CSV `team_color` column. For the known NE-KC game_01 metadata, the display maps
red to KC and white to NE; the CSV also includes `team_name`. The optional feature uses HSV color evidence and votes
across frames, not trained model weights. It rejects ambiguous overlapping bodies
and does not identify individual teammates or jersey numbers. Team names outside
the known game_01 uniform mapping remain unknown. A
referee or sideline person can still resemble a jersey; the color label alone is
not proof of being a player. Other uniform combinations need a different profile
or a trained classifier with labeled torso examples. See
[OpenCV's color-space documentation](https://docs.opencv.org/4.10.0/df/d9d/tutorial_py_colorspaces.html).

### Helmet dataset and training

An experimental single-clip route is available in `train_helmet_pilot.py`. It
uses reviewed game_01 labels and explicitly reuses its training images for the
library's diagnostic validation. Those scores are **not held-out accuracy**.
per batch. No model is installed automatically by that training script.

When `models/football_helmet_pilot.pt` has been installed after a visual check,
the viewer's default `--target auto` selects helmet mode. The header shows
`HELMET + BODY (PILOT)`. Use `--target player` to explicitly select the old body
detector. A standard `models/football_helmet.pt` takes priority over the pilot.
For custom model paths, specify `--target helmet` explicitly. Red/white color

Use different game folders for training, validation, and optionally final testing:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --target helmet --train data/annotations/game_01 --val data/annotations/game_02 --test data/annotations/game_03 --output data/processed/helmet_dataset
.\.venv\Scripts\python.exe train_detector.py --check
.\.venv\Scripts\python.exe train_detector.py --epochs 50 --imgsz 1280 --batch 4 --device cpu --output outputs/training/helmet_v1
```

The game_02 and game_03 paths are examples: prepare and review footage from those
games first. Omit `--test` until you have a third game. Dataset export rejects
duplicate frames and records image/label hashes. Training refuses changed exports;
make corrections in the annotation tool and export to a new folder. `--check`
verifies integrity only; it does not certify sufficient data or label quality.

Training starts from local `models/yolo11n.pt`, adapts it to the single helmet
class, and saves `weights/best.pt`, `weights/last.pt`, learning curves, training
settings and an automatic validation report in the chosen run folder. CPU training
at 1280 can be slow; use `--device 0` only with a working CUDA installation.
Lowering `--imgsz` saves compute but may hurt small-helmet detection.

Validation helps select the model and confidence setting. After selecting them,
evaluate once on the separate test game:

```powershell
.\.venv\Scripts\python.exe evaluate_detector.py --model outputs/training/helmet_v1/weights/best.pt --split test --conf 0.25 --output outputs/evaluation/helmet_v1_test
```

Reports include standard mAP50 and mAP50-95, precision/recall, and fixed-confidence
TP/FP/FN counts in `summary.json`. `per_frame.csv` identifies missed detections and
false positives by original clip/frame. Up to 25 error images show missed truth
in yellow, false detections in red, matched truth in green and matched predictions
in cyan. Matching requires one prediction per truth box at IoU >= 0.5 by default;
duplicate predictions count as false positives. Standard precision/recall use the
library's selected threshold, while the separate operating-point report uses
`--conf`. These metrics evaluate detection, not stable track identities or schemes.

The viewer model is never automatically replaced. After inspecting real-game
results, explicitly load the trained weights:

```powershell
.\.venv\Scripts\python.exe main.py data/plays/game_01/pass_001.mp4 --target helmet --model outputs/training/helmet_v1/weights/best.pt
```

Developer smoke test: `python tests/smoke_training.py` prepares artificial
rectangles and prints a one-epoch training command. Its outputs live under
`outputs/synthetic_training_smoke`. It proves pipeline execution only; those
weights and metrics are not a football detector or evidence of football accuracy.
