# NFL Defensive Shell Analyzer

Computer-vision project focused on **pre-snap defensive shells and initial
post-snap safety rotation** in All-22 pass-play footage. Shells are not coverage calls.

The active dataset contains 15 pass clips. Run clips were removed at the user's
request; the clip extractor now skips non-pass metadata rows. Original game
metadata and raw footage remain available as source records.

The [snapshot review index](data/defensive_shells/index.html) contains one selected
pre-snap image per pass play: 9 one-high, 3 two-high, 1 zero-high, and 2 unknown.
These are preliminary training candidates from visual agent review. Verify the
labels before supervised training; unknown frames are held out.

The [post-snap coverage gallery](data/defensive_shells/coverages/index.html) adds
30 original snapshots with movement-sequence evidence: 8 Cover 1 man candidate
plays, 3 Cover 2 man candidate plays, and 4 unknown plays. All six requested
coverage folders exist; unsupported classes remain empty. These dataset labels
extend the reference collection, not the automatic shell classifier's capabilities.

The current baseline classifies manually prepared defensive coordinates. The
viewer supplies full-player detection and tracking for review. Automatic clip-to-
shell analysis is not yet validated or wired together; see the
[audit and implementation plan](docs/shell-audit-and-plan.md).

## Start here

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.shells docs/shell-observations.example.json
.\.venv\Scripts\python.exe main.py data/plays/game_01/pass_001.mp4 --pause
```

After manually confirming a snap timestamp, use `--snap SECONDS` to limit the
viewer to one second before and 2.5 seconds after the snap. For example, **if**
the snap is at 5 seconds:

```powershell
.\.venv\Scripts\python.exe main.py data/plays/game_01/pass_001.mp4 --snap 5 --headless --save-video outputs/shell-window.mp4 --save-csv outputs/shell-tracks.csv
```

That timestamp is illustrative, not a label for this play. Without `--snap`, the
viewer retains its exploratory playback behavior. `tracking_start_seconds` is
not a snap label. `--pre-seconds` and `--post-seconds` adjust the window;
`--buffer 30` retains missing tracks for 30 processed frames. Full-player YOLO
is now the default; helmet mode remains an explicit diagnostic option.

## Project layout

| Location | Purpose |
|---|---|
| `main.py` | Detection/tracking review and short-window video export |
| `src/shells/` | Small geometry baseline, rotation taxonomy, window calculation |
| `src/detection/`, `src/tracking/`, `src/viewer.py` | Existing detection and review components |
| `data/plays/`, `data/raw/` | Original footage; preserved |
| `data/defensive_shells/` | [Reference images and rotation clips](data/defensive_shells/README.md) |
| `data/metadata/` | Original game metadata, shell labels, and review queue |
| `tools/` | Optional labeling, detector training, evaluation, benchmarks |
| `scripts/` | Historical one-off diagnostic scripts |
| `docs/legacy/` | Historical experiments and detailed viewer controls |
| `outputs/` | Generated review artifacts; ignored by Git |
| `tests/` | Geometry, window, detector, and tracking regression tests |

Utility commands now use module syntax: `python -m tools.prepare_dataset`,
`python -m tools.label_helmets`, `python -m tools.train_detector`,
`python -m tools.evaluate_detector`, or `python -m tools.benchmark`.
Pass `--help` for options. Older commands in historical notes predate this move.

Run checks with `python -m unittest discover -s tests -v`.

## Baseline contract

`python -m src.shells INPUT.json --output outputs/shell.json` writes a new result
file and refuses to overwrite one. The example is synthetic, not film evidence.
Inputs specify lateral distance from field center and depth from the LOS,
positive toward the defensive backfield. Defaults use yards; image-coordinate
users must first choose a consistent LOS-relative basis and supply all thresholds
in those units. No automatic homography is provided.

`complete: true` means a reviewer has confirmed both full secondary visibility
and reliable defensive-player observations. Otherwise the result is `unknown`.
Wide corner candidates are excluded by a configurable lateral gate; threshold
borderlines, asymmetric structures, and three-high are rejected conservatively.
Confidence is a heuristic score, not a measured probability. Pre/post coordinate
comparison does not establish player identity or measure tracking quality.
