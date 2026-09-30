# NFL Defensive Shell Analyzer

All 15 pass plays now have provisional snap/field configs in `data/metadata`.
To review saved analyses at normal speed, with large shell and best-estimate
labels and a final frame held until Q:

```powershell
.\.venv\Scripts\python.exe -m tools.review_defense_batch --input outputs/all_plays_review_v1
```

To run new analysis interactively on every play:

```powershell
.\.venv\Scripts\python.exe -m src.shells.run_defense --config-dir data/metadata --output-root outputs/live_all_plays --show
```

Each play ends at approximately 2.3 seconds after the estimated snap. Q advances
to the next play (or closes the final play); Esc exits the batch; Space pauses
and resumes during playback. The final frame never auto-closes. Saved visual
review candidates are explicitly labeled when used as display fallback and
are never fed into automatic inference. The final frame can use a saved
whole-play review, including evidence later than the paused frame; it is
labeled WHOLE-PLAY FILM REVIEW, not presented as a live model prediction.
Unknown remains visible when the evidence does not support a coverage name.

An experimental role-classifier training run and receiver-following analysis
are now available. See [training results and usage](docs/role-training-and-following.md).
The role pilot reached 50% agreement on held-out weak labels and remains
experimental. New defense runs track offensive release candidates and save
temporal following evidence alongside staged coverage estimates.

The [player-role review gallery](data/defensive_shells/player_roles/index.html)
contains 30 pre/post clip screenshots with provisional safety, corner, and
linebacker points across all 15 plays. Eight exact user reference PNGs are saved
under `data/defensive_shells/coverages/references/` and linked in
`data/metadata/coverage_reference_playbook.json`. These are persistent reference
assets, not newly trained model weights. New defense reports include
`coverage_stages`: pre-snap family estimates, tracked safety rotation and revised
families, then corner-motion refinement with alternatives retained.

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

The coordinate baseline remains independently testable. An experimental
defense-only short-window runner connects filtered detection, tracking, camera
compensation, pre/post snapshots, high-safety counts, shell candidates, and
corner-motion comparisons.
It defaults to helmet + body because that produced more candidate defenders
than body-only in the configured `pass_001` check. Shell results stay `unknown`
until all 11 defenders have reliable positions; coverage remains
unknown unless a majority of at least two tracked corners move within/outside the
configured 5–7-yard radius. This is the requested provisional motion heuristic,
not proof of man/zone responsibilities. See the [audit and implementation plan](docs/shell-audit-and-plan.md).

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

For the defense-only report, use the per-play config. `defense_pass_001.json`
contains a visually estimated snap frame (uncertainty about three frames), not
a human-verified snap. It also contains manually estimated camera geometry that
needs review. Each run requires a new output folder:

```powershell
.\.venv\Scripts\python.exe -m src.shells.run_defense --config data/metadata/defense_pass_001.json --output outputs/defense_pass001_run002
```

The result includes original/annotated snapshots, movement CSV/JSON, a short
tracking video, and separate helmet/body color-filter counts. Pre/post overlays
show shell candidate, accepted shell, and high-safety count. The post overlay
draws 6-yard bubbles around corners only when a zone-like majority is measured.
The current pixels-per-yard values are provisional and must be calibrated from
visible field markings; incomplete corner tracks produce `unknown`, not bubbles.
Add `--show` to view detections and tracks while frames are processed. Press
Space to pause/resume; Q or Esc stops the current play and preserves partial
video/tracks.

For a batch, place reviewed `defense_<play_id>.json` configs in a directory and
run:

```powershell
.\.venv\Scripts\python.exe -m src.shells.run_defense --config-dir data/metadata/defense_configs --output-root outputs/defense_batch_001 --show
```

Each config stores a pre-snap frame, estimated snap frame, defense color, field
polygon, and approximate local camera geometry. All 15 clips are configured;
new snap estimates carry uncertainty of about eight frames and require human
verification. Batch mode processes all configs in the directory and creates
one output folder per play, preserving previous runs.

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
