# NFL Defensive Coverage Analyzer

This project reviews All-22 pass-play footage to surface defensive indicators:
pre-snap shell, deep-safety candidates, safety rotation, corner movement, and
provisional man/zone coverage estimates. It is designed to support film study,
not replace a coach's coverage call.

The active dataset contains 15 New England–Kansas City pass clips. Defensive
players are separated with team-color evidence, tracked through the play, and
shown in an interactive review viewer. The viewer refreshes player detections on
the final frame, highlights the deepest safety tier with approximate 15-yard
boxes, and retains other zone-defender boxes or man-following lines when the
underlying evidence supports them.

## Review saved plays

```powershell
.\.venv\Scripts\python.exe -m tools.review_full_plays --input outputs/all_plays_review_v1
```

The viewer plays each source clip to its end and then holds the snap +2.30-second
coverage frame. Hold **HOLD: 2x speed** to accelerate playback, click **Skip to
snap** or press **S** to jump to the snap, use Space to pause, Q for the next
play, and Escape to exit.

The large labels show the estimated shell and best coverage estimate. A displayed
film-review candidate is explicitly labeled as such; it is not presented as a
verified model prediction. Safety and coverage boxes are approximate visual aids,
not measured assignments.

## Run a new analysis

Install dependencies and run an individual configured play:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.shells.run_defense --config data/metadata/defense_pass_001.json --output outputs/defense_pass001_run002 --show
```

Run all configured plays into a new output folder:

```powershell
.\.venv\Scripts\python.exe -m src.shells.run_defense --config-dir data/metadata --output-root outputs/live_all_plays --show
```

The geometry in each `defense_<play_id>.json` is a reviewed local image basis,
not a calibrated field homography. The system keeps uncertain observations as
`unknown` rather than forcing a coverage label. Cover 0 requires a complete view
with no high safety; Cover 2 versus Cover 4 uses both deep-corner depth and flat
occupancy cues.

## Shell baseline example

```powershell
.\.venv\Scripts\python.exe -m src.shells data/examples/shell-observations.example.json --output outputs/shell.json
```

The synthetic example demonstrates the geometry-only shell baseline. Input
coordinates use lateral distance from field center and depth from the line of
scrimmage, with positive depth toward the defensive backfield.

## Reference data and training context

The [shell gallery](data/defensive_shells/index.html), [coverage gallery](data/defensive_shells/coverages/index.html), and [player-role gallery](data/defensive_shells/player_roles/index.html) preserve the visual reference set. The machine-readable coverage cues are in [coverage_reference_playbook.json](data/metadata/coverage_reference_playbook.json).

These assets are training and review context, not verified ground truth or
fine-tuned model weights. The experimental role pilot and receiver-following
analysis remain optional; their outputs must be validated against independently
labeled games before accuracy claims are made.

## Project layout

| Location | Purpose |
|---|---|
| `src/` | Detection, tracking, shell, coverage, and viewer logic |
| `tools/` | Review, training, evaluation, and dataset utilities |
| `data/plays/` | Input play clips |
| `data/metadata/` | Per-play geometry, labels, and review context |
| `data/defensive_shells/` | Persistent image references and galleries |
| `data/examples/` | Runnable synthetic examples |
| `models/` | Local detector weights |
| `outputs/` | Generated reports and review artifacts |
| `tests/` | Regression tests |

Run all checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
