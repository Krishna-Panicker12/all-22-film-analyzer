# Scope reset and code audit

## Pass-only dataset update — September 29, 2026

At the user's subsequent request, all 15 `run_*.mp4` clips were deleted from
`data/plays/game_01`. All 15 pass clips remain, the active review queue is pass-only,
and `src/data/parseClips.py` now requires `play_type` and skips non-pass rows.
The original metadata retains run rows as source history; raw footage is unchanged.

Every pass clip was visually inspected using timeline sheets and full-resolution
frames. One original pre-snap PNG per play was exported: 9 one-high, 3 two-high,
1 zero-high, 2 unknown. Source/frame provenance and preliminary label rationale
are in the shell label CSV and frame-selection JSON. Human verification and
exact snap times remain pending. The historical 30-clip findings below describe
the earlier audit, before the explicitly requested run-file deletion.

The supplied NFL Defensive Shell Analyzer spec is the design reference for this
reset. Its embedded dataset-agent prompt is not evidence that labels have been
reviewed. This pass implements the geometry foundation and organization, not the
full success definition of validated automatic video analysis.

## Findings

* The worktree already contained substantial uncommitted work. Existing code,
  raw clips, weights, annotations, and generated outputs were preserved.
* `main.py` previously selected helmet pilot weights automatically. It now defaults
  to the full-player/person YOLO detector. Generic person weights also detect
  officials and sideline people; football-specific accuracy remains unmeasured.
* `PlayerTracker` calls the detector once and updates the Ultralytics engine once
  per frame, including empty detections. Default engine is BoT-SORT; ByteTrack is
  selectable. Both use Kalman prediction internally. The installed implementation
  predicts active and lost tracks before association.
* `track_buffer=30` is passed to the engine. Confirmed tracks enter the lost pool
  and expire beyond `max_frames_lost`; unconfirmed tracks may be removed sooner.
  Missing tracks are intentionally exported without measured boxes. A missing
  rendered box does not prove that the track was deleted.
* IDs originate in the engine, with additional canonical aliases and body-only
  IDs in helmet mode. Resets restart the ID namespace and increment the segment.
  Helmet mode includes duplicate suppression, appearance matching, body bridging
  up to min(buffer, 12), and confirmation gates. These layers complicate diagnosis
  and are retained only as existing experiments, not the new baseline dependency.
* Detector confidence filtering can discard weak boxes before the engine's
  secondary association. Benchmark this before lowering thresholds globally.
  Longer buffers cannot recover absent detections or reliably distinguish teammates.
* Team separation is a red/white heuristic with a known NE/KC mapping, not a
  general offense/defense classifier. Field filtering and camera optical flow
  are not yard calibration. Formation slots and route eligibility are deferred.
* There are 30 play clips. Their CSV has `tracking`, not a verified snap timestamp,
  LOS annotation, or shell labels. No measured shell accuracy can be claimed.

## Changes in this pass

1. Added `src/shells` with explicit geometry, conservative rejection, limited
   rotation classes, structured results, and an original-frame window function.
2. Added `--snap`, `--pre-seconds`, and `--post-seconds` to the viewer. Known snap
   times constrain actual decoding/inference; the default window is 1 + 2.5 seconds.
   Exploratory playback remains available when the snap is not known.
3. Moved eight optional root utilities into `tools`, updating Python imports.
   Archived the old README and experiment documentation under `docs/legacy`.
   No broad tracker rewrite or deletion of prior work was needed.
4. Added shell and rotation reference folders, a single combined label schema,
   and a populated clip review queue. The spec's two proposed CSV schemas are
   reconciled by keeping predictions separate from human-reviewed final labels.

## Next milestones and acceptance checks

1. **Review data:** verify snap and pre-snap times, visibility, defense, and LOS.
   Export reference images with source provenance; uncertain evidence stays unknown.
   No automatic assignment based solely on the legacy tracking marker.
2. **Detection and team audit:** annotate full players in representative held-out
   frames. Measure missed deep defenders and false positives; visually compare
   player-only and optional helmet-assisted modes. Use metadata for team mapping.
3. **Wire video to geometry:** transform measured feet into a reviewed LOS-relative
   basis, pass only reliable defensive observations to the classifier, and return
   unknown when visibility/team/geometry evidence fails. Image thresholds require
   per-view calibration; manual coordinates are the current input contract.
4. **Short-window tracking:** compare ByteTrack/BoT-SORT and buffers 15/30 using
   manually identified deep defenders. Record survival, drops, identity switches,
   and reacquisition; do not use raw ID count as accuracy. Preserve predicted
   positions as predictions, never observed detections. Add deep-defender selection
   after the reliable defensive candidate interface exists.
5. **Rotation and evaluation:** use stable pre/post frame groups, not single-frame
   flicker. Export selected safety tracks and JSON together. Measure per-class
   shell accuracy, unknown rejection, and rotation accuracy using only verified
   final labels; report sample counts and keep games separated.
6. **Later:** field homography, three-high, rotation direction, and eventually
   coverage classification. Offensive routes, complete identities, run/pass,
   and play grading remain outside this MVP.

The current baseline's fixed .75 confidence expresses a rule match only. It must
not be reported as 75% accuracy. Wide-corner gating and LOS-relative thresholds
need empirical validation before automatic dataset population is trustworthy.

## Verification for this pass

* `python -m unittest discover -s tests -v`: 62 tests passed, including existing
  identity/occlusion checks and new shell rejection, rotation, and window checks.
* Synthetic CLI example returned `2_high` → `1_high`, `2_to_1` with explanations.
* A real `pass_001.mp4` headless smoke run with an illustrative `--snap 5`
  processed frames 120–224 inclusive (105 frames at 29.97 FPS) using local
  `yolo11n.pt` full-player weights. This verifies bounded execution, not snap
  timing or shell accuracy. Diagnostic CSV: `outputs/shell_window_smoke.csv`.
* Review queue contains all 30 available clips. Original videos were not modified.
