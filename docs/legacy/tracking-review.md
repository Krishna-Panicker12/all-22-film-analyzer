# Tracking review — 2026-09-29

## Update: automatic boundaries and correction labels

This update supersedes the earlier pre-tracker filtering implementation described
below. All detections now continue through association. The boundary filters
tracked display/export only, so boundary changes do not starve tracks and force
them to restart with IDs that bypass manual exclusions. The regression test keeps
an excluded object outside the polygon for 35 frames, then expands the polygon:
its identity remains unchanged and it stays hidden until U restores it.

Automatic sideline detection checks long, shallow image edges for turf on the
inside and less turf outside. It requires opposite near/far candidates, plausible
separation, and agreement in two consecutive frames. This is an OpenCV heuristic,
not online learning. A single visible sideline is shown as a candidate only. Once
both have been acquired, the boundary follows camera motion until tracking fails
or a new reliable pair updates it. Manual F overrides automation; A re-enables it.

Full-clip checks: `pass_001` first acquired the pair at frame 374; 57 of 449 frames
had an active automatic boundary. Frames 135/148 had no candidates; frame 355 had
only a far-side candidate. Inspected previews at 378, 414 and 426 show the field
boundary following the sideline as the camera moves. `run_001` had no reliable
pair in its 390 frames and therefore did not enable automatic filtering. These
are limited checks on two clips, not a claim of general sideline accuracy.

Manual labeling is available with L and drag boxes: player, non-player or uncertain.
S saves labels separately from detector/tracker output; V explicitly marks a frame
complete. The dataset exporter accepts only complete labels, rejects uncertain
regions and same-game validation splits, and never converts predictions to truth.
No detector weights have been trained or changed in this update. Linemen and
overlapping tackles remain a detector-training problem. Actual track identity
switches can still require exclusion of a new ID.

Twelve automated tests pass, covering the prior tests plus boundary/exclusion
continuity, paired automatic detection, manual overrides, annotation reset and
safe training export. The following sections retain the earlier baseline results.

Source: `data/plays/game_01/pass_001.mp4`, frames 120–427 processed in order.
Screenshot checkpoints: 135 (4.50 s), 306 (10.21 s), 378 (12.61 s), 427 (14.25 s).
The second comparison uses `run_001.mp4`, frames 90–180.

## Controlled settings comparison

All variants use 1280 inference size, BoT-SORT, no field filtering and no manual
exclusions. Each variant changes one setting from the baseline.

| Variant | Pass clip: distinct returned IDs | Run clip: distinct returned IDs |
|---|---:|---:|
| Nano, confidence .25, buffer 30 | 175 | 39 |
| Confidence .10 | 164 | 36 |
| Small model instead of nano | 135 | 23 |
| Buffer 60 | 174 | 38 |

These are counts of distinct IDs actually returned, not the highest allocated ID,
not ID-switch counts, and not accuracy. Lower counts can result from missed or
merged players. No ground-truth identity annotations were available. Complete
missed-player, duplicate, and identity-switch rates have therefore not been claimed.

On the pass clip, reducing confidence increased mean raw detections from 27.25
to 43.92 per frame. The larger model averaged 24.34 raw detections. Approximate
processing throughput was 8.1 FPS for nano and 3.0 FPS for small, including startup;
some comparisons overlapped on the CPU, so these are not controlled speed benchmarks.

Visual review of the checkpoint images:

- Frame 135: both models struggle with the crowded offensive interior. The small
  model removes the nano model's false detection on the black camera equipment.
- Frame 306: the small model gives a separate box to a red-shirted player in the
  central blocking engagement that the nano tracked output misses.
- Frame 378: the small model misses the isolated red-shirted defender near the
  midfield logo, while the nano output includes him. Both still enclose the
  overlapping pair near the near sideline in one box.
- Frame 427: the small model misses the tackle; nano produces one combined box.
  Both detect many sideline people without filtering.

Decision: keep the conservative nano/.25/30 defaults. Expose the alternatives
for review, but do not claim the larger model or lower threshold solves occlusion.

## Implemented review improvements

- Raw detector output and tracked observations come from the same inference.
- Raw mode keeps rejected people visible for diagnosis; tracked mode filters them.
- Short labels, optional confidence and trails; header no longer covers field pixels.
- Editable convex field polygon follows approximate background camera motion.
- Motion failure visibly disables the polygon instead of leaving it frozen.
- Right-click exclusions are per-session track IDs; reset clears them.
- Missing tracks have no measured position and export empty coordinates.
- Snapshots save clean frames, overlays and structured observations.
- Export identity includes a segment number so IDs reused after reset are distinct.

## Validation

Six automated tests pass: metadata offsets; occlusion/reacquisition with the real
ByteTrack implementation; filtering before association while preserving raw
detections; exclusion/restore and snapshots without advancing tracking; synthetic
camera translation and failure handling; invalid boundaries and non-mutating drawing.

The actual viewer opened and processed five frames. Its exported MP4 decoded as
five frames of 1280×832 (720 image rows plus 112 header rows).

A separate real-clip field test processed frames 378–427 continuously. The polygon
remained active for all 50 frames. Visual inspection at frame 427 showed removal of
the bottom sideline boxes while retaining the tackle detection. Its box still spans
two players. This is a local sanity check, not proof that boundaries never drift or
that all active players are retained across other views.

Detailed images, per-frame CSVs and summaries are under `outputs/benchmark_pass001/`
and `outputs/benchmark_run001/`. Field snapshots are under `outputs/field_validation/`.

## Remaining work

### User review checkpoints (saved snapshots)

All five records below were located in `outputs/review/` and their overlays
inspected. Times are clip-relative; IDs refer only to this recorded run.

| Frame | Seconds | User report | Snapshot evidence |
|---|---:|---|---|
| 141 | 4.705 | Suspended camera mistaken for a player | Camera boxed as ID 20. No field polygon selected. This is a person-detector false positive inside the projected field area. |
| 148 | 4.938 | Right tackle, tight end, right guard missing; left guard not tracked | Several crowded offensive players lack separate reliable boxes. Role names are user annotations, not verified model labels. No polygon selected; only ID 20 excluded. |
| 355 | 11.845 | Red defender behind ID 382 and defensive linemen obscured | Bottom overlapping pair enclosed by ID 382; crowded line also has overlapping boxes. No polygon selected, so field filtering did not cause these omissions. |
| 414 | 13.814 | Patriots player and tackler not tracked | Neither has a displayed box at the tackle, which lies well inside the selected boundary. |
| 426 | 14.214 | ID 758 represents two players | A single tracked box spans both grounded players, well inside the selected field area. |

These are failure-review notes, not training labels or independently verified
assignment/position labels. The suspended camera needs a negative training example
or explicit exclusion; a field polygon cannot distinguish an airborne object whose
image position projects inside the field. The line and tackle cases need separate
visible-player annotations and temporal review, not a tighter field boundary.

The saved raw detections at frame 414 contain no box intersecting the tackle
region (x=500–620, y=380–480 in original-image coordinates). This locates the
failure upstream of tracking, including possible detector confidence/NMS filtering.
At frame 426, raw output already contains one combined box at approximately
(512,392)–(592,444), confidence 0.266, corresponding to tracked ID 758. The tracker
did not merge two separate raw boxes at that location in this saved frame.

Create manually reviewed player boxes and identities around these checkpoints.
Fine-tuning needs football examples of crouched linemen, contact and partial
occlusion. Fully hidden players cannot supply observed positions from this angle.
Automatic team/referee classification, identity correction across gaps, calibrated
field coordinates and reliable multi-angle association are not implemented.
