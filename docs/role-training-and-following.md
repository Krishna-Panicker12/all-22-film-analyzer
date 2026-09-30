# Role training and receiver following

The first role pilot trained a small context-crop CNN for 60 CPU epochs on
73 candidate role points from nine plays. Whole plays were held out: validation
uses pass_004, pass_007, pass_011 (25 points); test uses pass_009, pass_012,
pass_015 (22 points). Pre/post frames of a play never cross splits. All footage
is from one game. Exact image hashes and sample records are in the run manifest.

The checkpoint selected by validation loss scored 60% validation agreement and
50% held-out test agreement with the weak labels. Always predicting the most
common training class (CB) scores 45.5% on test. Test recall is S 100% (6/6),
CB 40% (4/10), LB 16.7% (1/6). This is an experimental baseline, not a reliable
role detector, and it has not replaced any default model. Labels remain
unverified and excluded from the strict supervised detector export pipeline.

Artifacts: `outputs/training/role_pilot_v1/best.pt`, `training.json`, and
`manifest.json`. The CNN classifies a supplied player point using surrounding
image context. It does not locate players or predict coverage. User diagrams
were retained as references and excluded from pixel training.

Reproduce with a new output folder:

```powershell
.\.venv\Scripts\python.exe -m tools.train_role_pilot --output outputs/training/role_pilot_v2 --epochs 60 --allow-unverified
```

Use `python -m tools.predict_roles --help` to classify supplied points in an
original image. Scores are uncalibrated. The test split must remain untouched
when choosing architecture, thresholds, or new training settings; obtain another
game and verified labels for subsequent model selection and final evaluation.

## Receiver-following evidence

The real-clip run is saved at `outputs/receiver_following_pass001_v1/`, including
the tracking video, snapshots, and per-frame offensive/defensive tracks. It
returned unknown: neither of the two selected corner IDs survived to the post
snapshot, and no valid sustained follow was established. This run verifies the
pipeline executes, not that following detection is accurate on real footage.
`tools.replay_following` can recompute evidence and diagnostics from saved tracks.

The defense runner now also tracks opposite-uniform offensive players with an
independent player tracker. Tracks share the defense's camera compensation and
local field basis. They are saved as `receivers` in `tracks.jsonl`, but are only
receiver candidates: releasing blockers can still pass the movement filter.

Following requires at least one second of comparable feet tracks, 70% temporal
coverage, no gap over 0.2 seconds, a downfield release, defender movement,
correlated 0.2-second motion, and sustained proximity with limited separation
variation. Ambiguous competing receivers and shared-receiver conflicts are
reported rather than forced into pairings. Selected corner IDs scope the check.
Failed camera registration, missing IDs, and stationary players yield unknown.

`receiver_following` stores pair evidence and `coverage_stages.corner_refinement`
uses it. Following retains man and match-zone alternatives; failure to follow
does not establish zone. The legacy `coverage` field remains the older
displacement-only estimate for compatibility. Use `coverage_stages` for the new
combined interpretation. Set `receiver_following_enabled: false` in a per-play
config to omit the extra offensive detector pass.

All 15 pass clips now have provisional runtime geometry and estimated snap
configs. New snap estimates use visually reviewed 15-frame brackets with
eight-frame uncertainty. Geometry is an approximate local image basis, not a
calibrated homography. These configs enable analysis but do not verify its labels.
The default analysis endpoint is 2.3 seconds after the estimated snap, with a
manual final-frame hold in interactive mode. Q advances and Esc quits the batch.

Next useful dataset work: verify existing role points (especially linebackers),
label offensive receiver identities through routes and cuts, and label defender
assignment follows/exchanges over time. Gather another game's examples for an
independent evaluation. Current following thresholds are heuristics; no learned
following classifier or accuracy claim is made.
