# Defensive-shell references

Store reference images and short clips here, separate from full play clips.
The first pass-only collection contains 15 original 1280 × 720 PNG frames:
9 one-high, 3 two-high, 1 zero-high, and 2 unknown. Open [the visual index](index.html)
to inspect every image and its rationale. Three-high and rotation folders remain
empty because this pass labels only pre-snap structure.

A subsequent coverage review added [post-snap coverage references](coverages/README.md)
under `coverages/`: 30 original snapshots and 15 supporting movement sequences.
Coverage candidates are labeled separately from the pre-snap shell folders.
Use the [coverage gallery](coverages/index.html) to inspect them.

These are agent-labeled training candidates, not human-verified ground truth.
The two unknown frames (pass_006 and pass_011) have shallow/asymmetric structures
and are held out of supervised training. Staggered two-high candidates pass_001
and pass_015 especially warrant review. The lone zero-high example is goal-line
footage; collect conventional field-position examples before generalizing it.

```text
zero_high/     pre_snap/ and examples/
one_high/      pre_snap/ and examples/
two_high/      pre_snap/ and examples/
three_high/    pre_snap/ and examples/ (reserved; outside MVP)
unknown/      ambiguous frames or candidates awaiting review
rotations/
  stay_1/
  stay_2/
  two_to_one/
  one_to_two/
  two_to_zero/
  unknown_rotation/
```

Use `<game_id>__<play_id>_presnap.png` and
`<game_id>__<play_id>_rotation.mp4` to preserve provenance across games.
Store paths relative to the repository in metadata. Do not move original clips.

Start with `data/metadata/shell_review_queue.csv`: 15 pass clips are listed
with offense, defense, and the original tracking marker. Agent-selected
`pre_snap_frame_seconds` are filled in; exact `snap_time_seconds` remain unverified.
The tracking marker alone cannot establish that a frame is pre-snap or complete.
Export a clean frame using the viewer's S control (original plus overlay in
`outputs/review/`), then copy its original image into the appropriate folder.
Use the viewer's `--snap` and `--save-video` for short annotated rotation clips.

Record extracted references in `data/metadata/defensive_shell_labels.csv`.
Use canonical labels `0_high`, `1_high`, `2_high`, `unknown` in every shell
column; folder names are the human-readable equivalents. Rotation values are
`stay_1`, `stay_2`, `2_to_1`, `1_to_2`, `2_to_0`, `unknown_rotation`.
Keep `predicted_shell`, `pre_snap_shell`, `post_snap_shell`, and `rotation` as
preliminary model/agent evidence. Use `human_verified=false` initially and leave
`final_shell`, `final_post_snap_shell`, and `final_rotation` empty until reviewed.
Only human-verified final columns may serve as ground truth. Never overwrite
verified labels with a prediction. Confidence is a workflow score, not calibrated.

`data/metadata/shell_frame_selections.json` records the visual curation decisions.
The label CSV also contains `source_clip`, zero-based `frame_index`, `source_fps`,
`source_sha256`, `image_sha256`, `label_source`, and `training_status`. PNG images
contain no added labels, boxes, resizing, or crops. One frame per play avoids
inflating the sample count with adjacent near-duplicates. This single-game,
imbalanced starter set is not sufficient for held-out model evaluation; keep
all future frames from a play/game in the same split.

Unknown is appropriate for hidden safeties, cuts, detector failure, or unclear
alignment. Do not infer Cover 0/1/2/3/4/6 from a shell. Three-high folders are
reserved for future work; the MVP baseline returns unknown for that structure.

Initial collection targets: 30 each of one/two-high, 10–20 zero-high, 10 unknown;
then 20 each of two-to-one, stay-two, and stay-one. Separate games across future
training and evaluation splits, including all frames from the same play.
